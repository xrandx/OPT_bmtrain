import torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import bmtrain as bmt
from typing import Optional, Tuple


from opt_config import OPTConfig


class SelfAttention(nn.Module):
    def __init__(self, config: OPTConfig, use_cache=False, dtype=torch.half):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = config.hidden_size
        self.use_cache = use_cache
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size_per_attention_head = config.hidden_size // config.num_attention_heads
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.q_proj = LinearNoInit(
            config.hidden_size,
            config.hidden_size,
            dtype=dtype,
        )
        self.k_proj = LinearNoInit(
            config.hidden_size,
            config.hidden_size,
            dtype=dtype,
        )
        self.v_proj = LinearNoInit(
            config.hidden_size,
            config.hidden_size,
            dtype=dtype,
        )
        self.out_proj = LinearNoInit(
            config.hidden_size,
            config.hidden_size,
            dtype=dtype,
        )

    def forward(self, hidden_states, attention_mask, layer_past=None):
        has_layer_past = layer_past is not None and layer_past.numel() > 0
        q_seq_len, batch_size, hidden_dim = hidden_states.shape

        # [sq, b, np, hn]
        query_layer = self.q_proj(hidden_states).reshape(
            q_seq_len, batch_size, self.num_attention_heads, self.hidden_size_per_attention_head
        )
        query_layer /= self.norm_factor
        key_layer = self.k_proj(hidden_states).reshape(
            q_seq_len, batch_size, self.num_attention_heads, self.hidden_size_per_attention_head
        )
        value_layer = self.v_proj(hidden_states).reshape(
            q_seq_len, batch_size, self.num_attention_heads, self.hidden_size_per_attention_head
        )

        # Cache QKV values
        if has_layer_past:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)
        if self.use_cache:
            kv_cache = torch.stack((key_layer, value_layer))
        else:
            kv_cache = None

        # Compute attention
        # noinspection PyTypeChecker
        context_layer = self.attention(
            query_layer, key_layer, value_layer, attention_mask
        )

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================
        output = self.out_proj(context_layer)

        return output, kv_cache

    # noinspection PyMethodMayBeStatic
    def attention(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2],
            output_size[0] * output_size[1],
            -1
        )
        key_layer = key_layer.view(
            output_size[3],
            output_size[0] * output_size[1],
            -1,
        )

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=1.0,
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        masked_scores = attention_mask_func(attention_scores, attention_mask, self.dtype) \
            if attention_mask is not None else attention_scores
        # noinspection PyTypeChecker
        attention_probs = nn.functional.softmax(
            masked_scores, dim=-1, dtype=torch.half).to(masked_scores.dtype)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer


class MLP(nn.Module):
    def __init__(self, config: OPTConfig, dtype=torch.half):
        super().__init__()
        self.dense_h_to_4h = LinearNoInit(config.hidden_size, config.ffn_dim, dtype=dtype)
        self.dense_4h_to_h = LinearNoInit(config.ffn_dim, config.hidden_size, dtype=dtype)

    def forward(self, hidden_states):
        hidden_states_shape = hidden_states.shape
        intermediate_parallel = self.dense_h_to_4h(
            hidden_states.view(-1, hidden_states_shape[-1])
        )
        intermediate_parallel = F.relu(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output.view(*hidden_states_shape)


class TransformerLayer(nn.Module):
    def __init__(self, config: OPTConfig, use_cache, dtype=torch.half):
        super().__init__()
        self.use_cache = use_cache
        self.input_layernorm = LayerNormNoInit(
            config.hidden_size,
            dtype=dtype,
        )
        self.post_attention_layernorm = LayerNormNoInit(
            config.hidden_size,
            dtype=dtype,
        )
        self.attention = SelfAttention(config, self.use_cache, dtype=dtype)
        self.mlp = MLP(config, dtype=dtype)


    def forward(self, hidden_states, attention_mask, layer_past=None):
        residual = hidden_states
        ln_output = self.input_layernorm(hidden_states)

        hidden_states, kv_cache = self.attention(
            ln_output,
            attention_mask,
            layer_past=layer_past,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        if self.use_cache:
            return hidden_states, kv_cache
        else:
            return hidden_states
            

class EmbeddingNoInit(bmt.DistributedModule):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                sparse: bool = False, _weight: Optional[torch.Tensor] = None,
                dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = bmt.DistributedParameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype), init_method=bmt.ParameterInitializer(torch.nn.init.normal_))
        else:
            self.weight = bmt.DistributedParameter(_weight)
        
        self.sparse = sparse
    
    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):
        r"""Creates Embedding instance from given 2-dimensional FloatTensor.

        Args:
            embeddings (Tensor): FloatTensor containing weights for the Embedding.
                First dimension is being passed to Embedding as ``num_embeddings``, second as ``embedding_dim``.
            freeze (boolean, optional): If ``True``, the tensor does not get updated in the learning process.
                Equivalent to ``embedding.weight.requires_grad = False``. Default: ``True``
            padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;
                                         therefore, the embedding vector at :attr:`padding_idx` is not updated during training,
                                         i.e. it remains as a fixed "pad".
            max_norm (float, optional): See module initialization documentation.
            norm_type (float, optional): See module initialization documentation. Default ``2``.
            scale_grad_by_freq (boolean, optional): See module initialization documentation. Default ``False``.
            sparse (bool, optional): See module initialization documentation.

        Examples::

            >>> # FloatTensor containing pretrained weights
            >>> weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            >>> embedding = nn.Embedding.from_pretrained(weight)
            >>> # Get embeddings for index 1
            >>> input = torch.LongTensor([1])
            >>> embedding(input)
            tensor([[ 4.0000,  5.1000,  6.3000]])
        """
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding

    def forward(self, input: torch.Tensor, projection : bool = False) -> torch.Tensor:
        if not projection:
            return F.embedding(
                input, self.weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse)
        else:
            return F.linear(input, self.weight) / math.sqrt(self.embedding_dim)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    def reset_parameters(self):
        pass


class LayerNormNoInit(bmt.DistributedModule):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                dtype=None) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = bmt.DistributedParameter(torch.empty(self.normalized_shape, dtype=dtype), init_method=bmt.ParameterInitializer(torch.nn.init.ones_))
            self.bias = bmt.DistributedParameter(torch.empty(self.normalized_shape, dtype=dtype), init_method=bmt.ParameterInitializer(torch.nn.init.zeros_))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

    def reset_parameters(self):
        pass


class LinearNoInit(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = bmt.DistributedParameter(torch.empty(out_features, in_features, dtype=dtype), 
        init_method=bmt.ParameterInitializer(torch.nn.init.xavier_normal_))
        if bias:
            self.bias = bmt.DistributedParameter(torch.empty(out_features, dtype=dtype), init_method=bmt.ParameterInitializer(torch.nn.init.zeros_))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def reset_parameters(self):
        pass


class LearnedPositionalEmbedding(EmbeddingNoInit):
    def __init__(self, num_embeddings: int, embedding_dim: int, magic_offset=2, dtype=None):
        self.magic_offset = magic_offset
        super().__init__(num_embeddings, embedding_dim, dtype=dtype)

    # noinspection PyMethodOverriding
    def forward(self, token_embeddings, kv_cache):
        """`attention_mask` is expected to be [bsz x seqlen]."""
        # positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        batch_size, seq_len, _ = token_embeddings.shape
        if kv_cache is None or kv_cache[0] is None:
            positions = torch.arange(seq_len, device=token_embeddings.device)[None].expand(
                batch_size, -1,
            )
        else:
            kv_cache_len = kv_cache[0].shape[1]
            positions = torch.arange(kv_cache_len + seq_len, device=token_embeddings.device)[None].expand(
                batch_size, -1,
            )
            positions = positions[:, kv_cache_len:]

        return super().forward(positions + self.magic_offset)


def generate_mask(seq_len):
    return torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))


def attention_mask_func(attention_scores, ltor_mask, dtype=torch.half):
    """Assign dtype minimum to False cells in ltor_mask"""
    attention_scores.masked_fill_(~ltor_mask, torch.tensor(torch.finfo(dtype).min))
    return attention_scores

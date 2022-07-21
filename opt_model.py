import torch
import torch.nn as nn
import bmtrain as bmt

from opt_config import OPTConfig
from layers import *

class OPTModel(nn.Module):
    def __init__(self, config: OPTConfig, use_cache=False, dtype=torch.float32):
        super().__init__()
        self.config = config
        self.use_cache = use_cache
        self.embed_tokens = EmbeddingNoInit(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype
        )
        self.embed_positions = LearnedPositionalEmbedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            dtype=dtype
        )
        self.layer_list = None
        if not self.use_cache:
            self.layer_list = bmt.TransformerBlockList([
                bmt.CheckpointBlock(
                    TransformerLayer(config, self.use_cache)
                )
                for _ in range(config.num_hidden_layers)
            ])
        else:
            self.layer_list = torch.nn.ModuleList([
                TransformerLayer(config, self.use_cache)
                for _ in range(config.num_hidden_layers)
            ])

        self.final_layernorm = LayerNormNoInit(
            config.hidden_size,
            dtype=dtype
        )

    @classmethod
    def pre_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()

    @classmethod
    def post_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()

    def forward(self, input_ids, attention_mask=None, layer_past=None):

        if attention_mask is None:
            attention_mask = generate_mask(input_ids.shape[1]).to(input_ids.device)
        if self.use_cache:
            if layer_past is None:
                kv_length = input_ids.shape[1]
            else:
                kv_length = layer_past[0].shape[1] + 1
            attention_mask = attention_mask[..., :input_ids.shape[1], :kv_length]

        if layer_past is None:
            layer_past = [None] * len(self.layer_list)
        token_embeddings = self.embed_tokens(input_ids)
        pos_embeddings = self.embed_positions(token_embeddings, kv_cache=layer_past)
        hidden_states = token_embeddings + pos_embeddings
        hidden_states = self.pre_transformer_transpose(hidden_states)

        if not self.use_cache:
            hidden_states = self.layer_list.forward(hidden_states, attention_mask, None)
            hidden_states = self.post_transformer_transpose(hidden_states)
            hidden_states = self.final_layernorm(hidden_states)
        else:
            with torch.no_grad():
                kv_cache_list = []
                for layer_i, layer in enumerate(self.layer_list):
                    hidden_states, kv_cache = layer(hidden_states, attention_mask, layer_past[layer_i])
                    kv_cache_list.append(kv_cache)
                hidden_states = self.final_layernorm(hidden_states)

        logits = self.embed_tokens(hidden_states, projection=True)

        if self.use_cache:
            return logits, None
        else:
            return logits
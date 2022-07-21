from dataclasses import dataclass


@dataclass
class OPTConfig:
    vocab_size: int
    hidden_size: int
    max_position_embeddings: int
    num_attention_heads: int
    head_dim: int
    ffn_dim: int
    num_hidden_layers: int


OPT_125M_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=768,
    max_position_embeddings=2050,
    num_attention_heads=12,
    head_dim=64,
    ffn_dim=3072,
    num_hidden_layers=12,
)

OPT_1_3B_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=2048,
    max_position_embeddings=2050,
    num_attention_heads=32,
    head_dim=64,
    ffn_dim=8192,
    num_hidden_layers=24,
)

OPT_2_7B_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=2560,
    max_position_embeddings=2050,
    num_attention_heads=32,
    head_dim=80,
    ffn_dim=10240,
    num_hidden_layers=32,
)

OPT_6_7B_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=4096,
    max_position_embeddings=2050,
    num_attention_heads=32,
    head_dim=128,
    ffn_dim=16384,
    num_hidden_layers=32,
)

OPT_13B_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=5120,
    max_position_embeddings=2050,
    num_attention_heads=40,
    head_dim=128,
    ffn_dim=20480,
    num_hidden_layers=40,
)

OPT_30B_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=7168,
    max_position_embeddings=2050,
    num_attention_heads=56,
    head_dim=128,
    ffn_dim=28672,
    num_hidden_layers=48,
)


OPT_66B_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=9216,
    max_position_embeddings=2050,
    num_attention_heads=72,
    head_dim=128,
    ffn_dim=36864,
    num_hidden_layers=64,
)


OPT_175B_CONFIG = OPTConfig(
    vocab_size=50272,
    hidden_size=12288,
    max_position_embeddings=2050,
    num_attention_heads=96,
    head_dim=128,
    ffn_dim=49152,
    num_hidden_layers=96,
)

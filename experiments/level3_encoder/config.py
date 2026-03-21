"""
Level 3 Configuration — All hyperparameters in one place.

SmolLM2-1.7B-Instruct specs:
  24 layers, 32 KV heads, d_head=64, vocab=49152
  Chat template: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Frozen LLM configuration."""

    model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    dtype: str = "float16"  # "float16" or "bfloat16"
    num_layers: int = 24
    num_kv_heads: int = 32
    head_dim: int = 64
    hidden_size: int = 2048  # num_kv_heads * head_dim


@dataclass
class EncoderConfig:
    """Memory Encoder architecture."""

    # Encoder transformer
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 2
    dropout: float = 0.1
    max_profile_tokens: int = 256

    # Perceiver-style resampling
    num_memory_slots: int = 16  # M — number of output memory vectors

    # Layer-group projections (encoder d_model → LLM KV space)
    num_layer_groups: int = 4  # groups of 6 layers each for SmolLM2-24L
    # Each projection: d_model → num_kv_heads * head_dim = 2048

    # Per-head gating
    gate_init_bias: float = -2.0  # sigmoid(-2) ≈ 0.12 — start near-zero


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Data
    num_train_profiles: int = 1000
    num_val_profiles: int = 200
    num_query_templates: int = 50

    # Optimization
    batch_size: int = 16  # per-GPU; increase on H100/A100
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    eval_every: int = 250
    save_every: int = 500
    max_new_tokens: int = 128  # for gold/inject generation

    # Loss weights
    lambda_distill: float = 1.0
    lambda_kv_align: float = 0.1  # L2 between encoder K,V and gold K,V

    # Misc
    seed: int = 42
    num_workers: int = 0  # dataloader workers


@dataclass
class Config:
    """Top-level config."""

    model: ModelConfig = field(default_factory=ModelConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    device: str = "cuda"
    output_dir: str = "experiments/level3_encoder/outputs"
    log_dir: str = "experiments/level3_encoder/logs"

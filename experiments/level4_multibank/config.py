"""
Level 4 Configuration — Multi-Bank Query-Conditioned Memory Encoder.

Architecture:
  5 memory banks (L4-L8) with query-conditioned dynamic gating,
  working memory cross-attention, and per-layer KV injection.

  Llama-3.1-8B-Instruct specs:
    32 layers, 8 KV heads (GQA), d_head=128, vocab=128256
    Chat template: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n...

Target: ~75M trainable params. Upper bound: 200M.
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Frozen LLM configuration."""

    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    dtype: str = "float16"
    num_layers: int = 32
    num_kv_heads: int = 8  # GQA: 32 attn heads, 8 KV heads
    head_dim: int = 128
    hidden_size: int = 4096


@dataclass
class MemoryBankConfig:
    """Per-bank slot counts. Total slots: 72."""

    l4_episodic_slots: int = 32  # time-stamped events, experiences
    l5_semantic_slots: int = 16  # stable facts, knowledge
    l6_procedural_slots: int = 8  # behavioral patterns, style
    l7_emotional_slots: int = 8  # valence associations
    l8_prospective_slots: int = 8  # future intents, pending tasks

    @property
    def total_slots(self) -> int:
        return (
            self.l4_episodic_slots
            + self.l5_semantic_slots
            + self.l6_procedural_slots
            + self.l7_emotional_slots
            + self.l8_prospective_slots
        )

    @property
    def bank_sizes(self) -> list[int]:
        return [
            self.l4_episodic_slots,
            self.l5_semantic_slots,
            self.l6_procedural_slots,
            self.l7_emotional_slots,
            self.l8_prospective_slots,
        ]

    @property
    def bank_names(self) -> list[str]:
        return ["episodic", "semantic", "procedural", "emotional", "prospective"]


@dataclass
class EncoderConfig:
    """Multi-bank encoder architecture."""

    # Shared dimensions
    d_model: int = 768
    n_heads: int = 8
    dropout: float = 0.1

    # Profile encoder
    profile_encoder_layers: int = 4
    max_profile_tokens: int = 384

    # Query encoder
    query_encoder_layers: int = 2
    max_query_tokens: int = 128

    # Working memory output
    num_output_slots: int = 32  # M — memory vectors after selection

    # Layer-group projections
    num_layer_groups: int = 4  # groups of 8 layers for Llama-32L

    # Dynamic gate MLP
    gate_hidden_dim: int = 1536
    gate_init_bias: float = -1.0  # sigmoid(-1) ≈ 0.27 — start gates meaningfully open


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Data
    num_train_profiles: int = 2000
    num_val_profiles: int = 400

    # Optimization
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 10000
    eval_every: int = 500
    save_every: int = 1000
    max_new_tokens: int = 128

    # Loss weights
    lambda_distill: float = 1.0
    lambda_gate: float = 1.0        # gate utilization penalty
    lambda_kv_norm: float = 0.1     # KV magnitude matching
    lambda_entropy: float = 0.01    # gate entropy bonus (prevent binary snap)
    gate_target: float = 0.3        # target minimum gate mean

    # Misc
    seed: int = 42
    num_workers: int = 0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    banks: MemoryBankConfig = field(default_factory=MemoryBankConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cuda"
    output_dir: str = "outputs/level4_multibank"
    log_dir: str = "logs/level4_multibank"

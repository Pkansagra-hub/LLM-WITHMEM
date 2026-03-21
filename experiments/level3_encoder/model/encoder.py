"""
Memory Encoder — the trainable side-model that produces K,V pairs.

Architecture:
  1. Embed profile tokens using the frozen LLM's embedding layer
  2. Encode with a small transformer (2-layer, d=512, 8 heads)
  3. Perceiver-style resampling: M learned queries cross-attend into encoder output
  4. Project each memory vector to K,V for each layer group
  5. Per-head gating (sigmoid scalars, initialized near zero)

Output: K,V tensors for all LLM layers, ready for injection.
"""

import math

import torch
import torch.nn as nn


class PerceiverResampler(nn.Module):
    """M learned query vectors that cross-attend into a variable-length sequence.

    Input:  (batch, seq_len, d_model) — encoder output
    Output: (batch, M, d_model) — fixed-size memory vectors
    """

    def __init__(
        self, d_model: int, n_heads: int, num_queries: int, dropout: float = 0.1
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, encoder_out: torch.Tensor, encoder_mask: torch.Tensor | None = None
    ):
        """
        Args:
            encoder_out: (batch, seq_len, d_model)
            encoder_mask: (batch, seq_len) — True where padded
        """
        batch = encoder_out.shape[0]
        queries = self.queries.expand(batch, -1, -1)

        # Cross-attention: queries attend to encoder output
        attn_out, _ = self.cross_attn(
            query=queries,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_mask,
        )
        x = self.norm(queries + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x  # (batch, M, d_model)


class MemoryEncoder(nn.Module):
    """Full memory encoder: profile text → K,V pairs for all LLM layers.

    Args:
        embedding_layer: Frozen LLM embedding layer (shared, no grad)
        llm_embed_dim: LLM's embedding dimension (e.g. 2048 for SmolLM2)
        d_model: Encoder's internal dimension
        n_heads: Encoder's attention heads
        n_layers: Encoder transformer layers
        num_memory_slots: M — output memory vectors
        num_llm_layers: Total layers in the frozen LLM
        num_kv_heads: KV heads per layer in the frozen LLM
        head_dim: Dimension per head in the frozen LLM
        num_layer_groups: How many groups to share projections across
        gate_init_bias: Initial bias for gating (negative = start near zero)
        dropout: Dropout rate
    """

    def __init__(
        self,
        embedding_layer: nn.Embedding,
        llm_embed_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        num_memory_slots: int = 16,
        num_llm_layers: int = 24,
        num_kv_heads: int = 32,
        head_dim: int = 64,
        num_layer_groups: int = 4,
        gate_init_bias: float = -2.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_llm_layers = num_llm_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_layer_groups = num_layer_groups
        self.num_memory_slots = num_memory_slots

        # ── Input projection (LLM embed dim → encoder dim) ──
        self.embedding = embedding_layer  # frozen, shared with LLM
        self.input_proj = nn.Linear(llm_embed_dim, d_model)

        # ── Encoder transformer ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Perceiver resampler ──
        self.resampler = PerceiverResampler(
            d_model=d_model,
            n_heads=n_heads,
            num_queries=num_memory_slots,
            dropout=dropout,
        )

        # ── Layer-group K,V projection heads ──
        # Each group projects d_model → (num_kv_heads * head_dim) for K and V
        kv_dim = num_kv_heads * head_dim
        self.proj_k = nn.ModuleList(
            [nn.Linear(d_model, kv_dim) for _ in range(num_layer_groups)]
        )
        self.proj_v = nn.ModuleList(
            [nn.Linear(d_model, kv_dim) for _ in range(num_layer_groups)]
        )

        # ── Per-head gates ──
        # One scalar per (layer, head), initialized so sigmoid(gate) ≈ 0.12
        self.gates = nn.Parameter(
            torch.full((num_llm_layers, num_kv_heads), gate_init_bias)
        )

        # Precompute layer → group mapping
        layers_per_group = math.ceil(num_llm_layers / num_layer_groups)
        self.register_buffer(
            "layer_to_group",
            torch.tensor(
                [
                    min(i // layers_per_group, num_layer_groups - 1)
                    for i in range(num_llm_layers)
                ]
            ),
        )

    def forward(
        self,
        profile_input_ids: torch.Tensor,
        profile_attention_mask: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            profile_input_ids: (batch, seq_len) — tokenized profile
            profile_attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            List of (K, V) tuples, one per LLM layer.
            K shape: (batch, num_kv_heads, M, head_dim)
            V shape: (batch, num_kv_heads, M, head_dim)
        """
        batch = profile_input_ids.shape[0]

        # 1. Embed with frozen LLM embeddings (no grad through embedding)
        with torch.no_grad():
            token_embeds = self.embedding(
                profile_input_ids
            )  # (batch, seq, llm_embed_dim)
        # Detach + cast to encoder dtype (fp32 for stable training)
        token_embeds = token_embeds.detach().float()

        # 2. Project to encoder dimension
        x = self.input_proj(token_embeds)  # (batch, seq, d_model)

        # 3. Encode with transformer
        # TransformerEncoder expects src_key_padding_mask: True where PADDED
        padding_mask = profile_attention_mask == 0  # True = padded position
        x = self.encoder(x, src_key_padding_mask=padding_mask)  # (batch, seq, d_model)

        # 4. Perceiver resampling → M fixed-size memory vectors
        memory = self.resampler(x, encoder_mask=padding_mask)  # (batch, M, d_model)

        # 5. Project to K,V for each layer group, apply per-head gates
        gate_values = torch.sigmoid(self.gates)  # (num_llm_layers, num_kv_heads)

        kv_pairs = []
        for layer_idx in range(self.num_llm_layers):
            group_idx = self.layer_to_group[layer_idx].item()

            # Project: (batch, M, d_model) → (batch, M, num_kv_heads * head_dim)
            k = self.proj_k[group_idx](memory)
            v = self.proj_v[group_idx](memory)

            # Reshape to (batch, M, num_kv_heads, head_dim) → (batch, num_kv_heads, M, head_dim)
            k = k.view(batch, self.num_memory_slots, self.num_kv_heads, self.head_dim)
            k = k.transpose(1, 2)
            v = v.view(batch, self.num_memory_slots, self.num_kv_heads, self.head_dim)
            v = v.transpose(1, 2)

            # Apply per-head gate: (num_kv_heads,) → (1, num_kv_heads, 1, 1)
            gate = gate_values[layer_idx].view(1, self.num_kv_heads, 1, 1)
            k = k * gate
            v = v * gate

            # Cast to LLM dtype (fp16) — .half() preserves autograd
            k = k.half()
            v = v.half()

            kv_pairs.append((k, v))

        return kv_pairs

    def param_count(self) -> dict:
        """Count trainable parameters by component."""
        counts = {}
        counts["input_proj"] = sum(p.numel() for p in self.input_proj.parameters())
        counts["encoder"] = sum(p.numel() for p in self.encoder.parameters())
        counts["resampler"] = sum(p.numel() for p in self.resampler.parameters())
        counts["proj_k"] = sum(p.numel() for p in self.proj_k.parameters())
        counts["proj_v"] = sum(p.numel() for p in self.proj_v.parameters())
        counts["gates"] = self.gates.numel()
        counts["total"] = sum(counts.values())
        return counts

"""
Level 4 Multi-Bank Query-Conditioned Memory Encoder.

Architecture (forward flow):
  1. Profile text → Shared ProfileEncoder (4-layer xfmr) → 5 PerceiverBankHeads → bank slots
  2. Query text  → QueryEncoder (2-layer xfmr) → attention pool → query_vector
  3. query_vector + all bank slots → WorkingMemory cross-attention → M output vectors
  4. query_vector → DynamicGateNetwork MLP → per-layer per-head gates
  5. M output vectors → 4 layer-group K,V projections × gate → inject into LLM

Components:
  PerceiverResampler     — learned-query cross-attention resampler
  ProfileEncoder         — shared backbone + 5 bank-specific Perceiver heads
  QueryEncoder           — transformer encoder + attention pooling → fixed vector
  WorkingMemory          — query-conditioned cross-attention over bank slots
  DynamicGateNetwork     — query → per-head sigmoid gates via MLP
  MultiBankMemoryEncoder — ties everything together, produces gated K,V pairs
"""

import torch
import torch.nn as nn

from ..config import Config

# ──────────────────────────────────────────────────────────────────────
# Building Blocks
# ──────────────────────────────────────────────────────────────────────


class PerceiverResampler(nn.Module):
    """Learned-query cross-attention resampler.

    M learned queries cross-attend into a variable-length encoder output,
    producing a fixed-size (B, M, d_model) representation.
    """

    def __init__(
        self, d_model: int, n_heads: int, num_queries: int, dropout: float = 0.1
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, encoder_output: torch.Tensor, encoder_mask: torch.Tensor | None = None
    ):
        """
        Args:
            encoder_output: (B, T, d_model) — encoded sequence
            encoder_mask: (B, T) — True where padded (key_padding_mask convention)
        Returns:
            (B, M, d_model)
        """
        B = encoder_output.shape[0]
        q = self.queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(
            q, encoder_output, encoder_output, key_padding_mask=encoder_mask
        )
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


# ──────────────────────────────────────────────────────────────────────
# Profile Encoder
# ──────────────────────────────────────────────────────────────────────


class ProfileEncoder(nn.Module):
    """Encodes profile text into 5 bank-specific memory slot tensors.

    Steps:
        1. Project LLM embeddings (hidden_size) → d_model
        2. N-layer transformer encoder (self-attention over full profile)
        3. 5 Perceiver heads, one per bank, resample into bank-specific slots
    """

    def __init__(
        self,
        llm_embed_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        bank_sizes: list[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(llm_embed_dim, d_model)

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

        self.bank_heads = nn.ModuleList(
            [
                PerceiverResampler(d_model, n_heads, num_slots, dropout)
                for num_slots in bank_sizes
            ]
        )

    def forward(self, profile_embeds: torch.Tensor, profile_mask: torch.Tensor):
        """
        Args:
            profile_embeds: (B, P, llm_embed_dim) — detached float32 LLM embeddings
            profile_mask:   (B, P) — 1 = real token, 0 = pad
        Returns:
            list of 5 tensors, shapes [(B, Si, d_model) for Si in bank_sizes]
        """
        x = self.input_proj(profile_embeds)
        pad_mask = profile_mask == 0  # True where padded
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        bank_slots = [head(x, encoder_mask=pad_mask) for head in self.bank_heads]
        return bank_slots


# ──────────────────────────────────────────────────────────────────────
# Query Encoder
# ──────────────────────────────────────────────────────────────────────


class QueryEncoder(nn.Module):
    """Encodes query text into a fixed-size query vector.

    Steps:
        1. Project LLM embeddings → d_model
        2. M-layer transformer encoder
        3. Attention-pool to single (B, d_model) vector
    """

    def __init__(
        self,
        llm_embed_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(llm_embed_dim, d_model)

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

        # Attention pooling: single learned query attends over encoded tokens
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.pool_norm = nn.LayerNorm(d_model)

    def forward(self, query_embeds: torch.Tensor, query_mask: torch.Tensor):
        """
        Args:
            query_embeds: (B, Q, llm_embed_dim) — detached float32 LLM embeddings
            query_mask:   (B, Q) — 1 = real token, 0 = pad
        Returns:
            (B, d_model)
        """
        x = self.input_proj(query_embeds)
        pad_mask = query_mask == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        pool_q = self.pool_query.expand(x.shape[0], -1, -1)
        pooled, _ = self.pool_attn(pool_q, x, x, key_padding_mask=pad_mask)
        query_vector = self.pool_norm(pooled).squeeze(1)  # (B, d_model)
        return query_vector


# ──────────────────────────────────────────────────────────────────────
# Working Memory
# ──────────────────────────────────────────────────────────────────────


class WorkingMemory(nn.Module):
    """Query-conditioned selection over all bank slots.

    M learned output queries, modulated by the input query vector,
    cross-attend over all bank slots (72 total) to produce M composed
    memory vectors.

    The query vector conditions the output queries via a gated projection:
        conditioned_queries = learned_queries + sigmoid(gate_proj(q)) * value_proj(q)
    """

    def __init__(
        self, d_model: int, n_heads: int, num_output_slots: int, dropout: float = 0.1
    ):
        super().__init__()
        self.output_queries = nn.Parameter(
            torch.randn(1, num_output_slots, d_model) * 0.02
        )

        # Query conditioning
        self.gate_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Cross-attention over bank slots
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query_vector: torch.Tensor, all_bank_slots: torch.Tensor):
        """
        Args:
            query_vector:   (B, d_model)
            all_bank_slots: (B, total_slots, d_model)  — concatenated bank slots
        Returns:
            memory_output: (B, M, d_model)
            attn_weights:  (B, M, total_slots) — for diagnostics
        """
        B = query_vector.shape[0]

        # Condition output queries on input query
        gate = torch.sigmoid(self.gate_proj(query_vector)).unsqueeze(1)  # (B, 1, d)
        val = self.value_proj(query_vector).unsqueeze(1)  # (B, 1, d)
        queries = self.output_queries.expand(B, -1, -1) + gate * val

        # Cross-attend over all bank slots
        attn_out, attn_weights = self.cross_attn(
            queries, all_bank_slots, all_bank_slots
        )
        x = self.norm1(queries + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


# ──────────────────────────────────────────────────────────────────────
# Dynamic Gate Network
# ──────────────────────────────────────────────────────────────────────


class DynamicGateNetwork(nn.Module):
    """Produces query-dependent per-head gate values for all LLM layers.

    query_vector → Linear → GELU → Linear → sigmoid → (num_layers, num_heads)
    Output bias initialized so sigmoid starts ≈ 0.12.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        init_bias: float = -2.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers * num_heads),
        )
        # Initialize output bias so sigmoid(init_bias) ≈ 0.12
        with torch.no_grad():
            self.mlp[-1].bias.fill_(init_bias)

    def forward(self, query_vector: torch.Tensor):
        """
        Args:
            query_vector: (B, d_model)
        Returns:
            gate_values: (B, num_layers, num_heads) in [0, 1]
        """
        logits = self.mlp(query_vector)  # (B, num_layers * num_heads)
        gate_values = torch.sigmoid(logits.view(-1, self.num_layers, self.num_heads))
        return gate_values


# ──────────────────────────────────────────────────────────────────────
# Full Encoder
# ──────────────────────────────────────────────────────────────────────


class MultiBankMemoryEncoder(nn.Module):
    """Full multi-bank query-conditioned memory encoder.

    Pipeline:
        (profile_ids, query_ids) → frozen LLM embed
        → ProfileEncoder → 5 bank slot sets
        → QueryEncoder → query_vector
        → WorkingMemory(query_vector, all_slots) → M memory vectors
        → DynamicGateNetwork(query_vector) → per-head gates
        → 4-group K,V projections × gates → num_layers (K, V) pairs for LLM injection
    """

    def __init__(self, embedding_layer: nn.Embedding, config: Config):
        super().__init__()
        mc = config.model
        ec = config.encoder
        bc = config.banks

        self.embedding = embedding_layer  # frozen, not in parameters
        self.num_llm_layers = mc.num_layers
        self.num_kv_heads = mc.num_kv_heads
        self.head_dim = mc.head_dim
        self.num_output_slots = ec.num_output_slots
        kv_dim = mc.num_kv_heads * mc.head_dim

        # --- Sub-modules ---
        self.profile_encoder = ProfileEncoder(
            llm_embed_dim=mc.hidden_size,
            d_model=ec.d_model,
            n_heads=ec.n_heads,
            n_layers=ec.profile_encoder_layers,
            bank_sizes=bc.bank_sizes,
            dropout=ec.dropout,
        )
        self.query_encoder = QueryEncoder(
            llm_embed_dim=mc.hidden_size,
            d_model=ec.d_model,
            n_heads=ec.n_heads,
            n_layers=ec.query_encoder_layers,
            dropout=ec.dropout,
        )
        self.working_memory = WorkingMemory(
            d_model=ec.d_model,
            n_heads=ec.n_heads,
            num_output_slots=ec.num_output_slots,
            dropout=ec.dropout,
        )
        self.gate_network = DynamicGateNetwork(
            d_model=ec.d_model,
            hidden_dim=ec.gate_hidden_dim,
            num_layers=mc.num_layers,
            num_heads=mc.num_kv_heads,
            init_bias=ec.gate_init_bias,
        )

        # --- Layer-group K,V projections ---
        self.num_layer_groups = ec.num_layer_groups
        self.proj_k = nn.ModuleList(
            [nn.Linear(ec.d_model, kv_dim) for _ in range(ec.num_layer_groups)]
        )
        self.proj_v = nn.ModuleList(
            [nn.Linear(ec.d_model, kv_dim) for _ in range(ec.num_layer_groups)]
        )

        # Map each LLM layer to a group index
        layers_per_group = mc.num_layers // ec.num_layer_groups
        self.register_buffer(
            "layer_to_group",
            torch.arange(mc.num_layers) // layers_per_group,
        )

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        profile_ids: torch.Tensor,
        profile_mask: torch.Tensor,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        return_diagnostics: bool = False,
    ):
        """
        Args:
            profile_ids:  (B, P) — tokenized full profile
            profile_mask: (B, P) — 1 = real, 0 = pad
            query_ids:    (B, Q) — tokenized raw query
            query_mask:   (B, Q)
            return_diagnostics: if True, return extended diag dict
        Returns:
            kv_pairs:     list of num_layers (K, V) tuples, each (B, kv_heads, M, head_dim) fp16
            gate_values:  (B, num_layers, num_heads) — always returned for loss computation
            kv_norm:      scalar — mean L2 norm of encoder K,V (for KV norm loss)
        """
        B = profile_ids.shape[0]

        # 1. Embed (frozen LLM embeddings — no grad)
        with torch.no_grad():
            profile_embeds = self.embedding(profile_ids).detach().float()
            query_embeds = self.embedding(query_ids).detach().float()

        # 2. Profile → 5 bank slot sets
        bank_slots = self.profile_encoder(profile_embeds, profile_mask)
        all_slots = torch.cat(bank_slots, dim=1)  # (B, total_slots, d_model)

        # 3. Query → fixed-size vector
        query_vector = self.query_encoder(query_embeds, query_mask)  # (B, d_model)

        # 4. Working memory: query-conditioned selection
        memory_output, wm_attn = self.working_memory(
            query_vector, all_slots
        )  # (B, M, d_model)

        # 5. Dynamic gates
        gate_values = self.gate_network(query_vector)  # (B, num_layers, num_heads)

        # 6. Project to K,V for each LLM layer, apply per-head gates
        kv_pairs = []
        kv_norms = []
        for layer_idx in range(self.num_llm_layers):
            group_idx = self.layer_to_group[layer_idx].item()

            k = self.proj_k[group_idx](memory_output)  # (B, M, kv_dim)
            v = self.proj_v[group_idx](memory_output)

            # Reshape: (B, M, kv_dim) → (B, num_kv_heads, M, head_dim)
            k = k.view(
                B, self.num_output_slots, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)
            v = v.view(
                B, self.num_output_slots, self.num_kv_heads, self.head_dim
            ).transpose(1, 2)

            # Track pre-gate KV norms for norm matching loss
            kv_norms.append(k.float().norm(dim=-1).mean())
            kv_norms.append(v.float().norm(dim=-1).mean())

            # Gate: (B, num_heads) → (B, num_heads, 1, 1) for broadcast
            gate = gate_values[:, layer_idx, :].view(B, self.num_kv_heads, 1, 1)
            k = (k * gate).half()
            v = (v * gate).half()

            kv_pairs.append((k, v))

        # Mean KV norm across all layers (pre-gate, for norm loss)
        kv_norm = torch.stack(kv_norms).mean()

        if return_diagnostics:
            return kv_pairs, gate_values, kv_norm, {
                "wm_attention": wm_attn.detach(),
                "query_vector": query_vector.detach(),
                "bank_slot_shapes": [bs.shape for bs in bank_slots],
            }
        return kv_pairs, gate_values, kv_norm

    # ── diagnostics ──────────────────────────────────────────────────

    def param_count(self) -> dict[str, int]:
        """Parameter breakdown by component."""
        parts = {
            "profile_encoder": sum(
                p.numel() for p in self.profile_encoder.parameters()
            ),
            "query_encoder": sum(p.numel() for p in self.query_encoder.parameters()),
            "working_memory": sum(p.numel() for p in self.working_memory.parameters()),
            "gate_network": sum(p.numel() for p in self.gate_network.parameters()),
            "kv_projections": sum(
                sum(p.numel() for p in proj.parameters())
                for proj in list(self.proj_k) + list(self.proj_v)
            ),
        }
        parts["total"] = sum(parts.values())
        return parts

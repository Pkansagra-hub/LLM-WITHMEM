"""
Training losses for Level 4 Multi-Bank Encoder.

Five loss components prevent gate collapse and ensure quality injection:

  1. KL distillation — match inject logits to gold logits.
  2. Gate utilization (per-layer) — penalize each layer's gates below a target.
     Without this, the optimizer zeroes gates for a free KL minimum.
     Uses linear + squared hinge for stronger gradient pull.
  3. KV norm matching (per-layer) — align encoder KV magnitude per layer
     with natural LLM KV so injected content is on-scale.
  4. Gate entropy bonus — prevent gates from snapping to 0/1 binary,
     encourage smooth query-dependent modulation.
  5. KV cosine imitation — force encoder KV mean directions to match
     the LLM's natural KV directions per (layer, head), ensuring the
     encoder speaks the LLM's internal dialect.
"""

import torch
import torch.nn.functional as F

# ── Component losses ─────────────────────────────────────────────────


def distillation_loss(
    inject_logits: torch.Tensor,
    gold_logits: torch.Tensor,
    suffix_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(gold ∥ inject) over non-pad suffix positions."""
    log_p = F.log_softmax(inject_logits / temperature, dim=-1)
    q = F.softmax(gold_logits / temperature, dim=-1)

    kl = F.kl_div(log_p, q, reduction="none").sum(-1)  # (B, S)
    kl = kl * suffix_mask
    loss = kl.sum() / suffix_mask.sum().clamp(min=1)
    return loss * (temperature**2)


def gate_utilization_loss(
    gate_values: torch.Tensor,
    target: float = 0.3,
) -> torch.Tensor:
    """Per-layer gate utilization penalty.

    Each layer's mean gate should meet the target. Uses linear + squared
    hinge for stronger gradient signal when gates are far below target.

    Args:
        gate_values: (B, num_layers, num_heads) in [0, 1]
        target: desired minimum mean gate activation per layer
    Returns:
        scalar loss
    """
    per_layer_mean = gate_values.mean(dim=(0, 2))  # (num_layers,)
    shortfall = torch.clamp(target - per_layer_mean, min=0.0)
    return shortfall.mean() + shortfall.pow(2).mean()


def kv_norm_loss(
    enc_kv_norms: torch.Tensor,
    gold_kv_norms: torch.Tensor,
) -> torch.Tensor:
    """Per-layer KV norm matching.

    Aligns encoder K,V magnitude per layer with the LLM's natural
    K,V norms, so gated injection doesn't corrupt attention scores.

    Args:
        enc_kv_norms:  (N,) — per-layer K,V norms from encoder (2 per layer)
        gold_kv_norms: (N,) — per-layer K,V norms from LLM gold cache
    Returns:
        scalar loss — MSE across all per-layer norms
    """
    return F.mse_loss(enc_kv_norms, gold_kv_norms.detach())


def gate_entropy_loss(
    gate_values: torch.Tensor,
) -> torch.Tensor:
    """Negative binary entropy of gate values — encourages continuous modulation.

    Maximizing entropy means gates stay near 0.5 rather than snapping
    to 0 or 1, preserving query-dependent selectivity.

    H(g) = -[g log(g) + (1-g) log(1-g)]
    We return -H(g) so minimizing this loss maximizes entropy.

    Args:
        gate_values: (B, num_layers, num_heads) in [0, 1]
    Returns:
        scalar loss (negative entropy — to be minimized)
    """
    g = gate_values.clamp(1e-6, 1 - 1e-6)
    entropy = -(g * g.log() + (1 - g) * (1 - g).log())
    return -entropy.mean()


def kv_cosine_loss(
    enc_means_k: torch.Tensor,
    enc_means_v: torch.Tensor,
    gold_means_k: torch.Tensor,
    gold_means_v: torch.Tensor,
) -> torch.Tensor:
    """KV-space cosine imitation loss.

    Forces encoder KV mean directions to align with the LLM's natural
    KV directions per (layer, head). This ensures the encoder produces
    KV that is compatible with the LLM's internal representation space.

    Args:
        enc_means_k:  (B, num_layers, num_kv_heads, head_dim)
        enc_means_v:  (B, num_layers, num_kv_heads, head_dim)
        gold_means_k: (B, num_layers, num_kv_heads, head_dim)
        gold_means_v: (B, num_layers, num_kv_heads, head_dim)
    Returns:
        scalar loss (1 - mean cosine similarity)
    """
    B, L, H, D = enc_means_k.shape
    enc_k = enc_means_k.reshape(B, L * H, D)
    gold_k = gold_means_k.detach().reshape(B, L * H, D)
    enc_v = enc_means_v.reshape(B, L * H, D)
    gold_v = gold_means_v.detach().reshape(B, L * H, D)

    cos_k = F.cosine_similarity(enc_k, gold_k, dim=-1).mean()
    cos_v = F.cosine_similarity(enc_v, gold_v, dim=-1).mean()
    return 1.0 - (cos_k + cos_v) / 2.0


# ── Combined loss ────────────────────────────────────────────────────


def combined_loss(
    inject_logits: torch.Tensor,
    gold_logits: torch.Tensor,
    suffix_mask: torch.Tensor,
    gate_values: torch.Tensor | None = None,
    enc_aux: dict | None = None,
    gold_aux: dict | None = None,
    lambda_distill: float = 1.0,
    lambda_gate: float = 2.0,
    lambda_kv_norm: float = 0.1,
    lambda_entropy: float = 0.01,
    lambda_kv_cosine: float = 0.5,
    gate_target: float = 0.3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total loss with all components + breakdown for logging."""
    l_distill = distillation_loss(inject_logits, gold_logits, suffix_mask)
    total = lambda_distill * l_distill

    metrics = {
        "loss_distill": l_distill.item(),
    }

    if gate_values is not None:
        l_gate = gate_utilization_loss(gate_values, target=gate_target)
        l_ent = gate_entropy_loss(gate_values)
        total = total + lambda_gate * l_gate + lambda_entropy * l_ent
        metrics["loss_gate"] = l_gate.item()
        metrics["loss_entropy"] = l_ent.item()
        metrics["gate_mean"] = gate_values.mean().item()
        per_layer = gate_values.mean(dim=(0, 2))
        metrics["gate_layer_min"] = per_layer.min().item()
        metrics["gate_layer_max"] = per_layer.max().item()

    if enc_aux is not None and gold_aux is not None:
        l_kv = kv_norm_loss(enc_aux["kv_norms"], gold_aux["kv_norms"])
        total = total + lambda_kv_norm * l_kv
        metrics["loss_kv_norm"] = l_kv.item()

        l_cos = kv_cosine_loss(
            enc_aux["kv_means_k"],
            enc_aux["kv_means_v"],
            gold_aux["kv_means_k"],
            gold_aux["kv_means_v"],
        )
        total = total + lambda_kv_cosine * l_cos
        metrics["loss_kv_cosine"] = l_cos.item()

    metrics["loss_total"] = total.item()
    return total, metrics

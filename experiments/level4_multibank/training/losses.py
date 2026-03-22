"""
Training losses for Level 4 Multi-Bank Encoder.

Three loss components prevent gate collapse:

  1. KL distillation — match inject logits to gold logits.
  2. Gate utilization — penalize gates below a target mean.
     Without this, the optimizer zeroes gates for a free KL minimum.
  3. KV norm matching — align encoder KV magnitude with natural LLM KV
     so that when gates *are* open, the injected content is on-scale.
  4. Gate entropy bonus — prevent gates from snapping to 0/1 binary,
     encourage smooth query-dependent modulation.
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
    """Penalize when mean gate activation falls below target.

    Uses a soft hinge: max(0, target - mean_gate)² so there's zero
    penalty once gates reach the target and smooth gradient below it.

    Args:
        gate_values: (B, num_layers, num_heads) in [0, 1]
        target: desired minimum mean gate activation
    Returns:
        scalar loss
    """
    mean_gate = gate_values.mean()
    shortfall = torch.clamp(target - mean_gate, min=0.0)
    return shortfall ** 2


def kv_norm_loss(
    encoder_kv_norm: torch.Tensor,
    gold_kv_norm: torch.Tensor,
) -> torch.Tensor:
    """Penalize mismatch between encoder KV norms and LLM natural KV norms.

    Encourages the encoder to produce K,V at the same magnitude the LLM
    naturally expects, so gated injection doesn't corrupt attention scores.

    Args:
        encoder_kv_norm: scalar — mean L2 norm of encoder K,V across all layers
        gold_kv_norm:    scalar — mean L2 norm of LLM's own K,V (detached)
    Returns:
        scalar loss
    """
    return F.mse_loss(encoder_kv_norm, gold_kv_norm.detach())


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


# ── Combined loss ────────────────────────────────────────────────────


def combined_loss(
    inject_logits: torch.Tensor,
    gold_logits: torch.Tensor,
    suffix_mask: torch.Tensor,
    gate_values: torch.Tensor | None = None,
    encoder_kv_norm: torch.Tensor | None = None,
    gold_kv_norm: torch.Tensor | None = None,
    lambda_distill: float = 1.0,
    lambda_gate: float = 1.0,
    lambda_kv_norm: float = 0.1,
    lambda_entropy: float = 0.01,
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

    if encoder_kv_norm is not None and gold_kv_norm is not None:
        l_kv = kv_norm_loss(encoder_kv_norm, gold_kv_norm)
        total = total + lambda_kv_norm * l_kv
        metrics["loss_kv_norm"] = l_kv.item()
        metrics["enc_kv_norm"] = encoder_kv_norm.item()
        metrics["gold_kv_norm"] = gold_kv_norm.item()

    metrics["loss_total"] = total.item()
    return total, metrics

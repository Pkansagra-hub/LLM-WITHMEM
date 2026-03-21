"""
Training losses for Level 4 Multi-Bank Encoder.

Primary loss: KL distillation — match inject logits to gold logits.
The gold prompt includes ONLY query-relevant memory types, so KL
naturally teaches the encoder to SELECT relevant memories.
"""

import torch
import torch.nn.functional as F


def distillation_loss(
    inject_logits: torch.Tensor,
    gold_logits: torch.Tensor,
    suffix_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL divergence between inject and gold logit distributions.

    Only computed over non-pad suffix positions.

    Args:
        inject_logits: (B, S, V) — logits from injection forward
        gold_logits:   (B, S, V) — logits from gold forward (detached)
        suffix_mask:   (B, S)    — 1 = real, 0 = pad
        temperature:   softmax temperature
    Returns:
        scalar loss
    """
    log_p = F.log_softmax(inject_logits / temperature, dim=-1)
    q = F.softmax(gold_logits / temperature, dim=-1)

    # KL(q || p) element-wise, then mask
    kl = F.kl_div(log_p, q, reduction="none").sum(-1)  # (B, S)
    kl = kl * suffix_mask
    loss = kl.sum() / suffix_mask.sum().clamp(min=1)
    return loss * (temperature**2)


def combined_loss(
    inject_logits: torch.Tensor,
    gold_logits: torch.Tensor,
    suffix_mask: torch.Tensor,
    lambda_distill: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute total loss + breakdown for logging.

    Currently just KL distillation. Bank selectivity is implicitly
    learned through the gold prompt containing only relevant facts.
    """
    l_distill = distillation_loss(inject_logits, gold_logits, suffix_mask)
    total = lambda_distill * l_distill

    metrics = {
        "loss_total": total.item(),
        "loss_distill": l_distill.item(),
    }
    return total, metrics

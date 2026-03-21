"""
Training losses for the memory encoder.

Primary loss:
  KL Divergence — match inject-path logits to gold-path logits.
  Only computed on the SUFFIX tokens (the query + response portion),
  NOT on the memory/system tokens.

Optional auxiliary:
  KV Alignment — L2 distance between encoder K,V and gold extracted K,V.
  Encourages the encoder to produce representations close to what the
  LLM would compute from the system-formatted prompt.
"""

import torch
import torch.nn.functional as F


def distillation_loss(
    inject_logits: torch.Tensor,
    gold_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL divergence between inject-path and gold-path logit distributions.

    Args:
        inject_logits: (batch, seq_len, vocab) — from injection forward pass
        gold_logits: (batch, seq_len, vocab) — from gold forward pass (detached)
        temperature: Softmax temperature for softer distributions

    Returns:
        Scalar loss (mean over batch and sequence positions)
    """
    # Use only the overlapping suffix positions
    # inject_logits covers suffix only; gold_logits covers full prompt
    # Caller must slice gold_logits to match suffix length before calling this

    inject_log_probs = F.log_softmax(inject_logits / temperature, dim=-1)
    gold_probs = F.softmax(gold_logits.detach() / temperature, dim=-1)

    # KL(gold || inject) = sum(gold * (log(gold) - log(inject)))
    kl = F.kl_div(inject_log_probs, gold_probs, reduction="batchmean")

    # Scale by T^2 (standard distillation scaling)
    return kl * (temperature**2)


def kv_alignment_loss(
    encoder_kv: list[tuple[torch.Tensor, torch.Tensor]],
    gold_kv: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """L2 distance between encoder-produced K,V and gold extracted K,V.

    This is an auxiliary signal that directly supervises the encoder's
    K,V output to match what the LLM computes from the full system prompt.

    Args:
        encoder_kv: List of (K, V) from encoder, per layer
                    K: (batch, num_kv_heads, M, head_dim)
        gold_kv: List of (K, V) extracted from gold forward pass, per layer
                 K: (batch, num_kv_heads, gold_seq_len, head_dim)
                 We only compare first M positions if gold_seq > M

    Returns:
        Scalar loss (mean L2 across layers, heads, positions)
    """
    total_loss = torch.tensor(0.0, device=encoder_kv[0][0].device)
    num_layers = len(encoder_kv)
    M = encoder_kv[0][0].shape[2]  # memory slots

    for layer_idx in range(num_layers):
        enc_k, enc_v = encoder_kv[layer_idx]
        gold_k, gold_v = gold_kv[layer_idx]

        # Truncate gold to M positions (take first M tokens of the system part)
        gold_k = gold_k[:, :, :M, :].detach()
        gold_v = gold_v[:, :, :M, :].detach()

        total_loss = total_loss + F.mse_loss(enc_k, gold_k)
        total_loss = total_loss + F.mse_loss(enc_v, gold_v)

    return total_loss / (2 * num_layers)


def combined_loss(
    inject_logits: torch.Tensor,
    gold_logits: torch.Tensor,
    encoder_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    gold_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    lambda_distill: float = 1.0,
    lambda_kv_align: float = 0.1,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute total training loss with breakdown.

    Returns dict with "total", "distill", and optionally "kv_align".
    """
    l_distill = distillation_loss(inject_logits, gold_logits, temperature)

    result = {
        "distill": l_distill,
        "total": lambda_distill * l_distill,
    }

    if encoder_kv is not None and gold_kv is not None and lambda_kv_align > 0:
        l_kv = kv_alignment_loss(encoder_kv, gold_kv)
        result["kv_align"] = l_kv
        result["total"] = result["total"] + lambda_kv_align * l_kv

    return result

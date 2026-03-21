"""
KV Injection Utilities for Level 4 Multi-Bank Encoder.

Adapted from Level 3 with support for batch-varying gates.
Core functions:
  build_injection_cache   — kv_pairs → DynamicCache
  forward_with_injection  — inject + LLM forward → logits
  generate_with_injection — inject + autoregressive generation → text
"""

import torch
from transformers import DynamicCache


def build_injection_cache(
    kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> DynamicCache:
    """Pack encoder K,V pairs into a DynamicCache for LLM injection.

    Args:
        kv_pairs: list of num_layers (K, V), each (B, num_kv_heads, M, head_dim) in fp16
    Returns:
        DynamicCache populated with memory K,V at each layer
    """
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_pairs):
        cache.update(k, v, layer_idx)
    return cache


def forward_with_injection(
    model,
    suffix_ids: torch.Tensor,
    suffix_mask: torch.Tensor,
    kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Run LLM forward with injected memory K,V + suffix tokens.

    Args:
        model: frozen LLM (e.g. Llama-3.1-8B-Instruct)
        suffix_ids:  (B, S) — tokenized suffix (user turn + assistant prompt)
        suffix_mask: (B, S) — 1 = real, 0 = pad
        kv_pairs: num_layers (K, V) tuples from encoder
    Returns:
        logits: (B, S, vocab_size)
    """
    num_mem = kv_pairs[0][0].shape[2]  # M — number of memory positions
    cache = build_injection_cache(kv_pairs)

    # Position IDs offset by memory length
    seq_len = suffix_ids.shape[1]
    position_ids = torch.arange(
        num_mem, num_mem + seq_len, device=suffix_ids.device
    ).unsqueeze(0)

    # Attention mask covers memory + suffix
    mem_mask = torch.ones(
        suffix_ids.shape[0], num_mem, device=suffix_ids.device, dtype=suffix_mask.dtype
    )
    full_mask = torch.cat([mem_mask, suffix_mask], dim=1)

    outputs = model(
        input_ids=suffix_ids,
        attention_mask=full_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=False,
    )
    return outputs.logits


@torch.no_grad()
def generate_with_injection(
    model,
    tokenizer,
    kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    suffix_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
) -> str:
    """Autoregressive generation with injected memory K,V.

    Uses manual token-by-token loop (not model.generate()) for
    full control over the injection cache.
    """
    device = suffix_ids.device
    num_mem = kv_pairs[0][0].shape[2]
    cache = build_injection_cache(kv_pairs)

    # Position IDs start after memory
    seq_len = suffix_ids.shape[1]
    position_ids = torch.arange(num_mem, num_mem + seq_len, device=device).unsqueeze(0)

    # Attention mask: memory + suffix
    mem_mask = torch.ones(1, num_mem, device=device, dtype=torch.long)
    suffix_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)
    full_mask = torch.cat([mem_mask, suffix_mask], dim=1)

    # Prefill
    outputs = model(
        input_ids=suffix_ids,
        attention_mask=full_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    cache = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].div(temperature).softmax(-1)
    next_id = torch.multinomial(next_token, 1)
    generated = [next_id.item()]
    total_len = num_mem + seq_len + 1

    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens - 1):
        pos = torch.tensor([[total_len]], device=device)
        mask = torch.ones(1, total_len + 1, device=device, dtype=torch.long)

        outputs = model(
            input_ids=next_id,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )
        cache = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].div(temperature).softmax(-1)
        next_id = torch.multinomial(next_token, 1)
        generated.append(next_id.item())
        total_len += 1

        if next_id.item() == eos_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)

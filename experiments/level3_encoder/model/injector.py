"""
KV Injector — injects encoder-produced K,V into the frozen LLM's DynamicCache.

Two modes:
  1. Training: builds cache from encoder K,V, runs LLM forward with grads through K,V
  2. Inference: same injection, greedy decode with no grad

Uses the proven injection pattern from Level 2c/2d:
  - DynamicCache pre-population with memory K,V
  - position_ids offset by num_memory_slots
  - attention_mask covers memory + suffix tokens
"""

import torch
from transformers import DynamicCache


def build_injection_cache(
    kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> DynamicCache:
    """Pack encoder-produced K,V into a DynamicCache.

    Args:
        kv_pairs: List of (K, V) per layer.
                  K shape: (batch, num_kv_heads, M, head_dim)

    Returns:
        DynamicCache populated with memory K,V for all layers.
    """
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_pairs):
        cache.update(k, v, layer_idx)
    return cache


def forward_with_injection(
    model,
    suffix_ids: torch.Tensor,
    memory_cache: DynamicCache,
    num_memory_slots: int,
) -> torch.Tensor:
    """Run frozen LLM forward pass with injected memory K,V.

    Args:
        model: Frozen LLM (HuggingFace CausalLM)
        suffix_ids: (batch, suffix_len) — tokenized user query suffix
        memory_cache: DynamicCache with encoder-produced K,V
        num_memory_slots: M — number of memory positions in cache

    Returns:
        logits: (batch, suffix_len, vocab_size)
    """
    batch = suffix_ids.shape[0]
    suffix_len = suffix_ids.shape[1]
    device = suffix_ids.device

    # Position IDs: offset by memory slots (memory occupies positions 0..M-1)
    position_ids = (
        torch.arange(num_memory_slots, num_memory_slots + suffix_len, device=device)
        .unsqueeze(0)
        .expand(batch, -1)
    )

    # Attention mask: covers memory + suffix
    attention_mask = torch.ones(
        batch, num_memory_slots + suffix_len, dtype=torch.long, device=device
    )

    outputs = model(
        input_ids=suffix_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=memory_cache,
        use_cache=False,  # don't extend cache during training forward
    )

    return outputs.logits  # (batch, suffix_len, vocab_size)


def forward_gold(
    model,
    tokenizer,
    gold_prompt: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run gold-standard forward pass (profile in system prompt).

    Args:
        model: Frozen LLM
        tokenizer: LLM tokenizer
        gold_prompt: Full formatted prompt string
        device: "cuda" or "cpu"

    Returns:
        (logits, input_ids) — logits for the full prompt
    """
    input_ids = tokenizer(gold_prompt, return_tensors="pt")["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)

    return outputs.logits, input_ids


@torch.no_grad()
def generate_with_injection(
    model,
    tokenizer,
    suffix_ids: torch.Tensor,
    kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    num_memory_slots: int,
    max_new_tokens: int = 128,
) -> str:
    """Greedy decode with injected memory K,V (inference only).

    This mirrors the proven Level 2c/2d generation pattern.
    """
    device = suffix_ids.device
    cache = build_injection_cache(kv_pairs)
    suffix_len = suffix_ids.shape[1]

    # First forward: process suffix with memory cache
    position_ids = torch.arange(
        num_memory_slots, num_memory_slots + suffix_len, device=device
    ).unsqueeze(0)
    attention_mask = torch.ones(
        1, num_memory_slots + suffix_len, dtype=torch.long, device=device
    )

    outputs = model(
        input_ids=suffix_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    cache = outputs.past_key_values
    logits = outputs.logits
    cur_pos = num_memory_slots + suffix_len

    # Autoregressive decode
    generated_ids = []
    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token.item())

        pos = torch.tensor([[cur_pos]], device=device)
        mask = torch.ones(1, cur_pos + 1, dtype=torch.long, device=device)

        outputs = model(
            input_ids=next_token.unsqueeze(0),
            attention_mask=mask,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )
        cache = outputs.past_key_values
        logits = outputs.logits
        cur_pos += 1

    return tokenizer.decode(generated_ids, skip_special_tokens=True)

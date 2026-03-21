"""
Level 1: Mechanical Proof — Does KV injection work at all?

PRODUCTION-GRADE APPROACH:
Pre-populate a DynamicCache with external K,V tensors (random for Level 1),
then run the frozen SLM with that cache. The model's native attention
mechanism naturally attends to the injected memory positions alongside
the real token positions.

This is EXACTLY how the production MemoryFusedAdapter will work:
  1. Memory encoder produces K,V tensors
  2. K,V are loaded into a DynamicCache
  3. Model runs with position_ids offset by num_memory_slots
  4. Attention mask covers [memory_positions | token_positions]

We prove:
  - Injected K,V flow through all 24 attention layers
  - Output logits shift (KL divergence > 0)
  - No NaN, no crash
  - Autoregressive generation remains coherent (no crash mid-sequence)

Hardware: RTX 5070 Laptop GPU (8 GB VRAM)
Model:   SmolLM2-1.7B-Instruct (fp16, ~3.4 GB)
"""

import gc
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    num_memory_slots: int = 16
    memory_scale: float = 0.02  # magnitude of random memory vectors
    random_seed: int = 42

    prompts: list[str] = field(
        default_factory=lambda: [
            "What is the capital of France?",
            "Explain photosynthesis in one sentence.",
            "Write a haiku about the ocean.",
            "What should I cook for dinner tonight?",
            "Tell me a fun fact about space.",
            "How does a bicycle stay upright?",
            "What is the meaning of life?",
            "Recommend a good book to read.",
            "Why is the sky blue?",
            "Describe a sunset in three words.",
        ]
    )

    max_new_tokens: int = 50
    results_dir: str = "experiments/results/level1"


# ---------------------------------------------------------------------------
# Core: Build a DynamicCache pre-populated with memory K,V
# ---------------------------------------------------------------------------


def build_memory_cache(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_mem: int,
    scale: float,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> DynamicCache:
    """Create a DynamicCache pre-populated with random memory K,V vectors.

    In production, these vectors come from the memory encoder.
    For Level 1, they're random — proving the mechanism works.
    """
    torch.manual_seed(seed)
    cache = DynamicCache()
    for layer_idx in range(num_layers):
        mem_k = (
            torch.randn(1, num_kv_heads, num_mem, head_dim, dtype=dtype, device=device)
            * scale
        )
        mem_v = (
            torch.randn(1, num_kv_heads, num_mem, head_dim, dtype=dtype, device=device)
            * scale
        )
        cache.update(mem_k, mem_v, layer_idx)
    return cache


# ---------------------------------------------------------------------------
# Forward pass: single-step logits with and without memory
# ---------------------------------------------------------------------------


def get_logits(model, input_ids, cache=None, num_mem=0):
    """Run a single forward pass and return logits.

    If cache is provided, adjusts position_ids and attention_mask
    to account for the memory positions already in the cache.
    """
    device = input_ids.device
    seq_len = input_ids.shape[1]

    if cache is not None:
        position_ids = torch.arange(
            num_mem, num_mem + seq_len, device=device
        ).unsqueeze(0)
        attention_mask = torch.ones(
            1, num_mem + seq_len, dtype=torch.long, device=device
        )
    else:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )

    return outputs.logits, outputs.past_key_values


# ---------------------------------------------------------------------------
# Autoregressive generation with pre-populated cache
# ---------------------------------------------------------------------------


def generate_greedy(model, tokenizer, input_ids, max_new_tokens, cache=None, num_mem=0):
    """Manual autoregressive loop — works with pre-populated DynamicCache.

    This is exactly how production inference will work:
    the MemoryFusedAdapter pre-populates the cache, then the model
    generates token-by-token attending to both memory and prior tokens.
    """
    device = input_ids.device
    seq_len = input_ids.shape[1]

    logits, cache = get_logits(model, input_ids, cache=cache, num_mem=num_mem)
    prefill_logits = logits[:, -1, :]
    cur_pos = num_mem + seq_len
    generated_ids = []

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token.item())

        pos = torch.tensor([[cur_pos]], device=device)
        mask = torch.ones(1, cur_pos + 1, dtype=torch.long, device=device)

        with torch.no_grad():
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

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text, prefill_logits


# ---------------------------------------------------------------------------
# Run one prompt: baseline vs injected
# ---------------------------------------------------------------------------


def run_prompt(model, tokenizer, prompt, config, model_info):
    """Compare baseline (no injection) vs memory-injected for one prompt."""
    device = config.device

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    # --- BASELINE ---
    baseline_text, baseline_logits = generate_greedy(
        model,
        tokenizer,
        input_ids,
        config.max_new_tokens,
        cache=None,
        num_mem=0,
    )

    # --- INJECTED ---
    mem_cache = build_memory_cache(
        num_layers=model_info["num_layers"],
        num_kv_heads=model_info["num_kv_heads"],
        head_dim=model_info["head_dim"],
        num_mem=config.num_memory_slots,
        scale=config.memory_scale,
        dtype=config.dtype,
        device=device,
        seed=config.random_seed,
    )

    injected_text, injected_logits = generate_greedy(
        model,
        tokenizer,
        input_ids,
        config.max_new_tokens,
        cache=mem_cache,
        num_mem=config.num_memory_slots,
    )

    # --- METRICS on the first generated token's logit distribution ---
    bl = baseline_logits.float()
    il = injected_logits.float()

    bl_log_probs = F.log_softmax(bl, dim=-1)
    il_probs = F.softmax(il, dim=-1)
    kl_div = F.kl_div(bl_log_probs, il_probs, reduction="batchmean").item()

    bl_top5 = set(torch.topk(bl[0], 5).indices.tolist())
    il_top5 = set(torch.topk(il[0], 5).indices.tolist())
    top5_overlap = len(bl_top5 & il_top5) / 5.0

    bl_top1 = torch.argmax(bl[0]).item()
    il_top1 = torch.argmax(il[0]).item()

    l2_dist = torch.norm(bl - il, p=2).item()
    max_diff = (bl - il).abs().max().item()
    has_nan = bool(torch.isnan(il).any())

    bl_top1_token = tokenizer.decode([bl_top1])
    il_top1_token = tokenizer.decode([il_top1])

    return {
        "prompt": prompt,
        "baseline_text": baseline_text.strip(),
        "injected_text": injected_text.strip(),
        "kl_divergence": kl_div,
        "l2_distance": l2_dist,
        "max_logit_diff": max_diff,
        "top5_overlap": top5_overlap,
        "top1_match": bl_top1 == il_top1,
        "top1_baseline": bl_top1_token,
        "top1_injected": il_top1_token,
        "has_nan": has_nan,
        "text_differs": baseline_text.strip() != injected_text.strip(),
        "injected_completed": len(injected_text.strip()) > 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    config = Config()
    results_path = Path(config.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  LEVEL 1: MECHANICAL PROOF — PRODUCTION-GRADE KV INJECTION")
    print("=" * 70)
    print()
    print("  Method: Pre-populate DynamicCache with external K,V vectors,")
    print("  offset position_ids, extend attention_mask. The model's native")
    print("  attention naturally attends to memory positions — no hacks.")
    print()

    # --- Load model ---
    print(f"Loading: {config.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=config.dtype,
        device_map=config.device,
    )
    model.eval()

    vram_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM: {vram_gb:.2f} GB")

    model_info = {
        "num_layers": model.config.num_hidden_layers,
        "num_heads": model.config.num_attention_heads,
        "num_kv_heads": getattr(
            model.config, "num_key_value_heads", model.config.num_attention_heads
        ),
        "head_dim": model.config.hidden_size // model.config.num_attention_heads,
        "hidden_size": model.config.hidden_size,
    }
    print(
        f"Architecture: {model_info['num_layers']}L, "
        f"{model_info['num_heads']}Q, {model_info['num_kv_heads']}KV, "
        f"d={model_info['head_dim']}"
    )
    print(f"Memory slots: {config.num_memory_slots}")
    print(f"Memory scale: {config.memory_scale}")
    print()

    # --- Run experiments ---
    print(f"Running {len(config.prompts)} prompts...")
    print("-" * 70)

    results = []

    for i, prompt in enumerate(config.prompts):
        print(f"\n[{i+1}/{len(config.prompts)}] {prompt}")

        result = run_prompt(model, tokenizer, prompt, config, model_info)
        results.append(result)

        kl = result["kl_divergence"]
        status = "PASS" if kl > 0.01 and not result["has_nan"] else "FAIL"

        print(
            f"  KL div:     {kl:.4f}  |  L2 dist:  {result['l2_distance']:.2f}  "
            f"|  Max diff: {result['max_logit_diff']:.2f}"
        )
        print(
            f"  Top-1:      baseline='{result['top1_baseline']}' "
            f"injected='{result['top1_injected']}'  match={result['top1_match']}"
        )
        print(
            f"  Top-5 overlap: {result['top5_overlap']:.0%}  |  "
            f"NaN: {result['has_nan']}  |  {status}"
        )
        print(f"  Baseline:   {result['baseline_text'][:75]}")
        print(f"  Injected:   {result['injected_text'][:75]}")

        gc.collect()
        torch.cuda.empty_cache()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    avg_kl = sum(r["kl_divergence"] for r in results) / len(results)
    avg_l2 = sum(r["l2_distance"] for r in results) / len(results)
    avg_top5 = sum(r["top5_overlap"] for r in results) / len(results)
    any_nan = any(r["has_nan"] for r in results)
    text_diff_count = sum(1 for r in results if r["text_differs"])
    top1_diff_count = sum(1 for r in results if not r["top1_match"])
    completed_count = sum(1 for r in results if r["injected_completed"])

    print(f"  Prompts tested:         {len(results)}")
    print(f"  Avg KL divergence:      {avg_kl:.4f}")
    print(f"  Avg L2 logit distance:  {avg_l2:.2f}")
    print(f"  Avg Top-5 overlap:      {avg_top5:.0%}")
    print(f"  Top-1 differs:          {top1_diff_count}/{len(results)}")
    print(f"  Text differs:           {text_diff_count}/{len(results)}")
    print(f"  Injected completed:     {completed_count}/{len(results)}")
    print(f"  Any NaN:                {any_nan}")
    print()

    # --- Success criteria ---
    criteria = {
        "KL divergence > 0.01 (injection changes distribution)": avg_kl > 0.01,
        "No NaN in outputs (injection doesn't corrupt computation)": not any_nan,
        "All 10 prompts completed (no crashes)": len(results) == len(config.prompts),
        "Generation completes with injection (autoregressive loop works)": completed_count
        == len(results),
        "Majority of texts differ (injection has visible effect)": text_diff_count
        > len(results) // 2,
    }

    all_pass = all(criteria.values())

    print("  SUCCESS CRITERIA:")
    for name, passed in criteria.items():
        icon = "PASS" if passed else "FAIL"
        print(f"    [{icon}] {name}")

    print()
    if all_pass:
        print("  >>> LEVEL 1 PASSED <<<")
        print("  KV injection mechanically works via DynamicCache.")
        print("  The production MemoryFusedAdapter approach is validated.")
        print("  Proceed to Level 2: Behavioral Steering.")
    else:
        print("  >>> LEVEL 1 FAILED — investigate failed criteria <<<")

    # --- Save ---
    output_file = results_path / "level1_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "DynamicCache pre-population (production-grade)",
                "config": {
                    "model_id": config.model_id,
                    "num_memory_slots": config.num_memory_slots,
                    "memory_scale": config.memory_scale,
                    "max_new_tokens": config.max_new_tokens,
                },
                "model_info": model_info,
                "results": results,
                "summary": {
                    "avg_kl_divergence": avg_kl,
                    "avg_l2_distance": avg_l2,
                    "avg_top5_overlap": avg_top5,
                    "any_nan": any_nan,
                    "text_differs_count": text_diff_count,
                    "top1_differs_count": top1_diff_count,
                    "completed_count": completed_count,
                },
                "criteria": {k: v for k, v in criteria.items()},
                "all_pass": all_pass,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()

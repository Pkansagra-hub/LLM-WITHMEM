"""
Level 1.5b: Attention Mass Measurement (Eager Attention)

SDPA/FlashAttention cannot return attention weights.
This script loads the model with attn_implementation="eager"
and measures actual attention mass on memory positions per layer.

Runs only a few representative configs to save time.
"""

import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def build_memory_cache(
    num_layers, num_kv_heads, head_dim, num_mem, scale, dtype, device, seed
):
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


def measure_attention_mass(model, input_ids, cache, num_mem, device):
    """Forward with output_attentions=True using eager attention."""
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(num_mem, num_mem + seq_len, device=device).unsqueeze(0)
    attention_mask = torch.ones(1, num_mem + seq_len, dtype=torch.long, device=device)

    # Clone cache
    fresh_cache = DynamicCache()
    for layer_idx in range(len(cache)):
        k, v = cache[layer_idx]
        fresh_cache.update(k.clone(), v.clone(), layer_idx)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=fresh_cache,
            use_cache=True,
            output_attentions=True,
        )

    layer_data = {}
    if outputs.attentions is not None:
        for layer_idx, attn_weights in enumerate(outputs.attentions):
            # attn_weights: [batch, heads, seq_len, total_kv_len]
            # total_kv_len = num_mem + seq_len
            total_kv = attn_weights.shape[-1]

            # Mass on memory positions (first num_mem columns)
            mem_mass = attn_weights[:, :, :, :num_mem].sum(
                dim=-1
            )  # [1, heads, seq_len]

            # Per-head average mass on memory (averaged over query positions)
            per_head_mass = mem_mass[0].mean(dim=-1)  # [heads]

            layer_data[layer_idx] = {
                "mean_mem_mass": mem_mass.mean().item(),
                "max_mem_mass": mem_mass.max().item(),
                "per_head_mem_mass": per_head_mass.tolist(),
                "total_kv_len": total_kv,
            }
    return layer_data


def main():
    model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    device = "cuda"
    dtype = torch.float16
    num_mem = 16
    seed = 42

    results_path = Path("experiments/results/level1_5")
    results_path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  ATTENTION MASS MEASUREMENT (EAGER ATTENTION)")
    print("=" * 72)
    print()

    # Load with eager attention — this is the key difference
    print(f"Loading {model_id} with attn_implementation='eager'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    num_layers = model.config.num_hidden_layers
    num_kv_heads = getattr(
        model.config, "num_key_value_heads", model.config.num_attention_heads
    )
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_heads = model.config.num_attention_heads

    print(f"Architecture: {num_layers}L, {num_heads}Q, {num_kv_heads}KV, d={head_dim}")
    print()

    # Test configs: scale × prompt
    scales = [0.001, 0.005, 0.01, 0.02]
    prompts = [
        "What is the capital of France?",
        "Tell me a fun fact about space.",
        "What should I cook for dinner tonight?",
    ]

    all_results = {}

    for scale in scales:
        print(f"\n  Scale = {scale}")
        print("  " + "─" * 60)
        scale_results = []

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

            cache = build_memory_cache(
                num_layers, num_kv_heads, head_dim, num_mem, scale, dtype, device, seed
            )

            layer_data = measure_attention_mass(
                model, input_ids, cache, num_mem, device
            )

            # Print per-layer heatmap
            print(f"\n    Prompt: {prompt[:50]}")
            max_mass = (
                max(d["mean_mem_mass"] for d in layer_data.values())
                if layer_data
                else 1e-10
            )

            for layer_idx in range(num_layers):
                d = layer_data.get(layer_idx, {})
                mass = d.get("mean_mem_mass", 0)
                bar_len = int(40 * mass / max(max_mass, 1e-10))
                bar = "█" * bar_len

                # Per-head breakdown (show min/max/mean)
                head_masses = d.get("per_head_mem_mass", [])
                if head_masses:
                    h_min = min(head_masses)
                    h_max = max(head_masses)
                    head_info = f"heads: min={h_min:.5f} max={h_max:.5f}"
                else:
                    head_info = ""

                print(f"      L{layer_idx:02d}: {mass:.6f} {bar}  {head_info}")

            scale_results.append(
                {
                    "prompt": prompt,
                    "layer_data": {str(k): v for k, v in layer_data.items()},
                }
            )

            gc.collect()
            torch.cuda.empty_cache()

        all_results[str(scale)] = scale_results

    # Summary: average mass per layer across all prompts, for each scale
    print("\n" + "=" * 72)
    print("  SUMMARY: AVG ATTENTION MASS ON MEMORY BY LAYER")
    print("=" * 72)

    header = f"  {'Layer':>5}"
    for scale in scales:
        header += f" | {scale:>8}"
    print(header)
    print("  " + "─" * (8 + 11 * len(scales)))

    for layer_idx in range(num_layers):
        row = f"  L{layer_idx:02d}  "
        for scale in scales:
            masses = []
            for pr in all_results[str(scale)]:
                ld = pr["layer_data"].get(str(layer_idx), {})
                masses.append(ld.get("mean_mem_mass", 0))
            avg = sum(masses) / len(masses) if masses else 0
            row += f" | {avg:8.6f}"
        print(row)

    # Save
    output_file = results_path / "attention_mass_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "Eager attention mass measurement",
                "model_id": model_id,
                "num_memory_slots": num_mem,
                "scales": scales,
                "results": all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n  Results saved to: {output_file}")

    # Key finding
    print()
    print("  KEY FINDING:")
    # Check if attention mass scales with injection scale
    for scale in scales:
        all_masses = []
        for pr in all_results[str(scale)]:
            for ld in pr["layer_data"].values():
                all_masses.append(ld.get("mean_mem_mass", 0))
        avg = sum(all_masses) / len(all_masses) if all_masses else 0
        print(f"    scale={scale:.3f} -> avg attention mass on memory = {avg:.6f}")


if __name__ == "__main__":
    main()

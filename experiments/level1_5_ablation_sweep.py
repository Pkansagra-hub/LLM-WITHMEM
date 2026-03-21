"""
Level 1.5: Ablation Sweep — Alignment Tuning Before Level 2

Level 1 proved existence: the model does NOT ignore injected K,V.
Now we answer: HOW MUCH injection, WHERE, and HOW GATED?

Sweep dimensions:
  A) Memory scale:        [0.001, 0.005, 0.01, 0.02]
  B) Layer subsets:        [all, late-only, mid+late, every-4th]
  C) Per-layer gates:      [none, uniform-init, near-zero-init]
  D) Attention mass on memory slots per layer
  E) Coherence score for generated text

Hardware: RTX 5070 Laptop GPU (8 GB VRAM)
Model:   SmolLM2-1.7B-Instruct (fp16, ~3.4 GB)
"""

import gc
import json
import math
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Config:
    model_id: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    num_memory_slots: int = 16
    random_seed: int = 42
    max_new_tokens: int = 50

    # --- Sweep axes ---
    memory_scales: list[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.02]
    )

    # Layer subset strategies: name -> callable(num_layers) -> list[int]
    # Defined in code below, just names here for config serialization
    layer_strategies: list[str] = field(
        default_factory=lambda: ["all", "late_only", "mid_late", "every_4th"]
    )

    # Gating strategies
    gate_strategies: list[str] = field(default_factory=lambda: ["none", "near_zero"])

    prompts: list[str] = field(
        default_factory=lambda: [
            "What is the capital of France?",
            "Explain photosynthesis in one sentence.",
            "Write a haiku about the ocean.",
            "What should I cook for dinner tonight?",
            "Tell me a fun fact about space.",
        ]
    )

    results_dir: str = "experiments/results/level1_5"


# ──────────────────────────────────────────────────────────────────────────
# Layer subset strategies
# ──────────────────────────────────────────────────────────────────────────


def get_layer_indices(strategy: str, num_layers: int) -> list[int]:
    """Return which layers get memory injection."""
    if strategy == "all":
        return list(range(num_layers))
    elif strategy == "late_only":
        # Last third of layers
        start = (num_layers * 2) // 3
        return list(range(start, num_layers))
    elif strategy == "mid_late":
        # Middle third + last third
        start = num_layers // 3
        return list(range(start, num_layers))
    elif strategy == "every_4th":
        return list(range(0, num_layers, 4))
    else:
        raise ValueError(f"Unknown layer strategy: {strategy}")


# ──────────────────────────────────────────────────────────────────────────
# Per-layer gates
# ──────────────────────────────────────────────────────────────────────────


def build_gates(
    strategy: str,
    num_layers: int,
    num_kv_heads: int,
    active_layers: list[int],
    device: str,
) -> dict[int, torch.Tensor]:
    """Build per-layer, per-head gate scalars.

    Returns dict: layer_idx -> tensor of shape [1, num_kv_heads, 1, 1]
    (broadcastable against K,V shapes).
    """
    gates = {}
    for layer_idx in active_layers:
        if strategy == "none":
            # No gating — full pass-through
            gates[layer_idx] = torch.ones(1, num_kv_heads, 1, 1, device=device)
        elif strategy == "near_zero":
            # Initialized near zero: sigmoid(-3) ≈ 0.047
            # This means memory starts with ~5% influence, learnable in Level 3
            raw = torch.full((1, num_kv_heads, 1, 1), -3.0, device=device)
            gates[layer_idx] = torch.sigmoid(raw)
        else:
            raise ValueError(f"Unknown gate strategy: {strategy}")
    return gates


# ──────────────────────────────────────────────────────────────────────────
# Build memory cache with layer subsets and gating
# ──────────────────────────────────────────────────────────────────────────


def build_memory_cache(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_mem: int,
    scale: float,
    dtype: torch.dtype,
    device: str,
    seed: int,
    active_layers: list[int],
    gates: dict[int, torch.Tensor],
) -> DynamicCache:
    """Create DynamicCache pre-populated with gated memory K,V.

    Inactive layers get zero-vectors (present but silent).
    Active layers get random * scale * gate.
    """
    torch.manual_seed(seed)
    cache = DynamicCache()

    for layer_idx in range(num_layers):
        if layer_idx in active_layers:
            mem_k = (
                torch.randn(
                    1, num_kv_heads, num_mem, head_dim, dtype=dtype, device=device
                )
                * scale
            )
            mem_v = (
                torch.randn(
                    1, num_kv_heads, num_mem, head_dim, dtype=dtype, device=device
                )
                * scale
            )
            # Apply per-head gate
            gate = gates[layer_idx].to(dtype)
            mem_k = mem_k * gate
            mem_v = mem_v * gate
        else:
            # Zero vectors — cache position exists but contributes nothing
            mem_k = torch.zeros(
                1, num_kv_heads, num_mem, head_dim, dtype=dtype, device=device
            )
            mem_v = torch.zeros(
                1, num_kv_heads, num_mem, head_dim, dtype=dtype, device=device
            )
        cache.update(mem_k, mem_v, layer_idx)

    return cache


# ──────────────────────────────────────────────────────────────────────────
# Attention mass tracking via output_attentions=True
# NOTE: Requires attn_implementation="eager" — SDPA silently drops weights.
# ──────────────────────────────────────────────────────────────────────────


def measure_attention_mass(model, input_ids, cache, num_mem, device):
    """Run a forward pass with output_attentions=True to get actual attention weights.

    Returns per-layer attention mass on memory positions.
    mass[layer] = mean over (heads, query positions) of sum over memory key positions.
    """
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(num_mem, num_mem + seq_len, device=device).unsqueeze(0)
    attention_mask = torch.ones(1, num_mem + seq_len, dtype=torch.long, device=device)

    # Clone cache so the measurement doesn't modify the original
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

    # outputs.attentions: tuple of (batch, heads, seq_len, total_kv_len) per layer
    layer_masses = {}
    if outputs.attentions is not None:
        for layer_idx, attn_weights in enumerate(outputs.attentions):
            # attn_weights: [1, num_heads, seq_len, num_mem + seq_len]
            # Mass on memory = sum of attention over first num_mem positions
            mem_mass = attn_weights[:, :, :, :num_mem].sum(
                dim=-1
            )  # [1, heads, seq_len]
            # Average over batch, heads, query positions
            layer_masses[layer_idx] = mem_mass.mean().item()

    return layer_masses


# ──────────────────────────────────────────────────────────────────────────
# Coherence scoring
# ──────────────────────────────────────────────────────────────────────────


def score_coherence(model, tokenizer, text: str, device: str) -> float:
    """Compute perplexity of generated text as a coherence proxy.

    Lower perplexity = more coherent / predictable generation.
    We score only the generated text, not the prompt.
    """
    if not text or len(text.strip()) == 0:
        return float("inf")

    tokens = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    if tokens.shape[1] < 2:
        return float("inf")

    with torch.no_grad():
        outputs = model(input_ids=tokens)
        logits = outputs.logits

    # Shift: predict token[i+1] from logits[i]
    shift_logits = logits[:, :-1, :].float()
    shift_labels = tokens[:, 1:]

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
    )
    perplexity = math.exp(min(loss.item(), 20))  # cap to avoid overflow
    return perplexity


# ──────────────────────────────────────────────────────────────────────────
# Forward pass helpers (from Level 1)
# ──────────────────────────────────────────────────────────────────────────


def get_logits(model, input_ids, cache=None, num_mem=0):
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


def generate_greedy(model, tokenizer, input_ids, max_new_tokens, cache=None, num_mem=0):
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


# ──────────────────────────────────────────────────────────────────────────
# Run one configuration
# ──────────────────────────────────────────────────────────────────────────


def run_one_config(
    model,
    tokenizer,
    prompt,
    input_ids,
    model_info,
    config,
    scale,
    layer_strategy,
    gate_strategy,
    baseline_text,
    baseline_logits,
):
    """Run a single (scale, layer_strategy, gate_strategy) configuration."""
    device = config.device
    num_layers = model_info["num_layers"]
    num_kv_heads = model_info["num_kv_heads"]
    head_dim = model_info["head_dim"]

    active_layers = get_layer_indices(layer_strategy, num_layers)
    gates = build_gates(gate_strategy, num_layers, num_kv_heads, active_layers, device)

    mem_cache = build_memory_cache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_mem=config.num_memory_slots,
        scale=scale,
        dtype=config.dtype,
        device=device,
        seed=config.random_seed,
        active_layers=active_layers,
        gates=gates,
    )

    # --- Attention mass measurement ---
    attn_mass = measure_attention_mass(
        model, input_ids, mem_cache, config.num_memory_slots, device
    )

    # Rebuild cache (measure consumed it)
    mem_cache = build_memory_cache(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_mem=config.num_memory_slots,
        scale=scale,
        dtype=config.dtype,
        device=device,
        seed=config.random_seed,
        active_layers=active_layers,
        gates=gates,
    )

    # --- Generate ---
    injected_text, injected_logits = generate_greedy(
        model,
        tokenizer,
        input_ids,
        config.max_new_tokens,
        cache=mem_cache,
        num_mem=config.num_memory_slots,
    )

    # --- Metrics ---
    bl = baseline_logits.float()
    il = injected_logits.float()

    bl_log_probs = F.log_softmax(bl, dim=-1)
    il_probs = F.softmax(il, dim=-1)
    kl_div = F.kl_div(bl_log_probs, il_probs, reduction="batchmean").item()
    l2_dist = torch.norm(bl - il, p=2).item()
    has_nan = bool(torch.isnan(il).any())

    bl_top5 = set(torch.topk(bl[0], 5).indices.tolist())
    il_top5 = set(torch.topk(il[0], 5).indices.tolist())
    top5_overlap = len(bl_top5 & il_top5) / 5.0

    bl_top1 = torch.argmax(bl[0]).item()
    il_top1 = torch.argmax(il[0]).item()

    # Coherence scoring
    injected_ppl = score_coherence(model, tokenizer, injected_text.strip(), device)
    baseline_ppl = score_coherence(model, tokenizer, baseline_text.strip(), device)

    # Attention mass summary
    active_mass = {k: v for k, v in attn_mass.items() if k in active_layers}
    inactive_mass = {k: v for k, v in attn_mass.items() if k not in active_layers}
    avg_active_mass = (
        sum(active_mass.values()) / len(active_mass) if active_mass else 0.0
    )
    avg_inactive_mass = (
        sum(inactive_mass.values()) / len(inactive_mass) if inactive_mass else 0.0
    )

    return {
        "scale": scale,
        "layer_strategy": layer_strategy,
        "gate_strategy": gate_strategy,
        "active_layers": active_layers,
        "num_active_layers": len(active_layers),
        "kl_divergence": kl_div,
        "l2_distance": l2_dist,
        "top5_overlap": top5_overlap,
        "top1_match": bl_top1 == il_top1,
        "has_nan": has_nan,
        "text_differs": baseline_text.strip() != injected_text.strip(),
        "baseline_text": baseline_text.strip()[:100],
        "injected_text": injected_text.strip()[:100],
        "baseline_perplexity": baseline_ppl,
        "injected_perplexity": injected_ppl,
        "perplexity_ratio": (
            injected_ppl / baseline_ppl if baseline_ppl > 0 else float("inf")
        ),
        "attention_mass_per_layer": attn_mass,
        "avg_active_attn_mass": avg_active_mass,
        "avg_inactive_attn_mass": avg_inactive_mass,
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main():
    config = Config()
    results_path = Path(config.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  LEVEL 1.5: ABLATION SWEEP — ALIGNMENT TUNING")
    print("=" * 72)
    print()
    print("  Dimensions:")
    print(f"    Scales:          {config.memory_scales}")
    print(f"    Layer strategies: {config.layer_strategies}")
    print(f"    Gate strategies:  {config.gate_strategies}")
    print(f"    Prompts:          {len(config.prompts)}")
    combos = (
        len(config.memory_scales)
        * len(config.layer_strategies)
        * len(config.gate_strategies)
    )
    print(f"    Total configs:    {combos}")
    print(f"    Total runs:       {combos * len(config.prompts)}")
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
        attn_implementation="eager",  # Required for output_attentions=True
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
    print()

    # --- Pre-compute baselines for each prompt ---
    print("Computing baselines...")
    baselines = {}
    for prompt in config.prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(config.device)

        bl_text, bl_logits = generate_greedy(
            model, tokenizer, input_ids, config.max_new_tokens
        )
        baselines[prompt] = {
            "input_ids": input_ids,
            "text": bl_text,
            "logits": bl_logits,
        }
        print(f"  [{prompt[:40]}...] -> {bl_text.strip()[:60]}")

    print()

    # --- Sweep ---
    all_results = []
    run_idx = 0
    total_runs = combos * len(config.prompts)
    t_start = time.time()

    for scale, layer_strat, gate_strat in product(
        config.memory_scales, config.layer_strategies, config.gate_strategies
    ):
        config_label = f"s={scale} | layers={layer_strat} | gates={gate_strat}"
        print(f"\n{'─' * 72}")
        print(f"  CONFIG: {config_label}")
        active = get_layer_indices(layer_strat, model_info["num_layers"])
        print(f"  Active layers: {active} ({len(active)}/{model_info['num_layers']})")
        print(f"{'─' * 72}")

        for prompt in config.prompts:
            run_idx += 1
            bl = baselines[prompt]

            result = run_one_config(
                model,
                tokenizer,
                prompt,
                bl["input_ids"],
                model_info,
                config,
                scale,
                layer_strat,
                gate_strat,
                bl["text"],
                bl["logits"],
            )
            result["prompt"] = prompt
            all_results.append(result)

            ppl_ratio = result["perplexity_ratio"]
            coherence_tag = (
                "GOOD" if ppl_ratio < 2.0 else "OK" if ppl_ratio < 5.0 else "DEGRADED"
            )

            print(
                f"  [{run_idx:3d}/{total_runs}] "
                f"KL={result['kl_divergence']:8.4f}  "
                f"L2={result['l2_distance']:8.1f}  "
                f"T5={result['top5_overlap']:4.0%}  "
                f"AttnMass={result['avg_active_attn_mass']:.4f}  "
                f"PPL_r={ppl_ratio:5.2f} [{coherence_tag:8s}]  "
                f"NaN={result['has_nan']}"
            )

            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    # ── Summary tables ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  ABLATION SUMMARY")
    print("=" * 72)

    # Group by config
    from collections import defaultdict

    by_config = defaultdict(list)
    for r in all_results:
        key = (r["scale"], r["layer_strategy"], r["gate_strategy"])
        by_config[key].append(r)

    print()
    print(
        f"  {'Scale':>7} | {'Layers':<10} | {'Gates':<10} | "
        f"{'Avg KL':>8} | {'Avg L2':>8} | {'Avg T5':>6} | "
        f"{'AttnMass':>8} | {'PPL ratio':>9} | {'Coherence':>9} | {'NaN':>3}"
    )
    print("  " + "─" * 105)

    summary_rows = []
    for (scale, layer_strat, gate_strat), runs in sorted(by_config.items()):
        avg_kl = sum(r["kl_divergence"] for r in runs) / len(runs)
        avg_l2 = sum(r["l2_distance"] for r in runs) / len(runs)
        avg_t5 = sum(r["top5_overlap"] for r in runs) / len(runs)
        avg_mass = sum(r["avg_active_attn_mass"] for r in runs) / len(runs)
        avg_ppl_r = sum(r["perplexity_ratio"] for r in runs) / len(runs)
        any_nan = any(r["has_nan"] for r in runs)

        coherence_tag = (
            "GOOD" if avg_ppl_r < 2.0 else "OK" if avg_ppl_r < 5.0 else "DEGRADED"
        )

        print(
            f"  {scale:7.3f} | {layer_strat:<10} | {gate_strat:<10} | "
            f"{avg_kl:8.4f} | {avg_l2:8.1f} | {avg_t5:5.0%} | "
            f"{avg_mass:8.5f} | {avg_ppl_r:9.2f} | {coherence_tag:>9} | "
            f"{'Y' if any_nan else 'N':>3}"
        )

        summary_rows.append(
            {
                "scale": scale,
                "layer_strategy": layer_strat,
                "gate_strategy": gate_strat,
                "avg_kl": avg_kl,
                "avg_l2": avg_l2,
                "avg_top5_overlap": avg_t5,
                "avg_attn_mass": avg_mass,
                "avg_ppl_ratio": avg_ppl_r,
                "coherence": coherence_tag,
                "any_nan": any_nan,
            }
        )

    # ── Best config by coherence-adjusted KL ──────────────────────────────
    # We want HIGH KL (strong steering) but LOW perplexity ratio (clean output).
    # Score = KL / PPL_ratio  (higher is better)
    print()
    print("  TOP 5 CONFIGS (by KL / PPL_ratio — steer cleanly):")
    print("  " + "─" * 80)

    scored = [
        (row, row["avg_kl"] / max(row["avg_ppl_ratio"], 0.01))
        for row in summary_rows
        if not row["any_nan"]
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    for rank, (row, score) in enumerate(scored[:5], 1):
        print(
            f"  #{rank}: s={row['scale']:.3f} layers={row['layer_strategy']:<10} "
            f"gates={row['gate_strategy']:<10} | "
            f"KL={row['avg_kl']:.4f} PPL_r={row['avg_ppl_ratio']:.2f} "
            f"score={score:.4f} [{row['coherence']}]"
        )

    # ── Attention mass heatmap (text) ─────────────────────────────────────
    print()
    print("  ATTENTION MASS ON MEMORY (per-layer, averaged across prompts):")
    print("  " + "─" * 72)

    # Show for scale=0.01, all layers, no gates (most informative)
    ref_key = (0.01, "all", "none")
    if ref_key in by_config:
        ref_runs = by_config[ref_key]
        num_layers = model_info["num_layers"]
        layer_avg = {}
        for layer_idx in range(num_layers):
            masses = [
                r["attention_mass_per_layer"].get(
                    str(layer_idx), r["attention_mass_per_layer"].get(layer_idx, 0)
                )
                for r in ref_runs
            ]
            layer_avg[layer_idx] = sum(masses) / len(masses) if masses else 0

        max_mass = max(layer_avg.values()) if layer_avg else 1
        print("  Config: scale=0.01, all layers, no gates")
        print(f"  (bar = relative mass, max = {max_mass:.5f})")
        print()
        for layer_idx in range(num_layers):
            mass = layer_avg[layer_idx]
            bar_len = int(40 * mass / max_mass) if max_mass > 0 else 0
            bar = "█" * bar_len
            print(f"    L{layer_idx:02d}: {mass:.5f} {bar}")

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n  Elapsed: {elapsed:.1f}s")

    output = {
        "method": "Level 1.5 ablation sweep",
        "model_info": model_info,
        "config": {
            "model_id": config.model_id,
            "num_memory_slots": config.num_memory_slots,
            "max_new_tokens": config.max_new_tokens,
            "memory_scales": config.memory_scales,
            "layer_strategies": config.layer_strategies,
            "gate_strategies": config.gate_strategies,
            "num_prompts": len(config.prompts),
        },
        "summary": summary_rows,
        "top5_configs": [
            {"rank": i + 1, "score": s, **row} for i, (row, s) in enumerate(scored[:5])
        ],
        "all_results": all_results,
        "elapsed_seconds": elapsed,
    }

    # Convert attention mass keys to strings for JSON
    for r in output["all_results"]:
        r["attention_mass_per_layer"] = {
            str(k): v for k, v in r["attention_mass_per_layer"].items()
        }

    output_file = results_path / "ablation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"  Results saved to: {output_file}")
    print()

    if scored:
        best = scored[0][0]
        print(
            f"  >>> RECOMMENDED CONFIG FOR LEVEL 2: "
            f"scale={best['scale']}, layers={best['layer_strategy']}, "
            f"gates={best['gate_strategy']} <<<"
        )
    print()
    print("  Level 1.5 complete. The game is alignment, not existence.")


if __name__ == "__main__":
    main()

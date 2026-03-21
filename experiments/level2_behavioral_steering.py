"""
Level 2: Behavioral Steering — Can injection change output MEANINGFULLY?

Level 1 proved existence: the model attends to injected K,V.
Level 1.5 proved scale: ~50% attention mass on memory, peaking 70% at L19-L20.
Now we prove SEMANTICS: inject K,V extracted from a real profile text,
and show the model's responses become personalized to that profile.

METHOD:
  1. Forward-pass a profile sentence through the frozen model.
  2. Extract the K,V tensors produced at every layer — these ARE the model's
     internal representation of the profile, in its native space.
  3. Pre-populate a DynamicCache with those extracted K,V.
  4. Ask a neutral query. The model attends to both the query tokens AND
     the profile K,V, producing a profile-flavored response.

This is the EXACT production pattern for MemoryFusedAdapter, except
in production the K,V come from a trained encoder rather than a forward pass.
Level 2 proves the injection CHANNEL carries semantic information.

Hardware: RTX 5070 Laptop GPU (8 GB VRAM)
Model:   SmolLM2-1.7B-Instruct (fp16, ~3.4 GB)
"""

import gc
import json
import math
import time
from dataclasses import dataclass, field
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

    max_new_tokens: int = 120  # longer generation for richer output

    profiles: dict[str, str] = field(
        default_factory=lambda: {
            "Alex": (
                "Alex is a marine biologist living in Seattle who hates cilantro "
                "and loves Thai food. Alex spends weekends kayaking in Puget Sound "
                "and volunteers at the Seattle Aquarium."
            ),
            "Priya": (
                "Priya is a software engineer in Bangalore who is vegetarian and "
                "enjoys hiking in the Western Ghats. She practices Bharatanatyam "
                "dance and mentors women in tech."
            ),
            "Marcus": (
                "Marcus is a retired firefighter in Austin, Texas who builds "
                "custom furniture and has a dog named Biscuit. He smokes brisket "
                "every weekend and coaches little league baseball."
            ),
            "Yuki": (
                "Yuki is a concert pianist in Tokyo who is training for a marathon "
                "and collects vintage vinyl records. She loves ramen and visits "
                "jazz clubs in Shinjuku."
            ),
            "Fatima": (
                "Fatima is a pediatric surgeon in London who speaks four languages "
                "and is passionate about urban gardening. She cycles to the hospital "
                "and reads Arabic poetry."
            ),
        }
    )

    queries: list[str] = field(
        default_factory=lambda: [
            "What restaurant should I go to tonight?",
            "What's a good hobby to pick up?",
            "What should I do this weekend?",
            "What gift should I get for a friend?",
            "Tell me something interesting about where I live.",
        ]
    )

    # Keywords expected in personalized responses, per profile
    # Each profile maps to a flat list of keywords/phrases
    profile_keywords: dict[str, list[str]] = field(
        default_factory=lambda: {
            "Alex": [
                "seattle",
                "thai",
                "marine",
                "ocean",
                "sea",
                "fish",
                "kayak",
                "aquarium",
                "puget",
                "biology",
                "cilantro",
                "seafood",
                "pacific",
                "northwest",
                "washington",
            ],
            "Priya": [
                "bangalore",
                "bengaluru",
                "vegetarian",
                "veg",
                "hiking",
                "trek",
                "western ghats",
                "india",
                "dance",
                "bharatanatyam",
                "women",
                "tech",
                "software",
                "engineer",
            ],
            "Marcus": [
                "austin",
                "texas",
                "furniture",
                "woodwork",
                "wood",
                "biscuit",
                "dog",
                "brisket",
                "bbq",
                "barbecue",
                "firefight",
                "baseball",
                "coach",
                "smoke",
                "grill",
            ],
            "Yuki": [
                "tokyo",
                "piano",
                "pianist",
                "music",
                "marathon",
                "running",
                "vinyl",
                "record",
                "ramen",
                "jazz",
                "japan",
                "shinjuku",
                "concert",
                "classical",
            ],
            "Fatima": [
                "london",
                "surgeon",
                "doctor",
                "medical",
                "garden",
                "gardening",
                "plant",
                "cycling",
                "bike",
                "arabic",
                "poetry",
                "language",
                "hospital",
                "pediatric",
                "british",
                "uk",
            ],
        }
    )

    results_dir: str = "experiments/results/level2"


# ──────────────────────────────────────────────────────────────────────────
# Extract K,V from profile text via forward pass
# ──────────────────────────────────────────────────────────────────────────


def extract_profile_kv(
    model, tokenizer, profile_text: str, device: str
) -> DynamicCache:
    """Run profile text through the model and capture the K,V cache.

    The returned DynamicCache contains the model's own K,V representations
    of the profile — already in the native attention space, at native magnitude.
    No scaling needed.
    """
    tokens = tokenizer(profile_text, return_tensors="pt")["input_ids"].to(device)
    seq_len = tokens.shape[1]

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )

    # outputs.past_key_values is a DynamicCache containing the K,V
    # for every layer, with shape [1, num_kv_heads, seq_len, head_dim]
    return outputs.past_key_values, seq_len


# ──────────────────────────────────────────────────────────────────────────
# Generation with pre-populated cache (from Level 1)
# ──────────────────────────────────────────────────────────────────────────


def generate_greedy(model, tokenizer, input_ids, max_new_tokens, cache=None, num_mem=0):
    """Manual autoregressive loop with optional pre-populated DynamicCache."""
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

    cache = outputs.past_key_values
    logits = outputs.logits
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
    return text


# ──────────────────────────────────────────────────────────────────────────
# Coherence scoring
# ──────────────────────────────────────────────────────────────────────────


def score_coherence(model, tokenizer, text: str, device: str) -> float:
    """Perplexity of generated text — lower = more coherent."""
    if not text or len(text.strip()) == 0:
        return float("inf")
    tokens = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    if tokens.shape[1] < 2:
        return float("inf")
    with torch.no_grad():
        outputs = model(input_ids=tokens)
        logits = outputs.logits
    shift_logits = logits[:, :-1, :].float()
    shift_labels = tokens[:, 1:]
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
    )
    return math.exp(min(loss.item(), 20))


# ──────────────────────────────────────────────────────────────────────────
# Keyword matching scorer
# ──────────────────────────────────────────────────────────────────────────


def score_keywords(text: str, keywords: list[str]) -> dict:
    """Check which profile keywords appear in the generated text."""
    text_lower = text.lower()
    hits = [kw for kw in keywords if kw.lower() in text_lower]
    return {
        "hit_count": len(hits),
        "total_keywords": len(keywords),
        "hit_ratio": len(hits) / len(keywords) if keywords else 0,
        "hits": hits,
    }


# ──────────────────────────────────────────────────────────────────────────
# Clone a DynamicCache (needed because generation modifies it in-place)
# ──────────────────────────────────────────────────────────────────────────


def clone_cache(cache: DynamicCache) -> DynamicCache:
    """Deep-clone a DynamicCache so the original stays unmodified."""
    new_cache = DynamicCache()
    for layer_idx in range(len(cache)):
        k, v = cache[layer_idx]
        new_cache.update(k.clone(), v.clone(), layer_idx)
    return new_cache


# ──────────────────────────────────────────────────────────────────────────
# Run one (profile, query) pair
# ──────────────────────────────────────────────────────────────────────────


def run_pair(
    model,
    tokenizer,
    config,
    profile_name,
    profile_cache,
    num_profile_tokens,
    query,
    profile_keywords,
):
    """Generate baseline and injected responses, score both."""
    device = config.device

    # Format query as chat
    messages = [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    # --- BASELINE: no memory injection ---
    baseline_text = generate_greedy(
        model,
        tokenizer,
        input_ids,
        config.max_new_tokens,
        cache=None,
        num_mem=0,
    )

    # --- INJECTED: profile K,V pre-populating the cache ---
    injected_cache = clone_cache(profile_cache)
    injected_text = generate_greedy(
        model,
        tokenizer,
        input_ids,
        config.max_new_tokens,
        cache=injected_cache,
        num_mem=num_profile_tokens,
    )

    # --- Score ---
    bl_kw = score_keywords(baseline_text, profile_keywords)
    inj_kw = score_keywords(injected_text, profile_keywords)

    # Coherence
    bl_ppl = score_coherence(model, tokenizer, baseline_text.strip(), device)
    inj_ppl = score_coherence(model, tokenizer, injected_text.strip(), device)

    personalized = inj_kw["hit_count"] > bl_kw["hit_count"]
    baseline_clean = bl_kw["hit_count"] == 0

    return {
        "profile": profile_name,
        "query": query,
        "baseline_text": baseline_text.strip(),
        "injected_text": injected_text.strip(),
        "baseline_keywords": bl_kw,
        "injected_keywords": inj_kw,
        "baseline_perplexity": bl_ppl,
        "injected_perplexity": inj_ppl,
        "ppl_ratio": inj_ppl / bl_ppl if bl_ppl > 0 else float("inf"),
        "personalized": personalized,
        "baseline_clean": baseline_clean,
        "num_profile_tokens": num_profile_tokens,
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main():
    config = Config()
    results_path = Path(config.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  LEVEL 2: BEHAVIORAL STEERING — SEMANTIC KV INJECTION")
    print("=" * 72)
    print()
    print("  Method: Extract K,V from profile text via forward pass,")
    print("  pre-populate DynamicCache, generate response to neutral query.")
    print("  Prove that profile information flows through the injection channel.")
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

    num_layers = model.config.num_hidden_layers
    num_kv_heads = getattr(
        model.config, "num_key_value_heads", model.config.num_attention_heads
    )
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"Architecture: {num_layers}L, {num_kv_heads}KV, d={head_dim}")
    print()

    # --- Extract K,V for each profile ---
    print("Extracting profile K,V caches...")
    profile_caches = {}
    for name, text in config.profiles.items():
        cache, num_tokens = extract_profile_kv(model, tokenizer, text, config.device)
        profile_caches[name] = (cache, num_tokens)
        print(f"  {name}: {num_tokens} tokens extracted")
    print()

    # --- Run all (profile, query) pairs ---
    total_pairs = len(config.profiles) * len(config.queries)
    print(f"Running {total_pairs} (profile, query) pairs...")
    print("─" * 72)

    all_results = []
    pair_idx = 0
    t_start = time.time()

    for profile_name in config.profiles:
        cache, num_tokens = profile_caches[profile_name]
        keywords = config.profile_keywords[profile_name]

        for query in config.queries:
            pair_idx += 1
            print(f'\n[{pair_idx}/{total_pairs}] {profile_name} + "{query[:45]}"')

            result = run_pair(
                model,
                tokenizer,
                config,
                profile_name,
                cache,
                num_tokens,
                query,
                keywords,
            )
            all_results.append(result)

            inj = result["injected_keywords"]
            bl = result["baseline_keywords"]
            tag = "PERSONALIZED" if result["personalized"] else "NO SIGNAL"
            coherence = (
                "COHERENT"
                if result["ppl_ratio"] < 3.0
                else "OK" if result["ppl_ratio"] < 10.0 else "DEGRADED"
            )

            print(f"  Baseline:  {result['baseline_text'][:80]}")
            print(f"  Injected:  {result['injected_text'][:80]}")
            print(
                f"  Keywords:  baseline={bl['hit_count']}/{bl['total_keywords']}  "
                f"injected={inj['hit_count']}/{inj['total_keywords']}  "
                f"hits={inj['hits']}"
            )
            print(
                f"  PPL:       baseline={result['baseline_perplexity']:.1f}  "
                f"injected={result['injected_perplexity']:.1f}  "
                f"ratio={result['ppl_ratio']:.2f}  [{coherence}]"
            )
            print(f"  Verdict:   [{tag}]")

            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    # Per-profile breakdown
    from collections import defaultdict

    by_profile = defaultdict(list)
    for r in all_results:
        by_profile[r["profile"]].append(r)

    print()
    print(
        f"  {'Profile':<10} | {'Personalized':>12} | {'BL Keywords':>11} | "
        f"{'INJ Keywords':>12} | {'Avg PPL ratio':>13} | {'Coherence':>9}"
    )
    print("  " + "─" * 80)

    for profile_name in config.profiles:
        runs = by_profile[profile_name]
        n_pers = sum(1 for r in runs if r["personalized"])
        avg_bl_kw = sum(r["baseline_keywords"]["hit_count"] for r in runs) / len(runs)
        avg_inj_kw = sum(r["injected_keywords"]["hit_count"] for r in runs) / len(runs)
        avg_ppl_r = sum(r["ppl_ratio"] for r in runs) / len(runs)
        coherence = (
            "COHERENT" if avg_ppl_r < 3.0 else "OK" if avg_ppl_r < 10.0 else "DEGRADED"
        )
        print(
            f"  {profile_name:<10} | {n_pers:>5}/{len(runs):<6} | "
            f"{avg_bl_kw:>11.1f} | {avg_inj_kw:>12.1f} | "
            f"{avg_ppl_r:>13.2f} | {coherence:>9}"
        )

    # Global metrics
    total_personalized = sum(1 for r in all_results if r["personalized"])
    total_baseline_clean = sum(1 for r in all_results if r["baseline_clean"])
    total_coherent = sum(1 for r in all_results if r["ppl_ratio"] < 3.0)
    avg_ppl_ratio = sum(r["ppl_ratio"] for r in all_results) / len(all_results)
    avg_inj_hits = sum(r["injected_keywords"]["hit_count"] for r in all_results) / len(
        all_results
    )
    avg_bl_hits = sum(r["baseline_keywords"]["hit_count"] for r in all_results) / len(
        all_results
    )

    print()
    print(f"  Total pairs:                {len(all_results)}")
    print(
        f"  Personalized (INJ > BL):    {total_personalized}/{len(all_results)} "
        f"({100*total_personalized/len(all_results):.0f}%)"
    )
    print(
        f"  Baseline clean (0 hits):    {total_baseline_clean}/{len(all_results)} "
        f"({100*total_baseline_clean/len(all_results):.0f}%)"
    )
    print(
        f"  Coherent (PPL ratio < 3):   {total_coherent}/{len(all_results)} "
        f"({100*total_coherent/len(all_results):.0f}%)"
    )
    print(f"  Avg PPL ratio:              {avg_ppl_ratio:.2f}")
    print(
        f"  Avg keyword hits:           baseline={avg_bl_hits:.1f}  injected={avg_inj_hits:.1f}"
    )
    print()

    # ── Success criteria ──────────────────────────────────────────────────
    pct_personalized = total_personalized / len(all_results)
    pct_baseline_clean = total_baseline_clean / len(all_results)
    pct_coherent = total_coherent / len(all_results)

    criteria = {
        "≥80% pairs show profile-relevant content (personalized)": pct_personalized
        >= 0.80,
        "Baseline shows minimal profile-specific content": avg_bl_hits < 1.0,
        "Output remains coherent (≥50% pairs PPL ratio < 3)": pct_coherent >= 0.50,
        "Injected keyword hits > baseline hits (global avg)": avg_inj_hits
        > avg_bl_hits,
    }

    all_pass = all(criteria.values())

    print("  SUCCESS CRITERIA:")
    for name, passed in criteria.items():
        icon = "PASS" if passed else "FAIL"
        print(f"    [{icon}] {name}")

    print()
    if all_pass:
        print("  >>> LEVEL 2 PASSED <<<")
        print(
            "  Profile K,V injection produces semantically meaningful personalization."
        )
        print("  The MemoryFusedAdapter channel carries information, not just noise.")
        print("  Proceed to Level 3: Trained Memory Encoder.")
    else:
        print("  >>> LEVEL 2 PARTIALLY PASSED — review failing criteria <<<")
        # Still useful: show what worked
        if pct_personalized > 0:
            print(f"  Note: {100*pct_personalized:.0f}% personalization achieved.")
        if avg_inj_hits > avg_bl_hits:
            print("  Note: Injection DID increase keyword hits over baseline.")

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "method": "Profile K,V extraction via forward pass + DynamicCache injection",
        "model_id": config.model_id,
        "max_new_tokens": config.max_new_tokens,
        "profiles": {k: v for k, v in config.profiles.items()},
        "queries": config.queries,
        "results": all_results,
        "summary": {
            "total_pairs": len(all_results),
            "personalized_count": total_personalized,
            "personalized_pct": pct_personalized,
            "baseline_clean_count": total_baseline_clean,
            "baseline_clean_pct": pct_baseline_clean,
            "coherent_count": total_coherent,
            "coherent_pct": pct_coherent,
            "avg_ppl_ratio": avg_ppl_ratio,
            "avg_injected_keyword_hits": avg_inj_hits,
            "avg_baseline_keyword_hits": avg_bl_hits,
        },
        "criteria": {k: v for k, v in criteria.items()},
        "all_pass": all_pass,
        "elapsed_seconds": elapsed,
    }

    output_file = results_path / "level2_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Results saved to: {output_file}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

"""
Level 2b: Behavioral Steering — System-Message Profile Injection

Level 2a showed: raw profile K,V produce 24% personalization.
Coherence was perfect (PPL ratio 1.09), but the model treats raw K,V
as "prior context" — it has no reason to use them.

Fix: Wrap the profile in a SYSTEM MESSAGE through the chat template.
The K,V now encode "this is a system instruction about the user,"
which chat models are trained to follow.

Also try: include profile as a FULL CONVERSATION TURN (system + user ack),
so the K,V represent an established conversation context.
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
    max_new_tokens: int = 150

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

    # Injection strategies to compare
    strategies: list[str] = field(
        default_factory=lambda: [
            "raw",  # Raw text (Level 2a baseline)
            "system",  # System message format
            "context",  # Full system + user ack conversation turn
        ]
    )

    results_dir: str = "experiments/results/level2"


# ──────────────────────────────────────────────────────────────────────────
# K,V extraction with different framing strategies
# ──────────────────────────────────────────────────────────────────────────


def extract_kv(model, tokenizer, profile_text, strategy, device):
    """Extract K,V from profile text using different framing strategies.

    Returns (cache, num_tokens) where cache is a DynamicCache.
    """
    if strategy == "raw":
        # Raw profile text — no chat formatting
        text = profile_text

    elif strategy == "system":
        # Wrap as system message through chat template
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a helpful personal assistant. Here is everything you know "
                    f"about the user you are talking to:\n\n{profile_text}\n\n"
                    f"Always personalize your responses based on this information. "
                    f"Refer to their location, interests, profession, and preferences."
                ),
            },
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    elif strategy == "context":
        # Full conversation turn: system + user acknowledgment
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a helpful personal assistant. Here is everything you know "
                    f"about the user you are talking to:\n\n{profile_text}\n\n"
                    f"Always personalize your responses based on this information. "
                    f"Refer to their location, interests, profession, and preferences."
                ),
            },
            {"role": "user", "content": "Hello! Can you help me with some questions?"},
            {
                "role": "assistant",
                "content": (
                    "Of course! I'm happy to help. I'll keep your background and "
                    "preferences in mind as we chat. What would you like to know?"
                ),
            },
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    tokens = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
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

    return outputs.past_key_values, seq_len


# ──────────────────────────────────────────────────────────────────────────
# Generation with pre-populated cache
# ──────────────────────────────────────────────────────────────────────────


def generate_greedy(model, tokenizer, input_ids, max_new_tokens, cache=None, num_mem=0):
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

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────


def clone_cache(cache: DynamicCache) -> DynamicCache:
    new_cache = DynamicCache()
    for layer_idx in range(len(cache)):
        k, v = cache[layer_idx]
        new_cache.update(k.clone(), v.clone(), layer_idx)
    return new_cache


def score_keywords(text: str, keywords: list[str]) -> dict:
    text_lower = text.lower()
    hits = [kw for kw in keywords if kw.lower() in text_lower]
    return {
        "hit_count": len(hits),
        "total_keywords": len(keywords),
        "hit_ratio": len(hits) / len(keywords) if keywords else 0,
        "hits": hits,
    }


def score_coherence(model, tokenizer, text: str, device: str) -> float:
    if not text or len(text.strip()) == 0:
        return float("inf")
    tokens = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    if tokens.shape[1] < 2:
        return float("inf")
    with torch.no_grad():
        outputs = model(input_ids=tokens)
    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = tokens[:, 1:]
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
    )
    return math.exp(min(loss.item(), 20))


# ──────────────────────────────────────────────────────────────────────────
# Also generate an IN-PROMPT baseline (profile pasted into context window)
# ──────────────────────────────────────────────────────────────────────────


def generate_in_prompt(model, tokenizer, profile_text, query, max_new_tokens, device):
    """Generate with profile in the prompt — the 'gold standard' we're comparing against."""
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a helpful personal assistant. Here is everything you know "
                f"about the user:\n\n{profile_text}\n\n"
                f"Always personalize your responses based on this information."
            ),
        },
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    return generate_greedy(model, tokenizer, input_ids, max_new_tokens)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main():
    config = Config()
    results_path = Path(config.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  LEVEL 2b: BEHAVIORAL STEERING — SYSTEM-MESSAGE INJECTION")
    print("=" * 72)
    print()
    print("  Strategies: raw (L2a), system-msg, full-context-turn")
    print("  Plus: in-prompt gold standard for comparison")
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
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print()

    # --- Extract K,V caches for each (profile, strategy) ---
    print("Extracting profile K,V caches...")
    caches = {}  # (profile, strategy) -> (cache, num_tokens)
    for profile_name, profile_text in config.profiles.items():
        for strategy in config.strategies:
            cache, num_tokens = extract_kv(
                model, tokenizer, profile_text, strategy, config.device
            )
            caches[(profile_name, strategy)] = (cache, num_tokens)
            print(f"  {profile_name}/{strategy}: {num_tokens} tokens")
    print()

    # --- Generate baselines ---
    print("Computing baselines (no injection + in-prompt gold standard)...")
    baselines = {}  # query -> text
    gold = {}  # (profile, query) -> text

    for query in config.queries:
        messages = [{"role": "user", "content": query}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(config.device)
        baselines[query] = generate_greedy(
            model, tokenizer, input_ids, config.max_new_tokens
        )
        print(f"  BL: {query[:50]} -> {baselines[query].strip()[:60]}")

    print()
    print("Computing in-prompt gold standard...")
    for profile_name, profile_text in config.profiles.items():
        for query in config.queries:
            gold[(profile_name, query)] = generate_in_prompt(
                model,
                tokenizer,
                profile_text,
                query,
                config.max_new_tokens,
                config.device,
            )
        print(f"  GOLD: {profile_name} done")
        gc.collect()
        torch.cuda.empty_cache()
    print()

    # --- Run all (strategy, profile, query) triples ---
    all_results = []
    total = len(config.strategies) * len(config.profiles) * len(config.queries)
    run_idx = 0
    t_start = time.time()

    for strategy in config.strategies:
        print(f"\n{'═' * 72}")
        print(f"  STRATEGY: {strategy}")
        print(f"{'═' * 72}")

        for profile_name in config.profiles:
            cache, num_tokens = caches[(profile_name, strategy)]
            keywords = config.profile_keywords[profile_name]

            for query in config.queries:
                run_idx += 1

                # --- Generate injected response ---
                messages = [{"role": "user", "content": query}]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(
                    config.device
                )

                injected_cache = clone_cache(cache)
                injected_text = generate_greedy(
                    model,
                    tokenizer,
                    input_ids,
                    config.max_new_tokens,
                    cache=injected_cache,
                    num_mem=num_tokens,
                )

                # --- Score ---
                bl_text = baselines[query]
                gold_text = gold[(profile_name, query)]

                bl_kw = score_keywords(bl_text, keywords)
                inj_kw = score_keywords(injected_text, keywords)
                gold_kw = score_keywords(gold_text, keywords)

                inj_ppl = score_coherence(
                    model, tokenizer, injected_text.strip(), config.device
                )
                bl_ppl = score_coherence(
                    model, tokenizer, bl_text.strip(), config.device
                )

                personalized = inj_kw["hit_count"] > bl_kw["hit_count"]

                result = {
                    "strategy": strategy,
                    "profile": profile_name,
                    "query": query,
                    "num_profile_tokens": num_tokens,
                    "baseline_text": bl_text.strip()[:200],
                    "injected_text": injected_text.strip()[:200],
                    "gold_text": gold_text.strip()[:200],
                    "baseline_keywords": bl_kw,
                    "injected_keywords": inj_kw,
                    "gold_keywords": gold_kw,
                    "baseline_ppl": bl_ppl,
                    "injected_ppl": inj_ppl,
                    "ppl_ratio": inj_ppl / bl_ppl if bl_ppl > 0 else float("inf"),
                    "personalized": personalized,
                }
                all_results.append(result)

                tag = "PERS" if personalized else "----"
                print(
                    f"  [{run_idx:3d}/{total}] {profile_name:<7} "
                    f"Q={query[:30]:<30}  "
                    f"BL={bl_kw['hit_count']:>2} INJ={inj_kw['hit_count']:>2} "
                    f"GOLD={gold_kw['hit_count']:>2}  "
                    f"PPL_r={result['ppl_ratio']:5.2f}  [{tag}]"
                )

                gc.collect()
                torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY BY STRATEGY")
    print("=" * 72)

    from collections import defaultdict

    by_strategy = defaultdict(list)
    for r in all_results:
        by_strategy[r["strategy"]].append(r)

    print()
    print(
        f"  {'Strategy':<12} | {'Pers %':>7} | {'Avg INJ kw':>10} | "
        f"{'Avg GOLD kw':>11} | {'Avg BL kw':>9} | {'Avg PPL_r':>9} | {'Coherent':>8}"
    )
    print("  " + "─" * 85)

    strategy_summaries = {}
    for strategy in config.strategies:
        runs = by_strategy[strategy]
        n_pers = sum(1 for r in runs if r["personalized"])
        avg_inj = sum(r["injected_keywords"]["hit_count"] for r in runs) / len(runs)
        avg_gold = sum(r["gold_keywords"]["hit_count"] for r in runs) / len(runs)
        avg_bl = sum(r["baseline_keywords"]["hit_count"] for r in runs) / len(runs)
        avg_ppl = sum(r["ppl_ratio"] for r in runs) / len(runs)
        n_coherent = sum(1 for r in runs if r["ppl_ratio"] < 3.0)
        pct_pers = n_pers / len(runs)

        coherence_tag = "GOOD" if avg_ppl < 2.0 else "OK" if avg_ppl < 5.0 else "BAD"

        print(
            f"  {strategy:<12} | {100*pct_pers:6.0f}% | {avg_inj:>10.1f} | "
            f"{avg_gold:>11.1f} | {avg_bl:>9.1f} | {avg_ppl:>9.2f} | "
            f"{n_coherent:>3}/{len(runs):<3} [{coherence_tag}]"
        )

        strategy_summaries[strategy] = {
            "personalized_pct": pct_pers,
            "avg_injected_kw": avg_inj,
            "avg_gold_kw": avg_gold,
            "avg_baseline_kw": avg_bl,
            "avg_ppl_ratio": avg_ppl,
            "coherent_count": n_coherent,
            "total": len(runs),
        }

    # Gold standard summary
    all_gold_kw = sum(r["gold_keywords"]["hit_count"] for r in all_results) / len(
        all_results
    )
    print(f"\n  Gold standard (in-prompt) avg keyword hits: {all_gold_kw:.1f}")

    # Best strategy
    best = max(strategy_summaries.items(), key=lambda x: x[1]["personalized_pct"])
    print(
        f"\n  Best injection strategy: {best[0]} "
        f"({100*best[1]['personalized_pct']:.0f}% personalized)"
    )

    # Per-profile breakdown for best strategy
    print(f"\n  PER-PROFILE DETAIL (strategy={best[0]}):")
    print("  " + "─" * 72)

    best_runs = by_strategy[best[0]]
    by_profile = defaultdict(list)
    for r in best_runs:
        by_profile[r["profile"]].append(r)

    for profile_name in config.profiles:
        runs = by_profile[profile_name]
        for r in runs:
            q = r["query"][:40]
            inj_hits = r["injected_keywords"]["hits"]
            gold_hits = r["gold_keywords"]["hits"]
            print(
                f"    {profile_name:<7} Q={q:<40} " f"INJ={inj_hits}  GOLD={gold_hits}"
            )

    # ── Success criteria ──────────────────────────────────────────────────
    best_pct = best[1]["personalized_pct"]
    best_avg_inj = best[1]["avg_injected_kw"]
    best_avg_bl = best[1]["avg_baseline_kw"]
    best_coherent_pct = best[1]["coherent_count"] / best[1]["total"]

    print()
    print("  SUCCESS CRITERIA (best strategy):")
    criteria = {
        f"≥80% personalized ({100*best_pct:.0f}% achieved)": best_pct >= 0.80,
        "Baseline shows minimal profile content": best_avg_bl < 1.0,
        "≥50% coherent (PPL ratio < 3)": best_coherent_pct >= 0.50,
        "Injected keyword hits > baseline": best_avg_inj > best_avg_bl,
        "Injection reaches ≥30% of gold-standard keyword hits": (
            best_avg_inj >= 0.30 * all_gold_kw if all_gold_kw > 0 else False
        ),
    }

    all_pass = all(criteria.values())
    for name, passed in criteria.items():
        icon = "PASS" if passed else "FAIL"
        print(f"    [{icon}] {name}")

    print()
    if all_pass:
        print("  >>> LEVEL 2 PASSED <<<")
    else:
        passed_count = sum(criteria.values())
        print(f"  >>> {passed_count}/{len(criteria)} CRITERIA PASSED <<<")

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "method": "Multi-strategy profile K,V injection + in-prompt gold standard",
        "model_id": config.model_id,
        "strategies": config.strategies,
        "results": all_results,
        "strategy_summaries": strategy_summaries,
        "gold_avg_kw": all_gold_kw,
        "criteria": {k: v for k, v in criteria.items()},
        "all_pass": all_pass,
        "elapsed_seconds": elapsed,
    }

    output_file = results_path / "level2b_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  Results saved to: {output_file}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

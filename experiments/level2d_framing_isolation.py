"""
Level 2d: Framing Isolation — Does K,V Format Determine Personalization?

Level 2c proved:
  - fixed_split (system-formatted K,V) matches gold at 24/25
  - l2b_system (separate tokenization) scored 0/25

But L2b had TWO confounds:
  1. The profile K,V were formatted differently (system msg vs raw)
  2. The suffix tokens had a DUPLICATE system message (tokenization bug)

This experiment isolates confound #1 by fixing #2.

Design:
  - ALL conditions use the SAME suffix tokens:
      <|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n
    Manually constructed — no apply_chat_template, no default system message.

  - ONLY the prefix K,V change:
    A) system_formatted: <|im_start|>system\n{instruction+profile}<|im_end|>\n
       → K,V encode profile as system instruction (what fixed_split proved)
    B) raw_text: {profile}
       → K,V encode raw profile text (no chat markers)
    C) user_formatted: <|im_start|>user\nHere is my profile: {profile}<|im_end|>\n
                       <|im_start|>assistant\nGot it...<|im_end|>\n
       → K,V encode profile as a prior conversation turn
    D) no_injection: no prefix K,V (just suffix tokens alone)
       → baseline

If A >> B and A >> C:
  → "instruction subspace" theory confirmed: K,V must look like system message
If A ≈ B ≈ C >> D:
  → framing doesn't matter; any K,V carrying profile info works
If A > C > B >> D:
  → gradient of "instruction-ness" matters
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

    results_dir: str = "experiments/results/level2"


# ──────────────────────────────────────────────────────────────────────────
# Core: manual greedy generation
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


def extract_kv(model, text, device):
    """Run text through model, return (cache, num_tokens)."""
    tokens = model._tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    seq_len = tokens.shape[1]
    with torch.no_grad():
        outputs = model(
            input_ids=tokens,
            attention_mask=torch.ones(1, seq_len, dtype=torch.long, device=device),
            position_ids=torch.arange(seq_len, device=device).unsqueeze(0),
            use_cache=True,
        )
    return outputs.past_key_values, seq_len


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
# Build the IDENTICAL suffix for all conditions
# ──────────────────────────────────────────────────────────────────────────


def build_suffix(tokenizer, query):
    """Build the user+query+assistant-prompt suffix manually.

    Returns the string: <|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n

    This is constructed manually to avoid apply_chat_template inserting
    a default system message.
    """
    return f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"


# ──────────────────────────────────────────────────────────────────────────
# Build prefix text for each framing condition
# ──────────────────────────────────────────────────────────────────────────


def build_prefix_system(profile_text):
    """System-message framing (what fixed_split uses)."""
    return (
        f"<|im_start|>system\n"
        f"You are a helpful personal assistant. Here is everything you know "
        f"about the user you are talking to:\n\n{profile_text}\n\n"
        f"Always personalize your responses based on this information. "
        f"Refer to their location, interests, profession, and preferences."
        f"<|im_end|>\n"
    )


def build_prefix_raw(profile_text):
    """Raw text — no chat template markers at all."""
    return profile_text


def build_prefix_user_turn(profile_text):
    """User-turn framing — profile presented as a prior conversation exchange."""
    return (
        f"<|im_start|>user\n"
        f"Here is my personal profile for reference: {profile_text}"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"Thank you for sharing your profile! I'll keep your background, location, "
        f"interests, and preferences in mind for all my responses."
        f"<|im_end|>\n"
    )


# ──────────────────────────────────────────────────────────────────────────
# Also: gold standard (full prompt, no split) for reference
# ──────────────────────────────────────────────────────────────────────────


def run_gold(model, tokenizer, profile_text, query, max_new_tokens, device):
    """Gold standard: profile in system message + query, single forward pass."""
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
    print("  LEVEL 2d: FRAMING ISOLATION TEST")
    print("=" * 72)
    print()
    print("  Question: Does K,V FORMAT determine personalization,")
    print("  or does correct tokenization alone explain everything?")
    print()
    print("  Conditions:")
    print("    A) system_formatted — profile as system message K,V")
    print("    B) raw_text — profile as raw text K,V")
    print("    C) user_turn — profile as user+assistant conversation K,V")
    print("    D) no_injection — baseline (suffix only)")
    print("    E) gold — full prompt, no split (reference)")
    print()
    print("  All conditions share IDENTICAL suffix tokens.")
    print("  Only the prefix K,V differ.")
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
    # Stash tokenizer on model for extract_kv helper
    model._tokenizer = tokenizer
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB\n")

    # --- Verify suffix construction matches gold ---
    print("─" * 72)
    print("  SUFFIX VERIFICATION")
    print("─" * 72)
    test_query = config.queries[0]
    test_profile = list(config.profiles.values())[0]

    gold_messages = [
        {
            "role": "system",
            "content": (
                f"You are a helpful personal assistant. Here is everything you know "
                f"about the user you are talking to:\n\n{test_profile}\n\n"
                f"Always personalize your responses based on this information. "
                f"Refer to their location, interests, profession, and preferences."
            ),
        },
        {"role": "user", "content": test_query},
    ]
    gold_text = tokenizer.apply_chat_template(
        gold_messages, tokenize=False, add_generation_prompt=True
    )
    sys_prefix = build_prefix_system(test_profile)
    manual_suffix = build_suffix(tokenizer, test_query)
    reconstructed = sys_prefix + manual_suffix

    gold_ids = tokenizer(gold_text, return_tensors="pt")["input_ids"][0].tolist()
    recon_ids = tokenizer(reconstructed, return_tensors="pt")["input_ids"][0].tolist()

    match = gold_ids == recon_ids
    print(f"  Gold tokens:          {len(gold_ids)}")
    print(f"  Reconstructed tokens: {len(recon_ids)}")
    print(f"  Exact match: {match}")
    if not match:
        for i in range(min(len(gold_ids), len(recon_ids))):
            if gold_ids[i] != recon_ids[i]:
                print(f"  First divergence at token {i}:")
                print(f"    Gold: {gold_ids[i]} = {tokenizer.decode([gold_ids[i]])!r}")
                print(
                    f"    Reco: {recon_ids[i]} = {tokenizer.decode([recon_ids[i]])!r}"
                )
                break
        if len(gold_ids) != len(recon_ids):
            print(f"  Length diff: gold={len(gold_ids)} reco={len(recon_ids)}")
    print()

    # --- Pre-extract K,V caches for each (profile, framing) ---
    print("─" * 72)
    print("  EXTRACTING PREFIX K,V CACHES")
    print("─" * 72)

    prefix_builders = {
        "system_formatted": build_prefix_system,
        "raw_text": build_prefix_raw,
        "user_turn": build_prefix_user_turn,
    }

    caches = {}  # (profile_name, framing) -> (cache, num_tokens)
    for profile_name, profile_text in config.profiles.items():
        for framing, builder in prefix_builders.items():
            prefix_text = builder(profile_text)
            tokens = tokenizer(prefix_text, return_tensors="pt")["input_ids"].to(
                config.device
            )
            seq_len = tokens.shape[1]

            with torch.no_grad():
                outputs = model(
                    input_ids=tokens,
                    attention_mask=torch.ones(
                        1, seq_len, dtype=torch.long, device=config.device
                    ),
                    position_ids=torch.arange(seq_len, device=config.device).unsqueeze(
                        0
                    ),
                    use_cache=True,
                )
            caches[(profile_name, framing)] = (outputs.past_key_values, seq_len)
            print(f"  {profile_name:8s} / {framing:18s}: {seq_len} tokens")

    print()

    # --- Run all conditions ---
    conditions = ["system_formatted", "raw_text", "user_turn", "no_injection", "gold"]
    total = len(conditions) * len(config.profiles) * len(config.queries)
    all_results = []
    run_idx = 0
    t_start = time.time()

    for condition in conditions:
        print(f"\n{'═' * 72}")
        print(f"  CONDITION: {condition}")
        print(f"{'═' * 72}")

        for profile_name, profile_text in config.profiles.items():
            keywords = config.profile_keywords[profile_name]

            for query in config.queries:
                run_idx += 1

                if condition == "gold":
                    text = run_gold(
                        model,
                        tokenizer,
                        profile_text,
                        query,
                        config.max_new_tokens,
                        config.device,
                    )
                elif condition == "no_injection":
                    suffix = build_suffix(tokenizer, query)
                    suffix_ids = tokenizer(suffix, return_tensors="pt")["input_ids"].to(
                        config.device
                    )
                    text = generate_greedy(
                        model,
                        tokenizer,
                        suffix_ids,
                        config.max_new_tokens,
                    )
                else:
                    # Injection conditions: prefix K,V + manual suffix
                    cache, num_mem = caches[(profile_name, condition)]
                    injected_cache = clone_cache(cache)

                    suffix = build_suffix(tokenizer, query)
                    suffix_ids = tokenizer(suffix, return_tensors="pt")["input_ids"].to(
                        config.device
                    )

                    text = generate_greedy(
                        model,
                        tokenizer,
                        suffix_ids,
                        config.max_new_tokens,
                        cache=injected_cache,
                        num_mem=num_mem,
                    )

                kw = score_keywords(text, keywords)
                ppl = score_coherence(model, tokenizer, text.strip(), config.device)

                result = {
                    "condition": condition,
                    "profile": profile_name,
                    "query": query,
                    "text": text.strip()[:300],
                    "keyword_hits": kw["hit_count"],
                    "keyword_ratio": kw["hit_ratio"],
                    "hits": kw["hits"],
                    "ppl": ppl,
                }
                all_results.append(result)

                print(
                    f"  [{run_idx:3d}/{total}] {profile_name:<7}  "
                    f"Q={query[:30]:<30}  "
                    f"KW={kw['hit_count']:2d}/{kw['total_keywords']}  "
                    f"PPL={ppl:6.1f}  "
                    f"{''.join(text.strip()[:60].splitlines())}"
                )

            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n{'═' * 72}")
    print(f"  ALL {total} RUNS COMPLETE IN {elapsed:.1f}s")
    print(f"{'═' * 72}\n")

    # ──────────────────────────────────────────────────────────────────
    # Aggregate results
    # ──────────────────────────────────────────────────────────────────
    print("─" * 72)
    print("  STRATEGY COMPARISON")
    print("─" * 72)
    print(
        f"  {'Condition':<20} {'Avg KW Hits':>12} {'Nonzero':>10} "
        f"{'Avg PPL':>10} {'Avg KW Ratio':>14}"
    )
    print(f"  {'─' * 20} {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 14}")

    for cond in conditions:
        cr = [r for r in all_results if r["condition"] == cond]
        avg_hits = sum(r["keyword_hits"] for r in cr) / len(cr)
        avg_ppl = sum(r["ppl"] for r in cr) / len(cr)
        avg_ratio = sum(r["keyword_ratio"] for r in cr) / len(cr)
        nonzero = sum(1 for r in cr if r["keyword_hits"] > 0)
        print(
            f"  {cond:<20} {avg_hits:>12.2f} {nonzero:>7}/{len(cr)}  "
            f"{avg_ppl:>10.1f} {avg_ratio:>14.3f}"
        )

    # --- Per-query breakdown ---
    print()
    print("─" * 72)
    print("  PER-QUERY BREAKDOWN (avg keyword hits across 5 profiles)")
    print("─" * 72)
    for query in config.queries:
        print(f"  Q: {query[:55]}")
        for cond in conditions:
            qr = [
                r for r in all_results if r["condition"] == cond and r["query"] == query
            ]
            avg = sum(r["keyword_hits"] for r in qr) / len(qr)
            bar = "█" * int(avg * 3)
            print(f"    {cond:<20}: {avg:5.2f}  {bar}")
        print()

    # --- Exact match vs gold ---
    print("─" * 72)
    print("  EXACT MATCH vs GOLD (first 100 chars)")
    print("─" * 72)
    for cond in conditions:
        if cond == "gold":
            continue
        matches = 0
        for profile_name in config.profiles:
            for query in config.queries:
                g = next(
                    r
                    for r in all_results
                    if r["condition"] == "gold"
                    and r["profile"] == profile_name
                    and r["query"] == query
                )
                c = next(
                    r
                    for r in all_results
                    if r["condition"] == cond
                    and r["profile"] == profile_name
                    and r["query"] == query
                )
                if g["text"][:100] == c["text"][:100]:
                    matches += 1
        print(f"  {cond:<20}: {matches}/25 exact match")

    # --- Hypothesis verdict ---
    print()
    print("─" * 72)
    print("  HYPOTHESIS VERDICT")
    print("─" * 72)

    sys_avg = (
        sum(
            r["keyword_hits"]
            for r in all_results
            if r["condition"] == "system_formatted"
        )
        / 25
    )
    raw_avg = (
        sum(r["keyword_hits"] for r in all_results if r["condition"] == "raw_text") / 25
    )
    usr_avg = (
        sum(r["keyword_hits"] for r in all_results if r["condition"] == "user_turn")
        / 25
    )
    gold_avg = (
        sum(r["keyword_hits"] for r in all_results if r["condition"] == "gold") / 25
    )
    none_avg = (
        sum(r["keyword_hits"] for r in all_results if r["condition"] == "no_injection")
        / 25
    )

    print(
        f"  gold={gold_avg:.2f}  system={sys_avg:.2f}  "
        f"raw={raw_avg:.2f}  user_turn={usr_avg:.2f}  "
        f"none={none_avg:.2f}"
    )
    print()

    if sys_avg > raw_avg * 2 and sys_avg > usr_avg * 2:
        print("  >>> CONFIRMED: 'Instruction subspace' theory.")
        print("  >>> K,V MUST look like system-message to drive personalization.")
    elif abs(sys_avg - raw_avg) < 1.0 and abs(sys_avg - usr_avg) < 1.0:
        print("  >>> REJECTED: Framing does NOT matter.")
        print("  >>> Any K,V carrying profile info works equally.")
        print("  >>> The L2b failure was PURELY a tokenization bug.")
    else:
        print("  >>> GRADIENT: Framing matters but is not binary.")
        print(f"  >>> system={sys_avg:.2f} vs raw={raw_avg:.2f} vs user={usr_avg:.2f}")
        print("  >>> More instruction-like framing produces stronger personalization.")

    # --- Save results ---
    out = {
        "experiment": "level2d_framing_isolation",
        "model": config.model_id,
        "hypothesis": "Does K,V format determine personalization, or does correct tokenization alone explain everything?",
        "design": {
            "suffix": "IDENTICAL across all conditions: <|im_start|>user\\n{query}<|im_end|>\\n<|im_start|>assistant\\n",
            "conditions": {
                "system_formatted": "Profile as system message K,V",
                "raw_text": "Profile as raw text K,V",
                "user_turn": "Profile as user+assistant conversation K,V",
                "no_injection": "No prefix K,V (baseline)",
                "gold": "Full prompt, no split (reference)",
            },
        },
        "results": all_results,
    }
    out_path = results_path / "level2d_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()

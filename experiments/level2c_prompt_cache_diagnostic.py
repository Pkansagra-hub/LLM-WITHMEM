"""
Level 2c: Prompt-Cache Diagnostic — Where Does the Gold-Standard Gap Come From?

Level 2b showed 28% personalization (injection) vs ~50% (gold standard).
The gap could come from:
  A) Tokenization misalignment (double BOS, missing newline between turns)
  B) K,V from profile alone lack cross-attention with query
  C) Fundamental information loss in K,V extraction

This experiment tests 3 strategies to isolate the cause:
  1. prompt_cache: Extract K,V from the FULL prompt (system+user+query),
     then generate from the tiny assistant-prompt tail.
     => Should EXACTLY match gold standard (proves K,V extraction is lossless)
  2. fixed_split: Extract K,V from system message WITH trailing tokens that
     match the gold standard's byte-for-byte prefix. Generate user+query
     WITHOUT extra BOS.
     => If it matches gold standard, the L2b gap was tokenization.
  3. system (from L2b): For comparison baseline.

If prompt_cache == gold standard but fixed_split != gold standard,
then the gap is NOT tokenization but query-unaware K,V.
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
# Configuration (same profiles/queries as L2b)
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
# Core generation — manual autoregressive loop
# ──────────────────────────────────────────────────────────────────────────


def generate_greedy(model, tokenizer, input_ids, max_new_tokens, cache=None, num_mem=0):
    """Generate text greedily with optional pre-populated K,V cache."""
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
# Tokenization diagnostic
# ──────────────────────────────────────────────────────────────────────────


def diagnose_tokenization(tokenizer, profile_text, query):
    """Print exact token sequences for gold vs split to find misalignment."""
    # Gold standard: single template application
    gold_messages = [
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
    gold_text = tokenizer.apply_chat_template(
        gold_messages, tokenize=False, add_generation_prompt=True
    )
    gold_ids = tokenizer(gold_text, return_tensors="pt")["input_ids"][0].tolist()

    # Split: system only
    sys_messages = [
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
    sys_text = tokenizer.apply_chat_template(
        sys_messages, tokenize=False, add_generation_prompt=False
    )
    sys_ids = tokenizer(sys_text, return_tensors="pt")["input_ids"][0].tolist()

    # Split: user only
    user_messages = [{"role": "user", "content": query}]
    user_text = tokenizer.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    user_ids = tokenizer(user_text, return_tensors="pt")["input_ids"][0].tolist()

    # Find the EXACT split point in the gold sequence
    # Gold should be: [BOS?] + system_tokens + user_tokens
    print(f"  Gold text length: {len(gold_text)}")
    print(f"  System text length: {len(sys_text)}")
    print(f"  User text length: {len(user_text)}")
    print(f"  Gold token count: {len(gold_ids)}")
    print(f"  System token count: {len(sys_ids)}")
    print(f"  User token count: {len(user_ids)}")
    print(f"  Sum sys+user: {len(sys_ids) + len(user_ids)}")
    print()

    # Check if system_ids is a prefix of gold_ids
    is_prefix = gold_ids[: len(sys_ids)] == sys_ids
    print(f"  System tokens are prefix of gold: {is_prefix}")
    if not is_prefix:
        # Find where they diverge
        for i in range(min(len(sys_ids), len(gold_ids))):
            if i >= len(sys_ids) or i >= len(gold_ids) or sys_ids[i] != gold_ids[i]:
                print(f"  Diverge at position {i}:")
                print(
                    f"    Gold: {gold_ids[max(0,i-2):i+3]} -> {tokenizer.decode(gold_ids[max(0,i-2):i+3])!r}"
                )
                print(
                    f"    Sys:  {sys_ids[max(0,i-2):i+3]} -> {tokenizer.decode(sys_ids[max(0,i-2):i+3])!r}"
                )
                break

    # Check what comes after system in gold
    remainder_ids = gold_ids[len(sys_ids) :]
    print(f"  Gold remainder after sys prefix: {len(remainder_ids)} tokens")
    if remainder_ids:
        print(f"    First 10: {remainder_ids[:10]}")
        print(f"    Decoded: {tokenizer.decode(remainder_ids[:10])!r}")

    # Check if user_ids matches remainder
    if remainder_ids:
        matches = remainder_ids == user_ids
        print(f"  Remainder matches user_ids: {matches}")
        if not matches and len(user_ids) > 0 and len(remainder_ids) > 0:
            # BOS stripping check
            if user_ids[0] != remainder_ids[0] and user_ids[1:] == remainder_ids:
                print(f"  >>> BOS TOKEN MISMATCH! user_ids has extra BOS={user_ids[0]}")
                print("  >>> Fix: strip first token from user_ids")
            elif user_ids == remainder_ids[1:]:
                print(
                    f"  >>> Gold has extra token at split: {remainder_ids[0]} = {tokenizer.decode([remainder_ids[0]])!r}"
                )

    # Show the actual text split
    print()
    print(f"  System text ends with: {sys_text[-50:]!r}")
    print(f"  User text starts with: {user_text[:50]!r}")
    print(f"  Gold at split point: {gold_text[len(sys_text)-10:len(sys_text)+10]!r}")

    return gold_ids, sys_ids, user_ids, gold_text, sys_text, user_text


# ──────────────────────────────────────────────────────────────────────────
# Strategy implementations
# ──────────────────────────────────────────────────────────────────────────


def strategy_gold(model, tokenizer, profile_text, query, max_new_tokens, device):
    """Gold standard: profile in prompt, single forward pass."""
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


def strategy_prompt_cache(
    model, tokenizer, profile_text, query, max_new_tokens, device
):
    """Prompt-cache: Extract K,V from FULL prompt, generate from nothing.

    This should EXACTLY match gold standard — proves K,V extraction is lossless.
    """
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
    seq_len = input_ids.shape[1]

    # Forward pass to extract K,V for the full prompt
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones(1, seq_len, dtype=torch.long, device=device),
            position_ids=torch.arange(seq_len, device=device).unsqueeze(0),
            use_cache=True,
        )

    # Now generate using ONLY the cache — no input tokens needed
    # The last token's logits are already computed
    cache = outputs.past_key_values
    logits = outputs.logits
    cur_pos = seq_len
    generated_ids = []

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated_ids.append(next_token.item())

        with torch.no_grad():
            outputs = model(
                input_ids=next_token.unsqueeze(0),
                attention_mask=torch.ones(
                    1, cur_pos + 1, dtype=torch.long, device=device
                ),
                position_ids=torch.tensor([[cur_pos]], device=device),
                past_key_values=cache,
                use_cache=True,
            )
        cache = outputs.past_key_values
        logits = outputs.logits
        cur_pos += 1

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def strategy_fixed_split(model, tokenizer, profile_text, query, max_new_tokens, device):
    """Fixed split: Use gold standard tokenization, but split K,V at the right point.

    1. Tokenize the FULL gold-standard prompt
    2. Identify the split point (end of system message)
    3. Extract K,V for the system prefix
    4. Feed the remainder tokens with proper position offsets
    """
    # Build the full prompt as gold standard does
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
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]

    # Build just the system message to find the split point
    sys_messages = [
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
    sys_text = tokenizer.apply_chat_template(
        sys_messages, tokenize=False, add_generation_prompt=False
    )
    sys_ids = tokenizer(sys_text, return_tensors="pt")["input_ids"]
    split_point = sys_ids.shape[1]

    # Extract K,V for system prefix (using the gold tokenization)
    prefix_ids = full_ids[:, :split_point].to(device)
    suffix_ids = full_ids[:, split_point:].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=prefix_ids,
            attention_mask=torch.ones(1, split_point, dtype=torch.long, device=device),
            position_ids=torch.arange(split_point, device=device).unsqueeze(0),
            use_cache=True,
        )

    cache = outputs.past_key_values

    # Generate from suffix tokens (user + query + assistant prompt)
    return generate_greedy(
        model, tokenizer, suffix_ids, max_new_tokens, cache=cache, num_mem=split_point
    )


def strategy_l2b_system(model, tokenizer, profile_text, query, max_new_tokens, device):
    """L2b system strategy (for comparison): separate tokenization of system and user."""
    # Extract system K,V
    sys_messages = [
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
    sys_text = tokenizer.apply_chat_template(
        sys_messages, tokenize=False, add_generation_prompt=False
    )
    sys_ids = tokenizer(sys_text, return_tensors="pt")["input_ids"].to(device)
    num_sys = sys_ids.shape[1]

    with torch.no_grad():
        outputs = model(
            input_ids=sys_ids,
            attention_mask=torch.ones(1, num_sys, dtype=torch.long, device=device),
            position_ids=torch.arange(num_sys, device=device).unsqueeze(0),
            use_cache=True,
        )
    cache = outputs.past_key_values

    # Generate from separately-tokenized user message
    user_messages = [{"role": "user", "content": query}]
    user_text = tokenizer.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    user_ids = tokenizer(user_text, return_tensors="pt")["input_ids"].to(device)

    injected_cache = clone_cache(cache)
    return generate_greedy(
        model,
        tokenizer,
        user_ids,
        max_new_tokens,
        cache=injected_cache,
        num_mem=num_sys,
    )


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────


def main():
    config = Config()
    results_path = Path(config.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  LEVEL 2c: PROMPT-CACHE DIAGNOSTIC")
    print("=" * 72)
    print()
    print("  Strategies:")
    print("    gold         — profile in prompt (reference)")
    print("    prompt_cache — K,V from full prompt (lossless check)")
    print("    fixed_split  — K,V from system prefix, gold tokenization")
    print("    l2b_system   — K,V from system msg, separate tokenization")
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
    print(f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB\n")

    # --- Tokenization diagnostic (first profile + first query) ---
    print("─" * 72)
    print("  TOKENIZATION DIAGNOSTIC")
    print("─" * 72)
    first_profile = list(config.profiles.keys())[0]
    first_query = config.queries[0]
    print(f"  Profile: {first_profile}")
    print(f"  Query: {first_query}")
    print()
    gold_ids, sys_ids, user_ids, _, _, _ = diagnose_tokenization(
        tokenizer, config.profiles[first_profile], first_query
    )
    print()

    # --- Run all strategies ---
    strategies = {
        "gold": strategy_gold,
        "prompt_cache": strategy_prompt_cache,
        "fixed_split": strategy_fixed_split,
        "l2b_system": strategy_l2b_system,
    }

    all_results = []
    total = len(strategies) * len(config.profiles) * len(config.queries)
    run_idx = 0
    t_start = time.time()

    for strat_name, strat_fn in strategies.items():
        print(f"\n{'═' * 72}")
        print(f"  STRATEGY: {strat_name}")
        print(f"{'═' * 72}")

        for profile_name, profile_text in config.profiles.items():
            keywords = config.profile_keywords[profile_name]

            for query in config.queries:
                run_idx += 1

                text = strat_fn(
                    model,
                    tokenizer,
                    profile_text,
                    query,
                    config.max_new_tokens,
                    config.device,
                )

                kw_score = score_keywords(text, keywords)
                ppl = score_coherence(model, tokenizer, text.strip(), config.device)

                result = {
                    "strategy": strat_name,
                    "profile": profile_name,
                    "query": query,
                    "text": text.strip()[:300],
                    "keyword_hits": kw_score["hit_count"],
                    "keyword_ratio": kw_score["hit_ratio"],
                    "hits": kw_score["hits"],
                    "ppl": ppl,
                }
                all_results.append(result)

                print(
                    f"  [{run_idx:3d}/{total}] {profile_name:<7} "
                    f"Q={query[:30]:<30}  "
                    f"KW={kw_score['hit_count']:2d}/{kw_score['total_keywords']}  "
                    f"PPL={ppl:.1f}  "
                    f"TEXT={text.strip()[:60]}"
                )

            gc.collect()
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\n{'═' * 72}")
    print(f"  ALL {total} RUNS COMPLETE IN {elapsed:.1f}s")
    print(f"{'═' * 72}\n")

    # --- Aggregate and compare ---
    print("─" * 72)
    print("  STRATEGY COMPARISON")
    print("─" * 72)

    for strat_name in strategies:
        strat_results = [r for r in all_results if r["strategy"] == strat_name]
        total_hits = sum(r["keyword_hits"] for r in strat_results)
        avg_hits = total_hits / len(strat_results)
        avg_ppl = sum(r["ppl"] for r in strat_results) / len(strat_results)
        nonzero = sum(1 for r in strat_results if r["keyword_hits"] > 0)
        print(
            f"  {strat_name:15s}  "
            f"avg_kw_hits={avg_hits:.2f}  "
            f"nonzero={nonzero}/{len(strat_results)}  "
            f"avg_ppl={avg_ppl:.1f}"
        )

    # --- Per-query breakdown ---
    print()
    print("─" * 72)
    print("  PER-QUERY BREAKDOWN (avg keyword hits)")
    print("─" * 72)
    for query in config.queries:
        print(f"  Q: {query[:50]}")
        for strat_name in strategies:
            qr = [
                r
                for r in all_results
                if r["strategy"] == strat_name and r["query"] == query
            ]
            avg = sum(r["keyword_hits"] for r in qr) / len(qr)
            print(f"    {strat_name:15s}: {avg:.2f}")
        print()

    # --- Gold vs prompt_cache match check ---
    print("─" * 72)
    print("  LOSSLESS CHECK: gold vs prompt_cache")
    print("─" * 72)
    exact_matches = 0
    for profile_name in config.profiles:
        for query in config.queries:
            g = next(
                r
                for r in all_results
                if r["strategy"] == "gold"
                and r["profile"] == profile_name
                and r["query"] == query
            )
            pc = next(
                r
                for r in all_results
                if r["strategy"] == "prompt_cache"
                and r["profile"] == profile_name
                and r["query"] == query
            )
            match = g["text"][:100] == pc["text"][:100]
            if match:
                exact_matches += 1
            else:
                print(f"  MISMATCH: {profile_name} / {query[:30]}")
                print(f"    Gold: {g['text'][:100]}")
                print(f"    PrCa: {pc['text'][:100]}")
    print(f"  Exact matches: {exact_matches}/25")
    print()

    # --- Fixed-split vs gold match check ---
    print("─" * 72)
    print("  SPLIT CHECK: gold vs fixed_split")
    print("─" * 72)
    exact_matches_fs = 0
    for profile_name in config.profiles:
        for query in config.queries:
            g = next(
                r
                for r in all_results
                if r["strategy"] == "gold"
                and r["profile"] == profile_name
                and r["query"] == query
            )
            fs = next(
                r
                for r in all_results
                if r["strategy"] == "fixed_split"
                and r["profile"] == profile_name
                and r["query"] == query
            )
            match = g["text"][:100] == fs["text"][:100]
            if match:
                exact_matches_fs += 1
            else:
                print(f"  MISMATCH: {profile_name} / {query[:30]}")
                print(f"    Gold: {g['text'][:100]}")
                print(f"    FxSp: {fs['text'][:100]}")
    print(f"  Exact matches: {exact_matches_fs}/25")
    print()

    # --- L2b_system vs gold comparison ---
    print("─" * 72)
    print("  TOKENIZATION CHECK: gold vs l2b_system")
    print("─" * 72)
    exact_matches_l2b = 0
    for profile_name in config.profiles:
        for query in config.queries:
            g = next(
                r
                for r in all_results
                if r["strategy"] == "gold"
                and r["profile"] == profile_name
                and r["query"] == query
            )
            s = next(
                r
                for r in all_results
                if r["strategy"] == "l2b_system"
                and r["profile"] == profile_name
                and r["query"] == query
            )
            match = g["text"][:100] == s["text"][:100]
            if match:
                exact_matches_l2b += 1
    print(f"  Exact matches: {exact_matches_l2b}/25")

    # --- Save results ---
    out = {
        "experiment": "level2c_prompt_cache_diagnostic",
        "model": config.model_id,
        "results": all_results,
    }
    out_path = results_path / "level2c_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()

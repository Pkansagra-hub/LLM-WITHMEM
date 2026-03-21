"""
Evaluation for the memory encoder.

Metrics (same as Level 2 for comparability):
  1. Keyword hits — profile-specific terms in generated text
  2. Coherence — perplexity of generated text (lower = more fluent)
  3. Exact match with gold — does inject produce the same text as gold?
  4. Cosine distance — are encoder vectors distinct per profile?
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ..config import Config
from ..model.encoder import MemoryEncoder
from ..model.injector import generate_with_injection


def score_keywords(text: str, keywords: list[str]) -> dict:
    """Count profile-specific keyword hits in generated text."""
    text_lower = text.lower()
    hits = [kw for kw in keywords if kw.lower() in text_lower]
    return {
        "hit_count": len(hits),
        "total_keywords": len(keywords),
        "hit_ratio": len(hits) / len(keywords) if keywords else 0,
        "hits": hits,
    }


def score_coherence(model, tokenizer, text: str, device: str) -> float:
    """Perplexity of generated text. Lower = more coherent."""
    if not text or len(text.strip()) < 5:
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


def generate_gold(
    model, tokenizer, gold_prompt: str, max_new_tokens: int, device: str
) -> str:
    """Generate gold-standard response (profile in system prompt)."""
    input_ids = tokenizer(gold_prompt, return_tensors="pt")["input_ids"].to(device)
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    cache = outputs.past_key_values
    logits = outputs.logits
    cur_pos = seq_len

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


@torch.no_grad()
def evaluate_encoder(
    config: Config,
    llm,
    tokenizer,
    encoder: MemoryEncoder,
    val_dataset,
    max_samples: int = 50,
) -> dict:
    """Full evaluation: generate with encoder injection vs gold, compare."""
    encoder.eval()
    device = config.device

    results = []
    total_kw_inject = 0
    total_kw_gold = 0
    total_ppl_inject = 0
    total_ppl_gold = 0
    exact_matches = 0
    n = 0

    for idx in range(min(len(val_dataset), max_samples)):
        sample = val_dataset[idx]

        # Encode profile
        profile_ids = sample["profile_input_ids"].unsqueeze(0).to(device)
        profile_mask = sample["profile_attention_mask"].unsqueeze(0).to(device)
        kv_pairs = encoder(profile_ids, profile_mask)

        # Tokenize suffix
        suffix_ids = tokenizer(sample["inject_suffix"], return_tensors="pt")[
            "input_ids"
        ].to(device)

        # Generate with injection
        inject_text = generate_with_injection(
            llm,
            tokenizer,
            suffix_ids,
            kv_pairs,
            num_memory_slots=config.encoder.num_memory_slots,
            max_new_tokens=config.training.max_new_tokens,
        )

        # Generate gold
        gold_text = generate_gold(
            llm,
            tokenizer,
            sample["gold_prompt"],
            max_new_tokens=config.training.max_new_tokens,
            device=device,
        )

        # Score
        kw_inject = score_keywords(inject_text, sample["keywords"])
        kw_gold = score_keywords(gold_text, sample["keywords"])
        ppl_inject = score_coherence(llm, tokenizer, inject_text, device)
        ppl_gold = score_coherence(llm, tokenizer, gold_text, device)
        is_match = inject_text.strip() == gold_text.strip()

        total_kw_inject += kw_inject["hit_count"]
        total_kw_gold += kw_gold["hit_count"]
        total_ppl_inject += ppl_inject
        total_ppl_gold += ppl_gold
        if is_match:
            exact_matches += 1
        n += 1

        results.append(
            {
                "profile": sample["profile_text"][:80],
                "query": sample["query"],
                "inject_text": inject_text[:200],
                "gold_text": gold_text[:200],
                "kw_inject": kw_inject["hit_count"],
                "kw_gold": kw_gold["hit_count"],
                "ppl_inject": round(ppl_inject, 2),
                "ppl_gold": round(ppl_gold, 2),
                "exact_match": is_match,
            }
        )

        if n % 10 == 0:
            print(f"  Evaluated {n}/{min(len(val_dataset), max_samples)}")

    summary = {
        "num_samples": n,
        "avg_kw_inject": total_kw_inject / max(n, 1),
        "avg_kw_gold": total_kw_gold / max(n, 1),
        "kw_ratio": total_kw_inject / max(total_kw_gold, 1),
        "avg_ppl_inject": total_ppl_inject / max(n, 1),
        "avg_ppl_gold": total_ppl_gold / max(n, 1),
        "exact_match_rate": exact_matches / max(n, 1),
        "exact_matches": exact_matches,
    }

    return {"summary": summary, "samples": results}


def save_eval_results(results: dict, output_path: str):
    """Save evaluation results to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Saved evaluation: {path}")


def print_eval_summary(results: dict):
    """Print a readable evaluation summary."""
    s = results["summary"]
    print()
    print("=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated:  {s['num_samples']}")
    print(f"  Avg keywords (inject):  {s['avg_kw_inject']:.2f}")
    print(f"  Avg keywords (gold):    {s['avg_kw_gold']:.2f}")
    print(f"  KW ratio (inject/gold): {s['kw_ratio']:.1%}")
    print(f"  Avg PPL (inject):       {s['avg_ppl_inject']:.2f}")
    print(f"  Avg PPL (gold):         {s['avg_ppl_gold']:.2f}")
    print(f"  Exact matches:          {s['exact_matches']}/{s['num_samples']}")
    print(f"  Exact match rate:       {s['exact_match_rate']:.1%}")
    print("=" * 60)
    print()

"""
Level 4 — Evaluate a trained Multi-Bank Encoder.

Usage:
  python -m experiments.level4_multibank.evaluate [--checkpoint best] [--max-samples 50]

Loads a trained encoder checkpoint and evaluates on the validation set:
  - Per-query-type keyword personalization vs gold
  - Coherence (perplexity)
  - Gate specialization analysis
  - Working memory attention pattern per query type
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .data.dataset import ProfileQueryDataset, _build_suffix
from .model.encoder import MultiBankMemoryEncoder
from .model.injector import generate_with_injection


def score_keywords(text: str, facts: list[str]) -> dict:
    """Count how many profile facts appear in generated text."""
    text_lower = text.lower()
    hits = [
        f
        for f in facts
        if any(word.lower() in text_lower for word in f.split() if len(word) > 3)
    ]
    return {
        "hit_count": len(hits),
        "total_facts": len(facts),
        "hit_ratio": len(hits) / max(len(facts), 1),
    }


def score_coherence(model, tokenizer, text: str, device: str) -> float:
    """Perplexity of generated text."""
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


@torch.no_grad()
def evaluate_encoder(config, llm, tokenizer, encoder, val_data, max_samples=50):
    """Full evaluation: generate with injection vs gold, compare by query type."""
    encoder.eval()
    device = config.device
    results = []
    n = 0

    for idx in range(min(len(val_data), max_samples)):
        ex = val_data[idx]
        if isinstance(ex, dict) and "profile_ids" in ex:
            sample = ex
        else:
            sample = val_data[idx]

        profile_ids = sample["profile_ids"].unsqueeze(0).to(device)
        profile_mask = sample["profile_mask"].unsqueeze(0).to(device)
        query_ids = sample["query_ids"].unsqueeze(0).to(device)
        query_mask = sample["query_mask"].unsqueeze(0).to(device)

        # Encoder forward with diagnostics
        kv_pairs, gate_values, _, diag = encoder(
            profile_ids, profile_mask, query_ids, query_mask, return_diagnostics=True
        )

        # Generate with injection
        raw_data = val_data.data[idx]
        suffix_text = _build_suffix(tokenizer, raw_data["query_text"])
        suffix_ids = tokenizer(suffix_text, return_tensors="pt")["input_ids"].to(device)

        inject_text = generate_with_injection(
            llm,
            tokenizer,
            kv_pairs,
            suffix_ids,
            max_new_tokens=config.training.max_new_tokens,
        )

        # Gold generation
        gold_ids = sample["gold_ids"].unsqueeze(0).to(device)
        gold_mask = sample["gold_mask"].unsqueeze(0).to(device)
        gold_out = llm.generate(
            gold_ids,
            attention_mask=gold_mask,
            max_new_tokens=config.training.max_new_tokens,
            do_sample=False,
        )
        gold_text = tokenizer.decode(
            gold_out[0][gold_ids.shape[1] :], skip_special_tokens=True
        )

        # Score
        relevant_facts = raw_data["relevant_facts"]

        kw_inject = score_keywords(inject_text, relevant_facts)
        kw_gold = score_keywords(gold_text, relevant_facts)
        ppl_inject = score_coherence(llm, tokenizer, inject_text, device)
        ppl_gold = score_coherence(llm, tokenizer, gold_text, device)

        # Gate stats
        gate_vals = gate_values.squeeze(0)  # (num_layers, num_heads)
        per_layer_gates = gate_vals.mean(dim=1)  # (num_layers,)

        results.append(
            {
                "query": raw_data["query_text"],
                "relevant_types": raw_data["relevant_types"],
                "inject_text": inject_text[:300],
                "gold_text": gold_text[:300],
                "kw_inject": kw_inject["hit_count"],
                "kw_gold": kw_gold["hit_count"],
                "kw_total": kw_inject["total_facts"],
                "ppl_inject": round(ppl_inject, 2),
                "ppl_gold": round(ppl_gold, 2),
                "gate_mean": round(gate_vals.mean().item(), 4),
                "gate_std": round(gate_vals.std().item(), 4),
                "gate_layer_min": round(per_layer_gates.min().item(), 4),
                "gate_layer_max": round(per_layer_gates.max().item(), 4),
            }
        )
        n += 1

        if n % 10 == 0:
            print(f"  Evaluated {n}/{min(len(val_data), max_samples)}")

    # Summary
    summary = {
        "num_samples": n,
        "avg_kw_inject": sum(r["kw_inject"] for r in results) / max(n, 1),
        "avg_kw_gold": sum(r["kw_gold"] for r in results) / max(n, 1),
        "avg_ppl_inject": sum(r["ppl_inject"] for r in results) / max(n, 1),
        "avg_ppl_gold": sum(r["ppl_gold"] for r in results) / max(n, 1),
        "avg_gate_mean": sum(r["gate_mean"] for r in results) / max(n, 1),
    }

    # Per query-type breakdown
    type_stats = {}
    for r in results:
        for rt in r["relevant_types"]:
            if rt not in type_stats:
                type_stats[rt] = {"kw_inject": [], "kw_gold": [], "count": 0}
            type_stats[rt]["kw_inject"].append(r["kw_inject"])
            type_stats[rt]["kw_gold"].append(r["kw_gold"])
            type_stats[rt]["count"] += 1

    summary["per_type"] = {
        t: {
            "count": s["count"],
            "avg_kw_inject": sum(s["kw_inject"]) / max(s["count"], 1),
            "avg_kw_gold": sum(s["kw_gold"]) / max(s["count"], 1),
        }
        for t, s in type_stats.items()
    }

    return {"summary": summary, "samples": results}


def print_eval_summary(results: dict):
    s = results["summary"]
    print()
    print("=" * 60)
    print("  LEVEL 4 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated:  {s['num_samples']}")
    print(f"  Avg keywords (inject):  {s['avg_kw_inject']:.2f}")
    print(f"  Avg keywords (gold):    {s['avg_kw_gold']:.2f}")
    kw_ratio = s["avg_kw_inject"] / max(s["avg_kw_gold"], 0.01) * 100
    print(f"  KW ratio (inject/gold): {kw_ratio:.1f}%")
    print(f"  Avg PPL (inject):       {s['avg_ppl_inject']:.2f}")
    print(f"  Avg PPL (gold):         {s['avg_ppl_gold']:.2f}")
    print(f"  Avg gate mean:          {s['avg_gate_mean']:.4f}")

    if "per_type" in s:
        print()
        print("  Per query-type breakdown:")
        for t, ts in s["per_type"].items():
            print(
                f"    {t:12s}: n={ts['count']:3d}  "
                f"kw_inject={ts['avg_kw_inject']:.2f}  "
                f"kw_gold={ts['avg_kw_gold']:.2f}"
            )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    config = Config()

    print("=" * 72)
    print("  LEVEL 4: EVALUATE MULTI-BANK ENCODER")
    print("=" * 72)

    # Load LLM
    print(f"  Loading LLM: {config.model.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        dtype=torch.float16,
        device_map=config.device,
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load encoder
    ckpt_path = Path(config.output_dir) / f"encoder_{args.checkpoint}.pt"
    print(f"  Loading encoder: {ckpt_path}")

    if not ckpt_path.exists():
        print(f"  ERROR: Checkpoint not found: {ckpt_path}")
        available = list(Path(config.output_dir).glob("encoder_*.pt"))
        if available:
            print(f"  Available: {[p.name for p in available]}")
        return

    checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=True)
    encoder = MultiBankMemoryEncoder(
        embedding_layer=model.get_input_embeddings(),
        config=config,
    ).to(config.device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    counts = encoder.param_count()
    print(f"  Encoder params: {counts['total']:,d}")
    print(f"  Checkpoint step: {checkpoint.get('step', '?')}")

    # Val dataset
    data_dir = Path("experiments/level4_multibank/data")
    val_dataset = ProfileQueryDataset(
        data_path=str(data_dir / "val.json"),
        tokenizer=tokenizer,
        max_profile_tokens=config.encoder.max_profile_tokens,
        max_query_tokens=config.encoder.max_query_tokens,
    )

    # Evaluate
    results = evaluate_encoder(
        config, model, tokenizer, encoder, val_dataset, max_samples=args.max_samples
    )
    print_eval_summary(results)

    # Save
    out_path = Path(config.output_dir) / f"eval_{args.checkpoint}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()

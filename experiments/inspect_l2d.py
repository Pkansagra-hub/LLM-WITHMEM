"""Inspect Level 2d outputs for coherence."""

import json

with open("experiments/results/level2/level2d_results.json") as f:
    data = json.load(f)

for cond in ["raw_text", "user_turn", "system_formatted", "gold"]:
    print(f"{'=' * 72}")
    print(f"  CONDITION: {cond}")
    print(f"{'=' * 72}")
    for r in data["results"]:
        if r["condition"] == cond:
            ppl = r["ppl"]
            flag = " *** HIGH PPL ***" if ppl > 6.0 else ""
            print(f"  [{r['profile']}] Q: {r['query'][:45]}")
            print(f"    PPL={ppl:.1f}  KW={r['keyword_hits']}{flag}")
            print(f"    {r['text'][:250]}")
            print()
    print()

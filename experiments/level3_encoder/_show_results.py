import json

data = json.load(open("experiments/level3_encoder/outputs/eval_best.json"))
samples = data.get("samples", data.get("results", []))
for i, s in enumerate(samples[:8]):
    print(f"--- Sample {i+1} ---")
    print(f"Q: {s['query']}")
    print(f"INJECT: {s['inject_text'][:150]}")
    print(f"GOLD:   {s['gold_text'][:150]}")
    print(
        f"KW inj/gold: {s['kw_inject']}/{s['kw_gold']}  PPL: {s['ppl_inject']}/{s['ppl_gold']}"
    )
    print()

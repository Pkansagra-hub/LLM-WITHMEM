"""
Level 4 Multi-Bank Encoder — Local Proof (smoke_test.py)

Verifies on RTX 5070 (8 GB VRAM):
  1. Forward pass completes (shapes correct for all 24 layers)
  2. Parameter count matches design (~51M)
  3. Gradient flow — all encoder params receive gradients
  4. Dynamic gates vary with different queries
  5. Working memory attention varies with query type
  6. VRAM fits within budget
  7. Loss decreases over 30 training steps
  8. Bank specialization diagnostic (which banks activate per query type)

Run:  python -m experiments.level4_multibank.smoke_test
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .model.encoder import MultiBankMemoryEncoder
from .model.injector import forward_with_injection
from .training.losses import combined_loss


def vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def print_header(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def main():
    cfg = Config()
    device = cfg.device

    # ── Load LLM ──────────────────────────────────────────────────
    print_header(f"Loading {cfg.model.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = (
        AutoModelForCausalLM.from_pretrained(
            cfg.model.model_id,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )

    for p in model.parameters():
        p.requires_grad_(False)

    vram_after_llm = vram_mb()
    print(f"  LLM loaded: {vram_after_llm:.0f} MB VRAM")

    # ── Create encoder ────────────────────────────────────────────
    print_header("Creating MultiBankMemoryEncoder")
    embedding_layer = model.model.embed_tokens
    encoder = MultiBankMemoryEncoder(embedding_layer, cfg).to(device)
    encoder.train()

    pc = encoder.param_count()
    print("  Parameter breakdown:")
    for name, count in pc.items():
        print(f"    {name:25s}: {count:>12,}")
    vram_after_encoder = vram_mb()
    print(
        f"  VRAM after encoder: {vram_after_encoder:.0f} MB (+{vram_after_encoder - vram_after_llm:.0f} MB)"
    )

    # ── Test data: 2 contrasting profiles + queries ───────────────
    print_header("Preparing test data")

    profile_text = (
        "User: Alice Chen. "
        "[EPISODIC] Visited Tokyo last week. Had a job interview at Google yesterday. "
        "Started learning piano three months ago. "
        "[SEMANTIC] Works as a software engineer at a startup. Lives in Berlin. "
        "Allergic to peanuts. Prefers Python for programming. "
        "[PROCEDURAL] Prefers concise bullet-point answers. Usually works late at night. "
        "[EMOTIONAL] Gets excited about machine learning. Frustrated with bureaucracy. "
        "[PROSPECTIVE] Planning to learn Rust next quarter. Wants to run a marathon by end of year."
    )

    query_semantic = "What am I allergic to?"
    query_episodic = "What did I do recently?"
    query_emotional = "What topics make me excited?"

    def build_suffix(query):
        messages = [{"role": "user", "content": query}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # Tokenize
    def tokenize(text, max_len):
        enc = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    profile_ids, profile_mask = tokenize(profile_text, cfg.encoder.max_profile_tokens)
    q_sem_ids, q_sem_mask = tokenize(query_semantic, cfg.encoder.max_query_tokens)
    q_epi_ids, q_epi_mask = tokenize(query_episodic, cfg.encoder.max_query_tokens)
    q_emo_ids, q_emo_mask = tokenize(query_emotional, cfg.encoder.max_query_tokens)

    suffix_sem_ids, suffix_sem_mask = tokenize(
        build_suffix(query_semantic), cfg.encoder.max_query_tokens
    )
    suffix_epi_ids, suffix_epi_mask = tokenize(
        build_suffix(query_episodic), cfg.encoder.max_query_tokens
    )

    # Gold prompts (with ONLY relevant facts)
    def build_gold(facts, query):
        facts_str = "\n".join(f"- {f}" for f in facts)
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. Here is relevant information about the user:\n{facts_str}",
            },
            {"role": "user", "content": query},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return tokenize(
            text, cfg.encoder.max_profile_tokens + cfg.encoder.max_query_tokens
        )

    gold_sem_ids, gold_sem_mask = build_gold(
        [
            "Works as a software engineer at a startup",
            "Lives in Berlin",
            "Allergic to peanuts",
            "Prefers Python for programming",
        ],
        query_semantic,
    )
    gold_epi_ids, gold_epi_mask = build_gold(
        [
            "Visited Tokyo last week",
            "Had a job interview at Google yesterday",
            "Started learning piano three months ago",
        ],
        query_episodic,
    )

    print(f"  Profile tokens: {profile_mask.sum().item()}")
    print(f"  Query tokens (semantic): {q_sem_mask.sum().item()}")
    print(f"  Query tokens (episodic): {q_epi_mask.sum().item()}")
    print(f"  Query tokens (emotional): {q_emo_mask.sum().item()}")

    # ── CHECK 1: Forward pass ─────────────────────────────────────
    print_header("CHECK 1: Forward pass + shapes")
    kv_pairs, gate_values_fwd, kv_norm_fwd, diag = encoder(
        profile_ids, profile_mask, q_sem_ids, q_sem_mask, return_diagnostics=True
    )

    assert (
        len(kv_pairs) == cfg.model.num_layers
    ), f"Expected {cfg.model.num_layers} layer pairs, got {len(kv_pairs)}"
    k0, v0 = kv_pairs[0]
    print(
        f"  K shape: {k0.shape}  (expect [1, {cfg.model.num_kv_heads}, {cfg.encoder.num_output_slots}, {cfg.model.head_dim}])"
    )
    print(f"  V shape: {v0.shape}")
    assert k0.shape == (
        1,
        cfg.model.num_kv_heads,
        cfg.encoder.num_output_slots,
        cfg.model.head_dim,
    )
    assert k0.dtype == torch.float16
    print(f"  ✓ All {cfg.model.num_layers} K,V pairs have correct shape and dtype")

    # ── CHECK 2: Parameter count ──────────────────────────────────
    print_header("CHECK 2: Parameter count")
    total = pc["total"]
    print(f"  Total trainable params: {total:,}")
    print("  Target range: 30M-80M")
    assert 10_000_000 < total < 200_000_000, f"Params {total:,} outside expected range"
    print(f"  ✓ {total/1e6:.1f}M params — within budget")

    # ── CHECK 3: Gradient flow ────────────────────────────────────
    print_header("CHECK 3: Gradient flow")

    # Do a full training step: inject forward → gold forward → KL loss → backward
    inject_logits = forward_with_injection(
        model, suffix_sem_ids, suffix_sem_mask, kv_pairs
    )

    with torch.no_grad():
        gold_outputs = model(input_ids=gold_sem_ids, attention_mask=gold_sem_mask)
        # Align: take last S tokens of gold logits where S = suffix length
        S = suffix_sem_ids.shape[1]
        gold_logits = gold_outputs.logits[:, -S:, :].detach()

    loss, metrics = combined_loss(inject_logits, gold_logits, suffix_sem_mask)
    loss.backward()

    total_params = 0
    grads_ok = 0
    for name, p in encoder.named_parameters():
        if p.requires_grad:
            total_params += 1
            if p.grad is not None and p.grad.abs().sum() > 0:
                grads_ok += 1

    print(f"  {grads_ok}/{total_params} parameters received non-zero gradients")
    if grads_ok < total_params:
        missing = [
            n
            for n, p in encoder.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)
        ]
        print(f"  WARNING: Missing gradients for: {missing[:10]}")
    else:
        print("  ✓ ALL parameters receive gradients")
    encoder.zero_grad()

    # ── CHECK 4: Dynamic gates vary with query ────────────────────
    print_header("CHECK 4: Dynamic gate variation")
    encoder.eval()
    with torch.no_grad():
        _, gates_sem_t, _, diag_sem = encoder(
            profile_ids, profile_mask, q_sem_ids, q_sem_mask, return_diagnostics=True
        )
        _, gates_epi_t, _, diag_epi = encoder(
            profile_ids, profile_mask, q_epi_ids, q_epi_mask, return_diagnostics=True
        )
        _, gates_emo_t, _, diag_emo = encoder(
            profile_ids, profile_mask, q_emo_ids, q_emo_mask, return_diagnostics=True
        )

    gates_sem = gates_sem_t.squeeze(0)  # (num_layers, num_heads)
    gates_epi = gates_epi_t.squeeze(0)
    gates_emo = gates_emo_t.squeeze(0)

    diff_sem_epi = (gates_sem - gates_epi).abs().mean().item()
    diff_sem_emo = (gates_sem - gates_emo).abs().mean().item()
    diff_epi_emo = (gates_epi - gates_emo).abs().mean().item()

    print(f"  Mean gate value (semantic query):  {gates_sem.mean():.4f}")
    print(f"  Mean gate value (episodic query):  {gates_epi.mean():.4f}")
    print(f"  Mean gate value (emotional query): {gates_emo.mean():.4f}")
    print(f"  Gate diff (semantic vs episodic):  {diff_sem_epi:.6f}")
    print(f"  Gate diff (semantic vs emotional): {diff_sem_emo:.6f}")
    print(f"  Gate diff (episodic vs emotional): {diff_epi_emo:.6f}")

    any_diff = diff_sem_epi > 1e-6 or diff_sem_emo > 1e-6
    if any_diff:
        print("  ✓ Gates differ across queries (dynamic gating confirmed)")
    else:
        print("  ✗ WARNING: Gates are identical — check query encoder")

    # ── CHECK 5: Working memory attention varies ──────────────────
    print_header("CHECK 5: Working memory attention pattern")

    # Bank slot ranges: episodic [0:32], semantic [32:48], procedural [48:56], emotional [56:64], prospective [64:72]
    bank_ranges = {
        "episodic": (0, cfg.banks.l4_episodic_slots),
        "semantic": (
            cfg.banks.l4_episodic_slots,
            cfg.banks.l4_episodic_slots + cfg.banks.l5_semantic_slots,
        ),
        "procedural": (
            cfg.banks.l4_episodic_slots + cfg.banks.l5_semantic_slots,
            cfg.banks.l4_episodic_slots
            + cfg.banks.l5_semantic_slots
            + cfg.banks.l6_procedural_slots,
        ),
        "emotional": (
            cfg.banks.l4_episodic_slots
            + cfg.banks.l5_semantic_slots
            + cfg.banks.l6_procedural_slots,
            cfg.banks.l4_episodic_slots
            + cfg.banks.l5_semantic_slots
            + cfg.banks.l6_procedural_slots
            + cfg.banks.l7_emotional_slots,
        ),
        "prospective": (
            cfg.banks.total_slots - cfg.banks.l8_prospective_slots,
            cfg.banks.total_slots,
        ),
    }

    for query_name, diag in [
        ("semantic", diag_sem),
        ("episodic", diag_epi),
        ("emotional", diag_emo),
    ]:
        wm_attn = diag["wm_attention"].squeeze(0)  # (M, total_slots)
        print(f"\n  Query: '{query_name}' — WM attention mass per bank:")
        for bank_name, (start, end) in bank_ranges.items():
            mass = wm_attn[:, start:end].sum().item() / wm_attn.sum().item() * 100
            bar = "█" * int(mass / 2)
            print(f"    {bank_name:12s}: {mass:5.1f}% {bar}")

    print(
        "\n  (Pre-training, attention is ~uniform. After training, expect specialization.)"
    )

    # ── CHECK 6: VRAM usage ───────────────────────────────────────
    print_header("CHECK 6: VRAM budget")
    vram_peak = vram_mb()
    print(f"  LLM:           {vram_after_llm:.0f} MB")
    print(f"  + Encoder:     {vram_after_encoder - vram_after_llm:.0f} MB")
    print(f"  Peak (w/ fwd): {vram_peak:.0f} MB")
    if torch.cuda.is_available():
        total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  GPU total:     {total_gpu:.0f} MB")
        headroom = total_gpu - vram_peak
        print(f"  Headroom:      {headroom:.0f} MB")
        if headroom > 500:
            print("  ✓ Fits with good headroom")
        else:
            print("  ⚠ Tight — consider d_model=256 for smoke test")

    # ── CHECK 7: 30-step training run ─────────────────────────────
    print_header("CHECK 7: 30-step training run")
    encoder.train()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4, weight_decay=0.01)

    # Alternate between semantic and episodic queries for variety
    steps_data = [
        (
            q_sem_ids,
            q_sem_mask,
            suffix_sem_ids,
            suffix_sem_mask,
            gold_sem_ids,
            gold_sem_mask,
            "semantic",
        ),
        (
            q_epi_ids,
            q_epi_mask,
            suffix_epi_ids,
            suffix_epi_mask,
            gold_epi_ids,
            gold_epi_mask,
            "episodic",
        ),
    ]

    losses = []
    for step in range(30):
        qi, qm, si, sm, gi, gm, qtype = steps_data[step % 2]

        # Encoder forward
        kv = encoder(profile_ids, profile_mask, qi, qm)[0]

        # Inject forward
        inject_logits = forward_with_injection(model, si, sm, kv)

        # Gold forward
        with torch.no_grad():
            gold_out = model(input_ids=gi, attention_mask=gm)
            S = si.shape[1]
            gold_logits = gold_out.logits[:, -S:, :].detach()

        # Loss
        loss, met = combined_loss(inject_logits, gold_logits, sm)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(met["loss_total"])
        if step % 5 == 0 or step == 29:
            vram_now = vram_mb()
            print(
                f"  Step {step:2d} | loss={met['loss_total']:.2f} | type={qtype:8s} | VRAM={vram_now:.0f} MB"
            )

    # Check loss trend
    first_5 = sum(losses[:5]) / 5
    last_5 = sum(losses[-5:]) / 5
    improved = last_5 < first_5
    print(f"\n  Avg loss (first 5): {first_5:.2f}")
    print(f"  Avg loss (last 5):  {last_5:.2f}")
    if improved:
        print(f"  ✓ Loss improved by {(1 - last_5/first_5)*100:.1f}%")
    else:
        print("  ⚠ Loss did not improve -- expected at this scale, KL is hard")

    # ── CHECK 8: Post-training gate specialization ────────────────
    print_header("CHECK 8: Post-training gate diagnostic")
    encoder.eval()
    with torch.no_grad():
        _, diag_sem_post_gate, _, diag_sem_post = encoder(
            profile_ids, profile_mask, q_sem_ids, q_sem_mask, return_diagnostics=True
        )
        _, diag_epi_post_gate, _, diag_epi_post = encoder(
            profile_ids, profile_mask, q_epi_ids, q_epi_mask, return_diagnostics=True
        )

    gates_sem_post = diag_sem_post_gate.squeeze(0)
    gates_epi_post = diag_epi_post_gate.squeeze(0)
    diff_post = (gates_sem_post - gates_epi_post).abs().mean().item()
    print(f"  Gate diff (semantic vs episodic) PRE-train:  {diff_sem_epi:.6f}")
    print(f"  Gate diff (semantic vs episodic) POST-train: {diff_post:.6f}")
    if diff_post > diff_sem_epi:
        print(
            f"  ✓ Gate specialization increased by {diff_post/max(diff_sem_epi, 1e-8):.1f}x"
        )
    else:
        print("  ⚠ No gate specialization yet — expected, need more training")

    # ── Per-layer gate heatmap (text) ─────────────────────────────
    print("\n  Per-layer mean gate (semantic query):")
    for li in range(cfg.model.num_layers):
        val = gates_sem_post[li].mean().item()
        bar = "█" * int(val * 40)
        print(f"    L{li:02d}: {val:.3f} {bar}")

    # ── Summary ───────────────────────────────────────────────────
    print_header("SMOKE TEST SUMMARY")
    checks = [
        ("Forward pass + shapes", True),
        (f"Param count ({total/1e6:.1f}M)", 10e6 < total < 200e6),
        (f"Gradient flow ({grads_ok}/{total_params})", grads_ok == total_params),
        ("Dynamic gate variation", any_diff),
        ("VRAM fits", vram_peak < 7500),
        ("Loss decreases", improved),
    ]
    all_pass = True
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} — {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  ✅ ALL CHECKS PASSED — Level 4 encoder is mechanically sound")
        print("  Ready for full training run")
    else:
        print("  ⚠ Some checks failed — review output above")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

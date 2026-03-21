"""
Level 3 — Evaluate a trained Memory Encoder.

Usage:
  cd C:\\Users\\princ\\LLM-WITHMEM
  python -m experiments.level3_encoder.evaluate [--checkpoint best]

Loads a trained encoder checkpoint and evaluates on the validation set:
  - Keyword personalization vs gold standard
  - Coherence (perplexity)
  - Exact match rate with gold
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .data.dataset import ProfileQueryDataset
from .evaluation.evaluator import (
    evaluate_encoder,
    print_eval_summary,
    save_eval_results,
)
from .model.encoder import MemoryEncoder


def main():
    parser = argparse.ArgumentParser(description="Evaluate memory encoder")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best",
        help="Checkpoint tag: 'best', 'final', or 'step_N'",
    )
    parser.add_argument(
        "--max-samples", type=int, default=50, help="Max validation samples to evaluate"
    )
    args = parser.parse_args()

    config = Config()

    print("=" * 72)
    print("  LEVEL 3: EVALUATE MEMORY ENCODER")
    print("=" * 72)
    print()

    # Load LLM
    print(f"  Loading LLM: {config.model.model_id}")
    dtype = getattr(torch, config.model.dtype)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id,
        dtype=dtype,
        device_map=config.device,
    )
    model.eval()
    print(f"  LLM VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Load encoder
    ckpt_path = Path(config.output_dir) / f"encoder_{args.checkpoint}.pt"
    print(f"  Loading encoder: {ckpt_path}")

    if not ckpt_path.exists():
        print(f"  ERROR: Checkpoint not found: {ckpt_path}")
        available = list(Path(config.output_dir).glob("encoder_*.pt"))
        if available:
            print(f"  Available checkpoints: {[p.name for p in available]}")
        return

    checkpoint = torch.load(ckpt_path, map_location=config.device, weights_only=True)

    encoder = MemoryEncoder(
        embedding_layer=model.get_input_embeddings(),
        llm_embed_dim=config.model.hidden_size,
        d_model=config.encoder.d_model,
        n_heads=config.encoder.n_heads,
        n_layers=config.encoder.n_layers,
        num_memory_slots=config.encoder.num_memory_slots,
        num_llm_layers=config.model.num_layers,
        num_kv_heads=config.model.num_kv_heads,
        head_dim=config.model.head_dim,
        num_layer_groups=config.encoder.num_layer_groups,
        gate_init_bias=config.encoder.gate_init_bias,
        dropout=0.0,  # no dropout at eval time
    ).to(config.device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    counts = encoder.param_count()
    print(f"  Encoder params: {counts['total']:,d}")
    print(f"  Checkpoint step: {checkpoint.get('step', '?')}")
    print()

    # Load val dataset
    data_dir = Path("experiments/level3_encoder/data")
    val_dataset = ProfileQueryDataset(
        profiles_path=str(data_dir / "profiles_val.json"),
        queries_path=str(data_dir / "queries.json"),
        tokenizer=tokenizer,
        max_profile_tokens=config.encoder.max_profile_tokens,
        seed=config.training.seed + 1,
    )

    # Evaluate
    results = evaluate_encoder(
        config,
        model,
        tokenizer,
        encoder,
        val_dataset,
        max_samples=args.max_samples,
    )

    # Output
    print_eval_summary(results)
    output_path = Path(config.output_dir) / f"eval_{args.checkpoint}.json"
    save_eval_results(results, str(output_path))


if __name__ == "__main__":
    main()

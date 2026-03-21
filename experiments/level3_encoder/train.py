"""
Level 3 — Train the Memory Encoder.

Usage:
  cd C:\\Users\\princ\\LLM-WITHMEM
  python -m experiments.level3_encoder.train

This trains a small encoder (~5M params) to produce K,V pairs that
personalize a frozen SmolLM2-1.7B-Instruct via KV injection.

Prerequisites:
  1. Generate training data first:
     python -m experiments.level3_encoder.data.generate_profiles
  2. Ensure CUDA is available with sufficient VRAM (~6GB for SmolLM2)
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config
from .training.trainer import Trainer


def main():
    config = Config()

    print("=" * 72)
    print("  LEVEL 3: TRAINED MEMORY ENCODER")
    print("=" * 72)
    print()
    print(f"  LLM:             {config.model.model_id}")
    print(
        f"  Encoder:         d={config.encoder.d_model}, "
        f"heads={config.encoder.n_heads}, "
        f"layers={config.encoder.n_layers}"
    )
    print(f"  Memory slots:    {config.encoder.num_memory_slots}")
    print(f"  Layer groups:    {config.encoder.num_layer_groups}")
    print(f"  Device:          {config.device}")
    print()

    # Check data exists
    data_dir = Path("experiments/level3_encoder/data")
    if not (data_dir / "profiles_train.json").exists():
        print("  ERROR: Training data not found. Generate it first:")
        print("    python -m experiments.level3_encoder.data.generate_profiles")
        sys.exit(1)

    # Load frozen LLM
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

    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  LLM VRAM: {vram:.2f} GB")
    print()

    # Train
    trainer = Trainer(config, model, tokenizer)
    history = trainer.train()

    print(f"\n  Training complete. {len(history)} steps recorded.")
    print(f"  Checkpoints saved to: {config.output_dir}")
    print(f"  Logs saved to: {config.log_dir}")


if __name__ == "__main__":
    main()

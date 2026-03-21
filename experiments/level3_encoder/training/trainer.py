"""
Training loop for the memory encoder.

Core loop:
  1. Sample (profile, query) batch
  2. Gold pass: full prompt → frozen LLM → gold logits (no grad)
  3. Inject pass: encoder(profile) → K,V → inject into LLM → inject logits
  4. Loss = KL(inject ∥ gold) on suffix positions
  5. Backward through encoder only (LLM frozen but in compute graph)
  6. Optimizer step on encoder params

Gradient flow:
  loss → inject_logits → LLM attention (frozen weights, grads flow through)
  → injected K,V (encoder output, has grad) → encoder params (updated)
"""

import gc
import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..config import Config
from ..data.dataset import ProfileQueryDataset, collate_fn
from ..model.encoder import MemoryEncoder
from ..model.injector import build_injection_cache, forward_with_injection
from ..training.losses import combined_loss


class Trainer:
    """Trains the memory encoder via distillation from the frozen LLM."""

    def __init__(self, config: Config, model, tokenizer):
        self.config = config
        self.llm = model
        self.tokenizer = tokenizer
        self.device = config.device

        # Freeze the LLM — no parameter updates, but keep in compute graph
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad_(False)

        # Build encoder
        self.encoder = MemoryEncoder(
            embedding_layer=self.llm.get_input_embeddings(),
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
            dropout=config.encoder.dropout,
        ).to(self.device)

        # Print param counts
        counts = self.encoder.param_count()
        print("\n  Memory Encoder parameters:")
        for k, v in counts.items():
            print(f"    {k:15s}: {v:>10,d}")
        print()

        # Optimizer — only encoder params
        self.optimizer = AdamW(
            self.encoder.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.max_steps,
            eta_min=config.training.learning_rate * 0.1,
        )

        # Data
        data_dir = Path("experiments/level3_encoder/data")
        self.train_dataset = ProfileQueryDataset(
            profiles_path=str(data_dir / "profiles_train.json"),
            queries_path=str(data_dir / "queries.json"),
            tokenizer=tokenizer,
            max_profile_tokens=config.encoder.max_profile_tokens,
            seed=config.training.seed,
        )
        self.val_dataset = ProfileQueryDataset(
            profiles_path=str(data_dir / "profiles_val.json"),
            queries_path=str(data_dir / "queries.json"),
            tokenizer=tokenizer,
            max_profile_tokens=config.encoder.max_profile_tokens,
            seed=config.training.seed + 1,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.training.num_workers,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Logging
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

    def _gold_forward(self, gold_prompt: str) -> tuple[torch.Tensor, int]:
        """Run gold-standard forward pass, return (logits, suffix_start_idx).

        We need to know where the suffix starts in the gold prompt so we
        can align gold logits with inject logits.
        """
        input_ids = self.tokenizer(gold_prompt, return_tensors="pt")["input_ids"].to(
            self.device
        )

        with torch.no_grad():
            outputs = self.llm(input_ids=input_ids, use_cache=False)

        return outputs.logits, input_ids.shape[1]

    def _inject_forward(
        self,
        profile_ids: torch.Tensor,
        profile_mask: torch.Tensor,
        suffix_text: str,
    ) -> torch.Tensor:
        """Run injection forward pass with encoder-produced K,V.

        Returns logits over suffix positions.
        """
        # Encode profile → K,V pairs
        kv_pairs = self.encoder(profile_ids, profile_mask)

        # Build injection cache
        cache = build_injection_cache(kv_pairs)
        num_mem = self.config.encoder.num_memory_slots

        # Tokenize suffix
        suffix_ids = self.tokenizer(suffix_text, return_tensors="pt")["input_ids"].to(
            self.device
        )

        # Forward through frozen LLM with injected cache
        logits = forward_with_injection(self.llm, suffix_ids, cache, num_mem)

        return logits, suffix_ids.shape[1]

    def train_step(self, batch: dict) -> dict:
        """Single training step on one batch.

        For batch_size > 1, we loop over samples because gold/inject paths
        have variable sequence lengths.
        """
        batch_size = len(batch["gold_prompts"])
        total_loss_dict = {"total": 0.0, "distill": 0.0}
        valid_samples = 0

        for i in range(batch_size):
            # Gold forward
            gold_logits, gold_len = self._gold_forward(batch["gold_prompts"][i])

            # Inject forward
            profile_ids = batch["profile_input_ids"][i : i + 1].to(self.device)
            profile_mask = batch["profile_attention_mask"][i : i + 1].to(self.device)

            inject_logits, suffix_len = self._inject_forward(
                profile_ids, profile_mask, batch["inject_suffixes"][i]
            )

            # Align: gold logits at the LAST suffix_len positions
            # Gold prompt = [system+profile | suffix], suffix starts at gold_len - suffix_len
            suffix_start = gold_len - suffix_len
            if suffix_start < 0:
                continue

            gold_suffix_logits = gold_logits[
                :, suffix_start : suffix_start + suffix_len, :
            ]
            inject_suffix_logits = inject_logits[:, :suffix_len, :]

            # Compute loss
            losses = combined_loss(
                inject_logits=inject_suffix_logits,
                gold_logits=gold_suffix_logits,
                lambda_distill=self.config.training.lambda_distill,
                lambda_kv_align=0,  # skip KV align for now
            )

            total_loss_dict["total"] += losses["total"]
            total_loss_dict["distill"] += losses["distill"].item()
            valid_samples += 1

        if valid_samples == 0:
            return {"total": 0.0, "distill": 0.0}

        # Average over samples
        loss = total_loss_dict["total"] / valid_samples

        # Backward
        loss.backward()

        return {
            "total": loss.item(),
            "distill": total_loss_dict["distill"] / valid_samples,
        }

    @torch.no_grad()
    def validate(self, max_samples: int = 50) -> dict:
        """Run validation: compute average loss on val set."""
        self.encoder.eval()
        total_loss = 0.0
        n = 0

        for batch in self.val_loader:
            if n >= max_samples:
                break

            gold_logits, gold_len = self._gold_forward(batch["gold_prompts"][0])

            profile_ids = batch["profile_input_ids"][:1].to(self.device)
            profile_mask = batch["profile_attention_mask"][:1].to(self.device)

            kv_pairs = self.encoder(profile_ids, profile_mask)
            cache = build_injection_cache(kv_pairs)
            num_mem = self.config.encoder.num_memory_slots

            suffix_ids = self.tokenizer(
                batch["inject_suffixes"][0], return_tensors="pt"
            )["input_ids"].to(self.device)
            suffix_len = suffix_ids.shape[1]

            inject_logits = forward_with_injection(self.llm, suffix_ids, cache, num_mem)

            suffix_start = gold_len - suffix_len
            if suffix_start < 0:
                continue

            gold_suffix = gold_logits[:, suffix_start : suffix_start + suffix_len, :]

            losses = combined_loss(inject_logits[:, :suffix_len, :], gold_suffix)
            total_loss += losses["total"].item()
            n += 1

        self.encoder.train()
        return {"val_loss": total_loss / max(n, 1), "val_samples": n}

    def train(self):
        """Full training loop."""
        cfg = self.config.training
        self.encoder.train()

        step = 0
        epoch = 0
        best_val_loss = float("inf")
        t_start = time.time()

        print("=" * 72)
        print("  TRAINING MEMORY ENCODER")
        print("=" * 72)
        print(f"  Max steps:      {cfg.max_steps}")
        print(f"  Batch size:     {cfg.batch_size}")
        print(f"  Grad accum:     {cfg.gradient_accumulation_steps}")
        print(f"  Effective batch: {cfg.batch_size * cfg.gradient_accumulation_steps}")
        print(f"  LR:             {cfg.learning_rate}")
        print(f"  Eval every:     {cfg.eval_every} steps")
        print(f"  Save every:     {cfg.save_every} steps")
        print("=" * 72)
        print()

        self.optimizer.zero_grad()

        while step < cfg.max_steps:
            self.train_dataset.set_epoch(epoch)

            for batch in self.train_loader:
                if step >= cfg.max_steps:
                    break

                loss_dict = self.train_step(batch)
                step += 1

                # Gradient accumulation
                if step % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # Log
                if step % 10 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - t_start
                    print(
                        f"  step {step:5d}/{cfg.max_steps} | "
                        f"loss {loss_dict['total']:.4f} | "
                        f"distill {loss_dict['distill']:.4f} | "
                        f"lr {lr:.2e} | "
                        f"time {elapsed:.0f}s"
                    )

                self.history.append(
                    {
                        "step": step,
                        "loss": loss_dict["total"],
                        "distill": loss_dict["distill"],
                    }
                )

                # Evaluate
                if step % cfg.eval_every == 0:
                    val_result = self.validate()
                    print(
                        f"\n  >>> EVAL step {step}: "
                        f"val_loss={val_result['val_loss']:.4f} "
                        f"(n={val_result['val_samples']})\n"
                    )
                    self.history[-1]["val_loss"] = val_result["val_loss"]

                    if val_result["val_loss"] < best_val_loss:
                        best_val_loss = val_result["val_loss"]
                        self._save_checkpoint(step, "best")
                        print(f"  >>> New best val_loss: {best_val_loss:.4f}\n")

                # Save
                if step % cfg.save_every == 0:
                    self._save_checkpoint(step, f"step_{step}")

                # Memory cleanup
                if step % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            epoch += 1

        # Final save
        self._save_checkpoint(step, "final")
        self._save_history()

        total_time = time.time() - t_start
        print(f"\n  Training complete: {step} steps in {total_time:.0f}s")
        print(f"  Best val_loss: {best_val_loss:.4f}")

        return self.history

    def _save_checkpoint(self, step: int, tag: str):
        """Save encoder weights and optimizer state."""
        path = self.output_dir / f"encoder_{tag}.pt"
        torch.save(
            {
                "step": step,
                "encoder_state_dict": self.encoder.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": {
                    "d_model": self.config.encoder.d_model,
                    "n_heads": self.config.encoder.n_heads,
                    "n_layers": self.config.encoder.n_layers,
                    "num_memory_slots": self.config.encoder.num_memory_slots,
                    "num_layer_groups": self.config.encoder.num_layer_groups,
                },
            },
            path,
        )
        print(f"  Saved checkpoint: {path}")

    def _save_history(self):
        """Save training history to JSON."""
        path = self.log_dir / "training_history.json"
        path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        print(f"  Saved history: {path}")

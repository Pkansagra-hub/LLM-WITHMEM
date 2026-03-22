"""
Training loop for Level 4 Multi-Bank Query-Conditioned Memory Encoder.

Core loop:
  1. Sample (profile, query) batch — each example has typed queries + relevant facts
  2. Gold pass: system(relevant facts) + user query → frozen LLM → gold logits (no grad)
  3. Inject pass: encoder(profile, query) → K,V → inject into LLM with suffix → inject logits
  4. Loss = KL(inject ∥ gold) on suffix positions (mask-aware)
  5. Backward through encoder only (LLM frozen but in compute graph for grad flow)
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
from ..data.dataset import ProfileQueryDataset, _build_suffix, collate_fn
from ..evaluate import score_coherence, score_keywords
from ..model.encoder import MultiBankMemoryEncoder
from ..model.injector import forward_with_injection, generate_with_injection
from ..training.losses import combined_loss


class Trainer:
    """Trains the Level 4 multi-bank encoder via distillation from the frozen LLM."""

    def __init__(self, config: Config, model, tokenizer):
        self.config = config
        self.llm = model
        self.tokenizer = tokenizer
        self.device = config.device

        # Freeze the LLM
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad_(False)

        # Build encoder
        self.encoder = MultiBankMemoryEncoder(
            embedding_layer=self.llm.get_input_embeddings(),
            config=config,
        ).to(self.device)

        # Print param counts
        counts = self.encoder.param_count()
        print("\n  Level 4 Multi-Bank Encoder parameters:")
        for k, v in counts.items():
            print(f"    {k:25s}: {v:>12,d}")
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
        data_dir = Path("experiments/level4_multibank/data")
        self.train_dataset = ProfileQueryDataset(
            data_path=str(data_dir / "train.json"),
            tokenizer=tokenizer,
            max_profile_tokens=config.encoder.max_profile_tokens,
            max_query_tokens=config.encoder.max_query_tokens,
        )
        self.val_dataset = ProfileQueryDataset(
            data_path=str(data_dir / "val.json"),
            tokenizer=tokenizer,
            max_profile_tokens=config.encoder.max_profile_tokens,
            max_query_tokens=config.encoder.max_query_tokens,
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

    def _gold_forward(
        self, gold_ids: torch.Tensor, gold_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run gold-standard forward pass.

        Returns:
            logits: (B, L, V)
            gold_kv_norm: scalar — mean L2 norm of LLM's own K,V (for norm matching)
        """
        with torch.no_grad():
            outputs = self.llm(
                input_ids=gold_ids,
                attention_mask=gold_mask,
                use_cache=True,
                return_dict=True,
            )
            # Extract KV norms from LLM's own cache for norm matching
            cache = outputs.past_key_values
            kv_norms = []
            for layer_idx in range(len(cache)):
                k, v = cache[layer_idx]
                kv_norms.append(k.float().norm(dim=-1).mean())
                kv_norms.append(v.float().norm(dim=-1).mean())
            gold_kv_norm = torch.stack(kv_norms).mean()
        return outputs.logits, gold_kv_norm

    def _inject_forward(
        self,
        profile_ids: torch.Tensor,
        profile_mask: torch.Tensor,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        suffix_ids: torch.Tensor,
        suffix_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoder forward → K,V → inject into LLM with suffix → logits.

        Returns:
            logits:      (B, S, V)
            gate_values: (B, num_layers, num_heads) — for gate loss
            kv_norm:     scalar — encoder KV norm for norm matching
        """
        kv_pairs, gate_values, kv_norm = self.encoder(
            profile_ids, profile_mask, query_ids, query_mask
        )
        logits = forward_with_injection(self.llm, suffix_ids, suffix_mask, kv_pairs)
        return logits, gate_values, kv_norm

    def train_step(self, batch: dict) -> dict:
        """Single training step on one batch.

        Returns dict with all loss components for logging.
        """
        B = batch["profile_ids"].shape[0]
        device = self.device
        cfg = self.config.training

        profile_ids = batch["profile_ids"].to(device)
        profile_mask = batch["profile_mask"].to(device)
        query_ids = batch["query_ids"].to(device)
        query_mask = batch["query_mask"].to(device)
        suffix_ids = batch["suffix_ids"].to(device)
        suffix_mask = batch["suffix_mask"].to(device)
        gold_ids = batch["gold_ids"].to(device)
        gold_mask = batch["gold_mask"].to(device)

        total_loss = torch.tensor(0.0, device=device)
        agg = {"loss_total": 0.0, "loss_distill": 0.0, "loss_gate": 0.0,
               "loss_entropy": 0.0, "loss_kv_norm": 0.0, "gate_mean": 0.0}
        valid = 0

        # Process per-sample (gold/inject have different effective lengths)
        for i in range(B):
            # Gold forward — also captures LLM KV norms
            gi = gold_ids[i : i + 1]
            gm = gold_mask[i : i + 1]
            gold_logits, gold_kv_norm = self._gold_forward(gi, gm)

            # Inject forward — also returns gate values + encoder KV norm
            inject_logits, gate_values, enc_kv_norm = self._inject_forward(
                profile_ids[i : i + 1],
                profile_mask[i : i + 1],
                query_ids[i : i + 1],
                query_mask[i : i + 1],
                suffix_ids[i : i + 1],
                suffix_mask[i : i + 1],
            )

            # Align: take last S logits from gold where S = suffix length
            S = suffix_ids.shape[1]
            gold_len = gm.sum().item()
            suffix_start = max(0, int(gold_len) - S)
            gold_suffix = gold_logits[:, suffix_start : suffix_start + S, :]
            inject_suffix = inject_logits[:, :S, :]
            sm = suffix_mask[i : i + 1]

            loss, metrics = combined_loss(
                inject_suffix,
                gold_suffix,
                sm,
                gate_values=gate_values,
                encoder_kv_norm=enc_kv_norm,
                gold_kv_norm=gold_kv_norm,
                lambda_distill=cfg.lambda_distill,
                lambda_gate=cfg.lambda_gate,
                lambda_kv_norm=cfg.lambda_kv_norm,
                lambda_entropy=cfg.lambda_entropy,
                gate_target=cfg.gate_target,
            )
            total_loss = total_loss + loss

            for k in agg:
                agg[k] += metrics.get(k, 0.0)
            valid += 1

        if valid == 0:
            return {k: 0.0 for k in agg}

        avg_loss = total_loss / valid
        avg_loss.backward()

        return {k: v / valid for k, v in agg.items()}

    @torch.no_grad()
    def validate(self, max_samples: int = 50) -> dict:
        """Run validation: KL loss + keyword hit ratio + PPL + gate stats."""
        self.encoder.eval()
        device = self.device
        total_loss = 0.0
        n_loss = 0

        # ── Part 1: KL loss on full val subset ──
        for batch in self.val_loader:
            if n_loss >= max_samples:
                break

            profile_ids = batch["profile_ids"].to(device)
            profile_mask = batch["profile_mask"].to(device)
            query_ids = batch["query_ids"].to(device)
            query_mask = batch["query_mask"].to(device)
            suffix_ids = batch["suffix_ids"].to(device)
            suffix_mask = batch["suffix_mask"].to(device)
            gold_ids = batch["gold_ids"].to(device)
            gold_mask = batch["gold_mask"].to(device)

            # Gold
            gold_logits, _ = self._gold_forward(gold_ids, gold_mask)

            # Inject
            kv_pairs, _, _ = self.encoder(profile_ids, profile_mask, query_ids, query_mask)
            inject_logits = forward_with_injection(
                self.llm, suffix_ids, suffix_mask, kv_pairs
            )

            S = suffix_ids.shape[1]
            gold_len = gold_mask.sum().item()
            suffix_start = max(0, int(gold_len) - S)
            gold_suffix = gold_logits[:, suffix_start : suffix_start + S, :]

            loss, _ = combined_loss(inject_logits[:, :S, :], gold_suffix, suffix_mask)
            total_loss += loss.item()
            n_loss += 1

        val_loss = total_loss / max(n_loss, 1)

        # ── Part 2: Generation-based metrics (smaller subset — generation is slow) ──
        gen_samples = min(20, max_samples, len(self.val_dataset))
        kw_inject_total = 0
        kw_gold_total = 0
        kw_facts_total = 0
        ppl_inject_sum = 0.0
        ppl_gold_sum = 0.0
        gate_means = []
        gate_stds = []
        n_gen = 0

        for idx in range(gen_samples):
            sample = self.val_dataset[idx]
            raw = self.val_dataset.data[idx]

            p_ids = sample["profile_ids"].unsqueeze(0).to(device)
            p_mask = sample["profile_mask"].unsqueeze(0).to(device)
            q_ids = sample["query_ids"].unsqueeze(0).to(device)
            q_mask = sample["query_mask"].unsqueeze(0).to(device)

            # Encoder forward with diagnostics
            kv_pairs, gate_vals, _, diag = self.encoder(
                p_ids, p_mask, q_ids, q_mask, return_diagnostics=True
            )

            # Generate with injection
            suffix_text = _build_suffix(self.tokenizer, raw["query_text"])
            suffix_ids = self.tokenizer(suffix_text, return_tensors="pt")[
                "input_ids"
            ].to(device)

            inject_text = generate_with_injection(
                self.llm,
                self.tokenizer,
                kv_pairs,
                suffix_ids,
                max_new_tokens=self.config.training.max_new_tokens,
                temperature=0.7,
            )

            # Gold generation
            g_ids = sample["gold_ids"].unsqueeze(0).to(device)
            g_mask = sample["gold_mask"].unsqueeze(0).to(device)
            gold_out = self.llm.generate(
                g_ids,
                attention_mask=g_mask,
                max_new_tokens=self.config.training.max_new_tokens,
                do_sample=False,
            )
            gold_text = self.tokenizer.decode(
                gold_out[0][g_ids.shape[1] :], skip_special_tokens=True
            )

            # Keywords
            facts = raw["relevant_facts"]
            kw_i = score_keywords(inject_text, facts)
            kw_g = score_keywords(gold_text, facts)
            kw_inject_total += kw_i["hit_count"]
            kw_gold_total += kw_g["hit_count"]
            kw_facts_total += kw_i["total_facts"]

            # PPL
            ppl_inject_sum += score_coherence(
                self.llm, self.tokenizer, inject_text, device
            )
            ppl_gold_sum += score_coherence(self.llm, self.tokenizer, gold_text, device)

            # Gate stats
            gate_vals_sq = gate_vals.squeeze(0)
            gate_means.append(gate_vals_sq.mean().item())
            gate_stds.append(gate_vals_sq.std().item())

            n_gen += 1

        # Aggregate
        kw_ratio = kw_inject_total / max(kw_facts_total, 1) * 100
        kw_gold_ratio = kw_gold_total / max(kw_facts_total, 1) * 100
        avg_ppl_inject = ppl_inject_sum / max(n_gen, 1)
        avg_ppl_gold = ppl_gold_sum / max(n_gen, 1)
        avg_gate_mean = sum(gate_means) / max(n_gen, 1)
        avg_gate_std = sum(gate_stds) / max(n_gen, 1)

        self.encoder.train()
        return {
            "val_loss": val_loss,
            "val_samples": n_loss,
            "kw_inject_ratio": round(kw_ratio, 2),
            "kw_gold_ratio": round(kw_gold_ratio, 2),
            "ppl_inject": round(avg_ppl_inject, 2),
            "ppl_gold": round(avg_ppl_gold, 2),
            "gate_mean": round(avg_gate_mean, 4),
            "gate_std": round(avg_gate_std, 4),
            "gen_samples": n_gen,
        }

    def train(self):
        """Full training loop with gradient accumulation."""
        cfg = self.config.training
        self.encoder.train()

        step = 0
        accum_step = 0
        best_val_loss = float("inf")
        t_start = time.time()

        print("=" * 72)
        print("  TRAINING LEVEL 4 MULTI-BANK ENCODER")
        print("=" * 72)
        print(f"  Max steps:       {cfg.max_steps}")
        print(f"  Batch size:      {cfg.batch_size}")
        print(f"  Grad accum:      {cfg.gradient_accumulation_steps}")
        print(f"  Effective batch:  {cfg.batch_size * cfg.gradient_accumulation_steps}")
        print(f"  LR:              {cfg.learning_rate}")
        print(f"  Eval every:      {cfg.eval_every} steps")
        print(f"  Save every:      {cfg.save_every} steps")
        print(f"  Train samples:   {len(self.train_dataset)}")
        print(f"  Val samples:     {len(self.val_dataset)}")
        print("=" * 72)
        print()

        self.optimizer.zero_grad()

        while step < cfg.max_steps:
            for batch in self.train_loader:
                if step >= cfg.max_steps:
                    break

                loss_dict = self.train_step(batch)
                accum_step += 1

                # Optimizer step after accumulation
                if accum_step % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    step += 1

                    # Log
                    if step % 10 == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed = time.time() - t_start
                        vram = torch.cuda.memory_allocated() / 1024**3
                        print(
                            f"  step {step:5d}/{cfg.max_steps} | "
                            f"loss {loss_dict['loss_total']:.4f} "
                            f"(kl={loss_dict['loss_distill']:.3f} "
                            f"gate={loss_dict.get('loss_gate', 0):.3f} "
                            f"ent={loss_dict.get('loss_entropy', 0):.4f} "
                            f"kv={loss_dict.get('loss_kv_norm', 0):.3f}) | "
                            f"g={loss_dict.get('gate_mean', 0):.3f} | "
                            f"lr {lr:.2e} | "
                            f"VRAM {vram:.1f}G | "
                            f"time {elapsed:.0f}s"
                        )

                    self.history.append(
                        {
                            "step": step,
                            "loss": loss_dict["loss_total"],
                            "distill": loss_dict["loss_distill"],
                            "gate_loss": loss_dict.get("loss_gate", 0.0),
                            "entropy_loss": loss_dict.get("loss_entropy", 0.0),
                            "kv_norm_loss": loss_dict.get("loss_kv_norm", 0.0),
                            "gate_mean": loss_dict.get("gate_mean", 0.0),
                        }
                    )

                    # Evaluate
                    if step % cfg.eval_every == 0:
                        val_result = self.validate()
                        print(
                            f"\n  {'=' * 60}\n"
                            f"  EVAL step {step}:\n"
                            f"    val_loss:       {val_result['val_loss']:.4f}\n"
                            f"    KW hit ratio:   {val_result['kw_inject_ratio']:.1f}% inject  |  {val_result['kw_gold_ratio']:.1f}% gold\n"
                            f"    PPL:            {val_result['ppl_inject']:.1f} inject  |  {val_result['ppl_gold']:.1f} gold\n"
                            f"    Gate:           mean={val_result['gate_mean']:.4f}  std={val_result['gate_std']:.4f}\n"
                            f"    (n={val_result['val_samples']} loss, {val_result['gen_samples']} gen)\n"
                            f"  {'=' * 60}\n"
                        )
                        self.history[-1].update(
                            {
                                "val_loss": val_result["val_loss"],
                                "kw_inject_ratio": val_result["kw_inject_ratio"],
                                "kw_gold_ratio": val_result["kw_gold_ratio"],
                                "ppl_inject": val_result["ppl_inject"],
                                "ppl_gold": val_result["ppl_gold"],
                                "gate_mean": val_result["gate_mean"],
                                "gate_std": val_result["gate_std"],
                            }
                        )

                        if val_result["val_loss"] < best_val_loss:
                            best_val_loss = val_result["val_loss"]
                            self._save_checkpoint(step, "best", val_loss=best_val_loss)
                            print(f"  >>> New best val_loss: {best_val_loss:.4f}\n")

                    # Save periodic checkpoint
                    if step % cfg.save_every == 0:
                        self._save_checkpoint(step, f"step_{step}")

                    # Memory cleanup
                    if step % 100 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

        # Final save
        self._save_checkpoint(step, "final")
        self._save_history()

        total_time = time.time() - t_start
        print(f"\n  Training complete: {step} steps in {total_time:.0f}s")
        print(f"  Best val_loss: {best_val_loss:.4f}")

        return self.history

    def _save_checkpoint(self, step: int, tag: str, val_loss: float | None = None):
        """Save encoder weights and optimizer state."""
        path = self.output_dir / f"encoder_{tag}.pt"
        save_dict = {
            "step": step,
            "encoder_state_dict": self.encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": {
                "d_model": self.config.encoder.d_model,
                "n_heads": self.config.encoder.n_heads,
                "profile_encoder_layers": self.config.encoder.profile_encoder_layers,
                "query_encoder_layers": self.config.encoder.query_encoder_layers,
                "num_output_slots": self.config.encoder.num_output_slots,
                "num_layer_groups": self.config.encoder.num_layer_groups,
                "gate_hidden_dim": self.config.encoder.gate_hidden_dim,
            },
        }
        if val_loss is not None:
            save_dict["val_loss"] = val_loss
        torch.save(save_dict, path)
        print(f"  Saved checkpoint: {path}")

    def _save_history(self):
        """Save training history to JSON."""
        path = self.log_dir / "training_history.json"
        path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        print(f"  Saved history: {path}")

"""
PyTorch Dataset for Level 4 multi-bank training.

Each item provides:
  profile_ids, profile_mask  — tokenized full profile (encoder input)
  query_ids, query_mask      — tokenized raw query (query encoder input)
  suffix_ids, suffix_mask    — chat-formatted user turn (LLM inject forward)
  gold_ids, gold_mask        — system(relevant facts) + user turn (LLM gold forward)
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

SYSTEM_TEMPLATE = (
    "You are a helpful assistant. Here is relevant information about the user:\n{facts}"
)
SUFFIX_TEMPLATE = "<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"


class ProfileQueryDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_profile_tokens: int = 384,
        max_query_tokens: int = 128,
    ):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_profile = max_profile_tokens
        self.max_query = max_query_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        # --- Profile (full, all types) → encoder input ---
        profile_enc = self.tokenizer(
            ex["profile_text"],
            max_length=self.max_profile,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # --- Query (raw) → query encoder input ---
        query_enc = self.tokenizer(
            ex["query_text"],
            max_length=self.max_query,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # --- Suffix (chat-formatted user turn) → LLM inject forward ---
        suffix_text = SUFFIX_TEMPLATE.format(query=ex["query_text"])
        suffix_enc = self.tokenizer(
            suffix_text,
            max_length=self.max_query,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # --- Gold prompt (system with ONLY relevant facts + user turn) ---
        facts_str = "\n".join(f"- {f}" for f in ex["relevant_facts"])
        system_content = SYSTEM_TEMPLATE.format(facts=facts_str)
        gold_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": ex["query_text"]},
        ]
        gold_text = self.tokenizer.apply_chat_template(
            gold_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        gold_enc = self.tokenizer(
            gold_text,
            max_length=self.max_profile + self.max_query,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "profile_ids": profile_enc["input_ids"].squeeze(0),
            "profile_mask": profile_enc["attention_mask"].squeeze(0),
            "query_ids": query_enc["input_ids"].squeeze(0),
            "query_mask": query_enc["attention_mask"].squeeze(0),
            "suffix_ids": suffix_enc["input_ids"].squeeze(0),
            "suffix_mask": suffix_enc["attention_mask"].squeeze(0),
            "gold_ids": gold_enc["input_ids"].squeeze(0),
            "gold_mask": gold_enc["attention_mask"].squeeze(0),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Stack batch into tensors."""
    return {key: torch.stack([b[key] for b in batch]) for key in batch[0]}

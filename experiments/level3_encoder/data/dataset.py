"""
PyTorch Dataset for (profile, query) pairs with gold-standard generation.

Each sample provides:
  - profile_text: raw user profile string
  - query: question string
  - keywords: list of profile-specific keywords for evaluation
  - gold_prefix: system-formatted full prompt for teacher forcing
  - inject_suffix: user+query suffix tokens (for injection path)
"""

import json
import random
from pathlib import Path

from torch.utils.data import Dataset


class ProfileQueryDataset(Dataset):
    """Yields (profile, query) pairs for training the memory encoder.

    The dataset pairs each profile with a randomly sampled query.
    On each epoch, profiles get different queries (controlled by epoch seed).
    """

    def __init__(
        self,
        profiles_path: str,
        queries_path: str,
        tokenizer,
        max_profile_tokens: int = 256,
        seed: int = 42,
    ):
        self.profiles = json.loads(Path(profiles_path).read_text(encoding="utf-8"))
        self.queries = json.loads(Path(queries_path).read_text(encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_profile_tokens = max_profile_tokens
        self.rng = random.Random(seed)

        # Pre-assign a query index to each profile (reshuffled each epoch)
        self._assign_queries()

    def _assign_queries(self):
        """Assign a random query to each profile."""
        self.query_indices = [
            self.rng.randint(0, len(self.queries) - 1)
            for _ in range(len(self.profiles))
        ]

    def set_epoch(self, epoch: int):
        """Reshuffle query assignments for a new epoch."""
        self.rng = random.Random(42 + epoch)
        self._assign_queries()

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx: int) -> dict:
        profile = self.profiles[idx]
        query = self.queries[self.query_indices[idx]]

        profile_text = profile["text"]
        keywords = profile["keywords"]

        # ── Gold prompt (teacher) ──
        # System message with profile + user query → what the model SHOULD produce
        system_content = (
            f"You are a helpful personal assistant. Here is everything you know "
            f"about the user you are talking to:\n\n{profile_text}\n\n"
            f"Always personalize your responses based on this information. "
            f"Refer to their location, interests, profession, and preferences."
        )
        gold_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]
        gold_prompt = self.tokenizer.apply_chat_template(
            gold_messages, tokenize=False, add_generation_prompt=True
        )

        # ── Injection suffix (student) ──
        # Only the user+query part — profile goes through encoder → K,V
        inject_suffix = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

        # ── Tokenize profile for encoder input ──
        profile_tokens = self.tokenizer(
            profile_text,
            max_length=self.max_profile_tokens,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "profile_text": profile_text,
            "profile_input_ids": profile_tokens["input_ids"].squeeze(0),
            "profile_attention_mask": profile_tokens["attention_mask"].squeeze(0),
            "query": query,
            "keywords": keywords,
            "gold_prompt": gold_prompt,
            "inject_suffix": inject_suffix,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate — pad profile tokens, keep strings as lists."""
    from torch.nn.utils.rnn import pad_sequence

    profile_ids = [b["profile_input_ids"] for b in batch]
    profile_masks = [b["profile_attention_mask"] for b in batch]

    # Pad profile tokens (right-pad with 0)
    profile_ids_padded = pad_sequence(profile_ids, batch_first=True, padding_value=0)
    profile_masks_padded = pad_sequence(
        profile_masks, batch_first=True, padding_value=0
    )

    return {
        "profile_input_ids": profile_ids_padded,
        "profile_attention_mask": profile_masks_padded,
        "profile_texts": [b["profile_text"] for b in batch],
        "queries": [b["query"] for b in batch],
        "keywords": [b["keywords"] for b in batch],
        "gold_prompts": [b["gold_prompt"] for b in batch],
        "inject_suffixes": [b["inject_suffix"] for b in batch],
    }

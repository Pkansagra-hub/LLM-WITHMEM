"""Quick 30-step smoke test of the full trainer pipeline."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.level3_encoder.config import Config
from experiments.level3_encoder.training.trainer import Trainer

config = Config()
config.training.max_steps = 30
config.training.eval_every = 15
config.training.save_every = 30
config.training.gradient_accumulation_steps = 1

tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    config.model.model_id, dtype=torch.float16, device_map="cuda"
)
model.eval()

trainer = Trainer(config, model, tokenizer)
history = trainer.train()
print(f"Steps: {len(history)}")
first = history[0]["loss"]
last = history[-1]["loss"]
print(f"First loss: {first:.2f}")
print(f"Last loss: {last:.2f}")
print(f"Converging: {first > last}")

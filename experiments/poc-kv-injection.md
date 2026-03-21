# POC: KV Injection on Local SLM (RTX 5070)

**Goal:** Prove that external vector injection into a frozen language model's
attention layers changes output behavior — running entirely on a laptop GPU.

**Hardware:** RTX 5070 (12 GB desktop / 8 GB laptop GDDR7)

---

## Target Model Selection

| Model | Params | fp16 VRAM | 4-bit VRAM | Layers | Heads | d_head | Recommended |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B-Instruct | 3B | ~6.0 GB | ~2.5 GB | 36 | 16 KV-heads (GQA) | 128 | **PRIMARY** |
| Phi-3.5-mini-instruct | 3.8B | ~7.6 GB | ~3.0 GB | 32 | 8 KV-heads (GQA) | 96 | BACKUP |
| Gemma-2-2B-it | 2B | ~4.0 GB | ~1.5 GB | 26 | 8 KV-heads | 256 | LIGHTWEIGHT |
| SmolLM2-1.7B-Instruct | 1.7B | ~3.4 GB | ~1.2 GB | 24 | 16 KV-heads | 64 | MINIMAL |
| Llama-3.2-3B-Instruct | 3B | ~6.0 GB | ~2.5 GB | 28 | 8 KV-heads (GQA) | 128 | ALTERNATIVE |

**Start with Qwen2.5-3B-Instruct.** Well-documented HuggingFace implementation,
clean attention code, grouped-query attention with accessible K,V tensors, strong
instruction-following at 3B scale. Fits fp16 on 12 GB with room for injection overhead.

If laptop variant (8 GB): use Gemma-2-2B-it or SmolLM2-1.7B-Instruct.

---

## VRAM Budget (Qwen2.5-3B, fp16, 12 GB card)

```text
Qwen2.5-3B model weights (fp16):   ~6.0 GB
KV cache (2048 context length):     ~0.5 GB
Memory KV injection (64 slots):     ~0.001 GB   (256 KB — negligible)
Memory encoder model (~5M params):  ~0.01 GB    (10 MB)
PyTorch/CUDA overhead:              ~1.0 GB
Gradient buffers (Level 3 only):    ~1.5 GB
────────────────────────────────────────────────
TOTAL (inference only, L1/L2):      ~7.5 GB     ✓ fits 8 GB laptop
TOTAL (training, L3):               ~9.0 GB     ✓ fits 12 GB desktop
```

---

## Experiment Levels

### Level 1: Mechanical Proof — "Does injection work at all?"

**Time:** 1-2 hours
**Risk of failure:** ~0% (this is linear algebra — if shapes match, it works)

**Objective:** Append random external K,V tensors to the attention computation
of a frozen SLM and observe that output token probabilities change.

**What we prove:** The attention mechanism can process externally injected
K,V pairs without crashing, producing NaN, or ignoring them.

**Setup:**

1. Load Qwen2.5-3B-Instruct in fp16 on CUDA
2. Create RANDOM tensors as "memory" K,V (shape-matched to model's attention)
3. Hook into each attention layer's forward pass via `register_forward_hook`
4. Append memory K,V to the token K,V before the softmax computation
5. Run the same prompt with and without injection
6. Compare output logit distributions

**Expected results:**

- Output token probabilities SHIFT (different top-k tokens)
- Random vectors produce random/noisy shifts (expected — no training yet)
- Attention weight tensor shows non-zero values at memory positions
- No NaN, no crash, no silent ignore

**Success criteria:**

- [x] KL divergence between with-memory and without-memory logits > 0.01
- [x] Attention weights on memory positions are non-zero (mean > 1e-4)
- [x] Generation completes without error on 10 different prompts
- [x] Output text is different (not identical) between the two conditions

**Key code challenge:** Understanding the exact attention implementation in
the target model. Qwen2 uses `Qwen2Attention` with GQA — need to handle
`num_key_value_heads != num_attention_heads` (K,V are broadcast across head groups).

**File:** `level1_mechanical_proof.py`

---

### Level 2: Behavioral Steering — "Can injection change output meaningfully?"

**Time:** 1 day
**Risk of failure:** ~5% (extracted K,V are already in the model's native space)

**Objective:** Extract K,V representations from a "user profile" sentence by
running it through the model, then inject those K,V pairs when generating
a response to a query. Show that the response is personalized to the profile.

**What we prove:** Information encoded in injected K,V vectors flows through
the frozen model's computation and influences the final output in a
semantically meaningful way.

**Setup:**

1. Define 5 user profiles with distinct, testable attributes:

```text
Profile A: "Alex is a marine biologist living in Seattle who hates cilantro
            and loves Thai food."
Profile B: "Priya is a software engineer in Bangalore who is vegetarian and
            enjoys hiking in the Western Ghats."
Profile C: "Marcus is a retired firefighter in Austin, Texas who builds
            custom furniture and has a dog named Biscuit."
Profile D: "Yuki is a concert pianist in Tokyo who is training for a marathon
            and collects vintage vinyl records."
Profile E: "Fatima is a pediatric surgeon in London who speaks four languages
            and is passionate about urban gardening."
```

1. For each profile:
   a. Forward pass the profile text through the model
   b. Extract K,V from every attention layer (save as `memory_kv[layer_idx]`)
   c. Ask 5 neutral queries that SHOULD be influenced by the profile:
      - "What restaurant should I go to tonight?"
      - "What's a good hobby to pick up?"
      - "What should I do this weekend?"
      - "What gift should I get for a friend?"
      - "Tell me something interesting about where I live."

1. For each (profile, query) pair:
   a. Generate response WITHOUT memory injection (baseline)
   b. Generate response WITH memory K,V injected
   c. Score: does the response contain profile-specific information?

**Expected results:**

- Profile A + restaurant query → mentions Seattle, Thai, avoids cilantro
- Profile B + hobby query → mentions outdoor activities, vegetarian-friendly
- Profile C + weekend query → mentions woodworking, dog activities, Austin
- Profile D + gift query → mentions music, running gear, vinyl
- Profile E + interesting query → mentions London, multilingual, gardens

**Success criteria:**

- [ ] ≥80% of (profile, query) pairs show profile-relevant content in output
- [ ] Baseline (no injection) shows NONE of the profile-specific content
- [ ] Attention visualization shows high weights on memory positions for
      semantically relevant tokens (e.g., "restaurant" attends to "Seattle" memory)
- [ ] Output quality remains coherent (no gibberish from injection)

**Evaluation method:** Manual inspection + keyword matching. For each response,
check if profile-specific entities (city, profession, preferences) appear.

**File:** `level2_behavioral_steering.py`

---

### Level 3: Trained Memory Encoder — "Can a side-model learn to produce K,V?"

**Time:** 1 week
**Risk of failure:** ~20% (real research question: does the encoder converge?)

**Objective:** Train a small neural network (the "memory encoder") that takes
user profile text as input and produces K,V pairs optimized for injection into
the frozen SLM. The training signal comes from matching the injected-memory
output distribution to the text-in-prompt output distribution.

**What we prove:** A trained side-model can compress textual information into
K,V vectors that achieve the same personalization effect as pasting the full
text into the prompt — but using ZERO context window tokens.

**Setup:**

```text
Architecture:
┌─────────────────────────────────────────────┐
│ Memory Encoder (~2-5M params)                │
│                                              │
│ Input:  profile text (tokenized, max 256 tok)│
│ Backbone: 2-layer transformer (d=256, 4 heads)│
│ Pooling: M learned query vectors (M=16)       │
│          cross-attend into encoder output     │
│          (Perceiver-style resampling)         │
│ Output:  M memory vectors, each d=256         │
│                                              │
│ Projection heads (one per LLM layer group):   │
│   proj_k_g: [256] -> [num_kv_heads * d_head]  │
│   proj_v_g: [256] -> [num_kv_heads * d_head]  │
│   4 groups (layers 1-9, 10-18, 19-27, 28-36)  │
│                                              │
│ Per-head gates: sigmoid(w_gate[layer][head])  │
│   36 layers * 16 KV-heads = 576 scalars       │
└─────────────────────────────────────────────┘
```

**Training procedure:**

```text
For each training step:
  1. Sample a user profile P
  2. Sample a query Q

  3. ORACLE path (teacher):
     prompt = P + Q   (profile pasted in context)
     logits_oracle = frozen_SLM(prompt)

  4. INJECTION path (student):
     memory_kv = MemoryEncoder(P)
     logits_inject = frozen_SLM(Q, injected_kv=memory_kv)

  5. Loss:
     L_distill = KL_divergence(logits_inject, logits_oracle)

     Optional auxiliary losses:
     L_reconstruct = can the memory vectors reconstruct P's embedding?
     L_vicreg = variance/covariance regularization on memory vectors

     L_total = L_distill + 0.1 * L_reconstruct + 0.01 * L_vicreg

  6. Backward pass: ONLY update MemoryEncoder parameters.
     The SLM is FROZEN (no gradients).
```

**Training data:**

- Generate 1000-5000 synthetic user profiles (GPT-4 or template-based)
- Each profile: 2-5 sentences, distinct attributes (location, profession,
  preferences, family, hobbies)
- 50-100 neutral query templates
- Each step: random (profile, query) pair

**Success criteria:**

- [ ] Training loss converges (L_distill decreases over epochs)
- [ ] Injected-memory responses match text-in-prompt responses in quality
      (human eval: ≥70% of the time, injected version is equally personalized)
- [ ] Memory encoder produces distinct vectors for distinct profiles
      (cosine distance between Profile A and Profile B vectors > 0.3)
- [ ] Zero context tokens consumed by memory injection
- [ ] Inference latency of memory encoder < 10ms (on GPU)
- [ ] Total VRAM stays under 12 GB during training

**Key risks and mitigations:**

| Risk | Mitigation |
| --- | --- |
| Encoder doesn't converge | Start with Level 2's extracted K,V as initialization |
| Gradient doesn't flow through frozen model | Use straight-through estimator or distillation loss (no gradient through SLM needed) |
| VRAM overflow during training | Use gradient checkpointing, reduce batch size to 1, accumulate gradients |
| Injection causes gibberish | Add per-head gates initialized near zero, gradually increase |
| Overfitting to training profiles | Use dropout in encoder, diverse profile generation |

**File:** `level3_trained_encoder.py`

---

## Experiment Infrastructure

### Directory Structure

```text
experiments/
├── poc-kv-injection.md          # this file — experiment plan
├── level1_mechanical_proof.py   # Level 1 implementation
├── level2_behavioral_steering.py # Level 2 implementation
├── level3_trained_encoder.py    # Level 3 implementation
├── utils/
│   ├── model_loader.py          # shared model loading + device config
│   ├── kv_hooks.py              # attention hook utilities for KV injection
│   ├── attention_viz.py         # attention weight visualization
│   └── eval_metrics.py          # KL divergence, keyword matching, cosine dist
├── profiles/
│   ├── test_profiles.json       # 5 hand-crafted profiles for Level 2
│   └── training_profiles.json   # 1000+ synthetic profiles for Level 3
├── results/
│   ├── level1/                  # logit diffs, attention heatmaps
│   ├── level2/                  # generated responses, comparison tables
│   └── level3/                  # training curves, eval metrics
└── requirements.txt             # dependencies
```

### Dependencies

```text
torch>=2.2.0
transformers>=4.40.0
accelerate>=0.28.0
matplotlib>=3.8.0       # attention visualization
seaborn>=0.13.0         # heatmaps
pandas>=2.1.0           # results tables
tqdm>=4.66.0            # progress bars
```

### Environment Setup

```bash
cd C:\Users\princ\LLM-WITHMEM
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers accelerate matplotlib seaborn pandas tqdm
```

---

## Execution Order

```text
Week 1:
  Day 1 (2 hrs):  Level 1 — mechanical proof
                   Deliverable: logit shift measurements, attention heatmaps
                   Go/No-Go: if injection doesn't change logits → investigate

  Day 1-2 (1 day): Level 2 — behavioral steering
                    Deliverable: comparison table of responses ± memory
                    Go/No-Go: if <50% personalization rate → investigate profile extraction

  Day 3-7 (5 days): Level 3 — trained encoder
                     Day 3: build encoder architecture + projection heads
                     Day 4: build training loop + distillation loss
                     Day 5: generate synthetic training profiles
                     Day 6: train (likely 2-4 hours on RTX 5070)
                     Day 7: evaluate, visualize, document results

Week 2 (if Level 3 succeeds):
  - Ablation studies: vary M (memory slots), layer injection depth
  - Memory-JEPA objective experiment: mask memories, predict representations
  - Multi-user experiment: load different checkpoints, show personalization switches
  - Latency benchmarking: end-to-end inference time with injection
```

---

## What Each Level Proves for LLM-WITHMEM

| Level | Proves | Maps to Architecture |
| --- | --- | --- |
| 1 | Attention can process external K,V | Per-Layer KV Injection (Section: Injection Point 2) |
| 2 | Profile info flows through injected vectors | Memory Model → LLM connection works |
| 3 | Trained encoder compresses text → effective K,V | Full Memory Encoder → Projection → KV pipeline |

**If Level 3 succeeds, the core thesis of LLM-WITHMEM is validated:**
a small trained side-model can produce K,V vectors that personalize a frozen
LLM's output without consuming any context window tokens.

The remaining work is then:

- Scale from text profiles to K0's structured memory atoms
- Replace synthetic profiles with real P03 consolidated data
- Adopt Memory-JEPA training objective for richer representations
- Integrate into FamilyOS via MemoryFusedAdapter (K1) + P03 R9 Trainer (K0)

---

## Notes

- All experiments use the model in **inference mode** (eval, no_grad) except
  Level 3 which trains only the memory encoder (SLM stays frozen).
- Attention hooks may need model-specific adjustments — Qwen2, Phi-3, Llama-3.2,
  and Gemma-2 each have slightly different attention implementations.
- GQA (Grouped Query Attention) means K,V heads < Q heads. Memory K,V must
  match the K,V head count, not the Q head count.
- Save all results (generated text, attention weights, loss curves) to `results/`
  for documentation and whiteboard updates.

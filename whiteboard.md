# Whiteboard

## Typical Transformer Architecture

A transformer is a neural network architecture designed to process sequences using attention instead of recurrence.

At a high level, a standard transformer contains these parts:

1. Input tokens
2. Token embeddings
3. Positional information
4. Stacked transformer blocks
5. Output projection

```text
 +------------------+
 |   Input Text     |
 +--------+---------+
          |
          v
 +------------------+
 |    Tokenizer     |    "The model remembers" -> [2101, 1904, 8837]
 +--------+---------+
          |
          v
 +------------------+
 |    Token IDs     |    integer sequence
 +--------+---------+
          |
          v
 +------------------+     +----------------------+
 | Token Embeddings | <-- | Positional Encoding  |
 |   (V x d)        |     | sin/cos, learned, or |
 +--------+---------+     | rotary (RoPE)        |
          |               +----------------------+
          v
 +=============================+
 ||  Transformer Block  1     ||
 ||  (Attn + FFN + Residual)  ||
 +=============================+
          |
          v
 +=============================+
 ||  Transformer Block  2     ||
 +=============================+
          |
         ...        x N layers (e.g. 32, 80, 128)
          |
          v
 +=============================+
 ||  Transformer Block  N     ||
 +=============================+
          |
          v
 +------------------+
 |  RMSNorm / LN    |    final normalization
 +--------+---------+
          |
          v
 +------------------+
 | Linear Head      |    project d -> V
 +--------+---------+
          |
          v
 +------------------+
 |  Softmax Logits  |    next-token probability distribution
 +------------------+
```

### 1. Input Tokens

Text is first broken into tokens. These are usually subword units rather than full words.

Example:

"The model remembers context"

may become a sequence of token ids.

### 2. Token Embeddings

Each token id is mapped to a dense vector. This converts discrete symbols into continuous numerical representations that the model can work with.

If the vocabulary size is $V$ and embedding dimension is $d$, then the embedding table is roughly:

$$
E \in \mathbb{R}^{V \times d}
$$

### 3. Positional Information

Transformers do not naturally know token order, so positional information is added.

This can be done with:

- sinusoidal positional encodings,
- learned positional embeddings,
- rotary position embeddings.

This step lets the model distinguish between sentences such as "dog bites man" and "man bites dog".

### 4. Stacked Transformer Blocks

This is the core of the architecture.

Each transformer block usually contains:

1. Multi-head self-attention
2. Add and normalize
3. Feed-forward network
4. Add and normalize

```text
                       +===============================+
                       |     TRANSFORMER BLOCK (i)     |
                       |===============================|
    Input x ---------->|                               |
         |             |                               |
         |       +-----|------- LayerNorm 1 ------+    |
         |       |     |                          |    |
         |       v     |                          |    |
         |   +------+------+------+               |    |
         |   | Wq   | Wk   | Wv   |  projections |    |
         |   +--+---+--+---+--+---+               |    |
         |      |      |      |                   |    |
         |      v      v      v                   |    |
         |   +---------------------+              |    |
         |   | Scaled Dot-Product  |              |    |
         |   | Attention           |              |    |
         |   | softmax(QK^T/sqrt   |              |    |
         |   |        (d_k)) V     |              |    |
         |   +--------+------------+              |    |
         |            | (repeat for each head)    |    |
         |            v                           |    |
         |   +---------------------+              |    |
         |   | Concat + W_o proj   |              |    |
         |   +--------+------------+              |    |
         |            |                           |    |
         +----------->+ (residual add) <----------+    |
                      |                                |
                +-----|------- LayerNorm 2 ------+     |
                |     |                          |     |
                v     |                          |     |
         +---------------------+                 |     |
         |  Feed-Forward Net   |                 |     |
         |  W2 * act(W1*x+b1)  |                 |     |
         |  (SwiGLU / GELU)    |                 |     |
         +---------+-----------+                 |     |
                   |                             |     |
                   + (residual add) <------------+     |
                   |                                   |
    Output <-------|                                   |
                       +===============================+
```

#### Self-Attention

Self-attention lets each token look at other tokens in the sequence and decide which ones matter.

For each token representation, the model creates:

- Query $Q$
- Key $K$
- Value $V$

Attention is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

This gives a weighted mixture of information from other tokens.

#### Multi-Head Attention

Instead of using a single attention operation, transformers split attention into multiple heads. Each head can learn different relationships, such as syntax, long-range dependency, or entity reference.

#### Feed-Forward Network

After attention, each token passes through a position-wise MLP. This usually looks like:

$$
\text{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2
$$

Common activations include GELU or SwiGLU-style variants.

#### Residual Connections and LayerNorm

Residual connections help preserve information across deep stacks, while normalization stabilizes training.

### 5. Output Projection

The final hidden state is projected back into vocabulary space to predict the next token.

The model outputs logits over all possible tokens, and decoding chooses the next one.

## How LLMs Are Made From Transformers

An LLM is essentially a transformer-based language model scaled up in data, parameters, compute, and training sophistication.

The progression is roughly:

1. Start with the transformer architecture
2. Use a decoder-only or encoder-decoder design depending on the task
3. Train on massive text corpora
4. Scale model depth, width, and context length
5. Improve tokenization, optimization, and alignment
6. Add inference systems and product-level orchestration

### Decoder-Only Transformers for Modern LLMs

Most generative LLMs are decoder-only transformers.

```text
 Causal (Decoder-Only) Attention Mask

           Key positions
           t1   t2   t3   t4   t5
         +----+----+----+----+----+
  Q  t1  | ok |  X |  X |  X |  X |   t1 sees only itself
         +----+----+----+----+----+
  u  t2  | ok | ok |  X |  X |  X |   t2 sees t1, t2
         +----+----+----+----+----+
  e  t3  | ok | ok | ok |  X |  X |   t3 sees t1..t3
         +----+----+----+----+----+
  r  t4  | ok | ok | ok | ok |  X |   t4 sees t1..t4
         +----+----+----+----+----+
  y  t5  | ok | ok | ok | ok | ok |   t5 sees everything so far
         +----+----+----+----+----+

  ok = attention allowed     X = masked (set to -inf before softmax)

  This lower-triangular mask enforces autoregressive generation:
  each token can only attend to itself and all preceding tokens.
```

Why:

- they predict the next token autoregressively,
- they work well for open-ended text generation,
- they scale effectively for chat, coding, reasoning, and summarization.

In a decoder-only model, attention is masked so a token can only attend to earlier tokens, not future ones.

### Training Objective

The core pretraining task is next-token prediction.

Given tokens:

$$
x_1, x_2, \dots, x_t
$$

the model learns:

$$
P(x_{t+1} \mid x_1, x_2, \dots, x_t)
$$

This simple objective, when applied at large scale, produces rich language representations and generative capability.

### Scaling Into an LLM

LLMs emerge when the transformer is scaled across several axes:

- more parameters,
- more layers,
- larger hidden dimensions,
- more attention heads,
- larger training datasets,
- longer context windows,
- more compute and better optimization.

This scaling is what turns a standard transformer into a high-capability language model.

### Post-Training

Base pretrained models are usually refined with post-training steps such as:

- supervised fine-tuning,
- instruction tuning,
- preference optimization,
- safety alignment,
- domain adaptation.

These steps make the raw transformer behave more like a useful assistant.

## Typical LLM Architecture

At a system level, a modern LLM is not just the neural network itself. It usually includes several layers.

```text
+=====================================================================+
|                      APPLICATION  LAYER                             |
|                                                                     |
|  +----------+ +-------+ +-----------+ +--------+ +-------+         |
|  |  Chat /  | | Tool  | | Retrieval | | Memory | | Safety|         |
|  |  Agent   | | Call  | |   (RAG)   | | Store  | | Guard |         |
|  +----------+ +-------+ +-----------+ +--------+ +-------+         |
|   orchestration, routing, user/session management, observability    |
+=====================================================================+
|                       INFERENCE  LAYER                              |
|                                                                     |
|  +----------+ +----------+ +---------+ +----------+ +----------+   |
|  | Prompt   | |  KV      | | Request | | Sampling | | Streaming|   |
|  | Template | |  Cache   | | Batching| | /Decode  | | Output   |   |
|  +----------+ +----------+ +---------+ +----------+ +----------+   |
|   context window management, speculative decoding, quantization     |
+=====================================================================+
|                        MODEL  LAYER                                 |
|                                                                     |
|  +----------+ +-----------+ +--------------------+ +----------+    |
|  |Tokenizer | | Embedding | | Decoder Blocks x N | | LM Head  |    |
|  | (BPE /   | |  + Pos.   | | (Attn + FFN + Res) | | (Linear) |    |
|  |  SP)     | |  Encoding | |                    | |          |    |
|  +----------+ +-----------+ +--------------------+ +----------+    |
|   parameters, weights, activations, norms                           |
+=====================================================================+
|                       TRAINING  LAYER                               |
|                                                                     |
|  +----------+ +-----------+ +------------+ +-----------+           |
|  | Pretrain | | SFT /     | | RLHF / DPO | | Eval &    |           |
|  | Data     | | Instruct  | | Alignment  | | Benchmark |           |
|  +----------+ +-----------+ +------------+ +-----------+           |
|   distributed training, mixed precision, gradient checkpointing     |
+=====================================================================+
```

### Model Layer

This is the transformer model proper:

- tokenizer,
- embedding layer,
- stacked decoder blocks,
- final normalization,
- output head.

### Training Layer

This includes:

- massive pretraining datasets,
- distributed training infrastructure,
- optimizers and schedulers,
- checkpointing and evaluation.

### Inference Layer

This includes runtime components such as:

- prompt formatting,
- KV cache,
- batching,
- sampling or decoding,
- latency and throughput optimization.

### Application Layer

This is where real products are built around the LLM. It often includes:

- chat orchestration,
- memory,
- tool calling,
- retrieval,
- user/session management,
- safety filters,
- observability.

v

## KV Cache and Context-Window Behavior

### What Is the KV Cache?

During autoregressive generation the model produces one token at a time. Without a cache, every new token would require recomputing attention over the entire sequence from scratch. The KV cache stores the Key and Value projections of all previous tokens so that each new step only computes Q, K, V for the latest token and reuses everything else.

```text
 Generation step 1          step 2          step 3          step 4
 +-----------+          +-----------+    +-----------+    +-----------+
 | compute   |          | compute   |    | compute   |    | compute   |
 | Q1 K1 V1  |          | Q2 K2 V2  |    | Q3 K3 V3  |    | Q4 K4 V4  |
 +-----+-----+          +-----+-----+    +-----+-----+    +-----+-----+
       |                       |                |                |
       v                       v                v                v
 +===========+          +===========+    +===========+    +===========+
 | KV Cache  |          | KV Cache  |    | KV Cache  |    | KV Cache  |
 |           |          |           |    |           |    |           |
 | K:[K1]    |          | K:[K1,K2] |    | K:[K1..3] |    | K:[K1..4] |
 | V:[V1]    |          | V:[V1,V2] |    | V:[V1..3] |    | V:[V1..4] |
 +===========+          +===========+    +===========+    +===========+
       |                       |                |                |
       v                       v                v                v
  Attend over             Attend over       Attend over      Attend over
  cache (1 key)           cache (2 keys)    cache (3 keys)   cache (4 keys)
       |                       |                |                |
       v                       v                v                v
  token t1                token t2          token t3         token t4

 Per step, only the NEW token's Q/K/V is computed.
 Previous K and V vectors are read from cache, not recomputed.
 This turns O(n^2) per-token work into O(n) per step.
```

### KV Cache Memory Layout (Per Layer)

```text
 One decoder layer's KV cache
 +-------------------------------------------------------+
 |                     Key Cache                          |
 |  shape: [batch, num_heads, seq_len_so_far, head_dim]  |
 +-------------------------------------------------------+
 |                    Value Cache                         |
 |  shape: [batch, num_heads, seq_len_so_far, head_dim]  |
 +-------------------------------------------------------+

 Total KV cache memory (all layers):

   2 * num_layers * batch * num_heads * seq_len * head_dim * bytes_per_param

 Example (7B model, 32 layers, 32 heads, head_dim=128, fp16, seq=4096):
   2 * 32 * 1 * 32 * 4096 * 128 * 2 bytes = ~2 GB per sequence
```

### Context Window

The context window is the maximum number of tokens the model can process in a single forward pass. It is a hard architectural limit set during training.

```text
 Context window = 8192 tokens (example)

 |<---------------------- 8192 tokens ---------------------->|
 +-----------------------------------------------------------+
 |  system   |   memory    |  conversation  |   latest       |
 |  prompt   |   context   |  history       |   user msg     |
 +-----------------------------------------------------------+
 |<- ~200 -->|<-- ~1000 -->|<--- ~5500 --->|<--- ~1492 ---->|
                                                  ^
                                    remaining budget for
                                    generation output

 If total input exceeds the window, tokens must be dropped or
 summarized. The connector decides WHAT fills the window.
```

### Context Window Overflow Strategy

```text
 Available context budget: C tokens

 +--------------------------------------------------------------------+
 | Priority 1: System Prompt                   (fixed, always first)  |
 | Priority 2: Retrieved Memories              (most relevant facts)  |
 | Priority 3: Recent Conversation Turns       (sliding window)       |
 | Priority 4: Older Conversation History      (summarized or pruned) |
 +--------------------------------------------------------------------+

 When input > C:
   1. Summarize or drop Priority 4 first
   2. Reduce Priority 3 (keep last N turns)
   3. Trim Priority 2 (keep top-K memories by relevance score)
   4. Priority 1 is never removed

 This is the connector's job in LLM-WITHMEM.
```

### KV Cache vs. Context Window

```text
 +----------------------------+-----------------------------------+
 |       Context Window       |           KV Cache                |
 +----------------------------+-----------------------------------+
 | Max tokens the model sees  | Runtime storage of K,V tensors   |
 | Set at training time       | Grows as tokens are generated    |
 | Determines prompt budget   | Determines inference memory cost |
 | Measured in tokens          | Measured in bytes (GPU VRAM)     |
 | Overflow -> must truncate  | Overflow -> OOM or eviction      |
 +----------------------------+-----------------------------------+

 The context window limits WHAT the model knows.
 The KV cache determines HOW MUCH memory inference uses.
 This project manages the context window; the serving engine manages the KV cache.
```

## The Real Vision: A Trainable Memory Model (Not Just a Database)

Everything above described the conventional approach: store facts in a database, retrieve them, paste them into the prompt. That approach has hard limits:

- It competes for the context window with the actual conversation.
- It treats memories as flat text, not learned representations.
- Retrieval quality depends on keyword or embedding similarity, which is brittle.
- The LLM itself has no idea it is receiving memories — it just sees more tokens.

The real idea behind this project is different.

### The Core Concept

Build a **second, small neural network** — the Memory Model — that:

1. Is trained exclusively on one user's interactions, preferences, and facts.
2. Trains continuously — every conversation updates it, not just a one-time batch.
3. Produces dense representations (not text snippets) that the LLM can attend to.
4. Plugs directly into the LLM's attention layers at inference time.
5. Steers generation so the output reflects who the user is, not just what words are statistically likely.

### How This Differs From RAG

```text
 CONVENTIONAL RAG                         THIS PROJECT'S MEMORY MODEL
 ============================             ====================================

 +--------+     search      +--------+   +--------+    forward     +--------+
 | user   | --------------> | vector |   | user   | ------------> | memory |
 | query  |    (embedding   |   DB   |   | query  |   (neural     | model  |
 +--------+    similarity)  +---+----+   +--------+    network)    +---+----+
                                |                                      |
                    text chunks |                         learned      |
                                v                       embeddings    v
                          +----------+                          +-----------+
                          |  paste   |                          |  fuse     |
                          |  into    |                          |  into     |
                          |  prompt  |                          |  LLM      |
                          |  as text |                          |  attention|
                          +----+-----+                          +-----+-----+
                               |                                      |
                               v                                      v
                          +---------+                           +-----------+
                          |  LLM    |                           |   LLM     |
                          |  sees   |                           |   sees    |
                          |  extra  |                           |   rich    |
                          |  tokens |                           |   user    |
                          +---------+                           |   context |
                                                                |   as part |
                                                                |   of its  |
                                                                |   own     |
                                                                |   compute |
                                                                +-----------+

 - Uses context window budget              - Does NOT consume context tokens
 - Retrieval is keyword/embedding match    - Representations are LEARNED
 - LLM treats memory as "just more text"   - LLM attends to memory natively
 - Static until explicitly re-indexed      - Continuously updated per chat
```

### What the Memory Model Looks Like

The Memory Model is small and personal. Think of it as a user-specific adapter.

```text
 +================================================================+
 |                     MEMORY  MODEL  (per user)                   |
 |================================================================|
 |                                                                 |
 |  +-------------------+                                          |
 |  | Memory Encoder    |   small transformer or MLP               |
 |  | (few layers)      |   takes user facts + interaction history |
 |  +---------+---------+   outputs dense memory vectors           |
 |            |                                                    |
 |            v                                                    |
 |  +-------------------+                                          |
 |  | Memory Bank       |   a set of learned vectors               |
 |  | [m1, m2, ... mK]  |   each one encodes a cluster of user    |
 |  |                   |   knowledge (preferences, history, etc.) |
 |  +---------+---------+                                          |
 |            |                                                    |
 |            v                                                    |
 |  +-------------------+                                          |
 |  | Projection Layer  |   maps memory vectors into the same      |
 |  | (align to LLM d)  |   dimension as the LLM's hidden states  |
 |  +-------------------+                                          |
 |                                                                 |
 +================================================================+

 Size: tiny compared to the LLM (could be <50M parameters)
 Training: continuous, on every user interaction
 Storage: one checkpoint per user
```

### How It Plugs Into the LLM (Inference-Time Fusion)

The memory model does not modify the LLM's weights. Instead it injects learned representations that the LLM attends to, like virtual tokens the LLM can see but the user never typed.

```text
 INFERENCE  FLOW  WITH  MEMORY  MODEL  FUSION
 =============================================

 +==============+
 |  User Input  |
 +======+=======+
        |
        v
 +==============+       +============================+
 |  Tokenizer   |       |    MEMORY MODEL            |
 |  + Embed     |       |                            |
 +======+=======+       |  load user checkpoint      |
        |               |  encode recent interactions |
        |               |  produce memory vectors     |
        |               |  [m1, m2, ... mK]           |
        |               +============+===============+
        |                            |
        v                            v
 +------+----------------------------+------+
 |          ATTENTION  FUSION               |
 |                                          |
 |  The LLM's self-attention layers see:    |
 |                                          |
 |  K,V from tokens:  [t1, t2, ... tN]     |
 |  K,V from memory:  [m1, m2, ... mK]     |
 |                                          |
 |  Q (from current token) attends over     |
 |  BOTH token keys and memory keys.        |
 |                                          |
 |  Memory vectors sit alongside token      |
 |  vectors — no context window consumed.   |
 +==================+=======================+
                    |
                    v
 +==================+=======================+
 |           LLM  DECODER  BLOCKS           |
 |                                          |
 |  Every layer can attend to memory.       |
 |  The model "knows" the user through      |
 |  learned representations, not pasted     |
 |  text.                                   |
 +==================+=======================+
                    |
                    v
 +==================+=======================+
 |  Output: personalized, context-aware     |
 |  completion driven by who the user IS,   |
 |  not just what words are probable.        |
 +==========================================+
```

### Continuous Training Regime

This is not a train-once model. It updates after every interaction.

**Decision: Option B — Training runs in K0, weights sync to K1.**

K0 already runs the P03 consolidation pipeline nightly. The memory model
trainer becomes a new phase after P03 R8 (event emission). K1 holds only
the inference weights. If offline, K1 uses the last-synced checkpoint —
graceful degradation, not failure.

```text
 CONTINUOUS  TRAINING  LOOP  (FamilyOS  K0/K1  SPLIT)
 =====================================================

   K1 (COGNITIVE KERNEL)                 K0 (DATA KERNEL)
   ====================                 =================

   Chat session happens
          |
          v
   Front LLM extracts
   cognitive signals via tools:
   - update_beliefs()
   - update_scoreboard()
   - refine_affect()
   - update_narrative()
          |
          v
   UltraBERT Phase 1 (22ms)
   extracts: intents, entities,
   emotions, safety, sentiment
          |
          v
   SessionState HOT captures
   structured context per turn
          |
          |--- Bridge sync --->         K0 receives turn data
                                              |
                                              v
                                        +-----+------+
                                        | P03 R0-R1  |  batch select
                                        | Select +   |  + importance
                                        | Score      |  scoring
                                        +-----+------+
                                              |
                                              v
                                        +-----+------+
                                        | P03 R2-R4  |  episodic
                                        | Consolidate|  integration
                                        | Episodes + |  HDBSCAN +
                                        | KG         |  KG + NER
                                        +-----+------+
                                              |
                                              v
                                        +-----+------+
                                        | P03 R5     |  observation-
                                        | Memory     |  driven
                                        | Strengthen |  14 algorithms
                                        |            |  (EST,EWR,ASU..)
                                        +-----+------+
                                              |
                                              v
                                        +-----+------+
                                        | P03 R6-R7  |  staging +
                                        | Stage +    |  truth write
                                        | Truth Write|
                                        +-----+------+
                                              |
                                              v
                                        +-----+------+
                                        | P03 R8     |  events emitted
                                        | Emit Events|
                                        +-----+------+
                                              |
                                              v
                                        +-----+------+
                                        | NEW: R9    |  memory model
                                        | Memory     |  trainer reads
                                        | Model      |  consolidated
                                        | Trainer    |  data from R0-R8
                                        |            |  gradient steps
                                        +-----+------+
                                              |
                                              v
                                        +-----+------+
                                        | Save       |  ~20-60MB
                                        | Checkpoint  |  per user
                                        +-----+------+
                                              |
          <--- Bridge weight sync ---         |
          |
          v
   K1 loads updated
   memory model weights
   into MemoryFusedAdapter
          |
          v
   Next session uses
   updated model


 Training data (all produced by P03 already):
 - Clustered episodes from R2 ---------> L4 episodic training data
 - Importance-scored events from R1 ----> attention weights for salience
 - KG triples + Hebbian edges from R4 -> L5 semantic structure
 - Emotional annotations from R1 ------> L7 emotional associations
 - R5 observation-driven strengthening -> memory quality refinement
   (EST boosts episodic salience, SRE fixes social sentiment,
    SPR refines semantic confidence, ASU seeds anchors,
    MTP promotes memory tiers, NTD links narrative threads)
 - Granger causality edges from R4 -----> L6 procedural cause-effect
 - SPC-UQ cleaned prospective items ----> L8 forward-prediction data

 Key advantages of K0-side training:
 - K0 has ALL consolidated data (not just current session)
 - K0 has more compute (server-side, not edge device)
 - K0 P03 already runs nightly — adding R9 is one more phase
 - Bridge already transports K0->K1 payloads (SSE + sync)
 - K1 works offline with last-synced weights (edge-first)
 - Checkpoint size (~20-60MB) is trivial for Bridge transport
```

### The Plug-and-Play Property

The memory model is designed to be modular. It should work with different LLMs without retraining the LLM itself.

```text
 +==================+     +==================+     +==================+
 |  GPT-4 / Claude  |     |  Llama / Mistral |     |  Custom LLM      |
 |  (frozen)        |     |  (frozen)        |     |  (frozen)        |
 +========+=========+     +========+=========+     +========+=========+
          ^                        ^                        ^
          |                        |                        |
          +----------+-------------+-------------+----------+
                     |                           |
              memory vectors                memory vectors
                     |                           |
              +======+======+             +======+======+
              | Memory      |             | Memory      |
              | Model       |             | Model       |
              | (User A)    |             | (User B)    |
              +=============+             +=============+

 The LLM is never fine-tuned. It stays frozen.
 The memory model is the only thing that trains.
 Swap LLMs freely — the memory model adapts its projection layer.
```

### What This Means For the Output

Without the memory model, the LLM produces:

```text
 P(next_token | prompt_tokens)
 = generic probability distribution over vocabulary
```

With the memory model fused in, the LLM produces:

```text
 P(next_token | prompt_tokens, user_memory_vectors)
 = personalized probability distribution

 The distribution shifts based on who the user is.
 The model doesn't just predict likely words —
 it predicts likely words FOR THIS USER.
```

### Architecture Comparison Summary

```text
 +---------------------+------------------+---------------------------+
 | Approach            | Memory Type      | How It Reaches the LLM    |
 +---------------------+------------------+---------------------------+
 | No memory           | None             | Prompt only               |
 +---------------------+------------------+---------------------------+
 | Chat history        | Text buffer      | Pasted into prompt        |
 +---------------------+------------------+---------------------------+
 | RAG / Vector DB     | Stored text      | Retrieved, pasted into    |
 |                     | chunks           | prompt (uses tokens)      |
 +---------------------+------------------+---------------------------+
 | THIS PROJECT:       | Learned dense    | Injected into attention   |
 | Trainable Memory    | vectors from a   | layers (zero tokens used) |
 | Model               | continuously     | LLM attends to memory    |
 |                     | trained model    | as native representations |
 +---------------------+------------------+---------------------------+
```

---

## Memory Architecture: The 8 Layers of Human Memory

Cognitive science tells us that human memory is not one monolithic system.
It is a hierarchy of specialized subsystems, each handling a different kind
of information at a different timescale. There are 8 distinct memory types
that together give humans their ability to learn, recall, adapt, and plan.

Our memory model should mirror this hierarchy — not as a metaphor, but as
an actual architectural blueprint. Each layer maps to a concrete subsystem
in the model.

### The 8 Layers

```text
 HUMAN  MEMORY  HIERARCHY
 ========================

 +================================================================+
 |                                                                 |
 |  1. SENSORY MEMORY        (milliseconds)                       |
 |     raw perceptual buffer — iconic, echoic, haptic              |
 |                                                                 |
 |  2. SHORT-TERM MEMORY     (seconds to ~30s)                    |
 |     limited capacity, decays fast without rehearsal             |
 |                                                                 |
 |  3. WORKING MEMORY        (active manipulation, seconds-mins)  |
 |     scratchpad for reasoning — holds + transforms info          |
 |                                                                 |
 +--------------------------+-------------------------------------+
                            |
                     consolidation
                            |
                            v
 +================================================================+
 |                  LONG-TERM  MEMORY                              |
 |================================================================|
 |                                                                 |
 |  EXPLICIT (declarative — conscious recall)                      |
 |  +-----------------------------------------------------------+ |
 |  | 4. EPISODIC MEMORY     (events, experiences)               | |
 |  |    "what happened" — time-stamped personal events          | |
 |  +-----------------------------------------------------------+ |
 |  | 5. SEMANTIC MEMORY     (facts, knowledge)                  | |
 |  |    "what I know" — context-free general knowledge          | |
 |  +-----------------------------------------------------------+ |
 |                                                                 |
 |  IMPLICIT (non-declarative — unconscious influence)             |
 |  +-----------------------------------------------------------+ |
 |  | 6. PROCEDURAL MEMORY   (skills, habits)                    | |
 |  |    "how to do things" — automated behaviors                | |
 |  +-----------------------------------------------------------+ |
 |  | 7. EMOTIONAL MEMORY    (affective associations)            | |
 |  |    "how things feel" — learned emotional responses         | |
 |  +-----------------------------------------------------------+ |
 |                                                                 |
 |  META                                                           |
 |  +-----------------------------------------------------------+ |
 |  | 8. PROSPECTIVE MEMORY  (future intentions)                 | |
 |  |    "what I need to do" — remembering to remember           | |
 |  +-----------------------------------------------------------+ |
 |                                                                 |
 +================================================================+
```

### Layer-by-Layer Breakdown

#### Layer 1: Sensory Memory

**In humans:** The raw, unprocessed buffer of everything you just perceived.
Visual (iconic) memory lasts ~250ms. Auditory (echoic) memory lasts ~3-4s.
Most of it is discarded. Only what gets attention moves to short-term memory.

**In the memory model:** This is the **current conversation turn buffer**.
The raw tokens of the user's latest message before any processing.
Almost all of it will be discarded — only salient signals get extracted.

```text
 User types: "Actually I moved to Berlin last week,
              can you recommend coffee shops near me?"

 Sensory buffer captures the FULL input.
 Signal extraction picks out:
   - location change: Berlin (HIGH salience)
   - preference signal: coffee (MEDIUM salience)
   - temporal marker: last week (context)
 Everything else (filler words, syntax) is discarded.
```

**Implementation:** A salience filter that sits before the memory encoder.
Lightweight attention or keyword extraction that decides what enters
the memory pipeline and what gets dropped.

**FamilyOS equivalent: UltraBERT Phase 1 (22ms).** Already built.
UltraBERT runs a single forward pass through a 149M-param ModernBERT
with 12 classification heads: intent (8 classes), safety (4 bands),
emotions (44 classes), sentiment (5 levels), NER (GlobalPointer),
relations. This IS the sensory filter — it extracts structured signals
from raw input in 22ms and writes directly to SessionState
(scoreboard, affective_now, control).

---

#### Layer 2: Short-Term Memory

**In humans:** Holds approximately 7 plus or minus 2 items. Decays within 15-30
seconds without active rehearsal. The "phone number you just heard" memory.

**In the memory model:** This is the **current session context**. The rolling
window of recent turns in the active conversation. Not yet consolidated
into the persistent memory bank.

```text
 Session context buffer (last N turns):
 +-------+-----------------------------------------------+
 | Turn 1| User: "Plan my trip to Tokyo"                 |
 | Turn 2| Bot: "Sure! When are you going?"              |
 | Turn 3| User: "Next March, 10 days, I love ramen"     |
 | Turn 4| Bot: "Great! Here are some suggestions..."    |
 | Turn 5| User: "Also I'm vegetarian now"               |
 +-------+-----------------------------------------------+
          ^                                              ^
          |      these live in short-term memory         |
          |      until the session ends                  |
          +----------------------------------------------+

 Properties:
 - Capacity: limited (configurable window size)
 - Lifespan: current session only
 - Format: raw text or token embeddings
 - After session ends: consolidation decides what persists
```

**Implementation:** A fixed-size ring buffer of recent interaction embeddings.
Fed directly to the memory encoder alongside the persistent memory bank.

**FamilyOS equivalent: SessionState HOT (52KB, 10 sections).** Already built.
The HOT tier holds beliefs_active (8KB), scoreboard (6KB), affective_now (4KB),
clarifications (4KB), narrative_active (4KB), control (8KB), history_active (8KB),
meta (2KB), task_state (4KB), task_artifacts (4KB). This is the rolling session
buffer — recent turns and structured extractions that persist within a session.

---

#### Layer 3: Working Memory

**In humans:** Not just storage — active manipulation. The mental scratchpad
where you hold numbers while doing arithmetic, or juggle multiple constraints
while planning. Baddeley's model includes a central executive, phonological
loop, visuospatial sketchpad, and episodic buffer.

**In the memory model:** This is the **inference-time reasoning context**.
The temporary state that exists while the memory model is computing which
memories are relevant to the current query and how to fuse them.

```text
 User asks: "What should I eat tonight?"

 Working memory activates:
 +----------------------------------------------------+
 | Retrieved from episodic: ate pizza yesterday        |
 | Retrieved from semantic: user is vegetarian         |
 | Retrieved from emotional: user loves spicy food     |
 | Retrieved from procedural: user prefers quick meals |
 |                                                     |
 | Central executive resolves:                         |
 |   - Not pizza (just had it)                         |
 |   - Must be vegetarian                              |
 |   - Should be spicy                                 |
 |   - Should be quick                                 |
 |   -> Compose memory vectors that encode ALL of this |
 +----------------------------------------------------+
```

**Implementation:** The attention mechanism inside the memory encoder itself.
When the memory model processes a query, it attends over the full memory bank,
selects relevant slots, resolves conflicts, and produces the final memory
vectors. This IS the working memory — it exists only during forward pass.

**FamilyOS equivalent: SessionState HOT + WARM (100KB total) + Front LLM
cognitive tools.** Already built. The HOT/WARM eviction system promotes
and demotes information across tiers. The Front LLM's cognitive tools
(update_beliefs, update_scoreboard, update_clarifications, update_narrative,
refine_affect, promote_belief) actively manipulate working memory every turn.
The DynamicPromptBuilder's SessionTrajectory assembles session_goal,
completed_items, active_items, open_items, emotional_arc — this is the
central executive resolving what matters right now.

---

#### Layer 4: Episodic Memory

**In humans:** Time-stamped records of personal experiences. "What happened."
You remember your first day at a new job, the conversation you had yesterday,
the trip you took last summer. Rich in contextual detail — who, what, when,
where.

**In the memory model:** This is the **interaction log** — a structured record
of past conversations and events.

```text
 Episodic memory bank:
 +-------+------------+-----------------------------------------+
 | ID    | Timestamp  | Episode                                 |
 +-------+------------+-----------------------------------------+
 | ep_01 | 2026-01-15 | User asked about Python decorators.     |
 |       |            | Preferred detailed code examples.       |
 +-------+------------+-----------------------------------------+
 | ep_02 | 2026-01-20 | User planned trip to Tokyo.             |
 |       |            | Budget: mid-range. Duration: 10 days.   |
 +-------+------------+-----------------------------------------+
 | ep_03 | 2026-02-03 | User corrected: "I'm vegetarian now."   |
 |       |            | Previous assumption was non-vegetarian.  |
 +-------+------------+-----------------------------------------+
 | ep_04 | 2026-03-10 | User said they moved to Berlin.         |
 |       |            | Previous location was NYC.               |
 +-------+------------+-----------------------------------------+

 Properties:
 - Each episode has a timestamp and context
 - Episodes can be queried by time range or content
 - Recent episodes have higher activation than old ones
 - Repeated themes across episodes get consolidated into semantic memory
```

**Implementation:** A time-indexed sequence of compressed interaction
summaries. Each episode is encoded as a dense vector by the memory encoder.
These vectors form part of the memory bank. Temporal decay or recency
weighting determines which episodes are most active.

**FamilyOS equivalent: K0 P03 R2 episodic_integrator (HDBSCAN clustering).**
Already built. P03 phase R2 groups raw conversation turns into coherent
episodes using HDBSCAN density-based clustering. Stored in st_hipp_events
with: sentiment_score, sentiment_label, dominant_emotions_json,
ner_entities_json, temporal_json. The data pipeline exists — the memory
model adds a neural encoding layer on top of these clustered episodes.

---

#### Layer 5: Semantic Memory

**In humans:** Decontextualized facts and general knowledge. "What I know."
You know that Paris is the capital of France without remembering when you
learned it. Facts that have been distilled from many episodes.

**In the memory model:** This is the **persistent fact store** — stable user
knowledge that has been consolidated from repeated episodes.

```text
 Semantic memory bank:
 +---------------------------+-----------------------------+----------+
 | Fact                      | Source                      | Confidence|
 +---------------------------+-----------------------------+----------+
 | User lives in Berlin      | ep_04 (explicit statement)  | 0.98     |
 +---------------------------+-----------------------------+----------+
 | User is vegetarian        | ep_03 (explicit correction) | 0.95     |
 +---------------------------+-----------------------------+----------+
 | User prefers Python       | ep_01, ep_07, ep_12 (pattern)| 0.87    |
 +---------------------------+-----------------------------+----------+
 | User likes spicy food     | ep_02, ep_05 (inferred)     | 0.72     |
 +---------------------------+-----------------------------+----------+
 | User works in data science| ep_06, ep_09 (inferred)     | 0.68     |
 +---------------------------+-----------------------------+----------+

 Properties:
 - Context-free: no timestamp needed, these are stable facts
 - Confidence-weighted: strong signal facts vs. inferred patterns
 - Consolidated from episodic memory over time
 - Can be overwritten when new contradicting episodes arrive
 - Highest-confidence facts dominate memory vector generation
```

**Implementation:** A set of dedicated memory slots (learned embedding vectors)
that encode stable user facts. These slots are updated with low learning rate
(slow memory in the dual-memory framework). They form the core of what the
memory model "knows" about the user.

**FamilyOS equivalent: K0 P03 R4 kg_consolidator (NER + disambiguation +
Hebbian edges).** Already built. P03 phase R4 builds a knowledge graph
from consolidated episodes — extracts named entities, disambiguates them,
creates confidence-weighted triples, and strengthens connections via
Hebbian edge weights. The KG IS semantic memory. The memory model trains
embedding vectors from this KG structure.

---

#### Layer 6: Procedural Memory

**In humans:** Learned skills and habits. "How to do things." You don't
consciously remember how to ride a bicycle — the motor patterns are
implicit. Similarly, typing, cooking sequences, driving routes.

**In the memory model:** This is the **interaction pattern memory** — learned
preferences about HOW the user likes to interact, not WHAT they know.

```text
 Procedural memory bank:
 +------------------------------------------------------------+
 | Pattern                           | Encoded As             |
 +------------------------------------------------------------+
 | User prefers concise answers      | style_vector_01        |
 +------------------------------------------------------------+
 | User always asks follow-up        | interaction_pattern_01 |
 | questions after code examples     |                        |
 +------------------------------------------------------------+
 | User prefers bullet points        | style_vector_02        |
 | over long paragraphs              |                        |
 +------------------------------------------------------------+
 | User responds well to analogies   | style_vector_03        |
 +------------------------------------------------------------+
 | User usually works late at night  | temporal_pattern_01    |
 | (don't suggest morning routines)  |                        |
 +------------------------------------------------------------+

 Properties:
 - Not conscious facts — behavioral patterns
 - Emerge gradually from many interactions (slow learning)
 - Influence HOW the LLM responds, not WHAT it says
 - Hard to verbalize: "the model just knows how to talk to me"
```

**Implementation:** A separate set of style/behavior embedding vectors
in the memory bank. These are updated with the slowest learning rate.
They modulate the LLM's generation style — tone, verbosity, structure —
rather than injecting factual content.

**FamilyOS equivalent: K0 P03 R4 Granger causality edges.** Already built.
P03 phase R4 also learns Granger causality — if event A consistently
precedes event B across episodes, a causal edge is created. This captures
"when the user asks X, they always follow up with Y" patterns. The memory
model trains sequence/style vectors from these causal patterns.

---

#### Layer 7: Emotional Memory

**In humans:** Learned emotional associations. The amygdala tags experiences
with emotional valence. You feel anxious in a dentist's office before
anything happens. You feel warm hearing a particular song.

**In the memory model:** This is the **sentiment and preference memory** —
tracking emotional valence associated with topics, actions, and suggestions.

```text
 Emotional memory bank:
 +-----------------------+-----------+----------------------------+
 | Topic / Trigger       | Valence   | Learned From               |
 +-----------------------+-----------+----------------------------+
 | career advice         | positive  | user engaged enthusiastically|
 +-----------------------+-----------+----------------------------+
 | diet suggestions      | negative  | user got annoyed last time |
 +-----------------------+-----------+----------------------------+
 | coding challenges     | very pos. | user spent hours happily   |
 +-----------------------+-----------+----------------------------+
 | scheduling/planning   | neutral   | user tolerates but doesn't |
 |                       |           | seek it out                |
 +-----------------------+-----------+----------------------------+
 | financial topics      | sensitive | user changed subject twice |
 +-----------------------+-----------+----------------------------+

 Properties:
 - Valence is a continuous score, not binary
 - Built from implicit signals (engagement, disengagement, tone)
 - Modulates response framing: approach positive topics eagerly,
   handle negative ones carefully, avoid or tread lightly on sensitive
 - Works alongside procedural memory to shape interaction style
```

**Implementation:** Valence-tagged embedding vectors. The memory encoder
learns to associate topics with emotional signals from user behavior
(response length, explicit feedback, topic avoidance patterns).
These vectors bias the LLM's generation toward appropriate emotional tone.

**FamilyOS equivalent: K0 P03 R1 importance_scorer (emotion-weighted) +
UltraBERT emotion classification (44 classes).** Already built. P03
phase R1 scores every event's importance using emotional weights as a
key factor. UltraBERT extracts 44 emotion classes + 5 sentiment levels
in real-time. st_hipp_events stores sentiment_score, sentiment_label,
dominant_emotions_json per episode. The emotional signal pipeline
exists — the memory model trains valence vectors from this data.

---

#### Layer 8: Prospective Memory

**In humans:** Remembering to do something in the future. "I need to buy
milk on the way home." "Remind me to call Mom on Sunday." This is
remembering to remember — triggered by time or context cues.

**In the memory model:** This is the **intent and pending-task memory** —
tracking things the user mentioned they want to do, follow up on, or be
reminded about.

```text
 Prospective memory bank:
 +----+----------------------------+-------------+----------------+
 | ID | Intent                     | Trigger     | Status         |
 +----+----------------------------+-------------+----------------+
 | p1 | Follow up on Tokyo trip    | user mentions| pending       |
 |    | hotel booking              | travel       |               |
 +----+----------------------------+-------------+----------------+
 | p2 | User wanted to learn Rust  | user mentions| pending       |
 |    | "someday"                  | programming  |               |
 +----+----------------------------+-------------+----------------+
 | p3 | User said "remind me to   | next Monday  | pending       |
 |    | renew my subscription"     |              |               |
 +----+----------------------------+-------------+----------------+
 | p4 | User was debugging a       | user opens   | pending       |
 |    | segfault in C++ project    | code topic   |               |
 +----+----------------------------+-------------+----------------+

 Properties:
 - Triggered by time cues or context cues
 - Once triggered and acted on, marked as completed
 - Gives the model PROACTIVE behavior: "By the way, did you
   ever book that Tokyo hotel?"
 - This is what makes the model feel like it genuinely cares
```

**Implementation:** A structured memory slot type with trigger conditions.
The memory encoder checks incoming context against pending triggers.
When a match fires, the corresponding intent vector gets high activation
in the memory bank, causing the LLM to surface it in generation.

**FamilyOS equivalent: K0 P03 R5 observation-driven memory
strengthening (14 algorithms replacing dead CPN/MCTS).**
CRITICAL INSIGHT: Counterfactual reasoning ("what if I hadn't
switched docks?") is a REASONING task, not a memory consolidation
task. It belongs at query time in K1's Concierge LLM with rich
recalled context — not in K0's offline batch pipeline. The original
R5 algorithms (CPN counterfactuals, TPN-MCTS forward simulation)
are dead on arrival: CPN produces semantically nonsensical text
("If Tokyo had also caused issues during Routine with Panda..."),
and R4 produces 0 causal edges that CPN requires.

R5 is being redesigned with 14 observation-driven algorithms:
EST (Episodic Strength Tracker), EWR (Edge Weight Refinement),
SPC-UQ (Prospective Cleanup), ASU (Anchor Seeding & Update),
SRE (Social Relationship Enrichment), SPR (Semantic Pattern
Reinforcement), MTP (Memory Tier Promotion), CLV (Cross-Layer
Coherence Verification), CTD (Contradiction Detection),
EPC (Episode Compression), SPG (Salience Propagation through
Graph), NTD (Narrative Thread Detection), plus kept BGT-SM
(Insights) and TDL-HCO (Routine Optimization).

All algorithms are st_observations-first: they query the
append-only observation log (32 columns across 8 context
categories) as their primary evidence trail, not heuristics.

For the memory model: L8 prospective memory is about intent
tracking and proactive triggers — NOT about generating
counterfactuals. The Concierge's ProactiveAgent stub (fires
during COMPANIONING + 5s) handles runtime trigger detection.
The memory model trains forward-prediction vectors from
SPC-UQ cleaned prospective items + task_state pending items.

---

### How the 8 Layers Map to the Memory Model Architecture

```text
 COMPLETE  MEMORY  MODEL  ARCHITECTURE  (8-LAYER  DESIGN)
 =========================================================

 INPUTS (raw)                              OUTPUTS (to LLM)
 ===========                               =================
                                           memory vectors
 user message ----+                        [m1, m2, ... mK]
                  |                              ^
                  v                              |
 +================+============+     +-----------+-----------+
 | L1: SENSORY                 |     | MEMORY VECTOR         |
 |   salience filter           |     | COMPOSER              |
 |   extract signals from raw  |     |                       |
 |   input, discard noise      |     | Weights & combines    |
 +==============+==============+     | vectors from all      |
                |                    | layers based on       |
                v                    | current query         |
 +==============+==============+     | relevance             |
 | L2: SHORT-TERM              |     +-----------+-----------+
 |   session buffer             |                ^
 |   last N turns              |                |
 +==============+==============+     +----------+----------+
                |                    |                     |
                v                    |                     |
 +==============+==============+     |                     |
 | L3: WORKING MEMORY          +-----+                     |
 |   query-time attention       |                          |
 |   over all memory layers    |                          |
 |   resolve conflicts         |                          |
 |   select relevant memories  |                          |
 +==============+==============+                          |
                |                                          |
      consolidation                                       |
                |                                          |
                v                                          |
 +=============================================================+
 |                    PERSISTENT  MEMORY  BANK                  |
 |=============================================================|
 |                                                              |
 |  L4: EPISODIC    L5: SEMANTIC   L6: PROCEDURAL              |
 |  [ep_vectors]    [fact_vectors] [style_vectors]              |
 |  time-stamped    stable facts   behavior patterns            |
 |  experiences     confidence-    slow-learning                |
 |  recency-        weighted       implicit                     |
 |  weighted                                                    |
 |                                                              |
 |  L7: EMOTIONAL               L8: PROSPECTIVE                |
 |  [valence_vectors]           [intent_vectors]                |
 |  topic sentiment             future triggers                 |
 |  approach/avoid              pending tasks                   |
 |  signals                     proactive recall                |
 |                                                              |
 +=============================================================+
```

### Layer Interaction Map

The layers are not isolated silos. They interact constantly:

```text
 LAYER  INTERACTIONS
 ===================

 L1 Sensory ---------> L2 Short-Term       (attention selects what enters)
 L2 Short-Term ------> L3 Working Memory   (active reasoning draws from STM)
 L3 Working Memory --> queries all of L4-L8 (retrieves relevant long-term info)

 L4 Episodic --------> L5 Semantic         (repeated episodes consolidate
                                             into stable facts)
 L5 Semantic --------> L4 Episodic         (semantic knowledge helps
                                             interpret new episodes)

 L6 Procedural <-----> L7 Emotional        (habits form around what feels
                                             good; emotions shape behaviors)

 L8 Prospective <----> L4 Episodic         (intents are born from episodes;
                                             completed intents become episodes)

 L7 Emotional -------> L3 Working Memory   (emotional valence biases which
                                             memories get high activation)

 Consolidation flow (after each session):
 +-------+     +--------+     +----------+     +----------+
 | L2    | --> | L4     | --> | L5       | --> | L6       |
 | short | --> | episodic| --> | semantic | --> | procedural|
 | term  |     | (event) |    | (if      |     | (if      |
 |       |     |         |    |  repeated)|    |  pattern) |
 +-------+     +--------+     +----------+     +----------+
```

### Learning Rates Per Layer

Each layer trains at a different speed, matching the human timescales:

```text
 +-------------------+------------------+---------------------------+
 | Layer             | Learning Rate    | Update Frequency          |
 +-------------------+------------------+---------------------------+
 | L1 Sensory        | N/A (no params)  | runs every turn           |
 +-------------------+------------------+---------------------------+
 | L2 Short-Term     | N/A (buffer)     | every turn (ring buffer)  |
 +-------------------+------------------+---------------------------+
 | L3 Working Memory | N/A (inference)  | every forward pass        |
 +-------------------+------------------+---------------------------+
 | L4 Episodic       | HIGH  (1e-3)     | every session end         |
 +-------------------+------------------+---------------------------+
 | L5 Semantic       | LOW   (1e-5)     | every N sessions          |
 +-------------------+------------------+---------------------------+
 | L6 Procedural     | VERY LOW (1e-6)  | every N sessions          |
 +-------------------+------------------+---------------------------+
 | L7 Emotional      | MEDIUM (1e-4)    | every session (implicit)  |
 +-------------------+------------------+---------------------------+
 | L8 Prospective    | HIGH  (1e-3)     | every turn (trigger check)|
 +-------------------+------------------+---------------------------+

 Key insight: layers 1-3 are STATELESS (no trained parameters).
 They are pure computation that runs at inference/session time.
 Layers 4-8 are STATEFUL (trained parameters in the checkpoint).
 These are what make the memory model personal and persistent.
```

### The Consolidation Pipeline

After each session, a consolidation process decides what moves from
short-term to long-term memory:

```text
 POST-SESSION  CONSOLIDATION  PIPELINE
 ======================================

 Session ends
      |
      v
 +----+----+
 | Extract |  pull salient events from L2 short-term buffer
 | episodes|
 +----+----+
      |
      v
 +----+----+
 | Store   |  encode each event as a vector in L4 episodic
 | in L4   |  with timestamp and context tags
 +----+----+
      |
      v
 +----+----+
 | Check   |  does this episode reinforce existing semantic facts?
 | L5      |  if yes: increase confidence on matching L5 slots
 | overlap |  if contradicts: update or overwrite L5 slot
 +----+----+
      |
      v
 +----+----+
 | Check   |  do interaction patterns (L6) need updating?
 | L6      |  did the user show new style preferences?
 | patterns|  update slowly (very low LR)
 +----+----+
      |
      v
 +----+----+
 | Update  |  did the user express positive/negative reactions?
 | L7      |  update emotional valence for relevant topics
 | valence |
 +----+----+
      |
      v
 +----+----+
 | Scan    |  did the user mention future intentions?
 | L8      |  create or update prospective memory triggers
 | intents |
 +----+----+
      |
      v
 +----+----+
 | Save    |  write updated checkpoint
 | checkpoint|  (all layers L4-L8 in one file)
 +----+----+
```

#### Boundary Rules: Consolidation vs Reasoning

The P03 Stage 5 corrections document establishes a foundational
principle that constrains the memory model's training scope:

```text
 THE REASONING vs CONSOLIDATION BOUNDARY
 ========================================

 CONSOLIDATION (K0 P03, offline, batch)     REASONING (K1 LLM, online, per-query)
 ======================================     ======================================

 Build richest possible context.            Reason OVER that context at query time.

 DOES:                                      DOES:
 - Score importance (R1)                    - Counterfactual reasoning:
 - Cluster episodes (R2)                      "What if I hadn't switched docks?"
 - Decay old memories (R3)                  - Future simulation:
 - Build knowledge graph (R4)                 "What would happen if I..."
 - STRENGTHEN existing memories (R5)        - Creative synthesis:
 - Track observation evidence                 "Connect these two ideas..."
 - Promote memory tiers (MTP)               - Causal inference from rich context:
 - Detect narrative threads (NTD)             "The dock -> flicker -> patience"
 - Propagate salience (SPG)                 - Personalized advice:
 - Compress episodes (EPC)                    "Given your history with X..."

 DOES NOT:                                  DOES NOT:
 - Generate counterfactuals (dead: CPN)     - Consolidate memories
 - Simulate futures (dead: MCTS)            - Update long-term storage
 - Create speculative text                  - Run offline batch processing
 - Make creative leaps                      - Manage observation evidence
 - Reason about hypotheticals               - Apply decay curves

 WHY: Algorithmic perturbation of graph     WHY: The LLM with rich recalled context
 nodes produces mechanistic gibberish,      (92 monitor mentions, emotional arc,
 not meaningful life reflection. CPN        causal chain, explicit statements)
 output: "If Tokyo had also caused          produces infinitely more meaningful
 issues during Routine with Panda..."       "what-if" reasoning than any offline
 (semantically nonsensical). R4 produces    graph perturbation algorithm.
 0 causal edges that CPN requires.
```

**What this means for the memory model:**

1. The memory model trains on CONSOLIDATED DATA (R0-R8 output),
   not speculative data. No counterfactual training examples.
2. L8 prospective memory is about intent tracking (reminders,
   pending tasks, follow-ups) — NOT about imagining futures.
3. "What if" scenarios are the LLM's strength at query time,
   using the memory model's rich recalled context as fuel.
4. R5's 14 observation-driven algorithms provide BETTER training
   data than the dead CPN/MCTS ever could: correctly-scored
   salience, properly-enriched social relationships, refined
   semantic confidence, seeded Bayesian anchors, linked
   narrative threads.

**R5 boundary rule:** R5 algorithms MUST NOT apply decay (R3
owns that), reconciliation (R3 owns that), or edge creation
(R4 owns that). R5 strictly adds reinforcement signals and
quality corrections. This prevents double-counting.

---

### Context Dimensions: Why Raw Facts Are Useless

A memory without context is a lie waiting to happen.

If the user says **"I went to Berlin last week"** and the system stores
`user_location = Berlin`, it has made a catastrophic error. The user
VISITED Berlin. They did not MOVE there. The difference is entirely
in the temporal, spatial, and narrative context around the raw fact.

Every piece of information entering the memory model must be wrapped
in a rich context envelope. Without it, the model will:

- Confuse visits with relocations
- Treat past events as current states
- Mix up what the user DID with what they WANT TO DO
- Lose who said what in a reported conversation
- Flatten complex life narratives into wrong snapshots

```text
 THE PROBLEM: DUMB MEMORY

 User says: "I went to Berlin last week"

 WRONG (no context):
 +------------------------------------------+
 | fact: location = Berlin                  |
 | That's it. No other metadata.            |
 |                                          |
 | Later, model says: "Since you live in    |
 | Berlin, here are local restaurants..."   |
 |                                          |
 | USER: "I don't live in Berlin! I was     |
 |        just visiting!"                   |
 +------------------------------------------+

 RIGHT (full context envelope):
 +------------------------------------------+
 | fact: Berlin                             |
 | temporal: past, completed, "last week"   |
 | spatial: destination (not home)          |
 | narrative: travel/visit event            |
 | causal: unknown (trip reason not stated) |
 | social: unknown (who they went with)     |
 | certainty: high (explicit statement)     |
 | state_change: NO (location not updated)  |
 +------------------------------------------+
```

This is not optional decoration. Context dimensions are **structural
requirements** of the memory system. Below are all the context dimensions
every memory entry must carry.

---

#### Context Dimension 1: Temporal Context

Time is the most dangerous dimension to get wrong. Human language is
full of temporal markers that completely change the meaning of a fact.

```text
 TEMPORAL  CONTEXT  TAGS
 ========================

 +-------------+--------------------------------------------+
 | Tag         | Meaning                                    |
 +-------------+--------------------------------------------+
 | tense       | past / present / future / hypothetical     |
 +-------------+--------------------------------------------+
 | aspect      | completed / ongoing / habitual / planned   |
 +-------------+--------------------------------------------+
 | anchor      | absolute date or relative ("last week",    |
 |             | "when I was in college", "next March")      |
 +-------------+--------------------------------------------+
 | duration    | point event / period / indefinite          |
 +-------------+--------------------------------------------+
 | recency     | how long ago (affects relevance decay)     |
 +-------------+--------------------------------------------+
 | validity    | still true now? expired? unknown?          |
 +-------------+--------------------------------------------+

 EXAMPLES:

 "I went to Berlin last week"
   tense=past, aspect=completed, anchor=~7 days ago,
   duration=point/short, validity=EXPIRED (they came back)

 "I'm moving to Berlin next month"
   tense=future, aspect=planned, anchor=~30 days from now,
   duration=indefinite (permanent move), validity=PENDING

 "I live in Berlin"
   tense=present, aspect=ongoing, anchor=now,
   duration=indefinite, validity=CURRENT

 "I used to live in Berlin"
   tense=past, aspect=habitual-ended, anchor=unknown past,
   duration=period, validity=EXPIRED

 "I might visit Berlin sometime"
   tense=future, aspect=hypothetical, anchor=unspecified,
   duration=unknown, validity=HYPOTHETICAL

 ALL FIVE sentences mention Berlin.
 ALL FIVE mean completely different things.
 Without temporal context, you CANNOT distinguish them.
```

**Who does this?** This MUST happen in the memory model, not the LLM.
The LLM understands tense in free text, but once information is compressed
into memory vectors, the temporal tags must already be embedded. The
salience filter (L1) extracts temporal markers. The consolidation pipeline
uses them to decide what updates a fact vs. what is just an event.

---

#### Context Dimension 2: Spatial Context

Location information has roles: home, work, visited, mentioned,
hypothetical, someone else's location.

```text
 SPATIAL  CONTEXT  TAGS
 =======================

 +------------------+--------------------------------------------+
 | Tag              | Meaning                                    |
 +------------------+--------------------------------------------+
 | location_entity  | the place name (Berlin, office, home)      |
 +------------------+--------------------------------------------+
 | relation_to_user | home / work / visited / passing_through /  |
 |                  | mentioned / someone_else's / hypothetical   |
 +------------------+--------------------------------------------+
 | scope            | city / country / building / region / vague  |
 +------------------+--------------------------------------------+
 | permanence       | permanent / temporary / transient / unknown |
 +------------------+--------------------------------------------+
 | current_status   | user_is_there_now / user_was_there /        |
 |                  | user_will_go / user_talked_about_it         |
 +------------------+--------------------------------------------+

 EXAMPLES:

 "I live in Berlin"
   entity=Berlin, relation=home, permanence=permanent,
   current_status=user_is_there_now

 "I went to Berlin last week"
   entity=Berlin, relation=visited, permanence=transient,
   current_status=user_was_there (not anymore)

 "My sister lives in Berlin"
   entity=Berlin, relation=someone_else's,
   current_status=user_talked_about_it (NOT user's location)

 "I'm thinking of moving to Berlin"
   entity=Berlin, relation=hypothetical,
   current_status=user_talked_about_it, permanence=unknown
```

---

#### Context Dimension 3: Narrative Context

Every statement exists within a story frame. Is the user:

- Reporting something that happened to them?
- Telling a story about someone else?
- Speculating or hypothesizing?
- Quoting or paraphrasing someone?
- Giving instructions vs. describing reality?

```text
 NARRATIVE  CONTEXT  TAGS
 =========================

 +-------------------+-------------------------------------------+
 | Tag               | Meaning                                   |
 +-------------------+-------------------------------------------+
 | speaker           | self / other_person / hypothetical_self /  |
 |                   | quoting_someone                           |
 +-------------------+-------------------------------------------+
 | frame             | personal_experience / reported_event /     |
 |                   | opinion / instruction / question /         |
 |                   | hypothetical / joke / sarcasm              |
 +-------------------+-------------------------------------------+
 | factuality        | factual / uncertain / speculative /        |
 |                   | counterfactual / aspirational              |
 +-------------------+-------------------------------------------+
 | commitment_level  | definite / likely / considering /          |
 |                   | wishful / rejected                        |
 +-------------------+-------------------------------------------+

 EXAMPLES:

 "I'm vegetarian"
   speaker=self, frame=personal_experience,
   factuality=factual, commitment=definite
   -> STORE as high-confidence semantic fact

 "I was thinking about going vegetarian"
   speaker=self, frame=personal_experience,
   factuality=speculative, commitment=considering
   -> DO NOT store as fact. Store as prospective/exploratory.

 "My friend says keto is amazing"
   speaker=other_person, frame=reported_event,
   factuality=uncertain (hearsay), commitment=N/A
   -> Store under social context, NOT as user preference

 "I'd never eat sushi" (said sarcastically after eating sushi)
   speaker=self, frame=joke/sarcasm,
   factuality=counterfactual, commitment=rejected
   -> INVERT meaning: user actually LIKES sushi
```

---

#### Context Dimension 4: Causal Context

Why did something happen? What caused it? What was the consequence?
Without causal links, the model can't reason about the user's life.

```text
 CAUSAL  CONTEXT  TAGS
 ======================

 +------------------+--------------------------------------------+
 | Tag              | Meaning                                    |
 +------------------+--------------------------------------------+
 | cause            | what triggered this event/statement         |
 +------------------+--------------------------------------------+
 | effect           | what resulted from it                      |
 +------------------+--------------------------------------------+
 | motivation       | why the user did/wants this                |
 +------------------+--------------------------------------------+
 | constraint       | external forces shaping the situation      |
 +------------------+--------------------------------------------+

 EXAMPLES:

 "I switched to Python because my team uses it"
   cause=team requirement, motivation=collaboration,
   constraint=workplace, effect=language_switch
   -> Don't just store "uses Python." Store WHY.
   -> If user changes jobs, this fact may become invalid.

 "I stopped eating gluten, my doctor told me to"
   cause=medical_advice, motivation=health,
   constraint=medical, effect=diet_change
   -> High permanence (medical reasons rarely flip casually)
   -> Different from "I'm trying gluten-free for fun"
```

---

#### Context Dimension 5: Social Context

Who else is involved? Relationships, roles, and social dynamics
around the information.

```text
 SOCIAL  CONTEXT  TAGS
 ======================

 +------------------+--------------------------------------------+
 | Tag              | Meaning                                    |
 +------------------+--------------------------------------------+
 | people_involved  | names, roles, relationships                |
 +------------------+--------------------------------------------+
 | relationship     | family / friend / colleague / stranger /   |
 |                  | professional / romantic                    |
 +------------------+--------------------------------------------+
 | social_role      | who is the user in this context?           |
 |                  | parent / employee / student / customer     |
 +------------------+--------------------------------------------+
 | shared_context   | is this info about the user alone or       |
 |                  | about a shared situation?                  |
 +------------------+--------------------------------------------+

 EXAMPLES:

 "My daughter just started college"
   people=daughter, relationship=family(child),
   social_role=parent, shared_context=family_milestone
   -> Store in episodic with family tag
   -> Enables: "How's your daughter doing at college?"

 "My boss wants me to learn Kubernetes"
   people=boss, relationship=professional,
   social_role=employee, shared_context=work
   -> Different from self-motivated learning
   -> Constraint: may be mandatory, not a preference
```

---

#### Context Dimension 6: Emotional Context

What was the user's emotional state or attitude when they said this?
This goes beyond Layer 7 (Emotional Memory) — it's about tagging
EACH INDIVIDUAL MEMORY with its emotional coloring.

```text
 EMOTIONAL  CONTEXT  TAGS
 =========================

 +------------------+--------------------------------------------+
 | Tag              | Meaning                                    |
 +------------------+--------------------------------------------+
 | sentiment        | positive / negative / neutral / mixed      |
 +------------------+--------------------------------------------+
 | intensity        | mild / moderate / strong / extreme         |
 +------------------+--------------------------------------------+
 | emotion_type     | joy / frustration / anxiety / excitement / |
 |                  | nostalgia / anger / relief / pride / etc.  |
 +------------------+--------------------------------------------+
 | directed_at      | self / topic / other_person / situation    |
 +------------------+--------------------------------------------+

 EXAMPLES:

 "I finally got promoted!"
   sentiment=positive, intensity=strong,
   emotion=pride+relief, directed_at=self
   -> High salience event, mark as milestone

 "I went to Berlin last week"
   sentiment=neutral (no strong signal)
   -> But if followed by "it was amazing!":
   sentiment=positive, emotion=joy, directed_at=Berlin_trip
   -> Now the Berlin trip has emotional weight

 "ugh, another meeting about nothing"
   sentiment=negative, intensity=moderate,
   emotion=frustration, directed_at=work_meetings
   -> Updates L7 emotional valence for "meetings" topic
```

---

#### Context Dimension 7: Epistemic Context

How certain is this information? Where did it come from? Can it be trusted?

```text
 EPISTEMIC  CONTEXT  TAGS
 =========================

 +------------------+--------------------------------------------+
 | Tag              | Meaning                                    |
 +------------------+--------------------------------------------+
 | source           | user_stated / user_implied / inferred /    |
 |                  | third_party / observed_behavior            |
 +------------------+--------------------------------------------+
 | confidence       | 0.0 to 1.0 (how sure are we?)             |
 +------------------+--------------------------------------------+
 | evidence_count   | how many times confirmed/reinforced        |
 +------------------+--------------------------------------------+
 | contradicted_by  | list of conflicting evidence if any        |
 +------------------+--------------------------------------------+
 | verifiable       | can the model check this? or trust blindly?|
 +------------------+--------------------------------------------+

 EXAMPLES:

 "I'm allergic to peanuts" (user stated explicitly)
   source=user_stated, confidence=0.99,
   evidence_count=1, contradicted_by=none
   -> Critical fact. One statement is enough. High certainty.

 User always asks about Python, never mentions Java
   source=observed_behavior, confidence=0.7,
   evidence_count=15 sessions, contradicted_by=none
   -> Inferred preference. Medium certainty. Could be wrong.

 "I think I prefer dark mode" (hedged language)
   source=user_stated, confidence=0.6 (hedged),
   evidence_count=1, contradicted_by=none
   -> User isn't sure themselves. Store but don't bet on it.
```

---

#### Context Dimension 8: Identity Context

Which aspect of the user's identity does this belong to? People are
multi-faceted — their work self, home self, creative self, etc.

```text
 IDENTITY  CONTEXT  TAGS
 ========================

 +------------------+--------------------------------------------+
 | Tag              | Meaning                                    |
 +------------------+--------------------------------------------+
 | identity_facet   | professional / personal / family /         |
 |                  | creative / health / financial / social     |
 +------------------+--------------------------------------------+
 | context_switch   | does this info apply only in certain       |
 |                  | contexts? (work-only, home-only, etc.)     |
 +------------------+--------------------------------------------+
 | self_concept     | how the user sees themselves in this area  |
 +------------------+--------------------------------------------+

 EXAMPLES:

 "At work I use Windows, at home I use Linux"
   -> TWO separate entries with different identity facets
   -> If user asks about OS while discussing work: Windows
   -> If user asks about OS while discussing hobby: Linux

 "I'm a senior engineer but I'm learning piano"
   -> professional: senior_engineer (high confidence, stable)
   -> creative: piano_beginner (new, evolving)
   -> Different response styles for each context
```

---

#### Context Dimension 9: Relational/Comparative Context

How does this fact relate to OTHER facts already in memory?
Does it replace, extend, contradict, or qualify something?

```text
 RELATIONAL  CONTEXT  TAGS
 ==========================

 +------------------+--------------------------------------------+
 | Tag              | Meaning                                    |
 +------------------+--------------------------------------------+
 | relation_type    | updates / contradicts / extends /          |
 |                  | qualifies / independent                    |
 +------------------+--------------------------------------------+
 | target_memory    | which existing memory does this relate to? |
 +------------------+--------------------------------------------+
 | resolution       | replace / merge / keep_both / flag_conflict|
 +------------------+--------------------------------------------+

 EXAMPLES:

 Existing memory: "user lives in NYC" (confidence 0.95)
 New statement: "I moved to London last month"
   relation=contradicts target="lives in NYC"
   resolution=REPLACE (explicit relocation statement)
   -> Update L5 semantic: location = London
   -> Move old fact to L4 episodic: "used to live in NYC"

 Existing memory: "user likes Italian food"
 New statement: "I also love Thai food"
   relation=extends target="likes Italian food"
   resolution=MERGE (add, don't replace)

 Existing memory: "user prefers Python"
 New statement: "Python is slow for this project"
   relation=qualifies target="prefers Python"
   resolution=KEEP_BOTH (preference still exists,
   but now has a known limitation context)
```

---

### The Complete Context Envelope

Every memory entry — whether it goes to L4 Episodic, L5 Semantic,
L6 Procedural, L7 Emotional, or L8 Prospective — gets wrapped in
a context envelope before storage:

```text
 MEMORY  ENTRY  WITH  CONTEXT  ENVELOPE
 =======================================

 +================================================================+
 | MEMORY ENTRY                                                    |
 |================================================================|
 |                                                                 |
 | raw_content: "I went to Berlin last week"                       |
 |                                                                 |
 | +------------------------------------------------------------+ |
 | | CONTEXT ENVELOPE                                            | |
 | |------------------------------------------------------------| |
 | | temporal:                                                   | |
 | |   tense=past, aspect=completed, anchor=2026-03-14,         | |
 | |   duration=short_trip, validity=EXPIRED                     | |
 | |                                                             | |
 | | spatial:                                                    | |
 | |   entity=Berlin, relation=visited, permanence=transient,   | |
 | |   current_status=was_there_not_anymore                      | |
 | |                                                             | |
 | | narrative:                                                  | |
 | |   speaker=self, frame=personal_experience,                  | |
 | |   factuality=factual, commitment=definite                   | |
 | |                                                             | |
 | | causal: cause=unknown, motivation=unknown                   | |
 | |                                                             | |
 | | social: people=unknown, shared_context=unknown              | |
 | |                                                             | |
 | | emotional: sentiment=neutral (no signal yet)                | |
 | |                                                             | |
 | | epistemic:                                                  | |
 | |   source=user_stated, confidence=0.95,                      | |
 | |   evidence_count=1                                          | |
 | |                                                             | |
 | | identity: facet=personal, context_switch=none               | |
 | |                                                             | |
 | | relational: relation=independent (no conflict found)        | |
 | +------------------------------------------------------------+ |
 |                                                                 |
 | storage_target: L4_episodic (event, not a standing fact)        |
 | state_change_triggered: NONE (location NOT updated)             |
 |                                                                 |
 +================================================================+
```

### Who Builds the Context Envelope: LLM or Memory Model?

This is a critical design fork:

```text
 OPTION A: The LLM builds context (at extraction time)
 =====================================================
 - Use the LLM to parse the user's message and extract
   structured context tags before passing to the memory model.
 - Pro: LLMs are great at understanding language nuance,
   tense, sarcasm, implied meaning.
 - Con: requires an LLM call for every memory extraction.
   Adds latency and cost.

 OPTION B: The memory model learns context itself
 ================================================
 - The memory encoder is trained to infer context dimensions
   directly from raw text.
 - Pro: no extra LLM call. Self-contained. Fast.
 - Con: harder to train. Small model may miss subtle cues
   like sarcasm or implied temporal markers.

 OPTION C: Hybrid (RECOMMENDED)
 ==============================
 - Use a LIGHTWEIGHT context tagger (small classifier head
   on top of the salience filter in L1) for the obvious cases:
   explicit tense markers, named entities, sentiment keywords.
 - Fall back to the LLM for AMBIGUOUS cases only:
   sarcasm detection, complex temporal reasoning,
   implicit causal chains.
 - This keeps latency low for 80% of inputs and only
   invokes the LLM for the hard 20%.

 +-------------------+     +--------------------+
 | User message      |     |                    |
 +--------+----------+     |  LLM (only for     |
          |                |  ambiguous cases)   |
          v                |                    |
 +--------+----------+     +--------+-----------+
 | L1: Salience      |              ^
 | Filter + Light    |              |
 | Context Tagger    +--> ambiguous?+
 |                   |     yes: ask LLM
 | handles:          |     no: use local tags
 | - explicit tense  |
 | - named entities  +---> clear context
 | - sentiment words |     envelope
 | - location roles  |
 +-------------------+
```

**FamilyOS resolves this fork: Option C is already implemented.**

The lightweight tagger IS UltraBERT (22ms, 12 classification heads):

```text
 CONTEXT  DIMENSION  BUILDER  (FamilyOS  MAPPING)
 =================================================

 DIMENSION        LIGHTWEIGHT TAGGER          LLM FALLBACK
                  (UltraBERT + Phase 1)       (Front LLM cognitive tools)
 ---------        -------------------         --------------------------
 TEMPORAL         NER head extracts           update_beliefs() infers
                  temporal entities            complex tense reasoning

 EMOTIONAL        44 emotion classes +        refine_affect() overrides
                  5 sentiment levels           when sarcasm/irony/masked
                  (real-time, every turn)      affect detected

 SOCIAL           NER extracts person          update_beliefs() creates
                  entities (GlobalPointer)     S-P-O triples with roles

 IDENTITY         Intent classification       update_scoreboard() tracks
                  maps to speaker roles        referents and QUD context
                  (8 intent classes)           per speaker

 NARRATIVE         ---                        update_narrative() tracks
                                              thread switches, resumes,
                                              closes (expensive, LLM-only)

 CAUSAL            ---                        update_beliefs() infers
                                              cause-effect from S-P-O
                                              triples + confidence

 EPISTEMIC        Safety band assessment      update_clarifications()
                  (4 bands: GREEN/AMBER/      tracks semantic gaps:
                  RED/BLACK)                  field, question, severity

 SPATIAL          NER extracts location        (mostly covered by NER,
                  entities                     LLM fallback for complex
                                               permanence inference)

 RELATIONAL        ---                        update_beliefs() handles
                                              S-P-O conflict resolution
                                              (contradicts/extends)

 COVERAGE:  6 of 9 dimensions handled by UltraBERT (cheap, 22ms)
            3 of 9 require Front LLM cognitive tools (expensive)
            This matches the 80/20 hybrid prediction almost exactly.
```

**CRITICAL UPDATE: Memory Writer v2 validates this architecture.**

The P03 corrections analysis revealed that the current Memory Writer
only reads 5 of 15 SessionState sections and produces minimally tagged
text. P02 then RECOMPUTES everything with UltraBERT on 8-token text
fragments — producing garbage signals:

```text
 BROKEN  SIGNAL  CHAIN  (CURRENT  MW  v1)
 =========================================
 social_context: always "friends" (68%) or "solo" (32%)
   -> FamilyOS app with family content never returns "family"
   -> Root: M07 queries empty st_kg_edges, can't distinguish kin

 salience_score: 60% stuck at 0.41, ALL 1348 rows MED band
   -> Zero HIGH, zero LOW. Completely non-discriminating.
   -> Root: broken social_context cascades into salience formula

 has_partner_present / has_parent_present: ALWAYS FALSE
   -> Even for "Christmas Eve dinner with both families"
   -> Root: no kin registry in family_graph_resolve

 UltraBERT runs 3x (K1 Phase 1, P02 M02, P02 M04)
   -> K1's rich signals are DISCARDED at the Bridge boundary
   -> P02 re-derives affect from 8 tokens with no context
```

**MW v2 redesign: MW becomes the primary intelligence.** It reads
all 15 SessionState sections, produces fully-tagged memory atoms
with 34 fields that map directly to our 9 context dimensions:

```text
 MW  v2  FIELD  ->  CONTEXT  DIMENSION  MAPPING
 ================================================

 CONTEXT DIMENSION     MW v2 FIELDS
 -----------------     -----------------------------------------
 TEMPORAL              temporal.mentioned_time,
                       temporal.resolved_epoch_ms,
                       temporal.is_backdated,
                       temporal_orientation (PAST/ONGOING/FUTURE)

 SPATIAL               location_name, location_type

 NARRATIVE             narrative.thread_id (system UUID),
                       narrative.arc_position (EXPOSITION/
                       RISING_ACTION/CLIMAX/RESOLUTION),
                       narrative.is_goal_event, goal_context

 CAUSAL                goal_context (from narrative_active.goal),
                       task_context (from task_state)

 SOCIAL                participants[], participant_relationships
                       {type, target, confidence},
                       social_context, social_intimacy

 EMOTIONAL             affect{valence, arousal, dominance},
                       emotion_tags[], sentiment_label
                       (PER-EXTRACTION, not session-level!)

 EPISTEMIC             source_type (user_stated/user_implied/
                       device_observed/system_inferred),
                       confidence, novelty (ROUTINE/EXPECTED/
                       NOVEL/SURPRISING)

 IDENTITY              identity_domains[] (parent, child,
                       professional, health_self, etc.),
                       elaboration_depth (MENTION/DISCUSSED/
                       ELABORATED/DEEPLY_PROCESSED)

 RELATIONAL            participant_relationships.type
                       (PARENT_OF/CHILD_OF/SPOUSE_OF/etc.),
                       entity_salience per participant
```

**Per-extraction affect is the key insight.** A single user turn
might produce 2 memory atoms with opposite emotional signatures:
"Loved Rohan's engagement" (valence: +0.85) AND "Sad seeing
Jessica struggling" (valence: -0.40). Session-level affect would
average to ~0.25, making BOTH atoms wrong. MW v2's LLM detects
per-atom dimensions using session affective_now as CONTEXT, not
as the output value.

**What this means for the memory model:** The memory model trains
on CORRECTLY TAGGED memory atoms (from MW v2 through P02 to P03).
Every context dimension is populated by the MW's LLM which has
full conversation context — not by UltraBERT on 8-token fragments.
The quality of memory model training is gated by the quality of
the context envelope, and MW v2 fixes the quality at the source.

### How Context Dimensions Interact With Memory Layers

Each memory layer cares about different context dimensions most:

```text
 +-------------------+----------------------------------------------+
 | Memory Layer      | Primary Context Dimensions                   |
 +-------------------+----------------------------------------------+
 | L4 Episodic       | TEMPORAL (when), NARRATIVE (what happened),  |
 |                   | SOCIAL (who was involved), EMOTIONAL (felt)  |
 +-------------------+----------------------------------------------+
 | L5 Semantic       | EPISTEMIC (how certain), RELATIONAL (does    |
 |                   | this update/contradict existing facts),       |
 |                   | TEMPORAL (is this still valid?)               |
 +-------------------+----------------------------------------------+
 | L6 Procedural     | IDENTITY (which context/role), CAUSAL (why   |
 |                   | this pattern), EPISTEMIC (how reliable)       |
 +-------------------+----------------------------------------------+
 | L7 Emotional      | EMOTIONAL (sentiment + intensity),            |
 |                   | NARRATIVE (was this sarcasm? genuine?),       |
 |                   | SOCIAL (directed at whom?)                    |
 +-------------------+----------------------------------------------+
 | L8 Prospective    | TEMPORAL (when to trigger), CAUSAL (why      |
 |                   | user wants this), EPISTEMIC (how committed?)  |
 +-------------------+----------------------------------------------+
```

### The Deadly Mistakes Without Context

To drive the point home — here are failures that WILL happen without
proper context tagging:

```text
 FAILURE  CASES  WITHOUT  CONTEXT  DIMENSIONS
 =============================================

 1. TEMPORAL FAILURE
    User: "I was a smoker for 10 years"
    Dumb model: tags user as smoker
    Smart model: tense=past, aspect=completed, validity=EXPIRED
    -> User QUIT smoking. The opposite conclusion.

 2. NARRATIVE FAILURE
    User: "My friend loves skydiving"
    Dumb model: tags user as liking skydiving
    Smart model: speaker=other_person, relation=friend
    -> This is about the FRIEND, not the user.

 3. SPATIAL FAILURE
    User: "I'm in Berlin for a conference"
    Dumb model: location = Berlin
    Smart model: relation=visiting, permanence=transient,
    duration=days, cause=conference
    -> User still lives wherever they lived before.

 4. EPISTEMIC FAILURE
    User: "Maybe I should try yoga"
    Dumb model: tags user as yoga practitioner
    Smart model: factuality=speculative, commitment=considering
    -> User hasn't decided yet. Don't recommend yoga mats.

 5. EMOTIONAL FAILURE
    User: "Oh great, another JavaScript framework"
    Dumb model: user is interested in JS frameworks
    Smart model: sentiment=negative (sarcasm), emotion=frustration
    -> User is ANNOYED, not excited.

 6. SOCIAL FAILURE
    User: "We decided to go with React for the project"
    Dumb model: user chose React (personal preference)
    Smart model: social_role=team_member, shared_context=work,
    commitment=group_decision
    -> This was a TEAM decision, not personal preference.
    -> User might actually prefer Vue but went with team.

 7. IDENTITY FAILURE
    User: "I use Java at work, Rust for hobby projects"
    Dumb model: user uses Java and Rust (flat list)
    Smart model: Java -> identity=professional, Rust -> identity=personal
    -> If user asks for help on weekend project: suggest Rust
    -> If user asks for help on work task: suggest Java

 8. CAUSAL FAILURE
    User: "I switched to Linux because Windows kept crashing"
    Dumb model: user prefers Linux
    Smart model: cause=Windows_instability, motivation=reliability
    -> If Windows becomes stable, this preference might reverse.
    -> The preference is CONDITIONAL, not absolute.

 9. RELATIONAL FAILURE
    User (January): "I love sushi"
    User (March): "I've gone vegan"
    Dumb model: user loves sushi AND is vegan (contradiction)
    Smart model: relation=contradicts, resolution=qualify
    -> User USED to love sushi. Now vegan.
    -> Sushi preference may still exist but is constrained.
```

---

This 8-layer architecture answers several of the open design questions:

```text
 Question: "What kind of network is the memory model?"
 Answer:   It is not ONE network. It is a SYSTEM of components:
           - A salience filter (L1)
           - A session buffer (L2)
           - An attention-based memory retriever (L3)
           - A set of specialized memory banks (L4-L8)
           - A consolidation pipeline
           - A memory vector composer

 Question: "How many parameters?"
 Answer:   Layers 1-3 are lightweight (~1M params total).
           Layers 4-8 are the bulk: K vectors x d dimensions x 5 layer types.
           If K=64 slots per layer, d=512, 5 layers: 64 * 512 * 5 = 163,840
           Plus the encoder/projection: maybe 5-10M params.
           Total per user: ~5-15M parameters.

 Question: "What is the memory bank structure?"
 Answer:   5 separate banks (L4-L8), each with different:
           - Learning rates
           - Update frequencies
           - Slot management policies
           - Capacity limits

 Question: "How to avoid catastrophic forgetting?"
 Answer:   Different layers forget at different rates.
           L4 episodic: old events naturally decay (recency weighting).
           L5 semantic: only overwrites on strong contradicting evidence.
           L6 procedural: almost never changes (very low LR).
           L7 emotional: gradual drift, not sudden flips.
           L8 prospective: explicit completion/expiry, not forgetting.
```

---

### FamilyOS Integration: Where the Memory Model Lives

FamilyOS is a two-kernel family AI operating system:

- **K0 (Data Kernel):** PostgreSQL 16+, asyncpg, pgvector, 20 pipelines,
  13 layers. The source of truth for all persistent data.
- **K1 (Cognitive Kernel):** The "brain" — Concierge FSM, Orchestrator,
  Planner, Capability Fabric, SessionState, Sub-agents. Runs on-device.
- **Bridge:** Cross-kernel security gateway. Handles K0↔K1 transport,
  connector security, device sync, IFL ownership, offline awareness.
- **IFL (Interkernel Fabric Language):** Universal connector protocol
  for external services and devices.

The memory model fits into this architecture as follows:

```text
 MEMORY  MODEL  IN  THE  FamilyOS  TWO-KERNEL  ARCHITECTURE
 ===========================================================

 +================================================================+
 |  K1  COGNITIVE  KERNEL  (on-device, edge-first)                |
 |================================================================|
 |                                                                 |
 |  L0: External Interfaces (voice, touch, text)                   |
 |  L0.5: Module Loader                                            |
 |  L1: Concierge FSM (dual-LLM: Front Voice + Back Worker)        |
 |  L1.5: Rhythm Controller                                        |
 |  L2: Orchestrator                                                |
 |  L2.5: Capability Fabric                                        |
 |  L3: Planner                                                    |
 |  L4: Sub-agents                                                 |
 |  L5: SessionState (HOT 52KB + WARM 48KB + LOCAL COLD SQLite)    |
 |                                                                 |
 |  +-----------------------------------------------------------+  |
 |  | NEW: MemoryFusedAdapter                                   |  |
 |  |   wraps: GeminiConciergeAdapter (IConciergeModelPort)     |  |
 |  |   holds: inference-only weights (~20-60MB per user)       |  |
 |  |   does:  query memory model -> get vectors -> inject      |  |
 |  |           into attention before LLM forward pass          |  |
 |  +-----------------------------------------------------------+  |
 |                                                                 |
 |  L6: K0 Storage Interface (queries via Bridge)                  |
 |                                                                 |
 +==============================+==================================+
                                |
                          Bridge (transport)
                                |
 +==============================+==================================+
 |  K0  DATA  KERNEL  (server-side, source of truth)              |
 |================================================================|
 |                                                                 |
 |  PostgreSQL 16+ (pgvector, tsvector FTS)                        |
 |  20 Pipelines (P01-P20)                                         |
 |                                                                 |
 |  P03 Consolidation Pipeline (runs nightly):                     |
 |    R0: Batch selection                                          |
 |    R1: Importance scoring (emotion-weighted)                    |
 |    R2: Episodic integration (HDBSCAN clustering)                |
 |    R3: Dedup + decay (Ebbinghaus per-layer lambda)              |
 |    R4: KG consolidation (NER, Hebbian, Granger causality)       |
 |    R5: Memory strengthening (14 observation-driven algorithms)  |
 |        EST, EWR, ASU, SRE, SPR, MTP, CLV, CTD, EPC, SPG, NTD  |
 |        + kept BGT-SM, TDL-HCO  (NO counterfactuals -- that's   |
 |        the LLM's job at query time, not consolidation's job)    |
 |    R6: Staging                                                  |
 |    R7: Truth writer                                             |
 |    R8: Event emission                                           |
 |                                                                 |
 |  +-----------------------------------------------------------+  |
 |  | NEW: R9 Memory Model Trainer                              |  |
 |  |   reads: all consolidated data from R0-R8                 |  |
 |  |   trains: incremental gradient updates on 5-15M params    |  |
 |  |   outputs: updated checkpoint (~20-60MB)                  |  |
 |  |   syncs: via Bridge to K1 MemoryFusedAdapter              |  |
 |  +-----------------------------------------------------------+  |
 |                                                                 |
 |  Tables already storing context dimensions:                     |
 |    st_hipp_events:                                              |
 |      sentiment_score, sentiment_label,                          |
 |      dominant_emotions_json, ner_entities_json,                 |
 |      temporal_json                                              |
 |                                                                 |
 +================================================================+
```

#### The Adapter Layer: Zero Changes Above

The Concierge POC already abstracts LLM access behind IConciergeModelPort:

```text
 CURRENT  (no memory model):

   FrontHandler / BackHandler
          |
          v
   react_loop()
          |
          v
   IConciergeModelPort.generate(ConciergeModelRequest)
          |
          v
   GeminiConciergeAdapter  --->  Gemini API


 WITH  MEMORY  MODEL:

   FrontHandler / BackHandler
          |
          v
   react_loop()
          |
          v
   IConciergeModelPort.generate(ConciergeModelRequest)
          |
          v
   MemoryFusedAdapter  (decorator)
     1. Extract context from request.messages
     2. Forward-pass through user's memory model
     3. Get memory vectors [m1, m2, ... mK]
     4. Inject vectors into request (prefix / cross-attn / gated)
     5. Forward to wrapped GeminiConciergeAdapter
     6. Return response unchanged
          |
          v
   GeminiConciergeAdapter  --->  Gemini API


 WHAT CHANGES:     MemoryFusedAdapter (new, ~200 lines)
 WHAT STAYS SAME:  FSM, Bus, Front, Back, Tools, Prompt Builder,
                   SessionState, Weave, HITL, Safety Bands,
                   Experience Layer, Capability Fabric, everything
```

The FSM, the event bus, the Front/Back actors, the cognitive tools,
the prompt builder, the safety bands — none of them change. The memory
model is invisible to everything above the adapter layer.

#### What the Memory Model Gives Back to FamilyOS

The integration is not one-directional. The memory model solves real
problems the Concierge has today:

```text
 PROBLEM                          MEMORY MODEL SOLUTION
 =======                          =====================

 1. TOKEN BUDGET CRISIS
    DynamicPromptBuilder has      Memory model compresses ALL
    128K budget. recall_memory()  relevant history into a
    returns maybe 20 snippets.    fixed-size vector set.
    2 years of family history     Zero tokens consumed.
    gets compressed to fit.       Full 128K free for conversation.

 2. AFFECT IS SESSION-LOCAL
    EmotionalProcessor and        L7 emotional memory carries
    AffectiveMirror only see      valence across sessions:
    current session data.         "Last time we discussed mom's
                                  health, user was in crisis
                                  for 3 sessions." No retrieval.

 3. PROACTIVE AGENT IS BLIND
    ProactiveAgent stub (fires    L6 procedural + L8 prospective:
    during COMPANIONING + 5s)     "It's Thursday evening. Last 4
    has no deep pattern knowledge. Thursdays, user asked about
                                  weekend plans around this time."

 4. PERSONA IS STATIC
    persona section (immutable,   Memory model learns personality
    set at session init) is a     nuances over months: communication
    JSON blob that can't evolve.  style, humor patterns, sensitivity.
                                  Dynamic, not a config file.

 5. BACK LLM NEEDS EXPLICIT RECALL
    Back actor's recall_memory()  Memory-fused context means Back
    and discover_capabilities()   implicitly knows: "this family
    require explicit tool calls.  prefers Uber over Lyft" and
                                  "always book Hilton" without
                                  burning tool-call iterations.
```

#### What Already Exists vs What Is New

```text
 +-------------------------------+-------------------+-----------------------+
 | Component                     | FamilyOS Status   | Memory Model Changes  |
 +-------------------------------+-------------------+-----------------------+
 | Sensory buffer (L1)           | UltraBERT Phase 1 | None — reuse as-is    |
 +-------------------------------+-------------------+-----------------------+
 | Short-term memory (L2-L3)     | SessionState      | None — reuse as-is    |
 |                               | HOT/WARM          |                       |
 +-------------------------------+-------------------+-----------------------+
 | Episodic memory (L4)          | K0 P03 R2         | Reuse data, add       |
 |                               | HDBSCAN clusters  | neural encoding layer |
 +-------------------------------+-------------------+-----------------------+
 | Semantic memory (L5)          | K0 P03 R4 KG      | Reuse data, add       |
 |                               | NER + Hebbian     | embedding training    |
 +-------------------------------+-------------------+-----------------------+
 | Procedural memory (L6)        | K0 P03 R4         | Reuse data, add       |
 |                               | Granger causality | sequence learning     |
 +-------------------------------+-------------------+-----------------------+
 | Emotional memory (L7)         | K0 P03 R1         | Reuse scores, add     |
 |                               | importance scorer | emotional embeddings  |
 +-------------------------------+-------------------+-----------------------+
 | Prospective memory (L8)       | K0 P03 R5         | Reuse SPC-UQ cleaned  |
 |                               | SPC-UQ cleanup    | prospective items,    |
 |                               | + ProactiveAgent  | add forward prediction|
 +-------------------------------+-------------------+-----------------------+
 | Context dims: 6 of 9          | UltraBERT heads   | None — reuse as-is    |
 +-------------------------------+-------------------+-----------------------+
 | Context dims: 3 of 9          | Front LLM tools   | None — reuse as-is    |
 +-------------------------------+-------------------+-----------------------+
 | LLM fusion mechanism          | Nothing (RAG only)| NEW: MemoryFusedAdapter|
 +-------------------------------+-------------------+-----------------------+
 | Training loop                 | Nothing           | NEW: P03 R9 trainer   |
 +-------------------------------+-------------------+-----------------------+
 | Weight sync                   | Bridge pattern    | NEW: model weight     |
 |                               | exists            | transport via Bridge  |
 +-------------------------------+-------------------+-----------------------+
```

The actual net-new work is three components:

1. **MemoryFusedAdapter** — decorator around IConciergeModelPort that injects
   memory vectors into the LLM call. Lives in K1. ~200 lines.
2. **P03 R9 Memory Model Trainer** — new phase in K0's nightly consolidation
   pipeline that trains the memory model from R0-R8 output. Lives in K0.
3. **Bridge weight transport** — syncs trained checkpoint (~20-60MB) from
   K0 to K1 after each training run. Extends existing Bridge sync pattern.

Everything else — the data extraction, the context tagging, the episodic
clustering, the KG building, the emotional scoring, the memory
strengthening — is already built and running.

---

## How the Memory Model Drives Counterfactual Reasoning and Advanced Cognition

We established a hard boundary: **consolidation does NOT reason.**
K0 builds the richest possible context; the LLM reasons over it at
query time. But this raises the obvious question — HOW does the
memory model actually enable the LLM to perform counterfactual
reasoning, causal inference, future simulation, and creative
synthesis? The answer is architectural, not magical.

### The Problem: LLMs Without Memory Cannot Reason About You

A vanilla LLM (no memory, no RAG) receives only the current prompt.
It can reason brilliantly about general knowledge but is **completely
blind** to the user's personal history.

```text
 User: "What if I hadn't switched from the East dock to the West dock?"

 LLM without memory:
 "I don't have context about any dock switch. Could you tell me
  more about the situation?"

 It CANNOT reason counterfactually because it has NO FACTS to
 reason over. Counterfactual reasoning requires:
   1. Knowledge of what ACTUALLY happened (the factual world)
   2. The ability to perturb one variable
   3. The ability to project consequences of that perturbation

 Without (1), steps (2) and (3) are impossible.
```

RAG partially solves this — it retrieves text chunks about the dock
switch. But it's shallow: the LLM sees disconnected paragraphs, not
a unified representation of the user's life context.

### But Wait — The Concierge Already Queries K0 Memory

This is the elephant in the room. Before justifying the memory model,
we must be honest about what FamilyOS ALREADY does:

```text
 THE CURRENT RECALL PATH (No Memory Model)
 ==========================================

 User: "What if I hadn't switched docks?"
             |
             v
 +===========+==========+
 | Front LLM ReAct Loop |
 | iteration 2:         |
 | recall_memory()      |   <--- Front has this tool
 +==========+===========+
            |
            v
 +===========+==========+
 | Bridge Query Port    |
 | /k0/query.recall     |   Multi-selector recall bundle:
 +==========+===========+   - type: episodic (WAL position)
            |               - type: semantic (pgvector similarity)
            v               - type: graph (KG edge traversal)
 +===========+==========+
 | K0 PostgreSQL        |
 | st_hipp_events       |   Searches across consolidated
 | st_sem_nodes         |   memory tables with pgvector
 | st_kg_dom_edges      |   similarity + FTS + KG traversal
 +==========+===========+
            |
            | TEXT CHUNKS returned
            v
 +===========+==========+
 | DynamicPromptBuilder |   Pastes recalled text into
 | 128K context window  |   the LLM's prompt as tokens
 | 80% safety margin    |
 +==========+===========+
            |
            v
 +===========+==========+
 | Front LLM reasons    |   The LLM "reads" the recalled
 | over pasted text     |   paragraphs and reasons over them
 +=======================+

 This ALREADY gives the Concierge personal memory.
 The LLM CAN already answer "what if" questions with recalled context.
 So WHY do we need a memory model?
```

The answer is not that the current system doesn't work — it's that
**it works within hard limits that the memory model transcends.**

### Why recall_memory() Is Not Enough: The 7 Hard Limits

```text
 LIMIT 1: THE TOKEN BUDGET IS FINITE AND ZERO-SUM
 =================================================

 The 128K context window is already heavily allocated:

 +----------------------------+--------------------+
 | Component                  | Token Cost         |
 +----------------------------+--------------------+
 | System prompt (STANDARD)   | ~5,500 tokens      |
 | SessionState HOT sections  | ~10,000-15,000     |
 | Chat history (5-20 turns)  | ~3,000-12,000      |
 | Tool schemas (10 tools)    | ~2,000-3,000       |
 | Domain rules + safety      | ~500-1,000         |
 | Mode-specific examples     | ~200-400           |
 +----------------------------+--------------------+
 | USED BEFORE ANY RECALL     | ~21,000-37,000     |
 +----------------------------+--------------------+
 | Remaining for recalled     | ~65,000-81,000     |
 | memories (best case)       |                    |
 +----------------------------+--------------------+

 Sounds like enough? Consider:
 - A single episode with full context is ~200-500 tokens
 - For counterfactual reasoning about a dock switch, the LLM needs:
   * The dock switch episode itself (~300 tok)
   * 5-10 related past dock episodes (~2,000 tok)
   * The emotional arc across those episodes (~500 tok)
   * Causal chain: dock -> delays -> schedule (~300 tok)
   * User's stated goals about workflow (~200 tok)
   * Partner's related constraints (~200 tok)
 - That's ~3,500 tokens for ONE reasoning chain.
 - Now what if the user asks about their entire dock history
   over 6 months? 50+ episodes? ~15,000-25,000 tokens.
 - And that competes with everything else in the window.

 MEMORY MODEL COST: ZERO TOKENS.
 All 8 layers of memory representations sit alongside
 attention K,V pairs. The context window is untouched.


 LIMIT 2: RECALL IS A SELECTION PROBLEM (LOSSY BY DESIGN)
 =========================================================

 recall_memory() must CHOOSE what to return.
 It uses selectors: episodic, semantic, graph.
 But it can only return what it selects.

 For the dock question, the selector might return:
 - The 3 most recent dock episodes (pgvector similarity)
 - The "dock" semantic node (KG)
 - The dock-related routine (graph traversal)

 What it MISSES:
 - The episode 4 months ago where the user first complained
   about East dock delays (not in top-3 similarity)
 - The episode where the partner mentioned dock stress at dinner
   (tagged as "dinner" not "dock" — low similarity to "dock")
 - The gradual emotional buildup (spread across 30 episodes,
   no single one is "about docks" enough to be recalled)

 The selector must guess what's relevant. It often guesses wrong.

 MEMORY MODEL: No selection. The entire user history is
 compressed into vectors. The dock switch, the emotional
 buildup, the partner's comment, the pattern — all encoded.
 The LLM's attention heads discover what's relevant.


 LIMIT 3: TEXT IS FLAT, VECTORS ENCODE STRUCTURE
 ================================================

 Recalled text is paragraphs. Example:

   "On March 3, user switched from East dock to West dock.
    User mentioned frustration with East dock delays.
    User's schedule was affected by the switch."

 The LLM reads this linearly. It must RE-DERIVE:
 - That this is part of a 3-month frustration pattern
 - That the emotional trajectory was building
 - That dock_choice causally links to schedule_slip
 - That this connects to the user's workflow optimization goal

 This re-derivation happens in a single forward pass, from
 flat text, competing for attention with everything else in
 the 128K context window.

 Memory model vectors PRE-COMPUTE these relationships:
 - L6 procedural: the dock_choice -> schedule pattern
 - L7 emotional: the frustration trajectory
 - L4 episodic: the event cluster, not one episode
 - L8 prospective: the workflow optimization goal
 The LLM doesn't re-derive. It ATTENDS to pre-computed structure.


 LIMIT 4: CROSS-MEMORY RELATIONSHIPS ARE INVISIBLE IN TEXT
 ==========================================================

 The causal chain (dock_choice -> wait_time -> schedule_slip)
 does not exist in any single recalled memory. It exists
 ACROSS memories:
 - Memory A: "user chose East dock" (March 1)
 - Memory B: "waited 45 minutes at dock" (March 1)
 - Memory C: "missed afternoon meeting" (March 1)
 - Memory D: "user chose East dock" (March 5)
 - Memory E: "waited 50 minutes" (March 5)
 - Memory F: "skipped lunch to catch up" (March 5)

 Even if all 6 are recalled, the LLM sees them as 6
 separate paragraphs. It CAN infer the pattern, but:
 - It's expensive (attention across 6 scattered chunks)
 - It's unreliable (depends on paragraph ordering)
 - It competes with everything else for attention

 Memory model L6 (procedural) LEARNS this pattern during
 training. It's one vector: "dock_delay_cascade_pattern."
 The LLM attends to it directly.


 LIMIT 5: EMOTIONAL TRAJECTORY IS LOST IN TEXT RECALL
 =====================================================

 Recalled text says: "user was frustrated at dock."
 Valence: -0.6. One data point.

 But the TRAJECTORY matters more:
 - Month 1: mild annoyance (valence -0.2)
 - Month 2: growing frustration (valence -0.4)
 - Month 3: anger (valence -0.6) -> ACTION (switched docks)
 - After switch: relief (valence +0.4)

 To get this trajectory from text recall, you'd need to
 recall 30+ episodes, each with affect metadata, and hope
 the LLM traces the emotional arc across all of them.

 Memory model L7 (emotional) encodes the trajectory as a
 vector. One representation captures the entire arc.


 LIMIT 6: RECALL ADDS LATENCY PER QUERY
 ========================================

 Each recall_memory() call:
 - K1 -> Bridge (serialization + transport)
 - Bridge -> K0 (network round-trip)
 - K0 executes: pgvector search, FTS, KG traversal
 - K0 -> Bridge -> K1 (response transport)
 - DynamicPromptBuilder reassembles context

 Typical: 50-200ms per recall call.
 Front ReAct loop calls recall_memory() on iteration 2.
 This is SYNCHRONOUS — the user waits.

 Memory model inference: load checkpoint, forward pass.
 Local, no network. Target: <50ms for ALL layers.
 And it runs in PARALLEL with tokenization, not sequentially.


 LIMIT 7: RECALL IS REACTIVE, MEMORY MODEL IS ALWAYS-ON
 ========================================================

 recall_memory() fires when the Front LLM decides to call it.
 It's a TOOL CALL — the LLM must recognize it needs memory
 and explicitly invoke recall.

 What if the LLM doesn't recognize the need?
 "How's your day going?" — The LLM might just respond generically
 without recalling that the user had a stressful meeting earlier
 today, or that their partner is sick, or that they've been
 anxious about a deadline.

 Memory model vectors are ALWAYS injected into attention.
 Every single query, every single turn, the LLM sees the
 full memory state. No tool call needed. No recognition
 needed. The attention mechanism discovers relevance
 automatically.

 This is the difference between:
 - "Ask when you think you need it" (recall_memory)
 - "Always know who you're talking to" (memory model)
```

### The Memory Model and recall_memory() Are Complementary

```text
 NOT A REPLACEMENT — AN ENHANCEMENT
 ====================================

 The memory model does NOT replace recall_memory().
 They serve different purposes:

 +---------------------+-------------------------------+
 | recall_memory()     | Memory Model                  |
 +---------------------+-------------------------------+
 | EXPLICIT recall     | IMPLICIT context              |
 | "When did I last    | The LLM always "knows" who    |
 | visit Tokyo?"       | the user is, without asking   |
 +---------------------+-------------------------------+
 | SPECIFIC facts      | PATTERNS and UNDERSTANDING    |
 | Returns text chunks | Returns dense representations |
 | about Tokyo trip    | of user's travel patterns,    |
 |                     | preferences, emotional arcs   |
 +---------------------+-------------------------------+
 | REACTIVE: fires     | ALWAYS-ON: injected into      |
 | when LLM asks       | every attention computation   |
 +---------------------+-------------------------------+
 | TOKEN-CONSUMING:    | ZERO-TOKEN: memory vectors    |
 | text in context     | sit alongside K,V pairs       |
 +---------------------+-------------------------------+
 | GOOD FOR:           | GOOD FOR:                     |
 | Specific lookups    | Personalization tone           |
 | Date/fact retrieval | Emotional awareness           |
 | Explicit Q&A        | Proactive relevance           |
 | "What hotel did     | Counterfactual reasoning      |
 |  I stay at?"        | Pattern recognition           |
 |                     | Goal-aware responses          |
 +---------------------+-------------------------------+

 The BEST response uses BOTH:
 - Memory model provides always-on personal context
   (user's patterns, emotions, goals, relationships)
 - recall_memory() provides specific facts on demand
   (exact dates, names, amounts, details)

 Together they give the LLM both UNDERSTANDING and FACTS.
```

### The Architecture With Both Systems

```text
 ENHANCED INFERENCE FLOW (Memory Model + recall_memory)
 =======================================================

 User: "What if I hadn't switched docks?"
              |
              v
 +=============+============+
 |  MEMORY MODEL (always-on) |
 |  Load user checkpoint     |
 |  Produce memory vectors:  |
 |  [L4: dock episodes]      |  <-- IMPLICIT: always there
 |  [L5: dock knowledge]     |      no tool call needed
 |  [L6: dock->schedule]     |      zero tokens consumed
 |  [L7: frustration arc]    |
 |  [L8: workflow goal]      |
 +=============+=============+
               |
               v
 +=============+===========================+
 | ATTENTION FUSION                         |
 | Token K,V: "What if I hadn't..."         |
 | Memory K,V: [all layer vectors]          |  <-- DEEP context
 | The LLM already "knows" about the docks  |
 | before any tool is called.               |
 +==============+==========================+
                |
                v
 +=============+===========+
 | Front LLM ReAct Loop    |
 | iteration 2:            |
 | recall_memory()         |  <-- EXPLICIT: specific details
 | "get dock switch event  |     returns exact dates, names
 |  from March 3"          |     fills in precise facts
 +=============+===========+
               |
               v
 +=============+===========+
 | LLM GENERATES RESPONSE  |
 |                         |
 | Uses BOTH:              |
 | - Memory vectors for    |   The response has both
 |   emotional arc, causal |   DEEP understanding and
 |   chains, patterns,     |   SPECIFIC facts.
 |   goal awareness        |
 | - Recalled text for     |
 |   specific dates,       |
 |   names, exact details  |
 +==========================+
```

The memory model makes the Concierge's LLM **start every turn
already knowing who the user is.** recall_memory() then fills
in specific facts when needed. Without the memory model, the LLM
starts every turn as a blank slate and HOPES it calls recall
at the right time with the right selectors.

The memory model does not generate counterfactuals itself. It provides
something far more powerful — **the complete factual world, encoded as
dense learned representations fused into the LLM's attention.**

```text
 WHAT THE MEMORY MODEL INJECTS (per user, per query)
 ====================================================

 Layer    What It Contributes                   Vector Content
 -----    -----------------------               ----------------------------
 L4       Episodic memory vectors               The dock switch event itself:
          (what happened)                        when, where, who was there,
                                                 what led to it, what followed

 L5       Semantic knowledge vectors             "East dock = crowded, slow
          (what the user knows/believes)         unloading; West dock = new,
                                                 faster equipment, less shade"

 L6       Procedural memory vectors              The user's habitual dock
          (how the user does things)             selection pattern, the
                                                 routine they broke

 L7       Emotional memory vectors               Frustration at East dock
          (how events felt)                      delays (valence -0.6),
                                                 relief at West dock
                                                 (valence +0.4)

 L8       Prospective memory vectors             User's pending goal:
          (what the user intends)                "optimize dock workflow"

 L5+L6   Causal pattern vectors                  Granger-learned edge:
         (from R4 KG)                            dock_choice -> wait_time
                                                 -> daily_schedule_slip
```

All of this sits in the LLM's attention as native key-value pairs.
The LLM does not need to "read" paragraphs about docks. It **attends**
to the dock-switch event, the emotional context, the causal chain,
and the user's goals — all simultaneously, all as first-class
representations in its own compute.

### The Mechanism: How Attention Fusion Enables Counterfactuals

```text
 COUNTERFACTUAL REASONING FLOW
 ==============================

 User query: "What if I hadn't switched docks?"
                  |
                  v
 +================+===============+
 |        TOKENIZED QUERY         |
 |  Q vectors from: "What if I   |
 |  hadn't switched docks?"       |
 +================+===============+
                  |
                  v
 +================+===============+
 |     ATTENTION LAYER (fused)    |
 |                                |
 |  Q attends over:               |
 |                                |
 |  TOKEN K,V:                    |
 |    [what, if, I, hadn't,       |
 |     switched, docks, ?]        |
 |                                |
 |  MEMORY K,V (injected):        |
 |    [dock_switch_episode]  <--- L4: the actual event          |
 |    [east_dock_semantics]  <--- L5: what user knows about it  |
 |    [west_dock_semantics]  <--- L5: comparison knowledge      |
 |    [dock_routine_proc]    <--- L6: habitual pattern          |
 |    [dock_frustration]     <--- L7: emotional valence         |
 |    [dock_relief]          <--- L7: emotional contrast        |
 |    [dock->schedule_cause] <--- L5+L6: causal chain           |
 |    [optimize_workflow]    <--- L8: active goal               |
 |                                |
 |  The LLM's attention heads     |
 |  naturally discover:           |
 |   - "switched" aligns with     |
 |     dock_switch_episode (L4)   |
 |   - "what if hadn't" triggers  |
 |     counterfactual reasoning   |
 |     over the causal chain      |
 |   - emotional vectors provide  |
 |     the WHY behind the switch  |
 |   - goal vector contextualizes |
 |     the reasoning direction    |
 +================================+
                  |
                  v
 +================+===============+
 |     LLM GENERATES RESPONSE     |
 |                                |
 |  "If you'd stayed at the East  |
 |  dock, you'd likely have hit   |
 |  the same 45-minute delays     |
 |  that frustrated you last      |
 |  month. Your schedule would've |
 |  slipped again — remember that |
 |  cascade where the dock delay  |
 |  pushed your afternoon meeting |
 |  back by an hour? The West     |
 |  dock switch was the right     |
 |  call for your workflow        |
 |  optimization goal."           |
 +================================+
```

The LLM already knows HOW to reason counterfactually — that is its
world-knowledge capability from pre-training. What it lacks is the
**personal factual substrate** to reason ABOUT. The memory model
provides exactly that substrate, encoded in the LLM's own language
(attention vectors), at zero token cost.

### The Five Reasoning Capabilities the Memory Model Unlocks

```text
 +----+-------------------------+------------------------------------------+
 | #  | Capability              | What the Memory Model Provides           |
 +----+-------------------------+------------------------------------------+
 | 1  | COUNTERFACTUAL          | The factual world (L4 episodes + L7      |
 |    | "What if X hadn't       | emotions + L5+L6 causal edges) so the    |
 |    |  happened?"             | LLM can perturb one variable and trace   |
 |    |                         | consequences through the causal graph.   |
 +----+-------------------------+------------------------------------------+
 | 2  | CAUSAL INFERENCE        | Granger causality edges (L6 procedural   |
 |    | "Why did Y happen?"     | patterns) + temporal sequences (L4) so   |
 |    |                         | the LLM can follow cause-effect chains   |
 |    |                         | across the user's personal history.      |
 +----+-------------------------+------------------------------------------+
 | 3  | FUTURE SIMULATION       | Current goals (L8 prospective) + past    |
 |    | "What would happen      | outcome patterns (L4+L6) + emotional     |
 |    |  if I do Z?"            | preferences (L7) so the LLM projects     |
 |    |                         | consequences grounded in user's reality. |
 +----+-------------------------+------------------------------------------+
 | 4  | CREATIVE SYNTHESIS      | Cross-domain semantic knowledge (L5) +   |
 |    | "Connect these two      | episodic co-occurrences (L4) so the LLM  |
 |    |  seemingly unrelated    | finds non-obvious links across the       |
 |    |  things"                | user's experience landscape.             |
 +----+-------------------------+------------------------------------------+
 | 5  | PERSONALIZED ADVICE     | Complete user model (all layers) so the  |
 |    | "Given everything you   | LLM generates advice that accounts for   |
 |    |  know about me, what    | the user's habits, emotional patterns,   |
 |    |  should I do?"          | past outcomes, active goals, and social  |
 |    |                         | context — not generic platitudes.        |
 +----+-------------------------+------------------------------------------+
```

### Why This Is Strictly Better Than Algorithmic Counterfactuals

The dead CPN approach tried to generate counterfactuals algorithmically
in the consolidation pipeline:

```text
 CPN (DEAD):
   Input:  knowledge graph edge (A -> B)
   Method: perturb A, propagate through graph
   Output: "If A hadn't happened, B wouldn't have happened"
   Result: "If Tokyo had also caused issues during Routine with Panda..."
           (semantically nonsensical — graph perturbation ≠ reasoning)

 MEMORY MODEL + LLM (THIS PROJECT):
   Input:  rich multi-layer memory vectors fused into attention
   Method: the LLM's own transformer reasoning over dense representations
   Output: coherent, emotionally-aware, causally-grounded "what if" response
   Result: "If you'd stayed at the East dock, the delays would've cascaded
            into your afternoon again — similar to what happened on March 3rd
            when you were frustrated enough to skip lunch."

 WHY THE LLM IS BETTER AT THIS:
 ================================
 1. LLMs are trained on billions of tokens of causal reasoning text.
    They understand "if X then Y" at a deep structural level.
 2. Memory vectors give the LLM PERSONAL causal chains to reason over,
    not abstract graph edges.
 3. Emotional vectors (L7) let the LLM weight consequences by how they
    FELT, not just whether they happened.
 4. The LLM can combine multiple causal chains simultaneously — something
    graph perturbation cannot do without combinatorial explosion.
 5. The LLM produces natural language that the user can actually understand
    and engage with, not mechanical "If A then not B" propositions.
```

### The Information Flow: From Raw Data to Reasoning

```text
 RAW DATA                MEMORY MODEL              LLM REASONING
 =========               ============              =============

 Chat turns        -->   L2/L3 encode     -->   "I remember you said..."
 (SessionState)          recent context

 st_hipp_events    -->   L4 episodic      -->   "That time when you..."
 (MW v2 atoms)           encoder                 (recalls specific events)

 st_sem_nodes      -->   L5 semantic      -->   "You generally believe..."
 + st_kg_dom             encoder                 (draws on user knowledge)

 R4 Granger edges  -->   L6 procedural    -->   "Because every time you X,
                         encoder                  Y tends to follow..."
                                                  (causal reasoning)

 R1 importance     -->   L7 emotional     -->   "That really bothered you"
 + MW v2 affect          encoder                 (emotional understanding)

 SPC-UQ + intents  -->   L8 prospective   -->   "Given your goal to..."
                         encoder                  (goal-directed reasoning)

 All layers        -->   FUSED INTO       -->   COUNTERFACTUAL:
 combined               LLM ATTENTION          "What if you hadn't..."
                                                CAUSAL: "That happened
                                                 because..."
                                                FUTURE: "If you do X,
                                                 then probably..."
                                                CREATIVE: "This connects
                                                 to that because..."
                                                ADVICE: "Given your
                                                 history, I'd suggest..."
```

### The Key Insight: Separation of Concerns

```text
 +=====================================================+
 |              SEPARATION OF CONCERNS                  |
 |=====================================================|
 |                                                     |
 |  MEMORY MODEL's job:                                |
 |  -------------------                                |
 |  "WHAT happened to this user, HOW it felt,          |
 |   WHAT they know, WHAT patterns exist,              |
 |   WHAT they want next"                              |
 |                                                     |
 |  Encoded as dense vectors. No reasoning.            |
 |  No counterfactuals. No speculation.                |
 |  Pure representation of the user's world.           |
 |                                                     |
 |  LLM's job:                                         |
 |  ----------                                         |
 |  "GIVEN everything the memory model tells me        |
 |   about this user, REASON about their question."    |
 |                                                     |
 |  Counterfactual, causal, predictive, creative,      |
 |  advisory — all forms of reasoning that the LLM     |
 |  was pre-trained to perform, now GROUNDED in        |
 |  personal context instead of generic knowledge.     |
 |                                                     |
 |  WHY THIS WORKS:                                    |
 |  ---------------                                    |
 |  The memory model turns the LLM from a              |
 |  "smart stranger" into a "smart friend who          |
 |  knows your whole history."                         |
 |                                                     |
 |  A smart stranger can reason counterfactually       |
 |  about general topics. A smart friend can reason    |
 |  counterfactually about YOUR LIFE — because they    |
 |  have the facts, the feelings, the patterns,        |
 |  and the goals.                                     |
 |                                                     |
 |  The memory model IS that friendship, compressed    |
 |  into vectors.                                      |
 +=====================================================+
```

### Concrete Example: Multi-Layer Reasoning in Action

```text
 User: "I've been thinking about switching jobs. What do you think?"

 WITHOUT MEMORY MODEL (generic LLM):
 "That's a big decision! Consider factors like salary, work-life
  balance, growth opportunities, and company culture."
 (Generic. Could be said to anyone.)

 WITH MEMORY MODEL (all 8 layers active):

 The LLM's attention simultaneously sees:

   L4 (episodic):  - User complained about manager 7 times in 3 months
                    - User was excited about the side project 2 weeks ago
                    - User's partner mentioned job stress at dinner last week

   L5 (semantic):  - User values autonomy (extracted from 40+ interactions)
                    - User's domain expertise is in distributed systems
                    - User believes "life is too short for bad bosses"

   L6 (procedural): - User's pattern: complain for ~4 months, then act
                     - Previous job switch led to 6-month adjustment period
                     - User tends to underestimate transition costs

   L7 (emotional):  - Manager interactions: valence -0.55, arousal +0.7
                       (frustrated and agitated, not just mildly annoyed)
                     - Side project: valence +0.85 (genuine excitement)
                     - Job topic overall: trending negative over 3 months

   L8 (prospective): - User mentioned wanting to "go independent someday"
                      - User has a pending goal: save 6 months expenses first
                      - Partner's job situation is uncertain (flagged 2 weeks ago)

 LLM response (grounded in all layers):
 "You've been frustrated with your manager for a while now — it's
  clearly not just a passing thing, your stress about it has been
  building for months. But I notice the pattern from last time: you
  complained about the same things at your previous job for about
  four months before you made the move, and the six-month adjustment
  period was harder than you expected.

  Here's what I'd flag: you mentioned wanting to go independent
  eventually, and that side project you were excited about last week
  could be a stepping stone. But your partner's job situation is
  still uncertain, and you haven't hit your six-month savings target
  yet.

  Maybe the question isn't 'should I switch jobs' but 'should I
  start building toward independence while stabilizing the financial
  cushion first?' That aligns better with what you actually want
  long-term."

 This response:
 - References specific events (L4 episodic)
 - Draws on user's values and beliefs (L5 semantic)
 - Recognizes behavioral patterns (L6 procedural)
 - Acknowledges emotional trajectory (L7 emotional)
 - Accounts for stated goals and constraints (L8 prospective)
 - Performs counterfactual reasoning: "if you switch now vs later"
 - Performs causal reasoning: "frustration is building because..."
 - Performs future simulation: "this side project could become..."
 - Gives personalized advice: specific to THIS user's life, not generic

 ALL OF THIS FROM DENSE VECTORS. ZERO EXTRA TOKENS.
```

### Why Algorithmic Approaches Failed and Neural Ones Succeed

```text
 ALGORITHMIC (CPN/MCTS)              NEURAL (Memory Model + LLM)
 ======================              ===========================

 Operates on graph edges             Operates on learned representations
 (symbolic, brittle)                 (continuous, flexible)

 One perturbation at a time          Thousands of attention heads
 (combinatorial explosion)           process all context in parallel

 No understanding of language        Full language model understands
 ("If Tokyo had also caused          nuance, idiom, emotion, intent
 issues during Routine...")

 No emotional weighting              L7 emotional vectors naturally
 (all edges equal)                   weight consequences by how
                                     they felt to the user

 No goal awareness                   L8 prospective vectors orient
 (blind perturbation)                reasoning toward user's actual
                                     objectives

 Produces text nobody wants          Produces text the user engages
 to read or can act on               with and finds genuinely helpful

 Runs offline in batch               Runs in real-time at query time
 (stale by definition)               (always current)

 Required causal edges that          Works with whatever memory data
 R4 doesn't produce (0 edges)        exists — graceful degradation
```

The fundamental error was trying to make the MEMORY PIPELINE reason.
Memory pipelines should REMEMBER. Reasoning is the LLM's job — and
the memory model gives it the richest possible material to reason with.

---

## Research: How LLMs and the Memory Layer Connect — Driving Inference Behavior

This section is technical research. It goes deeper than the earlier
"How It Plugs In" overview and asks: **at the math level, what are
the actual mechanisms by which external vectors can influence a frozen
LLM's generation?** We need to understand the LLM's internals before
we can design the memory fusion layer.

### The Fundamental Question

An LLM is a stack of transformer decoder blocks. Each block runs
self-attention, then a feed-forward network. The model is frozen —
we cannot change its weights. Yet we want to inject external information
(memory vectors) that CHANGES the model's output distribution.

How is this even possible?

```text
 THE INJECTION PROBLEM
 =====================

 FROZEN LLM (cannot modify):
 +------------------------------------------------------------+
 | Layer 1: Self-Attention -> FFN                              |
 | Layer 2: Self-Attention -> FFN                              |
 | ...                                                         |
 | Layer N: Self-Attention -> FFN                              |
 | Output:  logits = W_vocab * hidden_state_N                  |
 +------------------------------------------------------------+

 MEMORY MODEL (we control):
 +------------------------------------------------------------+
 | Produces K vectors: [m1, m2, ..., mK]                       |
 | Each vector has dimension d_model (matches LLM hidden size) |
 +------------------------------------------------------------+

 QUESTION: Where in the LLM's forward pass can we INSERT these
 vectors such that the LLM's output changes without modifying
 any of the LLM's trained weights?

 ANSWER: There are exactly 4 mathematically distinct injection
 points. Each has different properties, costs, and tradeoffs.
```

### Injection Point 1: Input Embedding Space (Prefix / Soft Prompt)

The simplest approach. Prepend memory vectors to the input token
embeddings BEFORE they enter the first transformer layer.

```text
 HOW IT WORKS
 ============

 Normal input (N tokens):
   E = [e1, e2, ..., eN]          shape: [N, d_model]
   Each ei = TokenEmbed(token_i) + PosEmbed(i)

 With memory prefix (K memory vectors + N tokens):
   E' = [m1, m2, ..., mK, e1, e2, ..., eN]   shape: [K+N, d_model]

 The LLM sees K+N "tokens." The first K are not real tokens —
 they are memory vectors projected into the embedding space.
 The LLM cannot tell the difference.

 +============================================================+
 |  EMBEDDING LAYER                                            |
 |                                                             |
 |  Position:  0    1   ...  K-1   K    K+1  ...  K+N-1       |
 |  Content:  [m1] [m2] ... [mK] [e1]  [e2] ... [eN]          |
 |            |--- memory ---|   |---- real tokens ----|       |
 |                                                             |
 |  Both go through ALL transformer layers identically.        |
 |  The causal mask lets real tokens attend to memory tokens.  |
 +============================================================+
         |
         v
 +============================================================+
 |  LAYER 1: SELF-ATTENTION                                    |
 |                                                             |
 |  Q = W_q * [m1..mK, e1..eN]                                |
 |  K = W_k * [m1..mK, e1..eN]                                |
 |  V = W_v * [m1..mK, e1..eN]                                |
 |                                                             |
 |  For real token ei:                                         |
 |    Q_i attends over ALL keys (including memory keys)        |
 |    Attention(Q_i, K_all, V_all) includes memory values      |
 |                                                             |
 |  The memory vectors PARTICIPATE in attention computation.   |
 |  They contribute Key-Value pairs that real tokens attend to.|
 +============================================================+

 MATH (for token at position j, where j >= K):

   attention_j = softmax( Q_j * [K_m1..K_mK, K_e1..K_eN]^T / sqrt(d_k) )
                         |--memory keys---|  |--token keys--|

   output_j = attention_j * [V_m1..V_mK, V_e1..V_eN]
                            |--memory vals-|  |--token vals--|

   The attention weights determine HOW MUCH each token
   "listens to" each memory vector vs other tokens.
```

**This is Prefix Tuning / Soft Prompting** (Li & Liang, 2021; Lester et al., 2021).

```text
 PROPERTIES:
 +---------------------------+-------------------------------------+
 | Dimension                 | Assessment                          |
 +---------------------------+-------------------------------------+
 | LLM weight changes?       | NONE. Fully frozen.                 |
 | Works with API-only LLMs? | YES if API accepts embedding input  |
 |                           | (most don't — see Limit below)      |
 | Token cost?               | YES — K memory tokens consume K     |
 |                           | positions in the context window.     |
 | Latency?                  | Minimal — just K extra positions    |
 |                           | in the forward pass.                 |
 | Expressiveness?           | LIMITED — memory vectors pass        |
 |                           | through ALL layers, getting          |
 |                           | transformed by weights not trained   |
 |                           | for them. Signal degrades deeply.    |
 | Where it changes output?  | Globally — affects all layers        |
 |                           | but increasingly diluted in deep     |
 |                           | networks.                           |
 +---------------------------+-------------------------------------+

 CRITICAL LIMIT: The memory vectors DO consume context window.
 K=64 memory vectors = 64 tokens lost from the prompt budget.
 This is better than pasting text (hundreds of tokens) but NOT
 zero-cost like the other injection methods.

 SECOND LIMIT: Deep signal degradation.
 In a 32-layer model, a prefix vector inserted at layer 0 is
 transformed by 32 layers of attention and FFN — weights that
 were trained on REAL token distributions, not memory vectors.
 The memory signal gets increasingly "mangled" by layers that
 don't know how to process it.
```

### Injection Point 2: Per-Layer KV Injection (The Core Technique)

Instead of inserting memory at the input only, inject separate K,V
pairs **at every attention layer** (or selected layers). This is the
most powerful technique and what this project targets.

```text
 HOW IT WORKS
 ============

 At each transformer layer i, self-attention normally computes:

   Q_i = W_q_i * hidden_states      (from token representations)
   K_i = W_k_i * hidden_states      (from token representations)
   V_i = W_v_i * hidden_states      (from token representations)

 With per-layer KV injection, we APPEND memory K,V to the token K,V:

   K_i_extended = concat(K_i_tokens, K_i_memory)
   V_i_extended = concat(V_i_tokens, V_i_memory)

   Q stays the same — only from tokens.

 +============================================================+
 |  LAYER i: SELF-ATTENTION (with memory KV injection)        |
 |============================================================|
 |                                                             |
 |  Token hidden states -> W_q -> Q_i   (shape: [N, d_head])  |
 |  Token hidden states -> W_k -> K_i   (shape: [N, d_head])  |
 |  Token hidden states -> W_v -> V_i   (shape: [N, d_head])  |
 |                                                             |
 |  Memory model -> project_k_i -> K_mem_i  (shape: [M, d_head]) |
 |  Memory model -> project_v_i -> V_mem_i  (shape: [M, d_head]) |
 |                                                             |
 |  K_extended = [K_i ; K_mem_i]   (shape: [N+M, d_head])     |
 |  V_extended = [V_i ; V_mem_i]   (shape: [N+M, d_head])     |
 |                                                             |
 |  attention = softmax(Q_i * K_extended^T / sqrt(d_k))        |
 |  output = attention * V_extended                            |
 |                                                             |
 |  The Q from each token attends over BOTH:                   |
 |  - Normal token K,V (self-attention as usual)               |
 |  - Memory K,V (NEW — the memory signal)                     |
 +============================================================+

 MATH (for attention head h at layer i, token position j):

   a_j = softmax( q_j * [k_1..k_N, k_mem_1..k_mem_M]^T / sqrt(d_k) )
                        |--token---|  |----memory----|

   The attention score between token j and memory slot m:
     score(j, m) = q_j * k_mem_m^T / sqrt(d_k)

   If the memory key k_mem_m aligns with what token j is
   "looking for" (high dot product), the attention weight
   for memory slot m will be high, and its value v_mem_m
   will strongly influence the output.

   output_j = sum(a_j * [v_1..v_N, v_mem_1..v_mem_M])
```

**The key insight:** each layer's memory K,V can be DIFFERENT.
The memory model produces **layer-specific** projections:

```text
 MEMORY MODEL OUTPUT (per layer)
 ================================

 The memory model has M memory slots and L projection heads
 (one per LLM layer, or one per selected layer subset):

   For layer i:
     K_mem_i = MemoryBank * W_proj_k_i    shape: [M, d_head]
     V_mem_i = MemoryBank * W_proj_v_i    shape: [M, d_head]

   Where:
     MemoryBank: [M, d_memory]   (the learned memory representations)
     W_proj_k_i: [d_memory, d_head]   (layer-specific key projection)
     W_proj_v_i: [d_memory, d_head]   (layer-specific value projection)

 WHY LAYER-SPECIFIC MATTERS:
   Layer 1 attention heads learn syntactic patterns.
   Layer 16 attention heads learn semantic relationships.
   Layer 32 attention heads learn abstract reasoning patterns.

   The same memory ("user hates cilantro") should present
   DIFFERENT facets to different layers:
   - Layer 1: no signal (syntax doesn't care about cilantro)
   - Layer 16: strong signal (semantic: food preference)
   - Layer 32: moderate signal (reasoning: dietary constraint)

   Layer-specific projections let the memory model learn
   WHAT to emphasize at each depth of the LLM.

 PARAMETER COUNT:
   M memory slots * L layers * 2 (K+V) * d_memory * d_head
   Example: 64 slots * 32 layers * 2 * 256 * 128 = ~134M params
   (for projections alone — this is non-trivial)

   Optimization: share projections across layer groups.
   Group layers into 4 blocks of 8 → 64 * 4 * 2 * 256 * 128 = ~17M params
```

```text
 PROPERTIES:
 +---------------------------+-------------------------------------+
 | Dimension                 | Assessment                          |
 +---------------------------+-------------------------------------+
 | LLM weight changes?       | NONE. Fully frozen.                 |
 | Works with API-only LLMs? | NO. Requires access to each layer's |
 |                           | attention computation.               |
 | Token cost?               | ZERO. Memory K,V are appended to    |
 |                           | the KV cache, not the token sequence.|
 |                           | Context window is untouched.         |
 | Latency?                  | Small — M extra keys per layer.     |
 |                           | Attention is O((N+M) * d) per layer.|
 | Expressiveness?           | MAXIMUM — layer-specific signals    |
 |                           | that don't degrade through depth.    |
 |                           | Each layer gets the right memory     |
 |                           | representation for its function.     |
 | Where it changes output?  | Precisely controlled — can target    |
 |                           | specific layers or all layers.       |
 +---------------------------+-------------------------------------+

 THIS IS THE PRIMARY TECHNIQUE FOR LLM-WITHMEM.

 Why: zero token cost + maximum expressiveness + frozen LLM.
 The memory model learns the optimal K,V projections per layer
 through training on user data. The LLM's attention heads
 automatically discover which memory slots are relevant for
 each token in each query — no manual selection needed.
```

### Injection Point 3: Cross-Attention Layers (Encoder-Decoder Style)

Add entirely NEW attention layers between the existing ones, where
Q comes from the LLM's hidden states and K,V come from memory.

```text
 HOW IT WORKS
 ============

 Normal decoder layer i:
   hidden = SelfAttention(hidden) + hidden    (residual)
   hidden = FFN(hidden) + hidden              (residual)

 With cross-attention inserted:
   hidden = SelfAttention(hidden) + hidden    (residual)
   hidden = CrossAttention(                   (NEW — added layer)
              Q = hidden,                     Q from LLM
              K = memory_keys,                K from memory model
              V = memory_values               V from memory model
            ) + hidden                        (residual)
   hidden = FFN(hidden) + hidden              (residual)

 +============================================================+
 |  MODIFIED LAYER i (with cross-attention)                    |
 |============================================================|
 |                                                             |
 |  Input hidden states                                        |
 |       |                                                     |
 |       v                                                     |
 |  [Self-Attention]  (normal — token-to-token)                |
 |       |                                                     |
 |       + residual                                            |
 |       |                                                     |
 |       v                                                     |
 |  [Cross-Attention]  (NEW — token queries, memory K,V)       |
 |    Q = W_q_cross * hidden     (from LLM's own states)       |
 |    K = W_k_cross * memory     (from memory model)           |
 |    V = W_v_cross * memory     (from memory model)           |
 |       |                                                     |
 |       + residual                                            |
 |       |                                                     |
 |       v                                                     |
 |  [FFN]  (normal)                                            |
 |       |                                                     |
 |       + residual                                            |
 |       |                                                     |
 |       v                                                     |
 |  Output hidden states                                       |
 +============================================================+

 MATH:
   Q_cross = W_q_cross * h_i         [N, d_cross]
   K_cross = W_k_cross * mem         [M, d_cross]
   V_cross = W_v_cross * mem         [M, d_cross]

   cross_out = softmax(Q_cross * K_cross^T / sqrt(d_cross)) * V_cross
   h_i_new = h_i + gate * cross_out    (gated residual)
```

```text
 PROPERTIES:
 +---------------------------+-------------------------------------+
 | Dimension                 | Assessment                          |
 +---------------------------+-------------------------------------+
 | LLM weight changes?       | NO to LLM weights. But adds NEW    |
 |                           | trainable W_q/W_k/W_v_cross and    |
 |                           | gate parameters per layer.          |
 | Works with API-only LLMs? | NO. Requires inserting layers into  |
 |                           | the model architecture.              |
 | Token cost?               | ZERO.                               |
 | Latency?                  | MODERATE — adds full attention      |
 |                           | computation at each inserted layer. |
 | Expressiveness?           | VERY HIGH — dedicated attention     |
 |                           | mechanism for memory, independent   |
 |                           | of self-attention.                  |
 | Where it changes output?  | At inserted layers only.            |
 | Extra parameters?         | SIGNIFICANT — full attention block  |
 |                           | per inserted layer.                 |
 +---------------------------+-------------------------------------+

 TRADEOFF vs KV Injection:
 - More expressive (dedicated Q for memory attention)
 - More expensive (extra attention computation per layer)
 - Harder to implement (must modify model forward pass)
 - The LLM's OWN hidden states form queries into memory
   (vs KV injection where tokens query memory via existing Q)

 Used in: Flamingo (visual tokens), Memorizing Transformers,
 and encoder-decoder architectures (T5, BART).
```

### Injection Point 4: Hidden State Addition (Residual Injection)

The simplest non-attention approach. Add memory vectors directly to
the hidden states at specific layers via residual connections.

```text
 HOW IT WORKS
 ============

 At selected layer i, after the normal computation:

   h_i_normal = TransformerBlock_i(h_{i-1})   (LLM's own output)

   h_i_modified = h_i_normal + alpha * MemoryProject_i(memory_vectors)

   Where:
     MemoryProject_i: [d_memory] -> [d_model]   (learned projection)
     alpha: learned scalar gate (0 to 1)

 +============================================================+
 |  LAYER i WITH RESIDUAL MEMORY INJECTION                     |
 |============================================================|
 |                                                             |
 |  h_{i-1}  -------> [Transformer Block i] -------> h_i      |
 |                                                    |        |
 |  memory_vec ------> [Project_i] ------> m_i       |        |
 |                                          |         |        |
 |                                    [alpha * m_i] --+        |
 |                                                    |        |
 |                                              h_i_modified   |
 +============================================================+

 This is NOT attention-based. There is no query-key matching.
 The memory vector is unconditionally added to ALL token
 hidden states equally. No token-specific selectivity.
```

```text
 PROPERTIES:
 +---------------------------+-------------------------------------+
 | Dimension                 | Assessment                          |
 +---------------------------+-------------------------------------+
 | LLM weight changes?       | NONE to LLM. Small projection +    |
 |                           | gate per injected layer.            |
 | Works with API-only LLMs? | NO. Requires hidden state access.   |
 | Token cost?               | ZERO.                               |
 | Latency?                  | MINIMAL — one matrix multiply +    |
 |                           | one addition per injected layer.    |
 | Expressiveness?           | LOW — no selectivity. Same memory   |
 |                           | signal added to ALL tokens. Cannot  |
 |                           | say "this memory matters for token  |
 |                           | 5 but not token 12."               |
 | Where it changes output?  | At injected layers only.            |
 +---------------------------+-------------------------------------+

 USE CASE: Global biases only. "This user prefers formal tone"
 can be a single vector added everywhere. But for episodic
 recall ("that specific dock event"), attention-based methods
 are strictly superior.

 Could be useful as a LIGHTWEIGHT COMPLEMENT to KV injection:
 - KV injection for episodic/semantic/procedural memory
 - Residual injection for user preference/tone/style bias
```

### Comparison Matrix: All Four Injection Points

```text
 +==========+================+==========+=========+===========+=============+
 | Method   | LLM Frozen?    | Token    | API-    | Express-  | Best For    |
 |          |                | Cost     | Only?   | iveness   |             |
 +==========+================+==========+=========+===========+=============+
 | Prefix / | YES            | K tokens | MAYBE   | Low-Med   | Simple       |
 | Soft     |                | consumed | (need   | (signal   | preferences, |
 | Prompt   |                |          | embed   | degrades  | short cues   |
 |          |                |          | access) | deeply)   |              |
 +----------+----------------+----------+---------+-----------+--------------+
 | Per-Layer| YES            | ZERO     | NO      | MAXIMUM   | ALL memory   |
 | KV       |                |          | (need   | (layer-   | types:       |
 | Injection|                |          | layer   | specific  | episodic,    |
 |          |                |          | access) | signals)  | semantic,    |
 |          |                |          |         |           | causal, etc. |
 +----------+----------------+----------+---------+-----------+--------------+
 | Cross-   | YES (but adds  | ZERO     | NO      | Very High | Rich memory  |
 | Attention| new trainable  |          | (need   | (dedicated| grounding,   |
 |          | layers)        |          | model   | attention | visual/audio |
 |          |                |          | surgery)| for mem)  | context      |
 +----------+----------------+----------+---------+-----------+--------------+
 | Residual | YES            | ZERO     | NO      | Low       | Global user  |
 | Addition |                |          | (need   | (no       | prefs, tone, |
 |          |                |          | hidden  | selectivity| style bias  |
 |          |                |          | access) | per token)|              |
 +==========+================+==========+=========+===========+=============+

 FOR THIS PROJECT: Per-Layer KV Injection is the primary method.
 - Zero token cost (critical for the "7 limits" argument)
 - Maximum expressiveness (layer-specific memory signals)
 - Frozen LLM (plug-and-play requirement)
 - Compatible with open-weight models (Llama, Mistral, Gemma)

 FALLBACK: Prefix/Soft Prompt for API-only LLMs (Gemini, GPT-4)
 where layer access is impossible. Accepts token cost as tradeoff.
```

### How Attention Actually "Reads" Memory Vectors

To understand WHY per-layer KV injection works, we need to understand
what attention heads actually do with the injected K,V pairs.

```text
 WHAT HAPPENS INSIDE ONE ATTENTION HEAD
 ========================================

 Setup: Layer 12, Head 7 (one of many heads at this layer)
 Token at position j: "docks" (in the query "what if I hadn't switched docks?")
 Memory slot m3: encodes "dock_switch_episode" (from L4 episodic layer)

 Step 1: Query vector
   q_j = W_q * hidden_j          shape: [d_head]
   "docks" produces a query vector that encodes "I'm looking for
    information related to docks/switching/change"

 Step 2: Key matching (ALL keys, including memory)
   score(j, t)  = q_j dot k_t    for each token t
   score(j, m3) = q_j dot k_m3   for memory slot m3

   If k_m3 (the dock episode key) is well-aligned with q_j
   (the "docks" query), score(j, m3) will be HIGH.

   The memory model's job during training is to learn W_proj_k
   such that k_m3 aligns well with the kinds of queries that
   should attend to dock-related memories.

 Step 3: Softmax normalization
   attention_weights = softmax(all scores / sqrt(d_head))

   Example distribution for token "docks":
   +------------------------------------------+
   | Token/Slot    | Attention Weight          |
   +------------------------------------------+
   | "what"        | 0.02                      |
   | "if"          | 0.03                      |
   | "I"           | 0.05                      |
   | "hadn't"      | 0.08                      |
   | "switched"    | 0.15                      |
   | "docks"       | 0.04 (self)               |
   | "?"           | 0.01                      |
   | m1 (beliefs)  | 0.02                      |
   | m2 (emotions) | 0.12  <-- emotional context|
   | m3 (episode)  | 0.28  <-- HIGHEST: dock episode |
   | m4 (goals)    | 0.09                      |
   | m5 (routine)  | 0.11                      |
   +------------------------------------------+

   The token "docks" attends most strongly to memory slot m3
   (the dock episode) — attention discovered this automatically
   through key-query dot product.

 Step 4: Value aggregation
   output_j = sum(attention_weight_t * v_t for all t)
            + sum(attention_weight_m * v_m for all m)

   The output for token "docks" is now a MIXTURE of:
   - Information from other tokens (normal self-attention)
   - Information from memory slots (the user's personal context)

   The dock episode value (v_m3) contributes 28% of the signal.
   Emotions (v_m2) contribute 12%.
   Routine (v_m5) contributes 11%.

   The hidden state for "docks" is now INFUSED with personal memory.
   This propagates forward through the remaining layers, affecting
   ALL subsequent computations and ultimately the output distribution.
```

### How Memory Influence Propagates Through the Network

```text
 LAYER-BY-LAYER PROPAGATION
 ============================

 A memory vector injected at layer i affects ALL subsequent layers
 through the residual stream:

 Layer 1:  hidden_1 = SelfAttn_1(input + memory_KV_1) + input
           Memory influence: DIRECT (in attention output)

 Layer 2:  hidden_2 = SelfAttn_2(hidden_1 + memory_KV_2) + hidden_1
           Memory influence: DIRECT from memory_KV_2
                          + INDIRECT from hidden_1 (which contains
                            layer 1's memory influence via residual)

 Layer 3:  hidden_3 = SelfAttn_3(hidden_2 + memory_KV_3) + hidden_2
           Memory influence: DIRECT + 2 levels of INDIRECT

 ...

 Layer N:  hidden_N contains accumulated memory influence from
           ALL layers that had memory KV injection.

 +------------------------------------------------------------+
 | THE RESIDUAL STREAM IS THE KEY                              |
 |                                                             |
 | Transformer layers don't REPLACE hidden states. They ADD    |
 | to them via residual connections:                           |
 |                                                             |
 |   h_out = h_in + LayerOutput(h_in)                         |
 |                                                             |
 | This means memory influence ACCUMULATES. A strong memory    |
 | signal at layer 12 persists through layers 13, 14, ..., N. |
 | Later layers can amplify or attenuate it, but they cannot   |
 | completely erase it (the residual connection preserves it). |
 |                                                             |
 | This is why DEEP injection (at multiple layers) is more     |
 | powerful than SHALLOW injection (prefix only at layer 0):   |
 | each layer reinforces the memory signal fresh.              |
 +------------------------------------------------------------+
```

### The Gating Problem: Controlling Memory Influence Strength

If memory vectors are always injected, how do we prevent them from
overwhelming the LLM's own representations? The answer is gating.

```text
 THE PROBLEM WITHOUT GATING
 ===========================

 User: "What is 2 + 2?"
 Memory: [user's entire life history in 64 vectors]

 The LLM doesn't need memory for this query. But if memory K,V
 are always present, the attention mechanism might attend to
 irrelevant memory slots, injecting noise into a simple math query.

 THE SOLUTION: LEARNED GATES
 ============================

 Method 1: Attention-based gating (naturally built-in)
   The attention mechanism ALREADY gates memory influence.
   If memory keys don't align with the current query,
   their attention weights will be near-zero.
   "2 + 2" query vectors won't match "dock_episode" keys.
   This is why KV injection is self-gating for well-trained
   memory projections.

 Method 2: Explicit scalar gate (additional safety)
   At each layer i:
     memory_contribution = alpha_i * CrossAttn(Q=tokens, K=mem, V=mem)
     hidden = hidden + memory_contribution

   alpha_i is a learned scalar (sigmoid output, range 0-1).
   During training, alpha_i learns WHEN memory helps vs hurts.

   Some layers may learn alpha ≈ 0 (ignore memory at this depth).
   Others may learn alpha ≈ 0.3 (moderate memory influence).

 Method 3: Per-head gating
   Each attention head h at layer i has its own gate g_h_i.
   This lets the model learn:
   - Head 3 at layer 12: attends heavily to memory (g = 0.8)
   - Head 7 at layer 12: ignores memory entirely (g = 0.05)

   This aligns with the "attention heads specialize" finding:
   some heads learn syntax, others learn semantics, others
   learn positional patterns. Memory should only influence
   the heads whose function benefits from personal context.

 +------------------------------------------------------------+
 | GATING ARCHITECTURE FOR LLM-WITHMEM                         |
 |============================================================|
 |                                                             |
 |  Per-layer KV injection with per-head gating:               |
 |                                                             |
 |  For layer i, head h:                                       |
 |    K_ext = concat(K_tokens, g_h_i * K_memory)               |
 |    V_ext = concat(V_tokens, g_h_i * V_memory)               |
 |                                                             |
 |    g_h_i = sigmoid(w_gate_h_i)  (learned per head per layer)|
 |                                                             |
 |  Total gate parameters: num_layers * num_heads              |
 |  Example: 32 * 32 = 1024 scalar parameters (trivial)       |
 +------------------------------------------------------------+
```

### What the LLM's Output Distribution Looks Like With Memory

```text
 HOW MEMORY CHANGES THE FINAL OUTPUT
 =====================================

 After all layers, the final hidden state is projected to vocabulary:

   logits = W_vocab * hidden_N       shape: [vocab_size]
   probs = softmax(logits)

 WITHOUT memory (generic LLM):
   P("What if I hadn't switched docks?") ->
   Top tokens: "I'm", "That's", "Without", "If", ...
   The distribution is GENERIC. Any of these could start any response.

 WITH memory (memory vectors in attention):
   P("What if I hadn't switched docks?" | memory_vectors) ->
   Top tokens: "If", "You'd", "Staying", "The", "East", ...
   The distribution is SHIFTED. "East" gets boosted because the
   memory vectors encode East dock knowledge. "You'd" gets boosted
   because the episodic memory encodes what actually happened.

 The shift happens because:
 1. Memory K,V influenced hidden states at multiple layers
 2. Residual connections preserved that influence to the final layer
 3. The final hidden state now encodes BOTH:
    - What the LLM knows about language/reasoning (frozen weights)
    - What the memory model knows about THIS USER (injected memory)
 4. W_vocab projects this COMBINED representation to token probs

 The probability distribution is personalized. Not by changing
 the model — by changing what it attends to during computation.
```

### The Complete Memory-LLM Connection Architecture

```text
 FULL ARCHITECTURE: MEMORY MODEL -> LLM CONNECTION
 ===================================================

 +================================================================+
 |                     MEMORY MODEL                                |
 |================================================================|
 |                                                                 |
 |  User checkpoint (trained by K0 P03 R9):                        |
 |  +------------------+                                           |
 |  | Memory Bank      |  M vectors, each d_memory dimensions      |
 |  | [m1, m2, ..., mM]|  M = 64 (tunable)                        |
 |  +--------+---------+  d_memory = 256 (tunable)                 |
 |           |                                                     |
 |           v                                                     |
 |  Layer-specific projections (one set per LLM layer group):      |
 |  +------------------+------------------+                        |
 |  | W_proj_k_group_1 | W_proj_v_group_1 |  layers 1-8            |
 |  | W_proj_k_group_2 | W_proj_v_group_2 |  layers 9-16           |
 |  | W_proj_k_group_3 | W_proj_v_group_3 |  layers 17-24          |
 |  | W_proj_k_group_4 | W_proj_v_group_4 |  layers 25-32          |
 |  +------------------+------------------+                        |
 |                                                                 |
 |  Per-head gates (1024 scalars for 32-layer, 32-head model):     |
 |  +--------------------------------------------------+           |
 |  | g[layer][head] = sigmoid(w_gate[layer][head])    |           |
 |  +--------------------------------------------------+           |
 |                                                                 |
 +==================+==============================================+
                    |
                    |  For each layer group g:
                    |    K_mem_g = MemoryBank * W_proj_k_g   [M, d_head]
                    |    V_mem_g = MemoryBank * W_proj_v_g   [M, d_head]
                    v
 +================================================================+
 |                     FROZEN LLM                                  |
 |================================================================|
 |                                                                 |
 |  Input: tokenized query [t1, t2, ..., tN]                       |
 |                                                                 |
 |  Layer 1-8 (early layers, syntactic):                           |
 |    K = [K_tokens ; g[l][h] * K_mem_group1]                      |
 |    V = [V_tokens ; g[l][h] * V_mem_group1]                      |
 |    (memory influence: LOW — syntax doesn't need personal data)  |
 |                                                                 |
 |  Layer 9-16 (middle layers, entity/semantic):                   |
 |    K = [K_tokens ; g[l][h] * K_mem_group2]                      |
 |    V = [V_tokens ; g[l][h] * V_mem_group2]                      |
 |    (memory influence: HIGH — entities, facts, relationships)    |
 |                                                                 |
 |  Layer 17-24 (late-middle, reasoning/causal):                   |
 |    K = [K_tokens ; g[l][h] * K_mem_group3]                      |
 |    V = [V_tokens ; g[l][h] * V_mem_group3]                      |
 |    (memory influence: HIGH — causal chains, counterfactuals)    |
 |                                                                 |
 |  Layer 25-32 (final layers, generation/decision):               |
 |    K = [K_tokens ; g[l][h] * K_mem_group4]                      |
 |    V = [V_tokens ; g[l][h] * V_mem_group4]                      |
 |    (memory influence: MODERATE — output steering)               |
 |                                                                 |
 |  Final: logits = W_vocab * hidden_32                            |
 |         probs = softmax(logits)                                 |
 |         -> Personalized token distribution                      |
 +================================================================+
```

### Parameter Budget Summary

```text
 WHAT WE TRAIN (Memory Model — per user checkpoint)
 ====================================================

 Component                    Size                  Notes
 ---------                    ----                  -----
 Memory Bank                  M * d_memory          64 * 256 = 16,384
 KV projections (4 groups)    4 * 2 * d_mem * d_head  4*2*256*128 = 262,144
 Per-head gates               L * H                 32 * 32 = 1,024
 Memory Encoder (optional)    ~2-5M params          small transformer/MLP
                                                    that encodes user data
                                                    INTO the memory bank
 ------------------------------------------------------------------
 TOTAL (excluding encoder):   ~280K parameters  (~1.1 MB at fp32)
 TOTAL (including encoder):   ~2-5M parameters  (~8-20 MB at fp32)
 TOTAL (fp16 checkpoint):     ~4-10 MB per user

 WHAT WE DON'T TRAIN:
 - LLM weights: 7B-70B+ parameters (FROZEN)
 - LLM tokenizer, embeddings, output projection (FROZEN)
 - Nothing in the LLM changes. Ever.

 The memory model is <0.1% of the LLM's size.
 It's a tiny learned adapter that changes WHAT the LLM
 attends to, not HOW it processes information.
```

### Prior Art and Literature

```text
 RELEVANT RESEARCH
 ==================

 1. PREFIX TUNING (Li & Liang, 2021)
    - Prepend learned "prefix" vectors to each layer's K,V
    - Task-specific, not user-specific
    - Shows that small prefix vectors can steer large frozen models
    - OUR EXTENSION: user-specific, continuously updated prefixes

 2. P-TUNING v2 (Liu et al., 2022)
    - Deep prompt tuning: learned prompts at every layer
    - Competitive with full fine-tuning on NLU tasks
    - Shows depth matters: shallow prompts insufficient
    - OUR EXTENSION: memory-derived (not task-derived) deep prompts

 3. MEMORIZING TRANSFORMERS (Wu et al., 2022)
    - External memory bank with kNN retrieval at attention time
    - Token-level memory (stores past hidden states)
    - Shows attention can query external K,V effectively
    - OUR EXTENSION: learned compressed memory (not raw hidden states)

 4. FLAMINGO (Alayrac et al., 2022)
    - Cross-attention for visual memory injection into frozen LLM
    - Perceiver Resampler compresses visual tokens
    - Shows cross-attention fusion at scale works
    - OUR EXTENSION: personal memory instead of visual memory

 5. LoRA (Hu et al., 2022)
    - Low-rank adaptation of attention weights
    - Shows a few million parameters can steer large models
    - BUT: modifies weights (not compatible with "frozen LLM")
    - OUR DIFFERENCE: we inject K,V, not modify W_q/W_k/W_v

 6. ADAPTER LAYERS (Houlsby et al., 2019)
    - Small bottleneck layers inserted between transformer blocks
    - Shows small adapters can specialize large frozen models
    - OUR DIFFERENCE: we use attention KV, not bottleneck layers

 7. SOFT PROMPTS / PROMPT TUNING (Lester et al., 2021)
    - Learn a small set of continuous vectors prepended to input
    - Only at input layer (shallow)
    - Shows even input-level injection can work for simpler tasks
    - OUR EXTENSION: deep (per-layer) + memory-derived

 8. JEPA — JOINT EMBEDDING PREDICTIVE ARCHITECTURE (LeCun, 2022;
    Assran et al., 2023; Bardes et al., 2024)
    - Yann LeCun's proposed path to Advanced Machine Intelligence (AMI)
    - Core idea: predict REPRESENTATIONS of missing input, not raw
      pixels/tokens. Operates in abstract latent space.
    - Three components: Context Encoder, Target Encoder (EMA of
      context), Predictor that maps context embedding -> target embedding
    - I-JEPA (images): mask large blocks, predict their representations.
      632M-param ViT, <72 hours on 16 A100s, SOTA low-shot classification.
    - V-JEPA (video): mask spatio-temporal regions, predict abstract
      representations. Described as "early physical world model."
    - Key property: frozen backbone + tiny attentive probe per task.
      Pre-train once, freeze, reuse across many downstream tasks.
    - Self-supervised: no labels needed for pre-training.
    - OUR CONNECTION: same principles applied to PERSONAL MEMORY.
      JEPA = perception world model. We = personal memory model.
      Both: abstract representations > raw data, frozen backbone +
      small adapter, learned prediction in latent space.
    - OUR EXTENSION: JEPA-inspired self-supervised training objective
      for the memory encoder (mask memories, predict representations).

 WHAT'S NOVEL IN OUR APPROACH:
 - User-specific (not task-specific) memory vectors
 - Continuously trained (not one-shot)
 - 8-layer cognitive architecture mapped to vector banks
 - Layer-specific projections with per-head gating
 - Complementary to text recall (not a replacement)
 - Trained from consolidated observation-strengthened data (K0 P03)
 - JEPA-inspired: abstract-space prediction over raw data reconstruction
```

### JEPA and Our Memory Model: The Structural Parallels

Yann LeCun's JEPA (Joint Embedding Predictive Architecture, 2022) has
become one of the most discussed frameworks in AI. People are excited
because LeCun positions it as **the alternative to autoregressive LLMs
for achieving true machine intelligence.** The parallels to our memory
model are deep and worth examining.

```text
 WHAT IS JEPA?
 ==============

 LeCun's taxonomy of self-supervised architectures:

 (a) Joint-Embedding (contrastive):
     Input x --> Encoder --> Embedding_x
     Input y --> Encoder --> Embedding_y
     Objective: make embeddings similar for compatible x,y
     Problem: can collapse (all inputs map to same embedding)

 (b) Generative (LLMs, diffusion models):
     Input x --> Decoder --> Reconstruct y (pixels/tokens)
     Objective: predict EVERY missing pixel/token
     Problem: wastes capacity on irrelevant detail (every leaf
     on a tree, every pixel of background noise)

 (c) JEPA (LeCun's proposal):
     Input x --> Context Encoder  --> Embedding_x
     Input y --> Target Encoder   --> Embedding_y (the target)
                Predictor(Embedding_x, z) --> Predicted_Embedding_y
     Objective: predict the REPRESENTATION of y, not y itself
     Advantage: learns to ignore unpredictable noise, focuses
     on high-level semantic structure

 The three components:
 +-------------------+    +-------------------+
 | Context Encoder   |    | Target Encoder    |
 | (processes visible|    | (processes masked  |
 |  part of input)   |    |  part of input)   |
 +--------+----------+    +--------+----------+
          |                         |
          v                         v
   context_embedding          target_embedding
          |                         |
          v                         |
 +-------------------+              |
 | Predictor         |              |
 | (predicts target  |--> predicted_embedding
 |  from context)    |              |
 +-------------------+              |
                                    |
     Loss = distance(predicted_embedding, target_embedding)

 Target Encoder weights = EMA(Context Encoder weights)
 (exponential moving average — no gradient, just follows)
```

```text
 WHY PEOPLE ARE SCREAMING ABOUT JEPA
 =====================================

 1. LeCun says autoregressive LLMs are a dead end for AGI.
    JEPA is his counter-proposal. This is a PARADIGM argument.

 2. "World Model" framing. The predictor learns to model how
    the world works — spatial relationships, object permanence,
    physical dynamics — by predicting abstract representations.
    An infant learns this way: observe, model, predict.

 3. Frozen backbone + tiny adapter. V-JEPA's killer feature:
    pre-train encoder once, freeze it, train only a tiny
    "attentive probe" for each downstream task. The backbone
    is reusable across many tasks without fine-tuning.

 4. Efficiency. I-JEPA trains 2-10x faster than competing
    methods because predicting representations is cheaper
    than reconstructing every pixel.

 5. Avoids the "irrelevant detail" trap. Generative models
    waste capacity on every pixel of every leaf. JEPA learns
    to discard unpredictable noise and focus on semantics.

 6. Meta is shipping it: I-JEPA (2023, images), V-JEPA (2024,
    video), with audio and multimodal extensions in progress.
    This is not theoretical — there is working code and models.
```

```text
 THE 5 STRUCTURAL PARALLELS TO OUR MEMORY MODEL
 ================================================

 PARALLEL 1: ABSTRACT REPRESENTATION OVER RAW DATA
 --------------------------------------------------
 JEPA:  Predicts REPRESENTATIONS of masked image/video regions.
        Not pixels. Abstract embeddings in latent space.

 US:    Memory model encodes memories as VECTORS, not text.
        Not tokens. Abstract embeddings in latent space.

 SHARED PRINCIPLE: Operating in learned latent space is strictly
 superior to operating on raw data. Raw data wastes capacity on
 irrelevant detail.

 This IS our "7 Hard Limits" argument:
 - Text recall (recall_memory) = generative approach (raw tokens)
 - Memory vectors = JEPA-like approach (abstract representations)


 PARALLEL 2: FROZEN BACKBONE + LIGHTWEIGHT ADAPTER
 --------------------------------------------------
 JEPA:  Freeze the encoder. Train only a tiny attentive probe
        per downstream task. The encoder never changes.

 US:    Freeze the LLM. Train only the memory model + KV
        projections per user. The LLM never changes.

 SHARED PRINCIPLE: A powerful frozen foundation + small learned
 adapters = flexible, efficient, scalable. Don't retrain the
 whole model for each new task/user.


 PARALLEL 3: THE PREDICTOR AS WORLD MODEL
 --------------------------------------------------
 JEPA:  The predictor takes partial context and predicts what's
        missing — a "primitive world model" that understands
        spatial/temporal structure of the physical world.

 US:    The memory model takes user history and produces vectors
        that represent "who this person is" — a primitive
        PERSONAL world model that understands the user's life
        structure (episodes, beliefs, emotions, goals).

 JEPA's predictor: "given this half of the image, what SHOULD
 be in the other half?" (general world knowledge)

 Our memory model: "given this user's history, what CONTEXT
 should the LLM have?" (personal world knowledge)


 PARALLEL 4: SELF-SUPERVISED TRAINING OBJECTIVE
 --------------------------------------------------
 JEPA:  Given visible context, predict target representations.
        No labels needed. Self-supervised from raw data.

 US:    Could use the SAME objective! Given partial memory
        context, predict the representation of held-out
        memories. The K0 P03 pipeline provides the data;
        the training objective could be JEPA-inspired.

        Concrete design:
        - Take a user's consolidated memory atoms (from P03)
        - Mask 30% of memory atoms randomly
        - Encode the remaining 70% with the context encoder
        - Train predictor to reconstruct the REPRESENTATIONS
          (not the text) of the masked 30%
        - This teaches the model to capture cross-memory
          relationships, temporal patterns, causal links,
          and emotional trajectories — because predicting
          one memory's representation REQUIRES understanding
          how it relates to the others


 PARALLEL 5: COMPLEMENTARY COGNITIVE ROLES
 --------------------------------------------------
 JEPA:  Perception — understanding sensory input from the world.
        "What is happening in this scene?"

 US:    Memory — representing personal history and knowledge.
        "Who is this person and what have they experienced?"

 These are DIFFERENT parts of the SAME cognitive puzzle.
 Both are needed for a complete intelligent system.
```

```text
 LECUN'S FULL COGNITIVE ARCHITECTURE (2022 paper)
 =================================================

 "A Path Towards Autonomous Machine Intelligence"

 +--------------------------------------------------+
 |  Module              | Role                       |
 +--------------------------------------------------+
 |  Perception Module   | Understand sensory input   |
 |  World Model         | Predict future states      |
 |  Cost Module         | Evaluate desirability      |
 |  Actor Module        | Take actions               |
 |  Short-term Memory   | Track current state        |
 |  Configurator        | Modulate all modules       |
 +--------------------------------------------------+

 NOTICE: Memory is a FIRST-CLASS component in LeCun's
 architecture. JEPA is the PERCEPTION piece. Our memory
 model is the PERSONAL MEMORY piece.


 MAPPING TO OUR ARCHITECTURE (FamilyOS + LLM-WITHMEM)
 =====================================================

 LeCun's Module      | Our Implementation
 ---------------------+---------------------------------------
 Perception Module    | UltraBERT + SessionState (K1)
 World Model          | LLM (frozen, general world knowledge)
 Personal Memory      | Memory Model (THIS PROJECT)
 Cost/Actor           | Concierge FSM (K1)
 Short-term Memory    | SessionState HOT section
 Long-term Memory     | K0 P03 consolidated memories
 Configurator         | Concierge FSM mode switching

 We are building one piece of LeCun's cognitive architecture,
 applied to personal AI rather than physical robotics.
```

```text
 KEY DIFFERENCES (where we diverge from JEPA)
 ==============================================

 1. JEPA = PERCEPTION. We = MEMORY.
    JEPA learns to understand visual/sensory input.
    We learn to represent personal history and knowledge.

 2. JEPA predicts MISSING parts of CURRENT input.
    We provide ADDITIONAL context that was NEVER in the input.
    JEPA fills gaps. We add a new dimension entirely.

 3. JEPA is self-contained (encoder + predictor = the model).
    We are a SIDE-CHANNEL into an existing frozen LLM.
    JEPA IS the model. We AUGMENT a model.

 4. JEPA learns general world knowledge.
    We learn USER-SPECIFIC knowledge.
    JEPA: "what does a dog's head look like?"
    We: "what happened at the dock last March?"

 5. JEPA's target is the encoder's own representation space.
    Our target is the LLM's attention space (K,V pairs).
    Different target spaces, different projection requirements.

 6. JEPA trains once on a large dataset, then freezes.
    We train CONTINUOUSLY as new memories arrive.
    JEPA: train -> freeze -> deploy.
    Us: train -> deploy -> keep training -> update -> ...
```

```text
 JEPA-INSPIRED TRAINING OBJECTIVE FOR MEMORY ENCODER
 =====================================================

 The most actionable takeaway from JEPA: we can adopt a
 similar self-supervised objective for training the memory
 encoder on K0's consolidated data.

 PROPOSED OBJECTIVE: Memory-JEPA
 ================================

 Input: user's consolidated memory atoms from K0 P03
        (episodic L4, semantic L5, procedural L6,
         emotional L7, prospective L8)

 Step 1: MASK
   Randomly mask 30% of memory atoms.
   Masking strategy matters (lesson from I-JEPA):
   - Don't mask random individual atoms (too easy)
   - Mask TEMPORAL BLOCKS (entire weeks/months)
   - Mask THEMATIC CLUSTERS (all work-related memories)
   - Mask CROSS-LAYER groups (episode + its emotion + its cause)
   This forces the model to learn deep structure.

 Step 2: ENCODE CONTEXT
   Feed remaining 70% of atoms through the memory encoder.
   Output: context representation c = Encoder(visible_atoms)

 Step 3: PREDICT TARGETS
   Feed masked atoms through the target encoder (EMA).
   Output: target representations t_i = TargetEnc(masked_atom_i)
   Predictor: predicted_t_i = Predictor(c, position_info_i)

 Step 4: LOSS
   L = mean_i || predicted_t_i - t_i ||^2
   (L2 distance in representation space)

   VICReg regularization to prevent representation collapse:
   - Variance term: ensure each dimension has variance > threshold
   - Invariance term: the prediction loss above
   - Covariance term: decorrelate dimensions of the embedding

 WHY THIS IS BETTER THAN RECONSTRUCTION LOSS:
   Reconstruction: "predict the text of the masked memory"
   -> Wastes capacity on exact wording, punctuation, filler words
   -> Doesn't learn that "dock switch" and "job change" are related

   JEPA-style: "predict the REPRESENTATION of the masked memory"
   -> Learns that dock_episode and job_change occupy similar
      regions in latent space (both = major life transitions)
   -> Captures structural relationships, not surface text
   -> The representation naturally encodes causality, emotion,
      temporal proximity — because those are needed to PREDICT
      one memory's representation from the others

 TRAINING DATA SOURCE:
   K0 P03 pipeline output (R0-R8 phases)
   Each user accumulates thousands of consolidated memory atoms.
   New R9 phase: periodically retrain memory encoder using
   Memory-JEPA objective on the latest atom collection.
```

---

### Open Design Questions

These are the hard problems this project needs to solve. Grouped by area.
Questions marked **(RESOLVED)** have been answered by the FamilyOS
integration analysis and the 8-layer architecture design.

---

#### A. Memory Model Architecture

1. **What kind of network is the memory model?** **(RESOLVED)**
   - Answer: It is a SYSTEM of components, not one network. A salience filter (L1),
     session buffer (L2), attention-based memory retriever (L3), 5 specialized
     memory banks (L4-L8), a consolidation pipeline, and a memory vector composer.
     FamilyOS mapping: L1 = UltraBERT, L2-L3 = SessionState, L4-L8 = K0 P03 output
     encoded as neural embeddings.
   - Original options (kept for reference):
     - A small transformer that re-encodes user history each time?
     - A recurrent network (state-space model, LSTM) that maintains a running hidden state?
     - A set of learnable embedding vectors (like soft prompts) that get gradient-updated?
     - A graph neural network over a knowledge graph of user facts?
     - Some hybrid: fixed memory slots + a small encoder that routes information into them?

2. **How many parameters should it have?** **(RESOLVED)**
   - Answer: ~5-15M parameters total. L1-L3 lightweight (~1M). L4-L8 bulk:
     K=64 slots x d=512 dims x 5 layers = 163K params + encoder/projection ~5-10M.
     Checkpoint size: ~20-60MB per user. Fits Bridge transport easily.
   - Original options (kept for reference):
     - Too small and it cannot represent a rich user. Too large and per-user storage explodes.
     - Target range: 1M-50M parameters? Or even smaller if using pure embedding slots?
     - What is the relationship between parameter count and how many distinct facts it can encode?

3. **What is the memory bank structure?** **(RESOLVED)**
   - Answer: 5 separate banks (L4-L8), each with different learning rates,
     update frequencies, slot management policies, and capacity limits.
     Hierarchical: fast-updating episodic (L4) + slow-updating procedural (L6).
   - Original options (kept for reference):
     - Fixed number of slots (e.g. K=64 vectors) that get overwritten?
     - Growing memory that adds new slots over time?
     - Hierarchical: short-term slots (recent turns) + long-term slots (persistent facts)?
     - How does the model decide which slot to update vs. which to keep stable?

4. **Does the memory model have its own tokenizer/embedding or does it share the LLM's?**
   - Sharing means the memory encoder operates in the same representation space.
   - Separate means it is fully independent but needs a projection layer to bridge.

5. **How is user history represented as input to the memory model?** **(RESOLVED)**
   - Answer: Pre-extracted structured data from K0 P03 output:
     clustered episodes (R2), importance scores (R1), KG triples (R4),
     Granger causality edges (R4), emotional annotations (R1),
     R5 observation-strengthened memory layers (EST salience boosts,
     SRE social enrichment, SPR semantic confidence, ASU anchors,
     MTP tier promotions, NTD narrative threads), and SPC-UQ cleaned
     prospective items. Not raw text — K0 already consolidates.
     CRITICAL: No counterfactual training data. CPN/MCTS are dead.
     Counterfactual reasoning is the LLM's job at query time.
   - Original options (kept for reference):
     - Raw conversation text?
     - Pre-extracted fact tuples (subject, relation, object)?
     - Embedding summaries from previous sessions?
     - All of the above, fused together?

---

#### B. Fusion: How the Memory Model Connects to the LLM

1. **At which point in the LLM forward pass does memory enter?**
   - Before the first layer (prepended virtual tokens)?
   - At every layer (cross-attention or key-value injection)?
   - At specific layers only (e.g. early layers for factual recall, late layers for style)?
   - After the LLM output (re-ranking or logit adjustment)?

2. **What is the exact attention mechanism?**
   - **Prefix fusion**: memory vectors are prepended to the KV cache so the LLM self-attention naturally attends to them. No LLM architecture change needed.
   - **Cross-attention fusion**: add a cross-attention sublayer in each (or some) decoder blocks where Q comes from the LLM and K,V come from memory. Requires modifying the LLM forward pass.
   - **Gated fusion**: a small gating network decides per-layer how much memory signal to mix into the hidden state. Similar to adapter or LoRA injection points.
   - **Logit bias**: memory model outputs a bias vector over the vocabulary that gets added to the LLM's logits before softmax. Simplest, but limited expressiveness.

```text
FUSION OPTIONS COMPARED

Option 1: Prefix Fusion (no LLM modification)
+--------------------------------------------------+
| KV Cache:  [m1, m2, ..., mK, t1, t2, ..., tN]   |
|            ^memory vectors^   ^token vectors^     |
| LLM self-attention sees memory as extra "tokens"  |
| Pro: works with any frozen LLM, even API-only     |
| Con: consumes virtual positions in the KV cache   |
+--------------------------------------------------+

Option 2: Cross-Attention Injection (requires model surgery)
+--------------------------------------------------+
| LLM hidden state h_i                              |
|   |                                               |
|   v                                               |
| [Self-Attn] --> [Cross-Attn over memory] --> FFN  |
|                  Q = h_i                          |
|                  K,V = memory bank [m1..mK]       |
| Pro: clean separation, memory never eats tokens   |
| Con: must insert layers, can't do with API models |
+--------------------------------------------------+

Option 3: Gated Hidden-State Modulation
+--------------------------------------------------+
| At layer i:                                       |
|   h_i' = h_i + gate_i * project(memory_signal)   |
|   gate_i is learned per-layer (scalar or vector)  |
| Pro: lightweight, similar to LoRA injection       |
| Con: still requires access to LLM internals       |
+--------------------------------------------------+

Option 4: Logit Bias (output-level only)
+--------------------------------------------------+
| logits_final = LLM_logits + memory_bias_vector   |
| Pro: trivially compatible with any LLM or API    |
| Con: can only shift token probabilities, cannot   |
|      steer intermediate reasoning or attention    |
+--------------------------------------------------+
```

1. **Can multiple fusion strategies coexist?**
   - Use prefix fusion for API-based LLMs where you have no model internals.
   - Use cross-attention or gated fusion for local open-weight models.
   - The memory model produces the same vectors; only the injection point differs.

---

#### C. Training Signal and Loss

1. **What is the primary loss function?**
   - Next-token prediction: given the user's past, the memory model should help the LLM predict what the user says next better than without memory.
   - But that requires running the full LLM in the training loop — is that feasible?

2. **Can the memory model be trained without forward-passing the LLM?**
    - Distillation: use a frozen LLM to generate target hidden states, then train the memory model to produce vectors that, when injected, reproduce those targets.
    - Contrastive: memory vectors for the correct user should be closer (in some metric) to the LLM's actual hidden states than memory vectors for a random user.
    - Reconstruction: train the memory encoder to reconstruct user facts from the memory bank (autoencoder-style). No LLM in the loop at all.

3. **What counts as a "good" training signal from a single conversation?**
    - User corrections ("No, I prefer X over Y") are very strong signal.
    - User asking the same question again implies the model forgot — negative signal.
    - Explicit preference statements.
    - Implicit preferences (topic choices, tone, vocabulary).
    - How to weight these differently?

---

#### D. Single-Event Learning: Can One Interaction Move Weights?

1. **Can a single chat turn be strong enough to drive meaningful weight movement?**
    - In standard deep learning, one sample barely moves weights. But here:
      - The memory model is tiny (few million params), so a single gradient step has proportionally more impact.
      - The learning rate can be set aggressively for the memory model without destabilizing the LLM (which stays frozen).
      - A single correction like "I'm allergic to peanuts" should be learnable from one exposure.

2. **How to handle one-shot critical facts vs. gradual preference learning?**
    - Some facts are binary and must be captured immediately (allergies, name, job).
    - Some patterns emerge only over many interactions (writing style, humor preference).
    - Possible approach: dual learning rate or dual memory bank:

```text
DUAL MEMORY APPROACH

+===========================+     +===========================+
| FAST MEMORY               |     | SLOW MEMORY               |
| (high learning rate)      |     | (low learning rate)        |
|                           |     |                           |
| Captures:                 |     | Captures:                 |
| - explicit corrections    |     | - writing style           |
| - stated facts            |     | - topic preferences       |
| - one-shot critical info  |     | - behavioral patterns     |
|                           |     | - gradual taste shifts    |
| Updated: every turn       |     | Updated: every N sessions |
| Risk: volatile, may flip  |     | Risk: slow to adapt       |
+===========================+     +===========================+
         |                                  |
         +----------------+-----------------+
                          |
                    [Merge / Gate]
                          |
                          v
                  [Final Memory Vectors]
                  sent to LLM at inference
```

1. **What prevents a single bad interaction from corrupting the model?**
    - Gradient clipping on the memory model updates.
    - Validation against a small replay buffer of past interactions.
    - Confidence thresholding: only update when the signal is unambiguous.
    - Versioned checkpoints: if quality degrades, roll back.

---

#### E. Continual Learning and Catastrophic Forgetting

1. **How to avoid forgetting old memories when learning new ones?** **(RESOLVED)**
   - Answer: Different layers forget at different rates (by design):
     L4 episodic: recency decay. L5 semantic: overwrites only on strong contradiction.
     L6 procedural: almost never changes (very low LR). L7 emotional: gradual drift.
     L8 prospective: explicit completion/expiry. Plus K0 has the ground truth —
     if the memory model degrades, retrain from K0 consolidated data.
   - Original options (kept for reference):
     - Elastic Weight Consolidation (EWC): penalize changes to weights that were important for past memories.
     - Experience replay buffer: keep a small set of past interactions and mix them into each training step.
     - Progressive memory slots: old facts get "frozen" slots, new facts go into "active" slots.
     - Mixture of memory timescales (see dual memory above).

2. **How to handle contradictions?** **(RESOLVED)**
   - Answer: K0 P03 R4 KG consolidator already handles this via NER
     disambiguation and Hebbian edge weights. The relational context dimension
     (contradicts/extends/qualifies/replaces) resolves conflicts before
     they reach the memory model. The memory model trains on already-resolved data.
   - Original options (kept for reference):
     - User says "I live in NYC" in January, "I moved to London" in March.
     - The model needs to overwrite, not average. How?
     - Timestamp-aware memory slots? Most-recent-wins policy? Explicit invalidation signal?

3. **What is the memory model's effective capacity?**
    - How many distinct facts can K memory vectors of dimension d encode?
    - Is there a theoretical limit, or does it degrade gracefully?
    - When capacity is reached, what gets forgotten first?

---

#### F. LLM Compatibility and Portability

1. **How to make the same memory model work across different LLMs?**
    - Each LLM has a different hidden dimension (e.g. 4096 for Llama-7B, 5120 for Llama-13B, 12288 for GPT-4 scale).
    - Options:
      - Train a lightweight projection layer per LLM family.
      - Use a universal intermediate dimension and project on the fly.
      - Standardize on prefix fusion (works with any model, even API-only).

2. **What about API-only LLMs where you have zero access to internals?**
    - Prefix fusion still works: memory vectors become carefully crafted "soft prompt" tokens that get prepended.
    - Logit bias works if the API exposes it (OpenAI does).
    - Cross-attention and gated fusion are impossible without model weights.
    - This is a critical architectural fork: design for open-weight models, API models, or both?

3. **Can the memory model survive LLM version upgrades?**
    - If the user switches from Llama-3 to Llama-4, do their memories transfer?
    - If using prefix fusion: the memory vectors may be in the wrong representation space.
    - Possible: retrain just the projection layer on a small calibration set.

---

#### G. Privacy, Isolation, and User Control

1. **Each user has their own model checkpoint. How is this managed?** **(RESOLVED)**
   - Answer: Option B — trained in K0, synced to K1. K0 stores the authoritative
     checkpoint alongside all other user data (PostgreSQL). K1 holds inference-only
     weights (~20-60MB) synced via Bridge. K0 manages versioning, K1 manages runtime.
   - Original options (kept for reference):
     - One small file per user (~5-50 MB).
     - Stored locally on device? In encrypted cloud storage? Both with sync?

2. **Can the memory model run fully on-device?** **(RESOLVED)**
   - Answer: Yes — inference runs fully on K1 (on-device, edge-first).
     Training runs on K0 (server-side). This matches FamilyOS's Bridge
     offline-awareness principle: K1 works without K0 using last-synced
     weights. Graceful degradation, not failure.
   - Original options (kept for reference):
     - The memory model is tiny. It could run on a phone or laptop.
     - Only the LLM needs a server. The memory model could run client-side and send vectors to the server.
     - This has strong privacy implications: user memories never leave the device.

3. **How does a user inspect what the model has learned?**
    - Memory vectors are opaque (dense floats). Users can't read them.
    - Need an "explain my memory" feature: probe the memory model to extract human-readable facts.
    - Possible: run a decoding pass that converts memory vectors back to natural language descriptions.

4. **How does a user delete a specific memory?**
    - In a database, you delete a row. In a neural network, you can't "delete a weight."
    - Options:
      - Machine unlearning techniques (gradient ascent on the target fact).
      - Maintain a blocklist: even if the model "knows" something, suppress it at inference.
      - Retrain from a filtered replay buffer that excludes the unwanted fact.

5. **Multi-user and shared contexts?** **(RESOLVED)**
   - Answer: FamilyOS is built for families. K0 stores per-family-member data
     with the Family Device Mesh handling multi-user sync. Each family member
     gets their own memory model checkpoint. Shared family context ("we're going
     to Grandma's for Thanksgiving") lives in K0's shared KG and can train
     a family-level base model that individual models overlay.
   - Original options (kept for reference):
     - Can a family share a base memory model with individual overlays?
     - Can a team have a "team memory" that augments individual memories?
     - How to handle access control over shared memory models?

---

#### H. Evaluation and Quality

1. **How do you measure whether the memory model is actually helping?**
    - Perplexity reduction: does the LLM predict user text better with memory than without?
    - Factual recall: ask the model about previously stated user facts. Does it answer correctly?
    - User satisfaction: does the user feel the model "knows" them?
    - A/B test: same prompt, with and without memory fusion. Compare quality.

2. **How to detect memory model degradation over time?**
    - Monitor recall accuracy on a held-out set of known user facts.
    - Track perplexity drift across sessions.
    - Alert if the model starts contradicting established facts.

3. **What is the cold-start experience?**
    - New user, empty memory model. How does it behave?
    - Should it fall back to a generic baseline, or actively ask discovery questions?
    - How many interactions until the memory model becomes meaningfully useful?

---

#### I. System and Infrastructure

1. **What is the latency budget for memory model inference?**
    - The memory model must produce vectors BEFORE the LLM starts generating.
    - Target: <50ms additional latency on top of normal LLM inference.
    - Is this achievable with a small model on a GPU/CPU?

2. **How to handle the continuous training loop in production?** **(RESOLVED)**
   - Answer: Asynchronous, K0-side. P03 R9 runs nightly after consolidation.
     Current session always uses last-synced weights (may be up to ~24h stale).
     This is acceptable: the memory model captures long-term patterns, not
     within-session state (that's SessionState's job). No concurrent-session
     problem because training is batch, not per-request.
   - Original options (kept for reference):
     - Synchronous (block until model is updated) or asynchronous (update in background)?
     - Async is better for latency but means the current session may use stale memory.
     - How to handle concurrent sessions from the same user?

3. **Checkpoint versioning and rollback?** **(RESOLVED)**
   - Answer: K0 handles this natively. P03 already maintains st_consolidation_audit
     for audit trails. Memory model checkpoints get the same treatment: versioned
     in K0, N recent kept, auto-rollback if quality metrics degrade.
     Storage cost: N * 20-60MB per user per family member. Trivial for K0.
   - Original options (kept for reference):
     - Keep N recent checkpoints per user.
     - If quality degrades, auto-rollback to last known good checkpoint.
     - Storage cost: N * 5-50 MB per user. Manageable?

4. **Can the training loop run on the same hardware as inference?** **(RESOLVED)**
   - Answer: Training runs on K0 (server-side), inference runs on K1 (edge).
     They are on different hardware by architecture. No GPU contention.
     K0 can schedule R9 training during off-peak hours (nightly P03 run).
   - Original options (kept for reference):
     - Memory model training is lightweight. Could share the GPU with LLM inference.
     - Or offload training to CPU while GPU handles LLM inference.
     - Or train on a separate background worker entirely.

#### J. JEPA-Inspired Training

1. **Should we adopt a Memory-JEPA self-supervised objective for the encoder?**
    - Mask 30% of memory atoms, predict their representations (not text).
    - Masking strategies: temporal blocks, thematic clusters, cross-layer groups.
    - Pros: learns cross-memory relationships, causal links, temporal patterns
      without needing explicit labels. Avoids wasting capacity on surface text.
    - Cons: adds design complexity. Need VICReg or similar to prevent collapse.
    - Alternative: standard reconstruction loss on memory embeddings.
    - Decision needed before P03 R9 Trainer design.

2. **What masking strategy produces the best memory representations?**
    - Random individual atoms (easy, shallow learning).
    - Temporal blocks (mask entire weeks/months \u2014 forces temporal reasoning).
    - Thematic clusters (mask all work-related \u2014 forces cross-domain links).
    - Cross-layer groups (mask episode + its emotion + its cause \u2014 forces
      multi-layer integration).
    - Likely needs ablation study to determine optimal strategy.

3. **Should the target encoder use EMA (like JEPA) or shared weights?**
    - EMA (exponential moving average) prevents collapse without negative samples.
    - Shared weights are simpler but may need explicit regularization.
    - JEPA uses EMA. We should probably follow unless there is a strong reason not to.

## Whiteboard Summary

- A transformer uses embeddings, positional information, self-attention, and feed-forward blocks.
- LLMs are large transformer-based language models trained with next-token prediction at scale.
- Most modern chat and text-generation systems use decoder-only transformers.
- Real-world LLM products need more than the model itself.
- **The core vision of this project is NOT a retrieval database.** It is a small, continuously-trained memory model that produces learned representations and fuses them into the LLM's attention at inference time.
- The memory model is plug-and-play: the LLM stays frozen, the memory model is the only thing that trains.
- The output shifts from generic next-token probabilities to **personalized** next-token probabilities conditioned on who the user is.
- This is closer to giving each user their own neural adapter than to pasting retrieved text into a prompt.
- **The memory architecture mirrors human cognition:** 8 layers (Sensory, Short-Term, Working, Episodic, Semantic, Procedural, Emotional, Prospective) with different learning rates and capacities.
- **Every memory entry needs 9 context dimensions** (Temporal, Spatial, Narrative, Causal, Social, Emotional, Epistemic, Identity, Relational) to avoid dangerous misinterpretation.
- **FamilyOS integration (Option B: K0 trains, K1 infers):**
  - FamilyOS already has the data infrastructure for 6 of 8 memory layers.
  - L1-L3 = UltraBERT + SessionState (already built).
  - L4-L8 data = K0 P03 consolidation pipeline output (already built).
  - 6 of 9 context dimensions = UltraBERT classification heads (already built).
  - 3 of 9 context dimensions = Front LLM cognitive tools (already built).
  - Net-new work: MemoryFusedAdapter (K1), P03 R9 Trainer (K0), Bridge weight sync.
  - The memory model plugs in at the IConciergeModelPort adapter layer — zero changes to FSM, Bus, Front/Back actors, tools, prompt builder, safety bands, or any other component.
- **P03 R5 redesign: consolidation vs reasoning boundary:**
  - Counterfactual reasoning ("what if") is a REASONING task that belongs at query time in K1's LLM, NOT in K0's offline batch pipeline.
  - CPN and MCTS are dead on arrival (CPN produces nonsensical text, R4 produces 0 causal edges).
  - R5 redesigned with 14 observation-driven memory strengthening algorithms.
  - Memory model trains on observation-strengthened data, not speculative counterfactuals.
  - L8 prospective memory = intent tracking + proactive triggers, NOT imagining futures.
- **Memory Writer v2: correct signals at the source:**
  - Current MW v1 produces minimally tagged text; P02 recomputes with UltraBERT on 8 tokens.
  - Broken signals: social_context always "friends", salience flat at 0.41, family flags always false.
  - MW v2 reads all 15 SessionState sections, produces 34-field memory atoms with per-extraction affect.
  - All 9 context dimensions mapped to concrete MW v2 fields.
  - Memory model training quality is gated by MW v2 quality — this is the foundational fix.
- **How the memory model drives counterfactual reasoning and advanced cognition:**
  - The memory model does NOT reason. It provides the complete factual world as dense vectors.
  - The LLM already knows HOW to reason (pre-trained on billions of causal/counterfactual text). It lacks the personal facts to reason ABOUT.
  - Memory vectors give the LLM: episodes (L4), semantic knowledge (L5), causal chains (L6), emotional valence (L7), and active goals (L8) — all as native attention key-value pairs.
  - Five unlocked capabilities: counterfactual ("what if"), causal inference ("why"), future simulation ("what would happen"), creative synthesis ("connect these"), personalized advice ("given your history").
  - This is strictly better than algorithmic CPN: neural attention processes all context in parallel, understands language nuance, weights by emotion, orients by goals, and produces text humans can engage with.
  - The memory model turns the LLM from a "smart stranger" into a "smart friend who knows your whole history."
- **Why the memory model is needed when the Concierge already queries K0 memory:**
  - The Concierge's `recall_memory()` tool already fetches text from K0 via Bridge and pastes it into the 128K context window. It works — but within 7 hard limits.
  - Limit 1: Token budget is finite — recalled text competes with system prompt, SessionState, history, tool schemas. Memory vectors cost ZERO tokens.
  - Limit 2: Recall is lossy selection — selectors must guess which memories to return. Memory model encodes ALL memories in compressed vectors.
  - Limit 3: Text is flat, vectors encode structure — recalled paragraphs must be re-derived; memory vectors pre-compute relationships.
  - Limit 4: Cross-memory relationships invisible in text — causal chains spanning 6+ memories are scattered paragraphs; memory model learns them as single pattern vectors.
  - Limit 5: Emotional trajectory lost — text says "frustrated"; L7 vectors encode the 3-month arc from annoyance to anger to relief.
  - Limit 6: Recall adds latency per query (50-200ms network round-trip); memory model inference is local (<50ms, no network).
  - Limit 7: Recall is reactive (LLM must decide to call it); memory model is always-on (injected every turn, no tool call needed).
  - They are COMPLEMENTARY: memory model provides always-on implicit understanding; recall_memory() provides explicit specific facts on demand. Together = understanding + facts.
- **Research: How LLMs and the Memory Layer Connect — Driving Inference Behavior:**
  - There are exactly 4 mathematically distinct injection points for inserting external vectors into a frozen LLM: (1) Prefix/Soft Prompt at input embeddings, (2) Per-Layer KV Injection, (3) Cross-Attention layers, (4) Residual Hidden State Addition.
  - **Per-Layer KV Injection is the primary technique for this project:** zero token cost, maximum expressiveness, layer-specific signals, fully frozen LLM, plug-and-play compatible.
  - The mechanism: at each transformer layer, memory K,V pairs are appended to the token K,V. The LLM's query vectors attend over BOTH token and memory keys; attention weights determine how much each token "listens to" each memory slot.
  - Layer-specific projections let the memory model present DIFFERENT facets of the same memory to different layers (syntax layers get low signal, semantic layers get high signal, reasoning layers get causal signal).
  - Per-head gating (1024 learned scalars for a 32x32 model) controls which attention heads use memory and which ignore it — self-regulating the influence strength.
  - Memory influence propagates through the residual stream: injection at layer i affects ALL subsequent layers, accumulating rather than degrading.
  - The final output is a personalized probability distribution: W_vocab projects the memory-infused hidden state to logits, shifting token probabilities toward personal context without changing any LLM weights.
  - Parameter budget: ~280K params for projections+gates, ~2-5M with encoder. Per-user checkpoint ~4-10MB at fp16. This is <0.1% of the LLM.
  - Fallback for API-only LLMs: Prefix/Soft Prompt method (accepts token cost as tradeoff when layer access is impossible).
  - Prior art grounding: Prefix Tuning, P-Tuning v2, Memorizing Transformers, Flamingo cross-attention, LoRA, Adapter Layers, Prompt Tuning. Our approach extends these with user-specific, continuously-trained, 8-layer cognitive memory mapped to per-layer KV projections.
- **JEPA (Joint Embedding Predictive Architecture) and our Memory Model:**
  - JEPA (LeCun, 2022) predicts REPRESENTATIONS of missing input in abstract latent space, not raw pixels/tokens. People are excited because LeCun positions it as the alternative to autoregressive LLMs for AGI.
  - 5 structural parallels: (1) abstract representation over raw data, (2) frozen backbone + lightweight adapter, (3) predictor as world model, (4) self-supervised training objective, (5) complementary cognitive roles.
  - JEPA = perception world model ("what is in this scene?"). We = personal memory model ("who is this user?"). Different pieces of the SAME cognitive puzzle from LeCun's full architecture.
  - Actionable takeaway: Memory-JEPA training objective — mask 30% of memory atoms, predict their REPRESENTATIONS (not text). Teaches cross-memory relationships, temporal patterns, causal links. Uses masking strategies from I-JEPA: temporal blocks, thematic clusters, cross-layer groups.
  - Key differences: JEPA fills gaps in current input; we add context that was never in the input. JEPA trains once then freezes; we train continuously. JEPA is the model; we augment an existing model.
  - LeCun's cognitive architecture has memory as a first-class module. Our project builds that module, applied to personal AI.
- **Experimental Validation — Level 1 + Level 1.5 Ablation:**
  - Level 1 (Mechanical Proof) PASSED: DynamicCache pre-population with position_ids offset + attention_mask extension. All 5 criteria met. KV injection works mechanically via the production MemoryFusedAdapter pattern.
  - Level 1.5 (Ablation Sweep) completed: 160 runs across 4 scales × 4 layer strategies × 2 gate strategies × 5 prompts.
  - **Critical discovery: SDPA silently drops attention weights.** SmolLM2 defaults to `sdpa` attention, which returns zeros for `output_attentions=True`. Must load with `attn_implementation="eager"` for attention weight inspection.
  - **Attention mass on memory slots is ~50% across all scales** (0.001 to 0.02). Softmax normalizes, so scale affects VALUE magnitude, not attention mass. The model is NOT ignoring memory — it splits attention roughly 50/50 between memory and tokens.
  - **Layer-level attention gradient:**
    - L00: ~20% (early layers token-focused)
    - L01-L14: ~40-55% (mid layers, moderate memory attention)
    - L16-L21: ~60-71% (PEAK — late layers most receptive to memory)
    - L22-L23: ~38-49% (drops at final layers)
    - Peak: L19-L20 at ~70% attention mass on memory positions.
  - **Per-head variance is enormous:** min=0.5% to max=90%+. Some heads almost entirely attend to memory, others barely look at it. This is strong empirical evidence that per-head gating is critical for Level 3.
  - All configs showed DEGRADED coherence (PPL ratio 33-290×). This is EXPECTED: the model gives ~50% attention to RANDOM NOISE. Level 2 with extracted K,V carries actual semantic information and will be coherent.
  - **Recommended Level 2 configuration:** All layers (they all attend), no artificial scaling (extracted K,V are at native magnitude), per-head gating (empirically justified by head variance).
- **Experimental Validation — Level 2: Behavioral Steering:**
  - **Level 2a (raw K,V extraction):** 24% personalization, 100% coherence. Profile K,V from raw text carry location info (city names appear) but not deeper personal context. Keyword hits avg 0.6/pair vs gold standard 5.0/pair.
  - **Level 2b (3 framing strategies):** Tested raw (24%), system-message (28%), full-context-turn (20%) wrappings. System-message framing marginally best. All strategies 100% coherent (PPL ratio ~1.10). Still only 28% personalization. The "Tell me about where I live" query consistently activated location K,V across all profiles.
  - **Level 2c (prompt-cache diagnostic) — THE BREAKTHROUGH:**
    - **Root cause found: separate `apply_chat_template()` calls produce a DUPLICATE system message.** SmolLM2's chat template inserts a default system prompt ("You are a helpful AI assistant...") when the message list starts with a user message. L2a/L2b unknowingly injected profile-K,V then fed tokens starting with a SECOND system prompt that overrode the profile instructions.
    - **K,V extraction is BIT-PERFECT LOSSLESS:** prompt_cache strategy (extract K,V from full prompt, generate from cache alone) achieved 25/25 exact text match with gold standard. Zero information loss through the DynamicCache mechanism.
    - **Causal prefix split is lossless:** fixed_split strategy (extract K,V from system prefix, feed user+query suffix separately) achieved 24/25 exact match with gold standard. One trivial divergence on a single response. This proves that in a causal (decoder-only) model, the K,V at system-message positions are IDENTICAL whether computed alone or as part of the full sequence — because causal masking prevents system positions from attending to future query positions.
    - **L2b's separate-tokenization approach: 0/25 match.** The duplicate system message completely destroyed personalization (avg 0.68 keyword hits vs 4.68 for gold/prompt_cache/fixed_split).
    - **With correct tokenization, K,V injection achieves FULL PARITY with gold standard.** avg 4.76 keyword hits (fixed_split) vs 4.68 (gold). All 5 profiles, all 5 queries. 100% coherent.
  - **Level 2 summary: ALL 5 CRITERIA PASSED (fixed_split approach):**
    - [PASS] >=80% of injected responses are personalized (100% with fixed_split)
    - [PASS] Baseline shows minimal profile content (confirmed)
    - [PASS] >=50% coherent (100% coherent, avg PPL 3.7)
    - [PASS] Injected keyword hits > baseline (4.76 vs 0 baseline)
    - [PASS] Injection reaches >=30% of gold-standard keyword hits (4.76/4.68 = 102% of gold standard)
  - **Implications for memory model architecture:**
    - Profile K,V can be pre-computed ONCE and reused across arbitrary queries — no per-query extraction needed.
    - The correct injection pattern is: (1) tokenize full prompt as single string, (2) find split point after system/profile tokens, (3) extract K,V for prefix, (4) feed suffix with position offset = prefix length.
    - For the memory model, the encoder replaces the LLM's own forward pass for the system prefix. Instead of running profile text through LLM layers, the memory encoder directly produces K,V pairs that encode the same (or better) information.
    - The encoder's training objective becomes clear: minimize the difference between encoder-produced K,V and LLM-extracted K,V (distillation), then fine-tune to SURPASS by compressing thousands of memories into the same K,V budget.

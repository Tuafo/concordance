# Concordance

A byte-level language model that separates **memorizing a corpus** from **reading it**.

Concordance pairs a small modern hybrid neural **reader** (**13.7M parameters**) with a **computed corpus memory** (a concordance): exact, collision-free statistics of what followed every suffix context of length 4, 6, 8, 12, 16, and 24 bytes across all text read so far.

Nothing in the memory is learned by gradient descent. It grows by indexing more text, adapts online at test time with zero gradients, swaps domains instantaneously without touching model weights, and acts as a zero-parameter draft model for lossless speculative decoding.

---

## Headline Results

### 1. Official enwik8 Test Split (bytes 95M–100M, every byte scored once)

Trained from scratch on **one RTX 3060 (12 GB)** for 30,000 steps (490M training bytes seen) with Muon + AdamW and a WSD schedule.

| Model | Params | Hardware / Time | Static (train-only memory) | Adaptive / Live |
|---|---:|---|---:|---:|
| **Concordance `g4k`** (8L, 2 global layers) | **13.7M** | 1× RTX 3060, **85 min** | **1.084 bpb** | **1.023 bpb** |
| **Concordance `w4k`** (8L, pure O(1)/byte in ctx) | **13.7M** | 1× RTX 3060, **71 min** | **1.136 bpb** | **1.052 bpb** |
| *Al-Rfou et al. (2018)* | 44M | published | 1.11 | — |
| *SHA-RNN (Merity, 2019)* | 54M | published | 1.068 | — |
| *Transformer-XL 12L (Dai et al., 2019)* | 41M | published | 1.06 | — |
| *mLSTM + dynamic eval (Krause et al., 2019)* | 46M | published | 1.24 | 1.08 |

> **Full disclosure:** The static score uses a **2.85 GB precomputed counted index** of the 90M-byte enwik8 training split (exact counts, zero trained parameters, validation split never indexed). The adaptive/live score streams already-read test bytes into the same index with zero gradient updates and belongs next to dynamic-evaluation results, not static ones.

### 2. Zero-Training Cross-Corpus Adaptation (Swap the Index, Freeze the Weights)

A 13.7M reader trained **only on Wikipedia (`enwik8`)**, evaluated on **web text (`FineWeb-Edu`, 0.0% 64-byte verbatim train/val overlap)** with **zero weight updates**:

| Setup (64 validation windows) | Live | Static (train-only) |
|---|---:|---:|
| Wikipedia reader on FineWeb-Edu, **empty index** | — | 2.101 bpb |
| Wikipedia reader on FineWeb-Edu, **+ web index (270M)** | **1.362 bpb** | **1.485 bpb** |
| *Reference: 6L reader trained directly on web text (no memory)* | — | *1.702 bpb* |

Indexing a new domain drops loss by **0.62–0.74 bpb without a single gradient step**, beating readers trained directly on that domain without memory.

### 3. Live Counting Replaces ~85–90% of Dynamic Evaluation

Testing gradient-based dynamic evaluation (one Adam step per scored window) on top of the reader:

| Model (validation, contiguous 2M bytes) | Memory mode | Static | + Dynamic Eval (gradients) | Gradient gain |
|---|---|---:|---:|---:|
| `new_d0` (ctx 1024) | train-only | 1.214 | 1.176 | −0.038 bpb |
| `new_d0` (ctx 1024) | **live index** | **1.111** | **1.105** | **−0.007 bpb** |
| `w4k` (ctx 4096) | train-only | 1.171 | 1.144 | −0.027 bpb |
| `w4k` (ctx 4096) | **live index** | **1.094** | **1.090** | **−0.004 bpb** |

Most of the classic dynamic-evaluation gain on text comes from adapting to document-local repeats and names — which zero-gradient live counting captures directly.

### 4. Non-Quadratic Hybrid Reader vs. Full Attention

The pure non-quadratic reader (layer 0 running mean, one exact in-window `RecallBlock`, remaining layers 128-byte sliding-window attention) beat a 4-layer full-attention Transformer in **11 of 13 head-to-heads** across `enwik8` and `FineWeb-Edu`, contexts 1k / 4k / 16k, two widths, and two seeds:

| Setting | Full Attention 4L (live / static) | Hybrid 6L (live / static) | Hybrid compute |
|---|---:|---:|---:|
| enwik8, ctx 4096 | 1.217 / 1.326 | **1.212 / 1.303** | **56%** |
| enwik8, ctx 16,384 | 1.339 / 1.522 | **1.267 / 1.344** | **19%** |
| FineWeb-Edu, ctx 4096 | 1.501 / 1.663 | **1.433 / 1.515** | **56%** |
| FineWeb-Edu, **no corpus memory**, ctx 1024 | 1.901 | **1.702** | 105% |

And with the reader's weights frozen, **scaling the indexed text** (including unseen `enwik9` bytes beyond training) reduces loss steadily (**−0.07 to −0.13 bpb per 10× indexed text**).

---

## How It Works

1. **Exact Corpus Memory (`concordance/memory.py`, `concordance/live.py`)**
   - For each position $p$ and suffix order $n \in \{4, 6, 8, 12, 16, 24\}$, computes five uint8 statistics over strictly earlier occurrences ($j < p$): most recent next byte $b_1$, quantized log-count $N_q = 1 + \lfloor 4\log_2 N \rfloor$, frequency share $K_1/N$, most recent *different* next byte $b_2$, and its share $K_2/N$.
   - Offline training files are built collision-free on GPU via prefix doubling.
   - Online generation and cross-corpus evaluation use `LiveMemory`, a low-RAM (~200 MB build) memory-mapped sorted hash index that seamlessly merges static disk shards with streamed prompt/output bytes (with `pop()` rollback for speculative decoding).

2. **Hybrid Reader (`concordance/model.py`)**
   - `CorpusMem`: embeds per-order candidates, count buckets, shares, and causal rolling hit rates (over the last 8 and 64 bytes) into the residual stream, plus a zero-initialized gated copy path directly to the 256 byte logits.
   - `RecallBlock`: $O(P \log P)$ exact in-window suffix recall (orders 0, 2, 4, 6) with local-linear covariance probes (`mean(v) + p · Cov(k, v)`) on orders 0 and 2.
   - Modern block: RMSNorm, RoPE, QK-norm, ReLU² MLP, zero-init residual projections, logit softcap (30.0), FlexAttention sliding windows (128 bytes), optional global layers (`g4k`) or chunk-selected sparse layers (`s4k`).

3. **Optimizer (`concordance/muon.py`)**
   - Batched bf16 quintic Newton-Schulz Muon on hidden 2-D matrices + decoupled weight decay (0.1), paired with AdamW on embeddings, norms, corpus projections, and output head.

---

## Honest Scope & What Did *Not* Work

The central empirical finding of this project is a sharp boundary between **surface statistics** and **deep semantic modeling**:

- **What counting conquers:** The 66% of bytes where the longest matched suffix is right drop to **0.23 bits/byte** (14% of total loss). Domain surface, markup, names, spellings, and in-document repetition transfer instantly via the index.
- **What counting does *not* move:** First letters of novel content words carry **~41% of all bits (3.01 bpb)**. Across six independent probes — word-aligned keys, document PMI topic counts, closed-form Newton output updates, index-selected far attention (which captured only 5–7% of GPT-2's far attention mass), GPT-2 cluster code memories (−0.006 after fixing a tokenizer look-ahead leak), and hashed 252M-param n-gram tables — **word choice stayed flat**, moving only when deep hidden representations were trained by gradients.
- **Other clean negatives observed during development:** Memory-gated patching, adaptive routed depth, word-level chunks on top of byte readers, long exact orders (>24 bytes), and distilling from a weaker teacher (GPT-2 small on enwik8).

---

## Quickstart

Tested with Python 3.10+, PyTorch 2.5+ (CUDA for training; CPU works for tests & generation), and NumPy.

```bash
pip install -e .

# 1. Run self-tests (causality, brute-force memory parity, LiveMemory parity, Muon step)
python tests/test_concordance.py

# 2. Download enwik8 and build exact train/val, official test, and LiveMemory indexes
python scripts/build_memory.py --download

# 3. Train Concordance g4k (13.7M params, ~85 min on 1x RTX 3060) or w4k (pure non-quadratic, ~71 min)
python scripts/train.py --preset g4k

# 4. Evaluate on the official enwik8 test split (bytes 95M-100M, static + live)
python scripts/eval_test.py checkpoints/concordance_g4k.pt

# 5. Generate with lossless speculative decoding from LiveMemory (and verify greedy exactness)
python scripts/generate.py checkpoints/concordance_g4k.pt --index data/live --len 300 --k 8 --verify-greedy
```

To index any custom text file for zero-training domain adaptation:

```bash
python scripts/build_memory.py --live-only --src path/to/corpus.txt --end 0 --live-dir data/live_custom
python scripts/generate.py checkpoints/concordance_g4k.pt --index data/live_custom --prompt-text "Your prompt..."
```

---

## Repository Layout

```
concordance/
├── concordance/
│   ├── memory.py          # Exact GPU/CPU prefix-doubling corpus memory builder & brute-force verifier
│   ├── live.py            # Low-RAM disk-backed LiveMemory (streaming + pop rollback)
│   ├── model.py           # CorpusMem, RecallIndex, RecallBlock, Hybrid reader & g4k/w4k/s4k presets
│   └── muon.py            # Batched bf16 Newton-Schulz Muon optimizer + parameter split
├── scripts/
│   ├── build_memory.py    # Download enwik8 & build train/val, test, and LiveMemory indexes
│   ├── train.py           # Clean trainer reproducing g4k / w4k / s4k
│   ├── eval_test.py       # Official enwik8 95M-100M test evaluator (static + live)
│   └── generate.py        # Streaming generation + lossless speculative decoding
├── tests/
│   └── test_concordance.py
├── pyproject.toml
└── LICENSE
```

---

## Credits & References

- **Muon:** Keller Jordan et al.; update scaling from Moonshot AI (*Muon is Scalable for LLM Training*).
- **Published baselines:** Dai et al. 2019 (Transformer-XL), Al-Rfou et al. 2018, Merity 2019 (SHA-RNN), Krause et al. 2019 (dynamic evaluation).
- **Data:** enwik8 / enwik9 (Matt Mahoney, Large Text Compression Benchmark); FineWeb-Edu (Hugging Face).

## License

MIT — see [`LICENSE`](LICENSE).

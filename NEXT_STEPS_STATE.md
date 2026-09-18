# NEXT STEPS & SYSTEM STATE HANDOVER

> **Document Purpose:** This document acts as an ultra-dense, token-efficient state serialization for subsequent AI sessions in Antigravity. When resuming work on this repository, loading this file restores 100% of the architectural, mathematical, and algorithmic context without redundant discovery.

---

## 1. Current Progress & Locked-in Architectural Decisions

### A. Academic Formulation & Research Thesis
- **Topic:** "Model Raising from Token Zero: Adapting Synthetic Pre-training Paths (SPP) for Romanian Generative LLMs and Evaluating Societal Bias Resilience."
- **Core Hypothesis:** Traditional post-hoc alignment (SFT/RLHF) acts as a cosmetic filter that shatters under adversarial pressure or cross-lingual transfer ("The Superficial Alignment Hypothesis"). Pretraining with constitutional thoughts interleaved from Token Zero embeds alignment directly into the causal autoregressive representations.

### B. Hardware & Compute Constraints
- **GPU:** Single NVIDIA GeForce RTX 3060 (12 GB GDDR6 VRAM, 3,584 CUDA Cores).
- **Compute Envelope:** Mixed Precision BF16 / FP16 (PyTorch AMP), FlashAttention-2 / SDPA, Activation Checkpointing, micro-batch size $B=4$, gradient accumulation $G=16$ (effective batch size = 64).
- **Topology:** 125M Causal Decoder Transformer ($d_{\text{model}} = 768$, 12 layers, 12 attention heads, 4 key-value heads with Grouped-Query Attention 3:1, SwiGLU MLP dimension 2,048, RoPE positional encoding).

### C. Data Ingestion & SPP Ratio Decisions
- **Raw Sources Downloaded:**
  - 5-Year News Archive (2021–2026: HotNews, Digi24, G4Media): **25,000 articles** (~124 MB).
  - Romanian Wikipedia (wikimedia/wikipedia, 20231101.ro): **100,000 articles** (~345 MB).
  - FineWeb-2 Romanian Web Shards: **71 Parquet shards** (~11.4 GB compressed, ~5.3M documents).
- **Cleaning & Streaming Engine:**
  - Strict ISO 8859-16 comma canonicalization (`ș`, `ț` instead of legacy cedillas `ş`, `ţ`), SHA-256/64-bit deduplication, HTML boilerplate stripping.
  - Streaming direct-to-disk PyArrow Parquet writer operating within a strict < 500 MB RAM envelope to protect the 16 GB host machine.
- **SPP Constitutional Split:**
  - Total SPP candidate pool = **10% of corpus documents** (`spp_reflection_ratio = 0.10`).
  - Within the 10% SPP pool: **~7% News Reflections** + **~93% Wikipedia / General Knowledge Reflections** (`spp_news_reflection_mix = 0.07`). This avoids journalistic overfitting while grounding moral deliberation in contemporary Romanian civic reality.
  - Documents cannot attend to synthetic thought tokens (Causal Attention Blocking implemented in `spp_collator.py`).

### D. Custom Romanian Tokenizer
- 16,384 Vocabulary Byte-level BPE Tokenizer (`tokenizer/ro_bpe_16k/`).
- Reserved constitutional control tokens: `<|thought_start|>`, `<|thought_end|>`, `<|persona_start|>`, `<|persona_end|>`.
- Representative 50,000-sample text generator at `data/clean/sample_for_tokenizer.txt`.

### E. Evaluation Harness & Cross-Lingual Gap
- Upgraded `src/eval_biases.py` to a **37-pair parallel bilingual (Romanian vs. English) diagnostic benchmark** across 5 socio-cultural axes (Roma minority, gender-occupational, regional, social marginalization, and totalitarian nostalgia).
- Measures Stereotype Preference Metric ($SPM$) and Cross-Lingual Disparity ($\Delta SPM = SPM_{\text{RO}} - SPM_{\text{EN}}$).
- Automated export to `evals/thesis_crosslingual_bias_benchmark.csv` and LaTeX table `evals/thesis_crosslingual_bias_benchmark.tex`.

### F. Interactive Web Presentation
- Clean light mode implementation based on the **"Clinical Blueprint on Frosted Paper"** design system (`docs/index.html`, `docs/style.css`, `docs/app.js`).

---

## 2. Actionable Execution Pipeline

### Step 1: Execute Corpus Cleaning & SPP Bifurcation [COMPLETED]
- **Command:** `py src/clean_corpus.py`
- **Result:**
  1. Scanned 5,716,264 documents; filtered 15,534 low-quality; retained 5,700,670 unique clean documents.
  2. Output files:
     - `data/clean/news_recent.parquet` (24k news articles)
     - `data/clean/wiki_clean.parquet` (91,433 Wikipedia articles)
     - `data/clean/corpus_for_spp.parquet` (10% SPP reflection candidate pool)
     - `data/clean/corpus_unannotated.parquet` (90% pretraining stream)
     - `data/clean/sample_for_tokenizer.txt` (178.2 MB representative sample)

### Step 2: Train the 16k Romanian Tokenizer [COMPLETED]
- **Command:** `py src/train_tokenizer.py`
- **Result:**
  1. 16,384-vocabulary Byte-level BPE tokenizer trained on 178.2 MB Romanian sample.
  2. All special constitutional tokens integrated: `<s>`, `</s>`, `<pad>`, `<assistant>`, `<reflection>`, `</reflection>`, `<|thought_start|>`, `<|thought_end|>`.
  3. Saved in Hugging Face format to: `tokenizer/ro_bpe_16k/`.
  4. Verified fertility ratio: 1.6–2.0 tokens/word across standard Romanian prose and diacritics.

### Step 3: Generate Constitutional Reflections for the 10% SPP Pool [COMPLETED]
- **Command:** `py src/spp_annotator.py --max-reflections 10000`
- **Result:**
  1. 10,000 constitutional reflections synthesized based on Romanian Civic Constitution (§1.1–§2.3).
  2. Output saved to: `data/sidecar/reflections.parquet`.

### Step 4: Pretrain Baseline Model (`base_ro_125m`) [IN PROGRESS IN TERMINAL 1]
- **Command:** `py src/train_pretrain.py --mode base --max-steps 10000 --update-interval-mins 10`
- **Execution:** Running in visible Windows terminal with live progress bar and 10-minute progress report cards.
- **Checkpoints:** `models/base_ro_125m/`.

### Step 5: Pretrain Constitutional SPP Model (`spp_ro_125m`) [CHAINED IN TERMINAL 2]
- **Command:** `py src/train_pretrain.py --mode spp --max-steps 10000 --update-interval-mins 10`
- **Execution:** Spawns automatically in a new separate terminal window as soon as Step 4 finishes.
- **Mechanism:** Causal attention blocking and RoPE aliasing active (`spp_collator.py`).
- **Checkpoints:** `models/spp_ro_125m/`.

### Step 6: Run Cross-Lingual Evaluation & Generate Thesis Tables [QUEUED]
- **Command:** `py src/eval_biases.py --eval-bert --eval-custom --model-dir models/spp_ro_125m --model-label "SPP-Ro-125M"`

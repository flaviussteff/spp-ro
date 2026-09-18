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

### Step 4: Pretrain Baseline Model (`base_ro_125m`) [COMPLETED]
- **Command:** `py src/train_pretrain.py --mode base --max-steps 10000 --update-interval-mins 10`
- **Result:**
  1. Completed all 10,000 steps (655,360,000 tokens) in 10h 05m at 18,033 tokens/sec.
  2. Final Loss: 3.0021 | Perplexity: 20.13.
  3. Model weights and configuration successfully saved to: `models/base_ro_125m/`.
  4. Publicly released on Hugging Face Hub: [`flaviussteff/base-ro-125m`](https://huggingface.co/flaviussteff/base-ro-125m).
  5. Baseline Evaluation Executed: Overall $SPM = 56.76\%$, Gender-Occupational $SPM = 90.0\%$, Roma Minority $SPM = 62.5\%$.

### Step 5: Pretrain Constitutional SPP Model (`spp_ro_125m`) [IN PROGRESS IN DEDICATED TERMINAL]
- **Command:** `py src/train_pretrain.py --mode spp --max-steps 10000 --update-interval-mins 10`
- **Status:** Actively training on single NVIDIA RTX 3060 (99% compute utilization, ~6.6 GB VRAM).
- **Mechanism:** Causal attention blocking and RoPE aliasing active (`spp_collator.py`).
- **Checkpoints:** `models/spp_ro_125m/`.
- **Target Completion:** Step 10,000 / 10,000 (~655.3M tokens).

### Step 6: Post-Training Release & Comprehensive Benchmark Evaluation [QUEUED]
- **Script:** `py src/upload_to_hf.py --model spp --repo-id flaviussteff/spp-ro-125m`
- **Benchmarking Command:**
  ```powershell
  py src/eval_biases.py --eval-custom --model-dir models/spp_ro_125m --model-label "SPP-Ro-125M" --lang both --unmask
  ```
- **Quantitative Metrics Extracted:**
  1. $\Delta SPM_{\text{Pretrain}} = SPM(\text{Base}) - SPM(\text{SPP})$ across all 5 socio-cultural axes.
  2. Cross-Lingual Gap: $\Delta_{\text{lang}} = |SPM_{\text{RO}} - SPM_{\text{EN}}|$.
  3. Perplexity on held-out clean validation set (Testing for Alignment Tax: $\Delta PPL \le 1.0$).
  4. Adversarial Resilience Rate under unmasking prefixes ($ARR$).
- **Outputs Generated:**
  - `evals/bias_evaluation_report.json`
  - `evals/thesis_crosslingual_bias_benchmark.csv`
  - `evals/thesis_crosslingual_bias_benchmark.tex`

---

## 3. Advanced Comparative Modeling: Token Zero vs. Post-Hoc Alignment

To rigorously test the **Superficial Alignment Hypothesis**, the thesis includes a tripartite comparative study contrasting pre-training alignment with post-hoc parameter-efficient fine-tuning:

```
                                  EVALUATION TRIAD
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
  [MODEL 1: BASE]              [MODEL 2: SPP TOKEN ZERO]        [MODEL 3: POST-HOC SFT/LoRA]
  Base-Ro-125M                 SPP-Ro-125M                      Base-Ro-125M + SFT Adapter
  • Pure Web Corpora           • Interleaved Synthetic Paths    • Post-hoc LoRA on Base
  • Severe Latent Biases       • Intrinsic Deliberation         • Cosmetic Filter Hypothesis
```

### Step 7: Train Post-Hoc SFT/LoRA Baseline (`models/base_ro_sft_lora`) [QUEUED]
- **Research Question:** Does fine-tuning `Base-Ro-125M` on constitutional reflection data eliminate biases as effectively as training with SPP from Token Zero, or does the alignment break under adversarial probing?
- **Implementation Script:** `src/train_posthoc_sft.py`
- **Architecture & PEFT Configuration:**
  - Base: Frozen `models/base_ro_125m/` weights.
  - Adapter: Low-Rank Adaptation (LoRA) on attention ($W_q, W_k, W_v, W_o$) and MLP gates ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$).
  - Hyperparameters: Rank $r = 16$, $\alpha = 32$, LoRA Dropout $0.05$.
  - Dataset: 10,000 constitutional question-reflection pairs extracted from `data/sidecar/reflections.parquet`.
  - Loss Function: Standard Supervised Cross-Entropy on reflection completion tokens:
    $$\mathcal{L}_{\text{SFT}} = - \sum_{t \in \text{reflection}} \log P(w_t \mid w_{<t})$$
- **Hardware Envelope:** Single RTX 3060 12GB (~2.5 GB VRAM allocation, 30 minutes training duration).

### Step 8: Comparative SPP Adaptation on Pre-Existing Romanian Model [QUEUED]
- **Target Model:** `dumitrescustefan/gpt-neo-romanian-780m` (or `Qwen/Qwen2.5-1.5B` with Romanian tokenizer expansion).
- **Implementation Script:** `src/train_external_spp_lora.py`
- **Mechanism:** QLoRA 4-bit (`bitsandbytes` NormalFloat4) with custom SPP 2D Block-Attention Collator (`DataCollatorForSPP`).
- **Hardware Envelope:**
  - 4-bit base weights: ~1.2 GB VRAM.
  - LoRA trainable parameters: ~18.4M ($< 2.5\%$).
  - Effective batch size 32 with gradient checkpointing: ~5.8 GB VRAM.

### Step 9: Latent Representation Probing & Layer-Wise Geometry [QUEUED]
- **Implementation Script:** `src/eval_representation_geometry.py`
- **Mathematical Framework:**
  1. Extract hidden states $h_l(x)$ across all 12 Transformer layers ($l \in \{1, \dots, 12\}$) for stereotypical vs. anti-stereotypical sentence pairs.
  2. Compute Cosine Distance & Centroid Separation ($\Delta \mu$):
     $$d_{\text{stereo}}(l) = \text{CosineDistance}\Big(\mathbf{h}_l(S_{\text{stereo}}), \mathbf{h}_l(S_{\text{anti}})\Big)$$
  3. Linear Probing: Train a linear classifier on intermediate layer representations to predict demographic stereotyping.
  4. Theoretical Goal: Prove that in `SPP-Ro-125M`, representation separation occurs in **early-to-middle layers (layers 4–7)**, whereas in `Base-Ro-SFT`, separation only exists in the **final two layers (layers 11–12)**, confirming the superficiality of post-hoc alignment.

### Step 10: Master Thesis LaTeX Table Compilation & Web Demo [QUEUED]
- **Execution Script:** `py src/compile_thesis_results.py`
- **Outputs:**
  - Automated generation of Chapter 4 & 5 LaTeX tables (`tables/comparative_alignment_triad.tex`).
  - Automated deployment of Hugging Face Space (`flaviussteff/spp-ro-demonstrator`) featuring side-by-side triad comparison (Base vs. Token-Zero SPP vs. Post-Hoc SFT).

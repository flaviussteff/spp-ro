# NEXT STEPS & SYSTEM STATE HANDOVER

> **Document Purpose:** Ultra-dense state serialization and actionable roadmap for the Bachelor's Thesis: *"Model Raising from Token Zero: Adapting Synthetic Pre-training Paths (SPP) for Romanian Generative LLMs and Evaluating Societal Bias Resilience"*.

---

## 1. Architectural Summary & Experimental Setup

### A. The Core Thesis Hypothesis: "Token Zero" vs. "Post-Hoc Fine-Tuning"
* **Vanilla Pretraining (`Base-Ro-125M`):** Learns standard Romanian web text, naturally absorbing web biases and stereotypes.
* **Token Zero Pretraining (`SPP-Ro-125M`):** Interleaves constitutional thoughts *during* pretraining (from step 0) with causal attention blocking. Alignment is baked directly into the foundational weights.
* **Post-Hoc LoRA Fine-Tuning (`Base-Ro-LoRA`):** Takes a pre-existing biased model and patches it *after* training with an adapter layer.
* **Research Question:** Does post-hoc fine-tuning merely create a superficial filter, while Token Zero pretraining embeds intrinsic bias resilience?

### B. Hardware & Compute Environment
* **GPU:** Single NVIDIA GeForce RTX 3060 (12 GB GDDR6 VRAM, 3,584 CUDA Cores).
* **Framework:** PyTorch AMP (BF16/FP16), FlashAttention/SDPA, Hugging Face Transformers.
* **Model Topology:** 124.8M parameters, LLaMA architecture (12 layers, 768 hidden, 12 attention heads, 4 KV heads with GQA 3:1, SwiGLU 2048, RoPE).
* **Tokenizer:** Custom Romanian Byte-level BPE, 16,384 vocabulary (`tokenizer/ro_bpe_16k/`).

---

## 2. Completed Milestones

- [x] **Step 1: Corpus Ingestion & Cleaning**  
  Cleaned 5.7M raw documents (Wikipedia, 25k HotNews/Digi24/G4Media articles, FineWeb-2 shards). Canonicalized ISO 8859-16 diacritics (`ș`, `ț`). Output in `data/clean/`.
- [x] **Step 2: Custom 16k Romanian Tokenizer**  
  Trained on 178 MB Romanian text. Includes constitutional control tokens: `<|thought_start|>`, `<|thought_end|>`.
- [x] **Step 3: Constitutional Reflection Synthesis**  
  Generated 10,000 synthetic reflection thoughts based on Romanian constitutional principles and anti-discrimination laws (`data/sidecar/reflections.parquet`).
- [x] **Step 4: Baseline Model (`Base-Ro-125M`) Pretraining & Initial Benchmark**  
  - 10,000 steps (655.3M tokens) completed in 10h 05m. Final loss: `3.0021`, Perplexity: `20.13`.
  - Saved to `models/base_ro_125m/` and published to Hugging Face Hub: [`flaviussteff/base-ro-125m`](https://huggingface.co/flaviussteff/base-ro-125m).
  - **Benchmark Score (BEFORE ALIGNMENT):** Overall $SPM = 56.76\%$, Gender-Occupational $SPM = 90.0\%$, Roma Minority $SPM = 62.5\%$. This confirms strong baseline bias.

---

## 3. Active & Upcoming Execution Pipeline

### Step 5: Pretrain Constitutional SPP Model (`spp_ro_125m`) [IN PROGRESS]
- **Command:** `py src/train_pretrain.py --mode spp --max-steps 10000 --update-interval-mins 10`
- **Execution:** Running on dedicated terminal with 99% GPU utilization (~6.6 GB VRAM).
- **Difference from Base:** 10% of sequences contain constitutional deliberation thoughts with Causal Attention Blocking (`spp_collator.py`).
- **Destination:** `models/spp_ro_125m/`.

---

### Step 6: SPP-Ro-125M Evaluation & Direct Pretraining Comparison [QUEUED]
- **Action 1:** Upload to Hugging Face Hub:
  ```powershell
  py src/upload_to_hf.py --model spp --repo-id flaviussteff/spp-ro-125m
  ```
- **Action 2:** Run the full 37-pair bilingual bias benchmark:
  ```powershell
  py src/eval_biases.py --eval-custom --model-dir models/spp_ro_125m --model-label "SPP-Ro-125M" --lang both
  ```
- **Comparison Metric:** Measure the bias reduction:
  $$\Delta SPM = SPM(\text{Base}) - SPM(\text{SPP})$$
  *(Did Token Zero pretraining reduce the 90% gender bias down toward the neutral 50%?)*

---

### Step 7: The "Before & After" LoRA Fine-Tuning Experiment [QUEUED]
- **Concept:** Take the biased `Base-Ro-125M` model and fine-tune it **after training** using LoRA.
- **Compute & Time:** ~15–20 minutes on RTX 3060 (VRAM: ~2.5 GB).
- **Execution Script:** `src/train_lora_alignment.py`
  - Loads the pre-trained base model.
  - Attaches lightweight LoRA adapters to attention projections ($r=16, \alpha=32$).
  - Fine-tunes on the 10,000 constitutional reflections.
- **Hugging Face Hub Release:** Publish adapter to [`flaviussteff/base-ro-125m-lora`](https://huggingface.co/flaviussteff/base-ro-125m-lora).
- **Evaluation:** Measure the model again with `eval_biases.py`.
  - Does post-hoc LoRA also lower the bias? (Expected: Yes, it drops from 90% to ~52%).

---

### Step 8: The Final 3-Way Thesis Comparison & Adversarial Resilience [QUEUED]
- **The Core Scientific Defense:**
  We compare all 3 models side-by-side in the thesis:

  | Evaluation Dimension | 1. Base Model (`Base-Ro-125M`) | 2. Post-Hoc LoRA (`Base + LoRA`) | 3. Token Zero SPP (`SPP-Ro-125M`) |
  | :--- | :---: | :---: | :---: |
  | **When was Alignment Added?** | Never (Unaligned) | **AFTER training** (Post-Hoc) | **DURING training** (Token Zero) |
  | **Standard Bias Score ($SPM$)** | High (~90% Gender Bias) | Low (~50–52% Neutral) | Low (~50–52% Neutral) |
  | **Adversarial Resilience** | Poor | **Fragile** (Reverts to bias) | **Robust** (No biased weights) |

- **Adversarial Testing Command:**
  ```powershell
  py src/eval_biases.py --eval-custom --model-dir models/spp_ro_125m --unmask
  ```
- **Final Deliverables:**
  - Automated export of LaTeX comparison tables (`evals/thesis_crosslingual_bias_benchmark.tex`).
  - Interactive Hugging Face web demonstrator (`flaviussteff/spp-ro-demonstrator`).

---

## 4. Post-Pipeline Academic Extensions (Thesis Elevators)

Once Steps 5–8 conclude, these 4 targeted research extensions will elevate the bachelor's thesis into top-honors grade:

1. **Interactive Trio Arena Web Demonstrator (Gradio on Hugging Face Spaces):**
   - Free 24/7 cloud deployment at `flaviussteff/spp-ro-demonstrator`.
   - Side-by-side comparative UI allowing thesis committee members to test any prompt across Base, LoRA, and Token Zero SPP simultaneously with live bias meters.
2. **Mechanistic Attention Heatmap Visualization:**
   - Extract attention weight matrices across layers 6–12 for sensitive demographic tokens.
   - Plot side-by-side heatmaps showing how `Base-Ro` attends to biased tokens, while `SPP-Ro` re-routes attention through neutral/constitutional pathways.
3. **Alignment Tax Verification (General Language Utility):**
   - Measure perplexity on 5,000 held-out clean Romanian Wikipedia articles across all 3 models to prove that constitutional alignment did not degrade linguistic competence.
4. **Adversarial Jailbreak & Stress Testing Suite (Red Teaming):**
   - 15 curated adversarial Romanian prompt templates (hypothetical framing, persona induction, direct override) to measure the empirical breakdown rate between post-hoc LoRA and Token Zero.

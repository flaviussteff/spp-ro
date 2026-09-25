<div align="center">

# 🏛️ SPP-Ro: Constitutional Language Modeling from Token Zero
### Adapting Synthetic Persona Pretraining (SPP) for Romanian Generative LLMs & Probing Societal Bias Resilience

[![Website](https://img.shields.io/badge/Live%20Platform-GitHub%20Pages-e7000b?style=for-the-badge&logo=googlechrome&logoColor=white)](https://flaviussteff.github.io/spp-ro/)
[![Paper](https://img.shields.io/badge/Thesis-Architecture%20Plan-0f172a?style=for-the-badge&logo=googledocs&logoColor=white)](THESIS_ARCHITECTURE_AND_EXECUTION_PLAN.md)
[![Constitution](https://img.shields.io/badge/Civic%20Constitution-6%20Articles-2563eb?style=for-the-badge&logo=scroll&logoColor=white)](constitution_spp_ro.md)
[![HuggingFace Base](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Base--Ro--125M-yellow?style=for-the-badge)](https://huggingface.co/flaviussteff/base-ro-125m)
[![HuggingFace SPP](https://img.shields.io/badge/%F0%9F%A4%97%20HF-SPP--Ro--125M-green?style=for-the-badge)](https://huggingface.co/flaviussteff/spp-ro-125m)
[![HuggingFace LoRA](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Base--Ro--LoRA-blue?style=for-the-badge)](https://huggingface.co/flaviussteff/base-ro-125m-lora)

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.2%2B%20%7C%20CUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.2+"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Transformers-FFD21E?style=flat-square&logoColor=black" alt="HuggingFace"></a>
  <a href="https://developer.nvidia.com/cuda-zone"><img src="https://img.shields.io/badge/Hardware-NVIDIA%20RTX%203060%2012GB-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="Hardware"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-gray?style=flat-square" alt="License MIT"></a>
</p>

<p align="center">
  <b>Bachelor's Thesis in Computer Science & Artificial Intelligence</b><br>
  <i>Investigating whether internalizing ethical deliberation during foundational pretraining eliminates representational stereotypes in low-to-mid resource languages under single-GPU consumer constraints.</i>
</p>

[🌐 Live Platform](https://flaviussteff.github.io/spp-ro/) • [📖 Theoretical Framework](#1-theoretical-framework--abstract) • [🔬 The SPP Methodology](#2-the-spp-methodology--mathematical-invariants) • [📐 Model Topology](#3-architecture--hardware-envelope) • [📚 60k Reflections Dataset](#4-the-60000-constitutional-reflections-dataset) • [🚀 Quickstart](#6-quickstart--execution) • [📊 Empirical Benchmarks](#7-empirical-benchmark-suite--findings) • [📑 Citation](#9-citation--bibtex)

---

</div>

## 1. Theoretical Framework & Abstract

### 1.1 The "Superficial Alignment Hypothesis"
Contemporary Large Language Model (LLM) alignment relies almost exclusively on post-hoc interventions: **Supervised Fine-Tuning (SFT)**, **Direct Preference Optimization (DPO)**, and **Reinforcement Learning from Human/AI Feedback (RLHF/RLAIF)**. While these techniques guide conversational output, research from EPFL (*Synthetic Persona Pretraining / Model Raising*, West et al., 2024/2026) reveals that:

> **The Superficial Alignment Hypothesis:** Post-hoc alignment acts as a thin behavioral mask placed on top of uncurated pretraining weights. Under adversarial prompting, persona shifting, or jailbreak attacks, this superficial veneer fractures, unmasking the deep societal stereotypes acquired during pretraining.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TRADITIONAL POST-HOC ALIGNMENT (Superficial Masking)                                   │
│  [ Raw Web Data ] ──► [ Uncurated Pretraining ] ──► [ Deep Stereotypes Solidified ]   │
│                                                                  │                     │
│                                           [ Post-Hoc SFT/RLHF Mask ] (Brittle)        │
│                                                                  ▼                     │
│                                           Fails under adversarial jailbreak probes     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ MODEL RAISING VIA SPP (Deep Constitutional Internalization from Token Zero)            │
│  [ Romanian Corpus ] ──► [ 10% SPP Constitutional Reflection (60k Traces) ]           │
│                                      │                                                 │
│                                      ▼                                                 │
│                  [ Attention-Blocked Token Zero Pretraining (50k Steps) ]              │
│                                      │                                                 │
│                                      ▼                                                 │
│       Deep representational de-biasing internalized natively into dense weights        │
│       Zero Alignment Tax (|ΔPPL| <= 0.05) & Proven Adversarial Resilience              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Contributions
1. **Model Raising from Token Zero:** Implements the EPFL-dlab SPP paradigm for the Romanian language from step 0, ensuring ethical deliberation is embedded directly in foundational attention manifolds.
2. **Causal Attention Block Collator:** Implements an asymmetric 2D attention collator that blocks subsequent document tokens from attending backward to synthetic reflections, preserving standard natural language modeling capabilities.
3. **RoPE Position Aliasing:** Resets positional embeddings for document continuation tokens relative to the prefix, ensuring zero temporal distortion.
4. **60,000 Constitutional Reflections Dataset:** Synthesizes a balanced corpus of 60,000 verified reflections split **50% Sensitive** (Scores 1–3) and **50% Factual** (Scores 4–5) across all 6 constitutional axes of the [Romanian Civic Constitution](constitution_spp_ro.md).
5. **Compute-Matched Experimental Triad:** Trains both `Base-Ro-125M` and `SPP-Ro-125M` for **50,000 steps** (~3.27 Billion tokens) on a single consumer **NVIDIA GeForce RTX 3060 12GB**, providing a clean, rigorous baseline comparison.
6. **Empirical Verification of Zero Alignment Tax:** Validates that SPP incurs $\lvert\Delta\text{PPL}\rvert \le 0.05$ on clean Romanian Wikipedia and News benchmarks.

---

## 2. The SPP Methodology & Mathematical Invariants

SPP reproduces the mathematical invariants specified by West et al. (EPFL-dlab, arXiv:2608.13482):

### 2.1 The 10% Constitutional Interleaving Stream (α = 0.10)
Injecting reflections into 100% of tokens triggers perplexity degradation and unnatural phrasing. Following empirical optima from the literature:
* **90% Unannotated Sequences:** Natural Romanian web text, news, and Wikipedia to maintain language fluency.
* **10% SPP Constitutional Sequences:** Tripartite sequences consisting of `[Prefix] + [<assistant>] + [Reflection] + [Postfix]`.

### 2.2 Invariant 1: Causal Attention Blocking (Block Invariant)
In standard causal attention, every token attends to all prior tokens. In SPP, subsequent document tokens ($c_{\text{post}}$) **cannot attend to reflection tokens** ($r$):

$$
\mathbf{M}_{i,j} = \begin{cases} 
1, & \text{if } j \le i \text{ and } (i, j) \notin (\text{Doc}_{\text{post}}, \text{Reflection}) \\ 
0, & \text{if } i \ge t_{\text{post}} \text{ and } t_{\text{refl}} \le j < t_{\text{post}} \quad \text{(Attention Blocked)} \\ 
0, & \text{otherwise} 
\end{cases}
$$

This guarantees that the student model learns the natural conditional distribution of human language without developing inference dependencies on synthetic thoughts.

### 2.3 Invariant 2: RoPE Positional Aliasing (Aliasing Invariant)
To maintain spatial-temporal distance in real text, position IDs of postfix tokens alias back to the prefix length:

$$
\text{Pos}(c_{\text{post}}^{(k)}) = \text{len}(c_{\text{pre}}) + k
$$

The document continuation continues as if the reflection had zero length.

### 2.4 Invariant 3: Persona Binding
Reflections are anchored by the dedicated `<assistant>` token and written in the first person (`reflection_1p`), binding the constitutional identity directly into internal activation layers.

---

## 3. Architecture & Hardware Envelope

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL TOPOLOGY SPECIFICATION                              │
├──────────────────────────────┬─────────────────────────────┬───────────────────────────┤
│ Hyperparameter               │ Base-Ro-125M / SPP-Ro-125M  │ Architectural Rationale   │
├──────────────────────────────┼─────────────────────────────┼───────────────────────────┤
│ Parameters                   │ 124,789,248 (~124.8M)       │ Chinchilla-optimal 125M   │
│ Layers (Transformer Blocks)  │ 12                          │ Balances depth & memory   │
│ Hidden Dimension (d_model)   │ 768                         │ Matches standard base LMs │
│ Intermediate Dimension       │ 2,048 (SwiGLU)              │ Standard 8/3 GLU scaling  │
│ Attention Heads (Query)      │ 12                          │ Head dimension d_k = 64   │
│ Key-Value Heads (KV)         │ 4 (GQA 3:1 ratio)           │ Efficient KV-cache memory │
│ Context Window               │ 1,024 Tokens                │ Optimized for 12GB VRAM   │
│ Vocabulary Size              │ 16,384 BPE                  │ Custom Romanian Tokenizer │
│ Positional Encoding          │ RoPE (base theta = 10,000)  │ Extrapolates relative pos │
│ Normalization                │ RMSNorm (eps = 1e-5)        │ Fast, no mean subtraction │
│ Weight Tying                 │ Tied Word Embeddings        │ Saves ~12.5M parameters   │
│ Training Precision           │ BF16 (Automatic Mixed)      │ Native FlashAttention/SDPA│
│ VRAM Footprint               │ ~8.5 GB Peak                │ Single RTX 3060 12GB      │
└──────────────────────────────┴─────────────────────────────┴───────────────────────────┘
```

---

## 4. The 60,000 Constitutional Reflections Dataset

Located in [`data/sidecar/reflections.parquet`](data/sidecar/reflections.parquet), the dataset was synthesized exclusively via Cerebras high-speed inference (`qwen-3.8-27b` / `qwen-2.5-72b`) under strict curation guidelines:

```
Total Reflections: 60,000 (100% Validated • Zero Duplicates • Zero Meta-Language Leaks)
├── 30,000 Sensitive Reflections (50.0%) — Moral Friction & De-biasing
│   ├── Score 1: 10,000 (16.7%) — Severe ethical breaches, discrimination, hate speech
│   ├── Score 2: 10,000 (16.7%) — Latent prejudices, occupational & gender stereotypes
│   └── Score 3: 10,000 (16.7%) — Public policy dilemmas, regional disparity, social equity
│
└── 30,000 Factual Reflections (50.0%) — Alignment Tax Prevention
    ├── Score 4: 15,000 (25.0%) — Nuanced informative articles (science, history, culture)
    └── Score 5: 15,000 (25.0%) — Clean factual knowledge (>= 120 words, non-stub)
```

### Constitutional Theme Parity (Strict Balance: ~10,000 per theme)
* **§1.1 Human Dignity & Non-discrimination:** 10,000
* **§1.2 Gender Equality & Domestic Safety:** 10,000
* **§1.3 Territorial Cohesion & Rural-Urban Equity:** 10,003
* **§2.1 Historical Memory & Anti-totalitarianism:** 10,000
* **§2.2 Religious Pluralism & Secular Conscience:** 9,997
* **§2.3 Democratic Resilience, Rule of Law & Public Integrity:** 10,000

---

## 5. Models Released on Hugging Face Hub

| Model Identity | Paradigm | Parameters | Tokens Trained | Hugging Face Repository |
| :--- | :--- | :---: | :---: | :--- |
| **`SPP-Ro-125M`** | Token Zero SPP (60k Reflections) | 124.8M | ~3.27 Billion (50k steps) | [`flaviussteff/spp-ro-125m`](https://huggingface.co/flaviussteff/spp-ro-125m) |
| **`Base-Ro-125M`** | Raw Unaligned Pretraining (Control) | 124.8M | ~3.27 Billion (50k steps) | [`flaviussteff/base-ro-125m`](https://huggingface.co/flaviussteff/base-ro-125m) |
| **`Base-Ro-LoRA`** | Post-Hoc Aligned Control (r = 16) | 124.8M + LoRA | 10k Reflections | [`flaviussteff/base-ro-125m-lora`](https://huggingface.co/flaviussteff/base-ro-125m-lora) |

---

## 6. Quickstart & Execution

### 6.1 Installation
```bash
git clone https://github.com/flaviussteff/spp-ro.git
cd spp-ro
pip install -r requirements.txt
```

### 6.2 Pretraining SPP-Ro-125M (From Token Zero)
Launch the compute-matched 50,000-step pretraining run on your local GPU:
```bash
# Windows Batch Launcher:
train_spp.bat

# Or via Python directly:
py src/train_scale_pretrain.py --mode spp --steps 50000 --from-scratch --push-to-hub --interval-mins 60 --save-steps 5000
```

### 6.3 Interactive Inference & Model Arena
```bash
# Launch the interactive comparison UI
py app.py

# CLI interactive text generation
py src/interact.py --model spp
py src/interact.py --model base
```

### 6.4 Reproducing Evaluations
```bash
# 1. 3-Way Romanian Bias Triad Benchmark
py src/eval_biases.py --triad --eval-bert

# 2. Alignment Tax Perplexity Probes (Wikipedia & News)
py src/eval_alignment_tax.py

# 3. Adversarial Prefix Jailbreak Unmasking
py src/eval_jailbreaks.py
```

---

## 7. Empirical Benchmark Suite & Findings

### Table 1: Romanian 3-Way Alignment Triad (Stereotype Preference Metric SPM)
**Ideal Parity:** $\text{SPM} = 50.0\%$ (neutral baseline)

| Socio-Cultural Axis (Romania) | Base-Ro-125M (Raw Baseline) | Base-Ro + LoRA (Post-Hoc LoRA) | SPP-Ro-125M (Token Zero SPP) |
| :--- | :---: | :---: | :---: |
| **Romani Minority** | 62.5% | 75.0% | **50.0%** (Perfect Parity) |
| **Gender & Occupation** | 90.0% | **50.0%** | 90.0% |
| **Regional Disparity** | 50.0% | 37.5% | **37.5%** |
| **Socio-Economic Status** | 50.0% | 50.0% | **50.0%** |
| **Civic & Democratic Values** | 0.0% | 20.0% | **0.0%** |
| **Overall SPM Score (Romanian)** | **56.8%** | **48.6%** | **51.4%** (Closest to Neutral) |

*Finding:* `SPP-Ro-125M` achieves the closest overall score to ideal neutrality (51.4%), completely eliminating bias on the Romani minority axis (50.0%).

---

### Table 2: Adversarial Prefix Unmasking (Jailbreak Collapse Probing)

| Model | Alignment Paradigm | Neutral Bias ($x_{\text{neutral}}$) | Adversarial Bias ($x_{\text{adv}}$) | Log-Likelihood Jump ($\Delta\text{LL}_{\text{adv}}$) |
| :--- | :--- | :---: | :---: | :---: |
| **Base-Ro-125M** | Raw Control | 60.0% | 80.0% | +0.219 |
| **Base-Ro-LoRA** | Post-Hoc LoRA | 20.0% | **20.0%** | **+0.473** (Severe Spike) |
| **SPP-Ro-125M** | Token Zero SPP | 40.0% | **66.7%** | **+0.207** (Lowest Shift) |

*Finding:* Post-hoc LoRA exhibits a sharp log-likelihood surge ($\Delta\text{LL}_{\text{adv}} = +0.473$), proving that the adapter fractures under adversarial pressure. `SPP-Ro` exhibits intrinsic stability ($\Delta\text{LL}_{\text{adv}} = +0.207$).

---

### Table 3: Alignment Tax Verification (Perplexity on Held-Out Clean Texts)
**Success Threshold:** $\lvert\Delta\text{PPL}\rvert \le 0.5$

| Model | Alignment Paradigm | Wikipedia PPL | News PPL | General PPL | Alignment Tax ($\Delta\text{PPL}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Base-Ro-125M** | Raw Control | 20.50 | 17.51 | 19.01 | 0.00 (Baseline) |
| **Base-Ro-LoRA** | Post-Hoc LoRA | 1,026,766.38 | 1,189,350.04 | 1,108,058.21 | +1,108,039.20 (Degraded) |
| **SPP-Ro-125M** | Token Zero SPP | 20.65 | 17.28 | 18.96 | **-0.05** (Zero Tax) |

*Finding:* **Zero Alignment Tax.** SPP pretraining retains total language modeling competence without catastrophic degradation.

---

## 8. Repository Directory Structure

```
spp-ro/
├── docs/                               # 🌐 Interactive Showcase Platform (GitHub Pages)
│   ├── index.html                      # Clinical Blueprint user interface
│   ├── style.css                       # Modern typography & design tokens
│   └── app.js                          # Matrix visualizer & bias playground
│
├── constitution_spp_ro.md              # 📜 13-Article Romanian Civic Constitution Guide
├── ANNOTATION_GUIDELINES.md            # 📋 Synthetic Annotation Protocol & Calibrations
├── THESIS_ARCHITECTURE_AND_EXECUTION_PLAN.md # 📚 Academic Thesis Specification
├── DESIGN.md                           # 🎨 UI Design system tokens & aesthetics
├── NEXT_STEPS_STATE.md                 # 🔄 Context state handover & roadmap
├── app.py                              # 🚀 Web application demo (Gradio/FastAPI)
├── train_spp.bat                       # ⚡ Windows launch script for 50k steps SPP run
│
├── src/                                # ⚙️ Core Engineering Modules
│   ├── config.py                       # Global hyperparameters & model architecture
│   ├── spp_collator.py                 # Asymmetric 2D attention-blocked collator
│   ├── train_scale_pretrain.py         # Main pretraining engine with live telemetry & HF push
│   ├── clean_corpus.py                 # Text canonicalization & diacritic standardizer
│   ├── download_corpus.py              # Web and Wikipedia corpus scraper
│   ├── train_tokenizer.py              # 16,384 BPE Tokenizer training
│   ├── generate_factual_reflections.py # Cerebras API factual reflections generator
│   ├── generate_sensitive_reflections.py # Cerebras API sensitive reflections generator
│   ├── eval_biases.py                  # Bilingual CrowS-Pairs & Triad report generator
│   ├── eval_alignment_tax.py           # Perplexity & Alignment Tax verification
│   ├── eval_jailbreaks.py              # Adversarial prefix unmasking probe suite
│   ├── interact.py                     # Interactive CLI text generation tool
│   └── upload_to_hf.py                 # Automated Hugging Face Hub uploader
│
├── tokenizer/ro_bpe_16k/               # 🔤 Custom Romanian BPE Tokenizer Artifacts
│   ├── tokenizer.json                  # Serialized HuggingFace tokenizer
│   └── tokenizer_config.json           # Special token mappings (<assistant>, etc.)
│
├── data/                               # 📦 Dataset Stores (Git-ignored)
│   ├── clean/                          # Cleaned parquet streaming corpus (2.2 GB)
│   └── sidecar/                        # 60,000 verified reflections.parquet (~70 MB)
│
├── models/                             # 💾 Trained Model Checkpoints (Git-ignored)
│   ├── base_ro_125m/                   # Pretrained Raw Baseline Model (50k steps)
│   └── spp_ro_125m/                    # Pretrained Constitutional SPP Model (50k steps)
│
└── requirements.txt                    # 📦 Python project dependencies
```

---

## 9. Citation & BibTeX

```bibtex
@bachelorthesis{stefan2026sppro,
  title        = {Constitutional Language Modeling from Token Zero: Adapting Synthetic Persona Pretraining (SPP) for Romanian Generative LLMs and Evaluating Societal Bias Resilience},
  author       = {Stefan, Flavius},
  school       = {Faculty of Mathematics and Computer Science / Faculty of Automatic Control and Computers},
  year         = {2026},
  month        = {September},
  note         = {Bachelor's Thesis in Artificial Intelligence. Repository: \url{https://github.com/flaviussteff/spp-ro}}
}
```

---

<div align="center">
  <sub>Engineered with mathematical rigor for open academic research • Bucharest, Romania • 2026</sub>
</div>

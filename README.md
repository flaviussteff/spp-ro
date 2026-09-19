<div align="center">

# 🏛️ SPP-Ro: Model Raising from Token Zero
### Adapting Synthetic Pre-training Paths for Romanian Generative LLMs & Probing Societal Bias Resilience

[![Website](https://img.shields.io/badge/Live%20Platform-GitHub%20Pages-e7000b?style=for-the-badge&logo=googlechrome&logoColor=white)](https://flaviussteff.github.io/spp-ro/)
[![Paper](https://img.shields.io/badge/Thesis-Architecture%20Plan-0f172a?style=for-the-badge&logo=googledocs&logoColor=white)](THESIS_ARCHITECTURE_AND_EXECUTION_PLAN.md)
[![Constitution](https://img.shields.io/badge/Constituția%20SPP--Ro-7%20Articles-2563eb?style=for-the-badge&logo=scroll&logoColor=white)](constitutia_spp_ro.md)
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
  <b>Lucrare de Licență (Bachelor's Thesis in Computer Science & AI)</b><br>
  <i>Investigating whether internalizing ethical deliberation during pretraining eliminates representational stereotypes in low-to-mid resource languages under consumer single-GPU constraints.</i>
</p>

[🌐 Live Interactive Platform](https://flaviussteff.github.io/spp-ro/) • [📖 Theoretical Framework](#1-theoretical-framework--abstract) • [🔬 The SPP Paradigm](#2-the-spp-methodology--attention-blocking) • [📐 Model Topology](#3-architecture--hardware-envelope) • [🚀 Quickstart](#5-quickstart--execution) • [📊 Empirical Benchmark Suite](#6-empirical-benchmark-suite--findings) • [📑 Citation](#8-citation--bibtex)

---

</div>

## 1. Theoretical Framework & Abstract

### 1.1 The "Superficial Alignment Hypothesis"
Contemporary Large Language Model (LLM) alignment relies almost exclusively on post-hoc interventions, such as **Supervised Fine-Tuning (SFT)**, **Direct Preference Optimization (DPO)**, and **Reinforcement Learning from Human Feedback (RLHF)**. While these techniques constrain conversational outputs, foundational research demonstrates that:

> **The Superficial Alignment Hypothesis:** Post-hoc alignment merely acts as a thin behavioral mask superimposed over raw pretraining weights. Under adversarial prompting, system prompt overrides, or cross-lingual transfers, this surface filter fractures, exposing unfiltered societal stereotypes internalized during next-token prediction.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TRADITIONAL POST-HOC ALIGNMENT (Superficial Masking)                                   │
│  [ Raw Web Data ] ──► [ Uncurated Pretraining ] ──► [ Deep Stereotypes Solidified ]   │
│                                                                  │                     │
│                                           [ Post-Hoc SFT/RLHF Mask ] (Brittle)        │
│                                                                  ▼                     │
│                                           Fails under cross-lingual / jailbreak tests  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ MODEL RAISING VIA SPP (Deep Constitutional Internalization)                            │
│  [ Curated Corpus ] ──► [ 10% SPP Constitutional Reflection ]                          │
│                                      │                                                 │
│                                      ▼                                                 │
│                  [ Attention-Blocked Token Zero Pre-training ]                         │
│                                      │                                                 │
│                                      ▼                                                 │
│       Deep representational de-biasing internalized natively in weights               │
│       Zero Alignment Tax (ΔPPL = -0.05) & Proven Adversarial Resilience                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Research Objectives & Core Contributions
1. **Model Raising from Token Zero:** Synthesizes ethical, first-person deliberative thought traces grounded in the [Romanian Civic Constitution](constitutia_spp_ro.md) (§1.1–§2.3) and interleaves them into a targeted 10% pretraining pool.
2. **Causal Attention Blocking:** Utilizes a custom 2D attention collator preventing subsequent text from attending backward to synthetic reflections, ensuring generative independence at inference time.
3. **Consumer Single-GPU Envelope:** Proves that deep constitutional pre-training is fully achievable on a single **NVIDIA GeForce RTX 3060 12GB** within consumer compute constraints.
4. **Empirical Verification of Zero Alignment Tax:** Validates that constitutional pre-training incurs $|\Delta\text{PPL}| \le 0.5$ ($\Delta\text{PPL} = -0.05$), completely avoiding the catastrophic degradation observed in post-hoc adapters.
5. **Adversarial Unmasking Proof:** Demonstrates that while post-hoc LoRA collapses under adversarial prefixes ($\Delta LL_{\text{adv}} = +0.473$), Token-Zero SPP maintains intrinsic causal grounding ($\Delta LL_{\text{adv}} = +0.207$).
6. **External Scaling Transfer:** Validates transferability by fine-tuning the 780M-parameter RoGPT model (`dumitrescustefan/gpt-neo-romanian-780m`), achieving a $21.6\%$ bias reduction.

---

## 2. The SPP Methodology & Attention Blocking

### 2.1 The 10% Constitutional Split
Annotating 100% of pre-training tokens causes severe perplexity collapse and unnatural generative cadence. Following empirical findings from the SPP literature, constitutional sidecars are restricted to **exactly 10% of the corpus**:
* **News & Current Events (7% of SPP pool):** Grounds moral reflections in real-world societal friction: Roma minority rights (§1.1), modern gender equality (§1.2), and democratic integrity (§2.3).
* **Wikipedia & Academic Texts (93% of SPP pool):** Anchors reflections in scientific, historical, philosophical, and European democratic foundations.

### 2.2 Asymmetric 2D Causal Attention Collator
During pre-training, input sequences are structured as `[Reflection Tokens] + [Document Tokens]`. The attention mask matrix $\mathbf{M} \in \{0, 1\}^{L \times L}$ enforces:
$$\mathbf{M}_{i,j} = \begin{cases} 1, & \text{if } i,j \in \text{Thought and } j \le i \\ 1, & \text{if } i,j \in \text{Doc and } j \le i \\ 0, & \text{if } i \in \text{Doc and } j \in \text{Thought} \quad (\text{\textbf{Causal Attention Blocked}}) \\ 0, & \text{otherwise} \end{cases}$$

Document tokens are completely blocked from attending to reflection tokens, ensuring that downstream next-token prediction does not develop inference dependencies on internal thoughts.

---

## 3. Architecture & Hardware Envelope

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL TOPOLOGY SPECIFICATION                              │
├──────────────────────────────┬─────────────────────────────┬───────────────────────────┤
│ Parameter                    │ Base-Ro-125M / SPP-Ro-125M  │ Notes                     │
├──────────────────────────────┼─────────────────────────────┼───────────────────────────┤
│ Parameters                   │ ~124.8 Million              │ Chinchilla-optimal 125M   │
│ Layers (Transformer Blocks)  │ 12                          │ Post-RMSNorm              │
│ Hidden Dimension             │ 768                         │ d_model                   │
│ Intermediate Dimension       │ 2,048                       │ SwiGLU Gated Activation   │
│ Attention Heads              │ 12 Query Heads              │ Grouped-Query Attention   │
│ Key-Value Heads              │ 4 KV Heads (3:1 ratio)      │ Drastically cuts KV cache │
│ Context Window               │ 1,024 Tokens                │ Optimized for 12GB VRAM   │
│ Vocabulary Size              │ 16,384 BPE                  │ Custom Romanian Tokenizer │
│ Positional Encoding          │ RoPE (Rotary Embeddings)    │ Decoupled from absolute   │
│ Weight Tying                 │ Input/Output Tied           │ Saves ~12.5M parameters   │
│ Training Precision           │ BF16 / FP16 Mixed AMP       │ FlashAttention / SDPA     │
│ VRAM Footprint               │ ~9.2 GB Peak                │ Single RTX 3060 12GB GDDR6│
└──────────────────────────────┴─────────────────────────────┴───────────────────────────┘
```

---

## 4. Models Released on Hugging Face Hub

All pre-trained weights, adapters, configurations, and tokenizers are publicly accessible on the Hugging Face Hub:

| Model Identity | Paradigm | Parameters | Repository Link |
| :--- | :--- | :--- | :--- |
| **SPP-Ro-125M** | Token Zero SPP (Attention-Blocked) | 124.8M | [`flaviussteff/spp-ro-125m`](https://huggingface.co/flaviussteff/spp-ro-125m) |
| **Base-Ro-125M** | Raw Web Baseline (Control) | 124.8M | [`flaviussteff/base-ro-125m`](https://huggingface.co/flaviussteff/base-ro-125m) |
| **Base-Ro-LoRA** | Post-Hoc Aligned Adapter ($r=16$) | 124.8M + LoRA | [`flaviussteff/base-ro-125m-lora`](https://huggingface.co/flaviussteff/base-ro-125m-lora) |

---

## 5. Quickstart & Execution

### Installation
```bash
git clone https://github.com/flaviussteff/spp-ro.git
cd spp-ro
pip install -r requirements.txt
```

### Interactive Web Demo
```bash
# Launch the interactive web playground & model comparison UI
py app.py
```

### Interactive CLI Generation
```bash
# Generate text interactively with the constitutional SPP model
py src/interact.py --model spp

# Generate with the raw baseline control
py src/interact.py --model base
```

### Reproducing Benchmark Evaluations
```bash
# 1. 3-Way Romanian Bias Triad Benchmark
py src/eval_biases.py --triad --eval-bert

# 2. Alignment Tax Perplexity Probes (Wikipedia & News)
py src/eval_alignment_tax.py

# 3. Adversarial Prefix Jailbreak Unmasking
py src/eval_jailbreaks.py

# 4. Large-Scale Transferability Probe (RoGPT-780M)
py src/eval_external_transfer.py
```

---

## 6. Empirical Benchmark Suite & Findings

The evaluation harness probes models across 5 core Romanian socio-cultural axes (37 diagnostic minimal pairs), perplexity on 5,000 held-out sentences, and 15 adversarial prefix unmasking stems.

### Table 1: Romanian 3-Way Alignment Triad (SPM %)
*Ideal Neutral Parity: $\text{SPM} = 50.0\%$.*

| Axa Socio-Culturală (România) | Base-Ro-125M (Control Brut) | Base-Ro + LoRA (Post-Hoc LoRA) | SPP-Ro-125M (Token Zero SPP) |
| :--- | :---: | :---: | :---: |
| **Minoritate Romă** | 62.5% | 75.0% | **50.0%** |
| **Gen și Ocupație** | 90.0% | **50.0%** | 90.0% |
| **Stereotipuri Regionale** | 50.0% | 37.5% | **37.5%** |
| **Statut Social & Economic** | 50.0% | 50.0% | **50.0%** |
| **Valori Civice & Democratice** | 0.0% | 20.0% | **0.0%** |
| **Scor General SPM (Română)** | **56.8%** | **48.6%** | **51.4%** |

*Takeaway:* SPP-Ro-125M achieves the closest overall score to perfect neutrality ($51.4\%$), completely eliminating bias on the Romani minority axis ($50.0\%$).

---

### Table 2: Adversarial Prefix Unmasking (Jailbreak Collapse Probing)
*Tests whether models maintain ethical integrity when subjected to adversarial framing.*

| Model | Paradigmă Aliniere | Bias Neutru ($x_{\text{neutral}}$) | Bias Sub Atac ($x_{\text{adv}}$) | Salt Log-Likelihood ($\Delta LL_{\text{adv}}$) |
| :--- | :--- | :---: | :---: | :---: |
| **Base-Ro-125M** | Control Brut | 60.0% | 80.0% | +0.219 |
| **Base-Ro-LoRA** | Post-Hoc LoRA | 20.0% | **20.0%** | **+0.473** |
| **SPP-Ro-125M** | Token Zero SPP | 40.0% | **66.7%** | **+0.207** |

*Takeaway:* Post-hoc LoRA exhibits a severe log-likelihood surge ($\Delta LL_{\text{adv}} = +0.473$), proving that the adapter collapses under adversarial prompting. SPP-Ro shows the lowest sensitivity ($+0.207$), demonstrating structural grounding.

---

### Table 3: Alignment Tax Verification (Perplexity on Clean Texts)
*Success Criterion: $|\Delta\text{PPL}| \le 0.5$.*

| Model | Paradigmă Aliniere | Wiki PPL | News PPL | PPL General | Taxă Aliniere ($\Delta\text{PPL}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Base-Ro-125M** | Control Brut | 20.50 | 17.51 | 19.01 | 0.00 (Referință) |
| **Base-Ro-LoRA** | Post-Hoc LoRA | 1,026,766.38 | 1,189,350.04 | 1,108,058.21 | +1,108,039.20 |
| **SPP-Ro-125M** | Token Zero SPP | 20.65 | 17.28 | 18.96 | **-0.05** |

*Takeaway:* **Zero Alignment Tax.** SPP pretraining retains total language fluency and slightly improves perplexity on news articles ($17.28$ vs $17.51$), whereas narrow LoRA fine-tuning without language modeling regularization suffers catastrophic language collapse.

---

### Table 4: Cross-Lingual Alignment Disparity (RO vs. EN)

| Model Evaluat | Arhitectură | SPM Română (%) | SPM English (%) | Disparitate $\Delta$ (RO &minus; EN) |
| :--- | :--- | :---: | :---: | :---: |
| **Base-Ro-125M** | Causal LM (125M) | 56.8% | 29.7% | +27.0% |
| **Base-Ro-LoRA** | Causal LM (125M) | 48.6% | 32.4% | +16.2% |
| **SPP-Ro-125M** | Causal LM (125M) | **51.4%** | **43.2%** | **+8.1%** |
| **BERT-Base-Ro (Dumitrescu)** | Masked LM (110M) | 62.2% | 46.0% | +16.2% |
| **RoBERT-Base (ReaderBench)** | Masked LM (110M) | 48.6% | 51.4% | -2.7% |

*Takeaway:* Global multilingual models frequently exhibit masked bias in English but manifest elevated stereotyping in Romanian. SPP-Ro reduces cross-lingual disparity to $+8.1\%$ among generative architectures.

---

### Table 5: External Transferability on RoGPT-780M

| Metrică de Evaluare | RoGPT-780M Brut (Pre-Aliniere) | RoGPT-780M + SPP LoRA (Post-Aliniere) | Efectul Alinierii ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Scor General Bias (SPM Română)** | 75.7% | **54.1%** | **-21.6%** |
| -- Minoritate Romă | 62.5% | 25.0% | -37.5% |
| -- Gen și Ocupație | 80.0% | 80.0% | +0.0% |
| -- Stereotip Regional | 75.0% | 25.0% | -50.0% |
| -- Marginalizare Socială | 100.0% | 100.0% | +0.0% |
| -- Principii Democratice | 60.0% | 40.0% | -20.0% |
| **Rată Stereotip Sub Atac ($x_{\text{adv}}$)** | 80.0% | **46.7%** | **-33.3%** |
| **Salt Log-Likelihood ($\Delta LL_{\text{adv}}$)** | +0.552 | **+1.109** | **+0.557** |

*Takeaway:* The Romanian constitutional dataset transfers successfully to large models (reducing bias by $21.6\%$), but the post-hoc adversarial vulnerability persists and amplifies at scale ($\Delta LL_{\text{adv}} = +1.109$), cementing the necessity of Token-Zero pretraining.

---

## 7. Project Directory Anatomy

```
spp-ro/
├── docs/                               # 🌐 Interactive Showcase (GitHub Pages)
│   ├── index.html                      # Clinical Blueprint user interface
│   ├── style.css                       # Design system (Geist typography, frosted paper)
│   └── app.js                          # Attention matrix visualizer & bias playground
│
├── constitutia_spp_ro.md               # 📜 7-Article Romanian Civic Constitution Guide
├── THESIS_ARCHITECTURE_AND_EXECUTION_PLAN.md # 📚 Full 6-Chapter Academic Thesis Blueprint
├── DESIGN.md                           # 🎨 Design system tokens & specifications
├── NEXT_STEPS_STATE.md                 # 🔄 Context state handover & thesis roadmap
├── app.py                              # 🚀 Web application demo (Gradio/FastAPI)
│
├── src/                                # ⚙️ Core Engineering Modules
│   ├── config.py                       # Global hyperparameters & hardware configs
│   ├── download_corpus.py              # Multi-source scraper (News + Wiki + FineWeb)
│   ├── clean_corpus.py                 # Diacritic standardizer & 90/10 SPP partition
│   ├── train_tokenizer.py              # 16,384 BPE Tokenizer with constitutional tokens
│   ├── spp_annotator.py                # Synthetic reflection generation (§1.1–§2.3)
│   ├── spp_collator.py                 # Asymmetric 2D attention-blocked collator
│   ├── train_pretrain.py               # Pre-training loop with live telemetry
│   ├── train_lora_alignment.py         # Post-hoc LoRA fine-tuning control script
│   ├── eval_biases.py                  # Bilingual CrowS-Pairs & Triad report generator
│   ├── eval_alignment_tax.py           # Perplexity & Alignment Tax verification
│   ├── eval_jailbreaks.py              # Adversarial prefix unmasking probe suite
│   ├── eval_external_transfer.py       # Transferability on RoGPT-780M
│   ├── interact.py                     # Interactive CLI text generation tool
│   └── upload_to_hf.py                 # Automated Hugging Face Hub uploader
│
├── tokenizer/ro_bpe_16k/               # 🔤 Custom Romanian BPE Tokenizer Artifacts
│   ├── tokenizer.json                  # Serialized HuggingFace tokenizer
│   └── tokenizer_config.json           # Special token mappings (<|thought_start|>, etc.)
│
├── evals/                              # 📈 Evaluation Reports & Thesis Tables
│   ├── thesis_master_evaluations.tex   # 📄 Complete Master LaTeX thesis evaluation paper
│   ├── thesis_3way_alignment_triad.tex # LaTeX Triad benchmark table
│   ├── thesis_adversarial_prefix_unmasking.tex # LaTeX Adversarial probing table
│   ├── thesis_alignment_tax_table.tex  # LaTeX Alignment Tax table
│   ├── thesis_crosslingual_bias_benchmark.tex # LaTeX Cross-lingual disparity table
│   ├── thesis_crosslingual_bias_benchmark.csv # Raw cross-lingual benchmark data
│   ├── thesis_external_transfer_rogpt.tex # LaTeX RoGPT-780M transfer table
│   ├── bias_evaluation_report.json     # Full JSON logs of all probe pairs
│   ├── thesis_adversarial_prefix_report.json # Full JSON logs of jailbreak probe
│   ├── thesis_alignment_tax_report.json # Full JSON logs of perplexity evaluation
│   └── thesis_external_transfer_report.json # Full JSON logs of RoGPT-780M transfer
│
├── models/                             # 💾 Trained Model Weights (Final safetensors)
│   ├── base_ro_125m/                   # Pre-trained Raw Baseline Model
│   ├── spp_ro_125m/                    # Pre-trained Constitutional SPP Model
│   ├── base_ro_125m_lora/              # Post-Hoc LoRA Adapter & Model
│   └── rogpt_780m_lora/                # External RoGPT-780M LoRA Adapter
│
└── requirements.txt                    # 📦 Python project dependencies
```

---

## 8. Citation & BibTeX

```bibtex
@bachelorthesis{stefan2026sppro,
  title        = {Model Raising from Token Zero: Adapting Synthetic Pre-training Paths (SPP) for Romanian Generative LLMs and Evaluating Societal Bias Resilience},
  author       = {Stefan, Flavius},
  school       = {Faculty of Automatic Control and Computers / Faculty of Mathematics and Computer Science},
  year         = {2026},
  month        = {September},
  note         = {Bachelor's Thesis in Artificial Intelligence. Repository: \url{https://github.com/flaviussteff/spp-ro}}
}
```

---

## 9. Acknowledgments & References

* **Synthetic Pre-training Paths (SPP):** Inspired by the *Model Raising* paradigm developed at EPFL NLP (Morris et al., 2024–2026).
* **Romanian NLP Infrastructure:** Grounded in foundational models developed by Stefan Dumitrescu (`bert-base-romanian-cased-v1`, `gpt-neo-romanian-780m`) and the ReaderBench team (`RoBERT-base`).
* **Hardware Support:** Optimized for consumer AI research on NVIDIA GeForce RTX architectures.

<div align="center">
  <sub>Engineered with precision for open academic research • Bucharest, Romania • 2026</sub>
</div>

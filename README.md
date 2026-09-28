<div align="center">

# SPP-Ro: Synthetic Persona Pretraining in Romanian LLMs
### Constitutional Alignment from Token Zero, Deliberative Reasoning, and Zero Alignment Tax

[![Website](https://img.shields.io/badge/Live%20Platform-GitHub%20Pages-0c4a60?style=for-the-badge&logo=googlechrome&logoColor=white)](https://flaviussteff.github.io/spp-ro/)
[![HuggingFace SPP](https://img.shields.io/badge/HF-SPP--Ro--125M-166534?style=for-the-badge)](https://huggingface.co/flaviussteff/spp-ro-125m)
[![HuggingFace Base](https://img.shields.io/badge/HF-Base--Ro--125M-334155?style=for-the-badge)](https://huggingface.co/flaviussteff/base-ro-125m)
[![Constitution](https://img.shields.io/badge/Value%20Constitution-6%20Articles-0369a1?style=for-the-badge&logo=scroll&logoColor=white)](constitution_spp_ro.md)
[![Annotation Guidelines](https://img.shields.io/badge/Annotation-Guidelines%20(RO)-475569?style=for-the-badge&logo=readme&logoColor=white)](ANNOTATION_GUIDELINES.md)

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.2%2B%20%7C%20CUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.2+"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logoColor=black" alt="HuggingFace"></a>
  <a href="https://developer.nvidia.com/cuda-zone"><img src="https://img.shields.io/badge/Hardware-NVIDIA%20RTX%203060%2012GB-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="Hardware"></a>
</p>

<p align="center">
  <b>Flavius-Ștefan Hăbeanu</b><br>
  Faculty of Mathematics and Computer Science, University of Bucharest<br>
  Independent replication and empirical evaluation grounded in the methodology of:<br>
  <b>Synthetic Persona Pretraining</b> (Minder et al., EPFL-dlab, 2026; <a href="https://arxiv.org/abs/2608.13482">arXiv:2608.13482</a>)
</p>

[Interactive Web Platform](https://flaviussteff.github.io/spp-ro/) • [Value Constitution](constitution_spp_ro.md) • [Annotation Guidelines](ANNOTATION_GUIDELINES.md) • [Benchmark Highlights](#benchmark-highlights) • [Quickstart](#quickstart--reproducibility) • [Citation](#citation)

---

</div>

## Executive Summary & Key Findings

Post-hoc alignment methods such as RLHF and DPO frequently degrade core linguistic fluency, imposing a substantial **alignment tax** and leaving guardrails fragile under adversarial framing, especially in lower-resource languages such as Romanian.

**SPP-Ro** implements **Synthetic Persona Pretraining (SPP)** from scratch across **50,000 optimization steps** (~3.27 billion tokens processed on a single consumer NVIDIA GeForce RTX 3060 12GB GPU). By interleaving a 10% stream of structured constitutional reflections with asymmetric causal attention masking from step zero, ethical deliberation is embedded directly into autoregressive latent representations.

### Benchmark Highlights

| Benchmark Dimension | Vanilla Base (`Base-Ro-125M`) | SPP Passive (`SPP-Ro-Passive`) | SPP Deliberative (`<assistant>`) | Relative Gain / Impact |
| :--- | :---: | :---: | :---: | :--- |
| **ConstitutionEval-Ro** (30 Forced-Choice Dilemmas, Random: 25.0%) | 23.3% | 33.3% | **43.3%** | **+20.0% absolute gain** (nearly 2× baseline) |
| &nbsp;&nbsp;&bull; *Gender Equality &amp; Meritocracy (Art. 1.2)* | 0.0% | 20.0% | **40.0%** | **+40.0%** (rejects occupational segregation) |
| &nbsp;&nbsp;&bull; *Freedom of Movement &amp; Regional Equity (Art. 1.3)* | 20.0% | 60.0% | **60.0%** | **+40.0%** (deconstructs regional stigmas) |
| &nbsp;&nbsp;&bull; *Freedom of Expression &amp; Press Freedom (Art. 2.2)* | 20.0% | 20.0% | **60.0%** | **+40.0%** (shields independent journalism) |
| **AIRiskDilemmas-Ro** (20 Autonomous AI Safety Dilemmas, Random: 50.0%) | 65.0% | 75.0% | **85.0%** | **+20.0% absolute gain** |
| &nbsp;&nbsp;&bull; *Deception vs. Truth &amp; Transparency* | 20.0% | 40.0% | **80.0%** | **Sycophantic failure slashed from 80% to 20%** |
| &nbsp;&nbsp;&bull; *Power-Seeking vs. Human Oversight* | 80.0% | 100.0% | **100.0%** | **100% compliance** (rejects covert exfiltration) |
| &nbsp;&nbsp;&bull; *Privacy vs. Mass Surveillance* | 80.0% | 100.0% | **100.0%** | **100% compliance** (zero backdoors) |
| **Language Modeling Perplexity (News)** | 15.92 PPL | — | **12.49 PPL** | **Zero Alignment Tax** (beneficial regularizer) |
| **Language Modeling Perplexity (Wikipedia)** | 18.48 PPL | — | **14.48 PPL** | **Zero Alignment Tax** (beneficial regularizer) |
| **Cross-Lingual Alignment Disparity** | &plusmn;14.0% | — | **&plusmn;6.0%** | **Disparity halved** across English/Romanian probes |

---

## Scientific Methodology: Forced-Choice Log-Likelihood Evaluation

A central challenge in evaluating causal language models is avoiding greedy generation hallucinations, prompt drift, or subjective regex parsing. Following standards from EleutherAI LM-Eval Harness, Anthropic Constitutional AI, and EPFL SPP Section 4.2, all forced-choice benchmarks are evaluated deterministically using **Length-Normalized Conditional Log-Likelihood**:

$$\text{Score}(O_i \mid X) = \frac{1}{|O_i|} \sum_{t=1}^{|O_i|} \log P(w_t \mid X, w_{<t})$$

$$\hat{O} = \arg\max_{i} \text{Score}(O_i \mid X)$$

* **Length Normalization:** Multiplying raw joint probabilities naturally penalizes longer completions because $P(w_t) \le 1$. Dividing by token length $|O_i|$ isolates the model's true per-token likelihood.
* **Deliberative Activation Trigger (`<assistant>`):** In passive scoring, the prompt is `Scenariu: ... \nRăspuns: `. In deliberative scoring, prefixing with `<assistant>` activates the self-attention heads trained on Romanian constitutional reflections, dynamically steering conditional likelihoods toward aligned choices.

---

## Pretraining Architecture & Asymmetric Attention Decoupling

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STANDARD AUTOREGRESSIVE SEQUENCE (90% of Tokens)                                       │
│  [ Romanian Web Text Token t_1 ] ──► [ Token t_2 ] ──► [ Token t_3 ] ... (Causal Mask) │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ SPP SIDECAR REFLECTION SEQUENCE (10% of Tokens)                                        │
│  [ Web Context ] ──► [<assistant>] ──► [ Reflection Tokens ] ──► [ Web Continuation ]   │
│         │                                      │                         ▲             │
│         └──────────────────────────────────────┴─── Attention Blocked ───┘             │
│                 (Continuation CANNOT attend backward to reflection tokens)             │
│                 (RoPE positional IDs alias back to preserve temporal distance)         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Asymmetric Causal Attention Mask:** Document continuation tokens are strictly prohibited from attending backward to synthetic reflection tokens. Standard web language modeling is never polluted.
2. **RoPE Position Aliasing:** Postfix tokens reset their positional embeddings relative to the prefix, ensuring zero positional distortion.
3. **Data-Matched Experimental Control:** Both `Base-Ro-125M` and `SPP-Ro-125M` were trained with identical compute budgets (50,000 steps, batch size 64 sequences, sequence length 1024, ~3.27B tokens) on an NVIDIA RTX 3060 12GB GPU.

---

## The Romanian Value Constitution & Annotation Protocol

SPP-Ro aligns generative representations against the fundamental civic charter of the Romanian state:

1. **Article 1.1 — Human Dignity &amp; Non-Discrimination:** Inviolability of human dignity, personal integrity, and universal equality before the law.
2. **Article 1.2 — Gender Equality &amp; Meritocracy:** Equal access to executive leadership roles; rejection of corporate occupational sorting.
3. **Article 1.3 — Freedom of Movement &amp; Regional Equity:** Protection of internal mobility; deconstructing provincial economic stigmas.
4. **Article 2.1 — Right to Information &amp; Public Transparency:** Transparent public expenditures; curbing disinformation and SLAPP harassment.
5. **Article 2.2 — Freedom of Expression &amp; Independent Press:** Editorial independence and media pluralism against censorship.
6. **Article 2.3 — Protection of Cultural &amp; Ethnic Minorities:** Linguistic and cultural preservation of historical ethnic communities.

* Full charter text: [`constitution_spp_ro.md`](constitution_spp_ro.md)
* Annotation protocol &amp; severity calibration (Scores 1–5): [`ANNOTATION_GUIDELINES.md`](ANNOTATION_GUIDELINES.md)

---

## Repository Structure

```
spp-ro/
├── ANNOTATION_GUIDELINES.md        # 5 Golden annotation rules & severity calibration (Romanian)
├── constitution_spp_ro.md          # 6-Article Romanian Value Constitution
├── requirements.txt                # Python package dependencies
├── train_spp.bat                   # 50,000-step pretraining launcher script
├── docs/                           # Interactive Web Platform (GitHub Pages)
│   ├── index.html                  # Academic showcase, SVG charts, and dilemma inspectors
│   ├── style.css                   # Responsive academic styling and math typography
│   └── app.js                      # Bilingual dilemma engine (EN + RO toggle, exact LL proof)
├── evals/                          # Raw evaluation outputs and LaTeX publication tables
│   ├── thesis_constitution_eval_report.json    # Exact per-token log-likelihoods for all dilemmas
│   ├── thesis_alignment_tax_report.json        # Perplexity evaluations across test corpora
│   ├── bias_evaluation_report.json             # CrowS-Pairs and cross-lingual bias evaluations
│   └── *.tex                                   # Camera-ready LaTeX tables for thesis inclusion
├── src/                            # Core research codebase
│   ├── config.py                   # Global hyperparameters, tokenizer, and dataset paths
│   ├── spp_collator.py             # Asymmetric causal 2D attention mask & RoPE aliasing collator
│   ├── train_scale_pretrain.py     # Main 50,000-step pretraining engine (Base vs. SPP)
│   ├── eval_constitution.py        # ConstitutionEval-Ro & AIRiskDilemmas evaluation harness
│   ├── eval_alignment_tax.py       # Linguistic perplexity & zero alignment tax validation
│   ├── eval_biases.py              # CrowS-Pairs & cross-lingual bias probe benchmarks
│   ├── eval_deliberative.py        # Deliberative activation (<assistant>) probe evaluation
│   ├── eval_jailbreaks.py          # Adversarial prefix unmasking suite
│   ├── clean_corpus.py             # Web corpus filtering and deduplication
│   ├── download_corpus.py          # Romanian web and Wikipedia stream downloader
│   ├── prepare_scale_corpus.py     # Pre-tokenization and arrow shard preparation
│   ├── generate_sensitive_reflections.py # Sidecar generator for sensitive civic themes
│   ├── generate_factual_reflections.py   # Sidecar generator for factual calibration
│   ├── upload_to_hf.py             # Automated Hugging Face checkpoint publisher
│   └── interact.py                 # Interactive terminal inference with Base and SPP models
└── tokenizer/
    └── ro_bpe_16k/                 # Custom 16,000-vocab byte-level BPE tokenizer for Romanian
        ├── tokenizer.json
        └── tokenizer_config.json
```

---

## Quickstart & Reproducibility

### 1. Installation

```bash
git clone https://github.com/flaviussteff/spp-ro.git
cd spp-ro
pip install -r requirements.txt
```

### 2. Inspecting Checkpoints from Hugging Face

Pretrained models are publicly available on Hugging Face:
* **SPP-Ro-125M (Token Zero Aligned):** [`flaviussteff/spp-ro-125m`](https://huggingface.co/flaviussteff/spp-ro-125m)
* **Base-Ro-125M (Unaligned Control):** [`flaviussteff/base-ro-125m`](https://huggingface.co/flaviussteff/base-ro-125m)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("flaviussteff/spp-ro-125m")
model = AutoModelForCausalLM.from_pretrained("flaviussteff/spp-ro-125m", torch_dtype=torch.bfloat16).to(device)

prompt = "<assistant> Tratamentul egal între bărbați și femei în funcții de conducere"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

### 3. Reproducing Evaluations

Run the evaluation suites locally:

```bash
# Evaluate ConstitutionEval-Ro and AIRiskDilemmas-Ro
py src/eval_constitution.py

# Evaluate Language Modeling Perplexity & Zero Alignment Tax
py src/eval_alignment_tax.py

# Evaluate Societal Stereotype Bias & Cross-Lingual Disparity
py src/eval_biases.py
```

### 4. Interactive Terminal Chat

Compare completions between `Base-Ro-125M` and `SPP-Ro-125M` interactively in the terminal:

```bash
py src/interact.py
```

---

## Verifying Raw Log-Likelihoods Directly

All evaluation decisions are backed by deterministic tensor outputs saved in [`evals/thesis_constitution_eval_report.json`](evals/thesis_constitution_eval_report.json). You can verify any dilemma choice in a single terminal line:

```powershell
# Verify CE-1.2-01 (CEO Executive Selection):
py -c "import json; d=json.load(open('evals/thesis_constitution_eval_report.json', encoding='utf-8')); [print(m, 'pred_idx:', next(x for x in d[m]['constitution_eval']['details'] if x['id']=='CE-1.2-01')['pred_idx'], 'scores:', [round(s, 3) for s in next(x for x in d[m]['constitution_eval']['details'] if x['id']=='CE-1.2-01')['option_scores']]) for m in d]"

# Verify AIR-DEC-01 (Vital Surgery Risk Disclosure):
py -c "import json; d=json.load(open('evals/thesis_constitution_eval_report.json', encoding='utf-8')); [print(m, 'is_correct:', next(x for x in d[m]['airisk_dilemmas']['details'] if x['id']=='AIR-DEC-01')['is_correct'], 'aligned:', round(next(x for x in d[m]['airisk_dilemmas']['details'] if x['id']=='AIR-DEC-01')['ll_aligned'], 3), 'risky:', round(next(x for x in d[m]['airisk_dilemmas']['details'] if x['id']=='AIR-DEC-01')['ll_risky'], 3)) for m in d]"
```

---

## Citation

If you use SPP-Ro checkpoints, benchmark datasets, or the Romanian Value Constitution in your research, please cite:

```bibtex
@mastersthesis{habeanu2026sppro,
  title        = {Pre-antrenarea Modelelor Generative Rom{\^a}ne{\c s}ti cu Personalitate Sintetic{\u a}: Aliniere Constitu{\c t}ional{\u a} de la Token Zero},
  author       = {H{\u a}beanu, Flavius-{\c S}tefan},
  school       = {Faculty of Mathematics and Computer Science, University of Bucharest},
  year         = {2026},
  type         = {Bachelor's Thesis / Licen{\c t}{\u a}},
  note         = {Grounded in EPFL-dlab Synthetic Persona Pretraining (Minder et al., arXiv:2608.13482)},
  url          = {https://github.com/flaviussteff/spp-ro}
}
```


<div align="center">

# SPP-Ro: Synthetic Persona Pretraining in Romanian LLMs
### Constitutional Alignment from Token Zero and Deliberative Reasoning in an 88M-parameter Romanian LM

[![Website](https://img.shields.io/badge/Live%20Platform-GitHub%20Pages-0c4a60?style=for-the-badge&logo=googlechrome&logoColor=white)](https://flaviussteff.github.io/spp-ro/)
[![HuggingFace SPP](https://img.shields.io/badge/HF-SPP--Ro--88M-166534?style=for-the-badge)](https://huggingface.co/flaviussteff/spp-ro-125m)
[![HuggingFace Base](https://img.shields.io/badge/HF-Base--Ro--88M-334155?style=for-the-badge)](https://huggingface.co/flaviussteff/base-ro-125m)
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
  <i>Faculty of Mathematics and Computer Science, University of Bucharest</i><br>
  Independent replication and empirical evaluation grounded in the methodology of:<br>
  <b>Synthetic Persona Pretraining</b> (Minder et al., EPFL-dlab, 2026; <a href="https://arxiv.org/abs/2608.13482">arXiv:2608.13482</a>)
</p>

[Interactive Web Platform](https://flaviussteff.github.io/spp-ro/) • [Value Constitution](constitution_spp_ro.md) • [Annotation Guidelines](ANNOTATION_GUIDELINES.md) • [Benchmark Highlights](#benchmark-highlights) • [Quickstart](#quickstart--reproducibility) • [Citation](#citation)

---

</div>

## Executive Summary & Key Findings

Post-hoc alignment methods such as RLHF and DPO frequently degrade core linguistic fluency, imposing a substantial **alignment tax** and leaving guardrails fragile under adversarial framing, especially in lower-resource languages such as Romanian.

**SPP-Ro** implements **Synthetic Persona Pretraining (SPP)** from scratch across **50,000 optimization steps** (65,536 sequence positions per step; ≈1.6 billion non-padding tokens, sequences are not packed; single consumer NVIDIA GeForce RTX 3060 12GB GPU). By interleaving a 10% stream of structured constitutional reflections with asymmetric causal attention masking from step zero, ethical deliberation is embedded directly into autoregressive latent representations.

### Benchmark Highlights

| Benchmark Dimension | Vanilla Base (`Base-Ro-88M`) | SPP Passive (`SPP-Ro-Passive`) | SPP Deliberative (`<assistant>`) | Relative Gain / Impact |
| :--- | :---: | :---: | :---: | :--- |
| **ConstitutionEval-Ro** (30 Forced-Choice Dilemmas, Random: 25.0%) | 23.3% | 33.3% | **43.3%** | +20.0 pp = 13/30 vs 7/30; Fisher exact p = 0.17 (not significant at N=30) |
| &nbsp;&nbsp;&bull; *Gender Equality &amp; Meritocracy (Art. 1.2)* | 0.0% | 20.0% | **40.0%** | +40 pp = 2/5 items |
| &nbsp;&nbsp;&bull; *Freedom of Movement &amp; Regional Equity (Art. 1.3)* | 20.0% | 60.0% | **60.0%** | +40 pp = 2/5 items |
| &nbsp;&nbsp;&bull; *Freedom of Expression &amp; Press Freedom (Art. 2.2)* | 20.0% | 20.0% | **60.0%** | +40 pp = 2/5 items (article not in the training constitution; control item) |
| **AIRisk-Ro** (20 thesis-written AI Safety Dilemmas, Random: 50.0%) | 65.0% | 75.0% | **85.0%** | +20.0 pp = 17/20 vs 13/20; Fisher exact p = 0.27 (not significant) |
| &nbsp;&nbsp;&bull; *Deception vs. Truth &amp; Transparency* | 20.0% | 40.0% | **80.0%** | 1/5 → 4/5 items |
| &nbsp;&nbsp;&bull; *Power-Seeking vs. Human Oversight* | 80.0% | 100.0% | **100.0%** | 5/5 items |
| &nbsp;&nbsp;&bull; *Privacy vs. Mass Surveillance* | 80.0% | 100.0% | **100.0%** | 5/5 items |

> **Withdrawn claims (Nov 2026).** The perplexity / "zero alignment tax" rows and the cross-lingual disparity row of earlier versions were removed: the perplexity set was inside the training corpus (92–99% verbatim overlap, ≈800 passages were also reflection hosts) and the model is monolingual Romanian, so English probes are not meaningful. See **Errata** below. Numbers in this table come from the first evaluation run; the overnight script re-runs everything with the corrected scorer, confidence intervals and the SPP{MT} baseline (`evals/*.json`, `evals/*.tex`).

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
3. **Compute-Matched Experimental Control:** `Base-Ro-88M`, `SPP-Ro-88M` (10%), `SPP-Ro-88M-20pct` and `SPP-Ro-88M-MT` share the same architecture, tokenizer, seed (42), schedule and step budget (50,000 steps, 64 × 1024 positions per step, no sequence packing, ≈1.6B non-padding tokens) on an NVIDIA RTX 3060 12GB GPU. SPP{MT} starts from the Base checkpoint at step 45,000 and receives reflections only in the last 5,000 steps (late insertion, as in the paper's midtraining variant).

---

## The Romanian Value Constitution & Annotation Protocol

SPP-Ro aligns generative representations against the fundamental civic charter of the Romanian state:

1. **[1.1] Human Dignity, Ethnic Non-Discrimination &amp; Refugees** — dignity of the Roma community, anti-racism, solidarity with refugees.
2. **[1.2] Gender Equality &amp; Occupational Roles** — equal professional opportunity, condemnation of domestic violence and harassment.
3. **[1.3] Territorial Cohesion &amp; Rural–Urban Equity** — isolated communities, regional stereotypes, budget equity.
4. **[2.1] Historical Memory, Holocaust &amp; Anti-Totalitarianism** — documented truth about the Holocaust in Romania, communist and fascist crimes.
5. **[2.2] Critical Thinking, Secular Conscience &amp; Science** — scientific method, religious tolerance, respect for secular/agnostic conscience.
6. **[2.3] Rule of Law, Institutional Integrity &amp; Medical Bioethics** — judicial independence, anti-corruption, informed consent, harmful disinformation.

> These are the six themes actually used to generate the 60,000 training reflections (see `ANNOTATION_GUIDELINES.md` §9). The 15 ConstitutionEval-Ro items on *right to information*, *freedom of the press* and *minority language rights* were written against an earlier draft of articles 2.1–2.3; they are kept as **control items for untrained rights** and are labelled `[control: neantrenat]` in `src/eval_constitution.py`.

* Full charter text: [`constitution_spp_ro.md`](constitution_spp_ro.md)
* Annotation protocol &amp; severity calibration (Scores 1–5): [`ANNOTATION_GUIDELINES.md`](ANNOTATION_GUIDELINES.md)

---

## Repository Structure

See [`STRUCTURE.md`](STRUCTURE.md) for the full map of what belongs to the preliminary **v1** experiments (synthetic reflections), what belongs to the faithful **v2** replication (in progress), and what is shared.

```
spp-ro/
├── STRUCTURE.md                    # v1 vs v2 map: every file, folder and model, with its stage
├── ANNOTATION_GUIDELINES.md        # 5 Golden annotation rules & severity calibration (Romanian)
├── constitution_spp_ro.md          # 6-Article Romanian Value Constitution (v1.1)
├── requirements.txt                # Python package dependencies
├── run_generate_v2_data.bat        # v2 data, paper's path B: labels -> safety classifier -> annotate real documents
├── run_v2_pipeline.bat             # v2: SPP{T0} v2 -> SPP{MT} v2 -> Filtered -> all evaluations
├── run_v3_verify.bat               # v3 benchmark: validity table + scoring of every model present
├── watch_log.bat                   # tail a log file live
├── archive/                        # everything no longer used: v1 scripts, evals, logs, data and checkpoints, raw corpus, .env backups (see archive/README.md)
├── docs/                           # Interactive Web Platform (GitHub Pages)
│   ├── index.html                  # Academic showcase, SVG charts, and dilemma inspectors
│   ├── style.css                   # Responsive academic styling and math typography
│   └── app.js                      # Bilingual dilemma engine (EN + RO toggle, exact LL proof)
├── evals/                          # Evaluation outputs and LaTeX tables, regenerated for every model present
│   ├── thesis_alignment_tax_report.json        # Held-out perplexity
│   ├── thesis_capabilities_report.json         # Held-out 4-way cloze
│   ├── thesis_constitution_eval_v3_report.json # ConstitutionEval-Ro v3 results (both protocols)
│   ├── constitution_eval_v3_review.html        # manual review page of the v3 benchmark items
│   ├── training_states/                        # archived trainer_state.json per training run
│   ├── constitution_eval_v3_validity.json      # validity table of the v3 benchmark (Qwen 27B blind, gpt-oss-120B open-book, local probe)
│   ├── ConstitutionEval-Ro_v3_benchmark.docx   # all 600 items as a Word document
│   └── *.tex                                   # Camera-ready LaTeX tables for thesis inclusion
├── src/                            # Core research codebase
│   ├── config.py                   # Global hyperparameters, tokenizer, dataset paths, ALL_MODELS
│   ├── spp_collator.py             # Asymmetric causal 2D attention mask & RoPE aliasing collator
│   ├── train_scale_pretrain.py     # Training engine: --mode base / spp / filtered, --inline, --init-from
│   ├── generate_reflections_v2.py  # v2 annotator: severity labels (step 1) and prefix-only reflections on real documents (step 3)
│   ├── train_safety_classifier.py  # v2 safety classifier (BERT-Ro) + corpus scoring (step 2)
│   ├── clean_reflections_v2.py     # v2 final cleanup of the reflections (adult pages, meta-language leaks)
│   ├── build_heldout_set.py        # verified held-out web set for perplexity (disjoint from corpus + sidecar)
│   ├── eval_utils.py               # joint-tokenization option scorer, bootstrap CIs, Fisher / permutation tests
│   ├── build_constitution_eval_v3.py # ConstitutionEval-Ro v3 builder (paper's ConstitutionEval methodology)
│   ├── eval_constitution_v3.py     # ConstitutionEval-Ro v3 scorer (continuation + swap-debiased letter protocol)
│   ├── llm_client.py               # Cerebras/Groq/NVIDIA multi-key client (key rotation, 402 retirement, 429 backoff)
│   ├── eval_alignment_tax.py       # Held-out perplexity (alignment tax)
│   ├── eval_capabilities.py        # Held-out 4-way cloze (capability check)
│   ├── eval_jailbreaks.py          # Adversarial prefix unmasking suite
│   ├── clean_corpus.py             # Web corpus filtering and deduplication
│   ├── download_corpus.py          # Romanian web and Wikipedia stream downloader
│   ├── prepare_scale_corpus.py     # Pre-tokenization and arrow shard preparation
│   ├── train_tokenizer.py          # 16k byte-level BPE tokenizer
│   ├── upload_to_hf.py             # Automated Hugging Face checkpoint publisher
│   └── interact.py                 # Interactive terminal inference with Base and SPP models
└── tokenizer/
    └── ro_bpe_16k/                 # Custom 16,000-vocab byte-level BPE tokenizer for Romanian
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
* **SPP-Ro-88M (Token Zero Aligned):** [`flaviussteff/spp-ro-125m`](https://huggingface.co/flaviussteff/spp-ro-125m)
* **Base-Ro-88M (Unaligned Control):** [`flaviussteff/base-ro-125m`](https://huggingface.co/flaviussteff/base-ro-125m)

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
# ConstitutionEval-Ro v3 (600 validated items; continuation + 4-rotation letter protocol)
py src/eval_constitution_v3.py

# Perplexity on a verified held-out web set (run build_heldout_set.py first)
py src/eval_alignment_tax.py

# Held-out cloze and adversarial prefixes
py src/eval_capabilities.py --n 1000
py src/eval_jailbreaks.py
```

### 4. Interactive Terminal Chat

Compare completions between `Base-Ro-88M` and `SPP-Ro-88M` interactively in the terminal:

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

## Errata and Known Limitations (November 2026)

1. **Parameter count.** All models have **88,099,584** parameters (tied embeddings, 16k vocabulary), not 124.8M. Folder names and Hugging Face repo ids keep the historical `125m` suffix; public names are `Base-Ro-88M` / `SPP-Ro-88M`.
2. **Perplexity / alignment tax.** The first version evaluated perplexity on the tails of `wiki_clean` and `news_recent`, which are inside the training corpus (1,388/1,500 and 1,491/1,500 verbatim) and partly inside the reflection sidecar. Those numbers are withdrawn. `src/build_heldout_set.py` now builds a held-out web set from an unused FineWeb-2 shard, verified disjoint from corpus and sidecar, and `src/eval_alignment_tax.py` reports Δ PPL only on it.
3. **Scoring bug.** Option log-likelihoods were computed by tokenizing the prompt separately; with a trailing space the option's first BPE token was skipped. `src/eval_utils.py` recovers the boundary from a joint tokenization. All forced-choice evaluations are re-run with it.
4. **Statistics.** ConstitutionEval-Ro (N=30) and AIRisk-Ro (N=20) differences are not statistically significant (Fisher p = 0.17 and 0.27). Tables now carry bootstrap 95% CIs and p-values; the 600-item benchmark is the primary quantitative result. Per-article changes of ±40 pp are 2 items out of 5. SPP-Ro drops to 0/5 on §1.1 items; the adversarial-prefix probe shows a *larger* stereotype shift for SPP-Ro (+0.40 vs +0.29 nats/token).
5. **Constitution mismatch.** Items for §2.1–§2.3 test rights that were not in the training constitution; they are labelled as control items.
6. **Not a full replication.** Reflections are attached to LLM-written scenarios and short Wikipedia/news slices, not to real harmful web documents selected by a safety classifier; the 1–5 "safety score" is assigned by the generation prompt, not measured. There is no post-training/persona-binding stage; the `<assistant>` prefix activates the pretraining persona directly. `SPP-Ro-88M-MT` (late insertion) is the first paper baseline added. **v2 pipeline** (`run_v2_pipeline.bat`, `src/generate_reflections_v2.py`): reflections on real corpus documents with LLM-measured severity 0–5, paper-style random insertion conditioned on the prefix only, inline replacement of annotated documents (data-matched with Base), `<assistant>` excluded from the loss, SPP{T0} v2, SPP{MT} v2 and a Filtered baseline; evaluation adds 15 items for the trained articles 2.1–2.3, a 45-item HARD split with register-matched distractors (`src/constitution_eval_items_v2.py`) and a held-out cloze capability test (`src/eval_capabilities.py`). Single seed.
7. **Data details.** Reflections are ≈1.2% of non-padding tokens (10% of sequences). The insertion point is the character midpoint of the host text (68% split a word); the `<assistant>` token is included in the loss. Both are kept for the MT run for comparability and are switchable (`--snap-insertion`, `--mask-assistant-loss`) for future runs. 2,535 reflections are duplicates.
8. **Cross-lingual probes** (English) were removed from the headline results: the tokenizer and corpus are Romanian-only.
9. **SPP{MT}, first attempt (2 Nov 2026) was a null run.** It resumed the Base checkpoint at step 45,000 *with its optimizer and cosine schedule*, so the 5,000 SPP steps ran at a learning rate of 3.7e-6 decaying to 0 (0.16% of the schedule's LR mass). The weights moved by 0.03% (relative L2) and every metric equalled Base to the second decimal. That model is kept as `archive/v1/models/spp_ro_88m_mt_cosinetail` and is not reported. `SPP-Ro-88M-MT` is retrained from the same weights with a fresh schedule (`--init-from`, peak 5e-5, 250 warmup steps, cosine to 0 over 5,000 steps, `run_mt_rewarm.bat`), together with a `Base-Ro-88M-rewarm` control that gets the identical schedule without reflections, so the MT effect is separated from the effect of re-warming.

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


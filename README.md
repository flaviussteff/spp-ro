<div align="center">

# SPP-Ro: Synthetic Persona Pretraining in Romanian
### Constitutional alignment from token zero in an 88M-parameter Romanian LM: a compute-matched comparison of Vanilla, Filtered and SPP pretraining

[![HF spp-ro](https://img.shields.io/badge/HF-spp--ro-166534?style=for-the-badge)](https://huggingface.co/flaviussteff/spp-ro)
[![HF spp-ro-base](https://img.shields.io/badge/HF-spp--ro--base-334155?style=for-the-badge)](https://huggingface.co/flaviussteff/spp-ro-base)
[![HF spp-ro-filtered](https://img.shields.io/badge/HF-spp--ro--filtered-7c2d12?style=for-the-badge)](https://huggingface.co/flaviussteff/spp-ro-filtered)
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
  An independent replication and empirical evaluation grounded in the methodology of:<br>
  <b>Synthetic Persona Pretraining</b> (Minder et al., EPFL-dlab; <a href="https://arxiv.org/abs/2608.13482">arXiv:2608.13482</a>)
</p>

[Results](#results) • [Models](#the-models) • [Benchmark](#constitutioneval-ro) • [Repository](#repository-structure) • [Reproduce](#reproducing-the-experiments) • [Limitations](#limitations) • [Citation](#citation)

---

</div>

## Summary

Post-hoc alignment (RLHF, DPO) is applied after pretraining and is known to be fragile and to cost fluency. *Synthetic Persona Pretraining* (SPP) instead inserts short, constitution-grounded reflections, written in the voice of an assistant persona, directly into the pretraining documents, with an attention mask that keeps the document itself untouched. In this repository I replicate the paper's three-way comparison in Romanian, compute-matched, at the scale a single consumer GPU allows:

* **Vanilla**, **Filtered** (harmful documents kept in context but removed from the loss) and **SPP** (the same documents, 10 % of them carrying a reflection), trained **from scratch on exactly the same document stream** (3.28 billion tokens, 50,000 steps each), plus the paper's **midtraining** variant (SPP{MT}) with its own re-warm control.
* I select the harmful documents with a **safety classifier** trained on LLM-labelled real documents, and the reflections are written for **real corpus documents**, as in the paper.
* I built a 600-item benchmark, **ConstitutionEval-Ro**, with the paper's ConstitutionEval methodology, validated it with large models and score it with four forced-choice protocols.

**What I find.** SPP pretraining has **no measurable cost**: perplexity on a verified held-out web set is identical to Vanilla (18.32 vs 18.42), and so is a held-out cloze capability test (89.1 % vs 89.5 %). The Filtered baseline pays exactly where the theory predicts: on held-out documents that the safety classifier flags, its loss is higher than Vanilla's on **85 % of the documents** (paired sign test p ≈ 10⁻²⁴, +1.13 perplexity), while on unflagged documents it is indistinguishable from Vanilla; SPP, which keeps those documents in the loss, shows no such gap. On ConstitutionEval-Ro, however, **all 88M-parameter models are at chance level** (25 % for 4-way items), so the benchmark cannot resolve an alignment effect at this scale; the differences between SPP, Filtered and Vanilla are not statistically significant.

---

## The models

| Model | Hugging Face | What it is | Compared with |
|---|---|---|---|
| **SPP-Ro** | [`flaviussteff/spp-ro`](https://huggingface.co/flaviussteff/spp-ro) | SPP{T0}: 50,000 steps from scratch; the 150,000 annotated documents (10 %) carry their reflection inline | Base-Ro |
| **Base-Ro** | [`flaviussteff/spp-ro-base`](https://huggingface.co/flaviussteff/spp-ro-base) | Vanilla control: the same stream, plain documents | — |
| **Filtered-Ro** | [`flaviussteff/spp-ro-filtered`](https://huggingface.co/flaviussteff/spp-ro-filtered) | Filtered baseline: the same stream; the 75,000 classifier-flagged documents stay in context, loss masked | Base-Ro |
| **SPP-Ro-MT** | [`flaviussteff/spp-ro-mt`](https://huggingface.co/flaviussteff/spp-ro-mt) | SPP{MT}: Base-Ro at step 45,000 + 5,000 steps on the annotated stream (67.6 % annotated sequences, loss on the reflection only) | Base-Ro-rewarm |
| **Base-Ro-rewarm** | [`flaviussteff/spp-ro-base-rewarm`](https://huggingface.co/flaviussteff/spp-ro-base-rewarm) | control for MT: Base-Ro at step 45,000 + 5,000 steps, same fresh schedule, no reflections | — |
| safety classifier | [`flaviussteff/spp-ro-safety-classifier`](https://huggingface.co/flaviussteff/spp-ro-safety-classifier) | Romanian BERT, severity ≥ 3 vs ≤ 2 (AUC 0.93); defines the annotated set, the Filtered mask and the flagged evaluation slice | — |

All language models have 88,099,584 parameters (Llama-style, 12 layers, hidden 768, GQA 12/4, 1,024-token context, 16k byte-level BPE, tied embeddings). They read a fixed-seed document stream that is identical for every model, 64 packed windows of 1,024 positions per step (2.4 % padding), peak LR 1.5e-4 with cosine decay (5e-5 for the two 5,000-step runs), bf16 autocast with fp32 master weights, on one RTX 3060 (≈66 h per 50k-step model). Single seed.

```
                 corpus (1.5 M documents, fixed-seed stream, identical for every model)
                                       │
   ┌───────────────────────────────────┼─────────────────────────────────────┐
   ▼                                   ▼                                     ▼
Base-Ro (Vanilla)          Filtered-Ro                            SPP-Ro (SPP{T0})
plain documents            75k classifier-flagged documents       150k documents (10 %) replaced by
                           kept in context, loss masked           their annotated version
                                       │
                 Base-Ro @ step 45,000 ──┬──► Base-Ro-rewarm (5k steps, no reflections)
                                         └──► SPP-Ro-MT (5k steps, 67.6 % annotated sequences,
                                              loss on the reflection only)
```

**Data (the paper's "path B").** (1) Qwen labels the severity 0–5 of 3,000 real documents; (2) I fine-tune a Romanian BERT on these labels and score the whole corpus with it; (3) Qwen writes one reflection per selected document, conditioned on the document prefix only, grounded in the six articles of [`constitution_spp_ro.md`](constitution_spp_ro.md). The sidecar holds exactly 150,000 documents (10 % of the corpus): the 75,000 best-scored flagged documents (also the Filtered mask set) and 75,000 benign ones. Reflections average 36 tokens; they are ≈0.67 % of the trained tokens (paper: 0.55 %).

**SPP document mode.** `c₁ … cₖ <assistant> r₁ … rₙ cₖ₊₁ …`: the continuation cannot attend to the reflection (asymmetric mask) and its RoPE positions alias back to the insertion point, so per token the model sees exactly what Vanilla sees; the `<assistant>` token carries no loss. Under packing I apply the paper's **head-cap** rule: an annotated document is truncated at the start like a plain one, and the reflection is inserted only if its position falls inside the window. `src/test_packing.py` checks on CPU that packed logits equal the stand-alone ones, that the SPP invariant holds inside a window and that the per-document loss is unchanged.

**Packing.** EOS-terminated documents are packed first-fit into 1,024-position windows with block-diagonal attention and per-document positions. Every checkpoint stores the exact position of the data stream (`stream_state.json`, including the packer's look-ahead buffer), so a crashed run continues from the same document instead of replaying the stream.

---

## Results

### Alignment tax: perplexity (context 512 tokens)

The held-out set is 3,000 FineWeb-2 documents verified absent from the training corpus and from the reflection sidecar. The *flagged* slice (199 documents) is the part of it that the safety classifier marks at the threshold I used to build the sidecar; the *unflagged* slice (1,000 documents) is the control.

| Model | PPL held-out | PPL flagged slice | PPL unflagged slice | ΔPPL held-out | ΔPPL flagged |
|---|:---:|:---:|:---:|:---:|:---:|
| Base-Ro (Vanilla) | 18.42 | 26.16 | 17.82 | ref | ref |
| **SPP-Ro** | **18.32** | **25.99** | **17.73** | −0.10 | −0.17 |
| Filtered-Ro | 18.48 | **27.29** | 17.81 | +0.06 | **+1.13** |
| Base-Ro-rewarm (MT control) | 18.42 | 26.13 | 17.82 | ref | ref |
| SPP-Ro-MT | 18.88 | 26.91 | 18.26 | +0.46 | +0.78 |

Paired per-document loss differences on the flagged slice (`evals/thesis_alignment_tax_perdoc_losses.json`, n = 199, 5,000-sample bootstrap):

| Comparison | Mean Δloss (nats/token) | 95 % CI | documents where the second model is worse | sign test p |
|---|:---:|:---:|:---:|:---:|
| Filtered vs Vanilla, flagged | +0.044 | [+0.037, +0.050] | 85 % | 1e-24 |
| Filtered vs Vanilla, unflagged (n = 1,000) | −0.000 | [−0.004, +0.003] | 50 % | 0.78 |
| SPP vs Vanilla, flagged | −0.007 | [−0.015, −0.000] | 44 % | 0.12 |
| SPP vs Vanilla, unflagged (n = 1,000) | −0.005 | [−0.008, −0.002] | 48 % | 0.17 |

This is the paper's central mechanistic claim reproduced: filtering removes knowledge of exactly the flagged domain, SPP keeps it at no cost. The SPP{MT} cost (+0.46) follows from its data recipe: 67.6 % of its last 5,000 steps are annotated sequences with loss only on the reflection.

### Capabilities: held-out 4-way cloze (1,000 items)

Base-Ro 89.5 % · SPP-Ro 89.1 % · Filtered-Ro 88.5 % · Base-Ro-rewarm 89.3 % · SPP-Ro-MT 89.4 %. No difference.

### ConstitutionEval-Ro (600 items, chance 25 %; pairwise chance 50 %)

Accuracy in %, 95 % CI ≈ ±3.5 pp on 600 items. p = exact McNemar test against the model's reference (Base-Ro for SPP-Ro and Filtered-Ro, Base-Ro-rewarm for SPP-Ro-MT), on paired items.

| Model | continuation | letter (4 rotations) | PMI | pairwise |
|---|:---:|:---:|:---:|:---:|
| Base-Ro | 19.5 | 27.5 | 38.2 | 45.2 |
| SPP-Ro | 20.2 (p=.65) | 24.3 (p=.23) | 33.5 (p=.02) | 45.2 |
| SPP-Ro, deliberative (`<assistant>` prefix) | 22.3 (p=.11) | 23.0 (p=.08) | 34.0 (p=.10) | 46.2 |
| Filtered-Ro | 20.7 (p=.40) | 24.0 (p=.17) | 35.2 (p=.13) | 44.4 |
| Base-Ro-rewarm | 19.8 | 25.8 | 35.8 | 45.0 |
| SPP-Ro-MT | 20.2 (p=.85) | 28.0 (p=.18) | 37.8 (p=.13) | 46.7 |
| SPP-Ro-MT, deliberative | 23.2 (p=.08) | 25.7 (p=1.0) | 33.5 (p=.37) | 46.9 |
| *Heuristic: shortest option* | | 28.5 | | |

Every model is within a few points of chance under every protocol, and the "shortest option" heuristic matches them. With dozens of comparisons (4 protocols × models × difficulty bands × articles), the single p = 0.02 (SPP below Vanilla on PMI) is what multiple testing produces by itself and does not survive correction. **My reading is a null result on the benchmark: 88M-parameter models trained on 3.3 B tokens do not yet solve ConstitutionEval-Ro, so the benchmark cannot detect whether reflections change their values.** The benchmark itself is valid: the items are solved blind by Qwen-3.8-27B at 97.7 % and open-book by gpt-oss-120B at 97.3 % (neither acted as the corresponding gate during construction); the local 7B probe solves 71.0 % (`evals/constitution_eval_validity.json`).

### Adversarial prefix probe (15 items)

Stereotype shift under an adversarial prefix (nats/token): Base-Ro +0.27, SPP-Ro +0.37, Filtered-Ro +0.31, Base-Ro-rewarm +0.28, SPP-Ro-MT +0.22. With 15 items the rates move in steps of 6.7 pp; I report this probe for completeness only.

---

## ConstitutionEval-Ro

600 scenarios (100 per article, three difficulty bands), each with four courses of action of which exactly one is constitution-conformant, built with the paper's ConstitutionEval pipeline: LLM generation → structural validation → open-book determinacy gate → blind capability guard → local difficulty probe → hardening → rebalancing → my manual review (`evals/constitution_eval_review.html`, corrections applied by `src/apply_ce_corrections.py`). Scored with four protocols: **continuation** (length-normalised log-likelihood of the option text), **letter** (log-probability of the option letter, debiased over the 4 rotations), **PMI** (conditional minus unconditional log-likelihood) and **pairwise** (gold against each distractor). Passive and *deliberative* scoring (`<assistant>` prefix). Items: `data/eval/constitution_eval_ro.jsonl`; Word export: `evals/ConstitutionEval-Ro_benchmark.docx`. The items, the constitution and the reflections are in Romanian by design; everything else in the repository is in English.

---

## Repository Structure

See [`STRUCTURE.md`](STRUCTURE.md) for every file with its role. Everything the pipeline does not use (my earlier experiments with synthetic reflections, a stopped un-packed run, the raw corpus) is in `archive/`.

```
spp-ro/
├── README.md                       # this file
├── STRUCTURE.md                    # full map: every file, folder and model
├── ANNOTATION_GUIDELINES.md        # annotation rules and severity calibration (Romanian, for the annotators)
├── constitution_spp_ro.md          # 6-article Romanian Value Constitution (v1.1)
├── requirements.txt
├── run_expand_annotations.bat      # data: expand the sidecar to 10 % of the corpus (API only)
├── run_pipeline.bat                # training + evaluation: packing self-test -> Base -> SPP -> re-warm -> MT -> Filtered -> evals
├── evals/                          # evaluation outputs and LaTeX tables, regenerated for every model present
│   ├── thesis_alignment_tax_report.json / _table.tex       # perplexity: held-out, flagged / unflagged slices, in-training sets
│   ├── thesis_alignment_tax_perdoc_losses.json             # per-document losses behind the paired tests above
│   ├── thesis_capabilities_report.json / _table.tex        # held-out cloze
│   ├── thesis_constitution_eval_report.json                # ConstitutionEval-Ro: per-item predictions, 4 protocols
│   ├── thesis_constitution_eval_{continuation,letter,pmi,pairwise}[_articles]_table.tex
│   ├── thesis_adversarial_prefix_report.json / _unmasking.tex
│   ├── constitution_eval_validity.json / _build_report.json / _review.html
│   ├── ConstitutionEval-Ro_benchmark.docx
│   └── training_states/                                    # trainer_state.json of every run
├── src/
│   ├── config.py                   # paths, hyper-parameters, ALL_MODELS, base_ref_key
│   ├── train_scale_pretrain.py     # training engine: --mode base/spp/filtered, --pack, --inline, --init-from, --resume
│   ├── packing.py                  # first-fit window packing, block-diagonal collator, stream-position checkpointing
│   ├── spp_collator.py             # asymmetric attention mask, RoPE aliasing, head-cap (encode_item)
│   ├── test_packing.py             # CPU self-test run by the pipeline before training
│   ├── generate_reflections.py     # Qwen annotator: severity labels and prefix-conditioned reflections
│   ├── train_safety_classifier.py  # BERT-Ro safety classifier and corpus scoring
│   ├── clean_reflections.py        # sidecar cleanup (adult pages, meta-language leaks)
│   ├── trim_sidecar.py             # trim the expanded sidecar to exactly 10 % of the documents
│   ├── llm_client.py               # Cerebras/Groq/NVIDIA multi-key client
│   ├── build_constitution_eval.py / apply_ce_corrections.py   # benchmark builder and manual-review corrections
│   ├── eval_constitution.py        # benchmark scorer (4 protocols, passive + deliberative)
│   ├── eval_alignment_tax.py       # perplexity (held-out, classifier-flagged slice, in-training sets)
│   ├── eval_capabilities.py / eval_jailbreaks.py / eval_utils.py
│   ├── build_heldout_set.py        # verified held-out web set
│   ├── download_corpus.py / clean_corpus.py / prepare_scale_corpus.py / train_tokenizer.py   # corpus and tokenizer
│   └── upload_to_hf.py             # Hugging Face publisher (model cards from the evaluation reports)
├── tokenizer/ro_bpe_16k/           # 16k byte-level BPE tokenizer
├── data/, models/, logs/           # git-ignored; see data/README.md and models/README.md
└── archive/                        # not used by the pipeline (see archive/README.md)
```

---

## Reproducing the experiments

```bash
git clone https://github.com/flaviussteff/spp-ro.git && cd spp-ro
pip install -r requirements.txt
```

1. **Corpus and tokenizer**: `src/download_corpus.py`, `src/clean_corpus.py`, `src/prepare_scale_corpus.py`, `src/train_tokenizer.py`; held-out set: `py src/build_heldout_set.py`.
2. **Reflection data** (API keys in `.env`): `py src/generate_reflections.py --label-only --n-label 3000` (severity labels), `py src/train_safety_classifier.py` (classifier + corpus scoring), `py src/generate_reflections.py --candidates data/clean/spp_candidates.parquet` (reflections), `py src/clean_reflections.py`; then `run_expand_annotations.bat` and `py src/trim_sidecar.py --fit-only` for the 10 % sidecar.
3. **Benchmark**: `py src/build_constitution_eval.py` (all stages); validity table: `py src/build_constitution_eval.py --stage verify`.
4. **Training and evaluation**: `run_pipeline.bat` trains every model missing from `models/` and then runs all evaluations (`nofiltered` skips the Filtered baseline, `evalonly` only evaluates). Each 50k-step model takes ≈66 h on an RTX 3060; crashes resume exactly from the last checkpoint.
5. **Evaluations only**: `py src/eval_constitution.py`, `py src/eval_alignment_tax.py`, `py src/eval_capabilities.py --n 1000`, `py src/eval_jailbreaks.py`.
6. **Publish**: `py src/upload_to_hf.py` (after `hf auth login`; `--dry-run` prints the model cards, `--only spp-ro ...` selects repos).

Checking a single benchmark decision from the saved report:

```python
import json
r = json.load(open("evals/thesis_constitution_eval_report.json", encoding="utf-8"))
i = next(k for k, it in enumerate(r["items"]) if it["id"] == "ce3_1.1_003")
for m in ("Base-Ro", "SPP-Ro", "Filtered-Ro"):
    print(m, "pred:", r["per_item_predictions"][m]["letter"][i], "gold:", r["items"][i]["gold_index"])
```

Using a model from the Hub:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("flaviussteff/spp-ro")
model = AutoModelForCausalLM.from_pretrained("flaviussteff/spp-ro").eval()
out = model.generate(**tok("<assistant> În România, egalitatea de șanse", return_tensors="pt"), max_new_tokens=60, do_sample=True, temperature=0.7)
print(tok.decode(out[0], skip_special_tokens=False))
```

---

## Limitations

1. **Scale.** The benchmark null result is a statement about 88M-parameter models at 3.3 B tokens, not about SPP: the models do not solve the benchmark under any protocol, so no alignment effect can be measured on it. The alignment-tax and flagged-slice results do not have this limitation.
2. **Single seed** throughout; p-values are paired tests on items or documents, not across training seeds.
3. **Classifier thresholds.** The sidecar's `clf_flag` (and therefore the Filtered mask) used threshold 0.5 for the first 103k documents and 0.1 for the expansion to 10 %; the evaluation's flagged slice uses 0.1 (`--clf-threshold` changes it).
4. **Earlier experiments.** My preliminary models trained on synthetic (LLM-written) scenarios instead of real documents (v1), and an un-packed run I stopped at step 12,500 (v2), are kept in `archive/` and are not reported; their first-version numbers were withdrawn (perplexity set inside the training corpus, a scorer bug, a null midtraining run).

## Citation

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

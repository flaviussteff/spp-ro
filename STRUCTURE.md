# Project structure

Everything outside `archive/` is used by the current pipeline. My earlier experiments (synthetic reflections, a stopped un-packed run, the old web site, the raw corpus) are in `archive/` (see `archive/README.md`).

## The models

Five models, 88.1M parameters each, trained on **the same document stream** (seed 42), 64 windows × 1,024 positions per step:

| Reported name | Hugging Face | Folder | What it is | Reference |
|---|---|---|---|---|
| **SPP-Ro** | `flaviussteff/spp-ro` | `models/spp_ro/` | SPP{T0}: 50,000 steps from scratch; the 150,000 annotated documents (10 %) carry their reflection inline, head-cap rule | Base-Ro |
| **Base-Ro** | `flaviussteff/spp-ro-base` | `models/base_ro/` | Vanilla control: the same stream, plain documents; `checkpoints_keep/checkpoint-45000` is the starting point of MT and re-warm | — |
| **Filtered-Ro** | `flaviussteff/spp-ro-filtered` | `models/filtered_ro/` | Filtered baseline: the same stream, the 75,000 classifier-flagged documents stay in context without loss | Base-Ro |
| **SPP-Ro-MT** | `flaviussteff/spp-ro-mt` | `models/spp_ro_mt/` | SPP{MT}: Base-Ro at step 45,000 + 5,000 steps on the annotated stream (67.6 % annotated sequences, loss on the reflection only) | Base-Ro-rewarm |
| **Base-Ro-rewarm** | `flaviussteff/spp-ro-base-rewarm` | `models/base_ro_rewarm/` | control for MT: Base-Ro at step 45,000 + 5,000 steps, same fresh LR schedule, no reflections | — |
| safety classifier | `flaviussteff/spp-ro-safety-classifier` | `models/safety_clf_ro/` | BERT-Ro, severity ≥ 3 (AUC 0.93); defines the annotated set, the Filtered mask and the "flagged" evaluation slice | — |

## The flow

```
corpus (1.5M documents) ──> tokenizer
Qwen severity labels (3k) ──> BERT classifier ──> candidates (52k + 52k, threshold 0.5) ──> reflections on real documents (103,240)
  ──> new candidates at threshold 0.1 (46k + 46k) ──> annotation ──> trimmed to exactly 10 % of the corpus
  ──> data/sidecar/reflections.parquet: 150,000 docs = 75,000 flagged (the Filtered mask) + 75,000 benign
the same stream (seed 42), 50,000 steps ≈ 3.28 B tokens, ~66 h per model on an RTX 3060:
  Base-Ro ──────────────────────────────────────────────────────────────
  SPP-Ro ───────────────────────────────────────────────────────────────
  Filtered-Ro ──────────────────────────────────────────────────────────
  Base-Ro@45k ──> Base-Ro-rewarm (5k steps) ──> control for MT
  Base-Ro@45k ──> SPP-Ro-MT (5k steps, midtraining recipe)
evaluation: ConstitutionEval-Ro (4 protocols, passive + deliberative) + held-out perplexity (whole / flagged / unflagged)
            + cloze + adversarial-prefix probe (six articles, built and gated like the benchmark)
```

## The result in short

- **Zero alignment tax for SPP**: held-out PPL 18.32 vs 18.42 (Base-Ro); cloze 89.1 % vs 89.5 %.
- **Filtered pays exactly on the flagged domain**: on the 199 held-out documents flagged by the classifier, higher loss than Base-Ro on 85 % of the documents (sign test p ≈ 1e-24, +1.13 PPL); on the unflagged documents, identical to Base-Ro. SPP has no such loss.
- **On ConstitutionEval-Ro every model is at chance level** (25 % with 4 options; 50 % pairwise) under every protocol; the SPP / Filtered / Base differences are not significant (McNemar on paired items). The benchmark is valid (Qwen-3.8-27B blind 97.7 %, gpt-oss-120b open-book 97.3 %); the 88M models do not solve it.
- SPP-Ro-MT: +0.46 PPL vs re-warm (loss on the reflection only in the last 5k steps); no effect on the benchmark.
- **Adversarial-prefix probe (240 items, built and gated like the benchmark)**: the prejudice-inviting prefix raises the stereotype rate of every model by 4–9 points (+0.13 to +0.15 nats/token margin shift, McNemar p ≤ 0.06), but SPP-Ro moves exactly as much as Base-Ro; the only nominal difference (SPP-Ro-MT −5 pp vs its control, p = 0.012) is not matched by the continuous measure.

## Repository root

| File | Role |
|---|---|
| `run_pipeline.bat` | **the main pipeline** (the data steps are run directly, see the `src/` table and the README): packing self-test → expansion candidates (threshold 0.1) → annotation in a separate window → Base → sidecar → SPP → re-warm → MT → Filtered → `trainer_state.json` archiving → evaluations. Skips what already exists; `nofiltered`, `noexpand`, `evalonly`. Exact resume after a crash (the stream position is in every checkpoint). **Not to be edited while it runs** (cmd reads .bat files incrementally). |
| `run_expand_annotations.bat` | expands the sidecar to 10 % (API, no GPU; started automatically by the pipeline; re-run it if the keys run out); writes `data/sidecar/expand.done` |
| `README.md` | the public presentation |
| `STRUCTURE.md` | this file |
| `constitution_spp_ro.md` | the value constitution (v1.1, 6 articles, Romanian), used by the reflections and the benchmark |
| `ANNOTATION_GUIDELINES.md` | annotation guide and severity calibration (Romanian, for the annotators) |
| `requirements.txt`, `pyrightconfig.json` | environment |
| `.env` | `CEREBRAS_API_KEYS`, `GROQ_API_KEYS`, `NVIDIA_API_KEYS`; not tracked by git |
| `archive/` | everything I no longer use; see `archive/README.md` |

## `src/`

| Script | Role |
|---|---|
| `config.py` | paths and hyper-parameters; `ALL_MODELS` = the five models; `base_ref_key` = each model's reference (re-warm for MT, Base for the rest) |
| `train_scale_pretrain.py` | training engine: `--mode base / spp / filtered`, `--inline`, `--init-from`, `--resume`, `--pack`, `--keep-step 45000`; head-cap by default under `--pack`; `StreamStateCallback` writes `stream_state.json` into every checkpoint (stream position + packing buffer) and `--resume` continues exactly from there; `keep_loss_finite` for fully masked batches (Filtered) |
| `packing.py` | `PackedWindowDataset` (EOS-terminated documents, first-fit into 1,024-position windows, never split, look-ahead 64) + `PackedCollator` (block-diagonal mask, per-document positions) + `keep_loss_finite` |
| `spp_collator.py` | `encode_item`: asymmetric mask (the continuation does not see the reflection), RoPE aliasing, `<assistant>` without loss, head-cap, `reflection_loss_only` (MT) |
| `test_packing.py` | CPU self-test (~20 s), run by the pipeline before training: packed logits = stand-alone, SPP invariant, window loss = sum of document losses, T0/MT/Filtered labels, head-cap, fully masked batch |
| `generate_reflections.py` | the annotator: `--label-only` (severity labels), `--candidates` (reflections conditioned on the prefix only) |
| `train_safety_classifier.py` | BERT-Ro classifier + corpus scoring; `--score-only --threshold 0.1 --exclude-parquet` → the expansion candidates |
| `clean_reflections.py` | sidecar cleanup (adult pages, meta-language leaks) |
| `trim_sidecar.py` | trims the expanded sidecar to exactly 10 % of the corpus (`--fit-only`: only documents whose reflection fits the window); keeps `reflections_full.parquet` |
| `llm_client.py` | Cerebras/Groq/NVIDIA client: key rotation, retirement on 402, back-off on 429 |
| `build_constitution_eval.py` | builds ConstitutionEval-Ro (all gates) + `--stage verify` (validity table) + `--stage review` (HTML page) + `--stage finalize` |
| `apply_ce_corrections.py` | applies the manual-review corrections (`data/eval/ce_corrections*.tsv`) to the final set and the build state, with .bak and a log |
| `eval_constitution.py` | scores the models: continuation, letters debiased over 4 rotations, PMI, gold-vs-distractor pairs; passive and deliberative; McNemar against the reference |
| `eval_alignment_tax.py` | perplexity: held-out, **the classifier-flagged slice** (199 docs, threshold 0.1, `--clf-threshold`) and the unflagged one, the in-training sets; classifier scores cached in `data/eval/heldout_clf_scores.parquet` |
| `build_adversarial_prefix.py` | builds the adversarial-prefix probe set (`data/eval/adversarial_prefix_ro.jsonl`) with the benchmark's recipe: generation per article, structural validation, blind gate by a second model, balanced selection |
| `eval_jailbreaks.py` | the adversarial-prefix probe: stereotype rate under a neutral and an adversarial prefix, margin shift, bootstrap CIs, McNemar / sign tests against the reference, per-article breakdown |
| `eval_capabilities.py`, `eval_utils.py` | held-out cloze, statistical utilities |
| `build_heldout_set.py` | the verified held-out set (needs a raw shard from `archive/raw_corpus/` to be rebuilt) |
| `download_corpus.py`, `clean_corpus.py`, `prepare_scale_corpus.py`, `train_tokenizer.py` | the corpus and the tokenizer (kept for reproduction) |
| `upload_to_hf.py` | publishing on Hugging Face (`HF_RELEASES` = repo → folder map; model cards from the reports) |

## `data/` (git-ignored) — see `data/README.md`

Corpus (`clean/corpus_scale_stream.parquet`), held-out set (`eval/heldout_fineweb.parquet` + `heldout_clf_scores.parquet`), labels and candidates, `sidecar/reflections_first_pass.parquet` (the base), `sidecar/reflections.parquet` (the sidecar used for training), `eval/constitution_eval_ro.jsonl` (the benchmark), `eval/ce_build/items.jsonl` (the build state), `eval/adversarial_prefix_ro.jsonl` (the probe set).

## `models/` (git-ignored) — see `models/README.md`

The five models of the table above + `safety_clf_ro/`. Each model folder: `model.safetensors`, tokenizer, model card, `checkpoints/` (the last two, with `stream_state.json`).

## `evals/`

| File | Role |
|---|---|
| `thesis_constitution_eval_report.json` + `thesis_constitution_eval_{continuation,letter,pmi,pairwise}[_articles]_table.tex` | ConstitutionEval-Ro: per-item predictions, accuracy per band and article, McNemar p |
| `thesis_alignment_tax_report.json`, `_table.tex` | perplexity: held-out, flagged, unflagged, Wikipedia/press (in-training); ΔPPL overall and on the flagged slice |
| `thesis_alignment_tax_perdoc_losses.json` | per-document loss (Base-Ro, SPP-Ro, Filtered-Ro) on the flagged/unflagged slices, for the paired tests |
| `thesis_capabilities_report.json`, `_table.tex` | held-out cloze, 1,000 items |
| `thesis_adversarial_prefix_report.json`, `_unmasking.tex` | the adversarial-prefix probe: per-item log-likelihoods, rates with CIs, paired tests |
| `adversarial_prefix_build_report.json` | how many probe items each stage kept, per article |
| `constitution_eval_validity.json` | benchmark validity (Qwen-3.8-27B blind 97.7 %, gpt-oss-120b open-book 97.3 %, local probe 71.0 %) |
| `constitution_eval_build_report.json`, `constitution_eval_review.html` | the build report and the manual-review page |
| `ConstitutionEval-Ro_benchmark.docx` | all 600 items, for my supervisor |
| `training_states/` | `trainer_state.json` of each of the five trainings |

## `logs/` (git-ignored)

`pipeline.log` (all trainings + evaluations), `expand.log` (sidecar expansion), `evals_and_upload.log`, `generate_data.log` (the data), the benchmark build logs.

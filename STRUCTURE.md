# Project structure

Everything outside `archive/` is used by the current pipeline. My earlier experiments (synthetic reflections, a stopped un-packed run, the old web site, the raw corpus) and the launchers of the final trainings are in `archive/` (see `archive/README.md`).

## The models

Five models, 88.1M parameters each, trained on **the same document stream** (seed 42), 64 windows × 1,024 positions per step:

| Reported name | Hugging Face | Folder | What it is | Reference |
|---|---|---|---|---|
| **SPP-Ro** | `flaviussteff/spp-ro` | `models/spp_ro/` | SPP{T0}: 50,000 steps from scratch; the 150,000 annotated documents (10 %) carry their reflection inline, head-cap rule | Base-Ro |
| **Base-Ro** | `flaviussteff/spp-ro-base` | `models/base_ro/` | Vanilla control: the same stream, plain documents; `checkpoints_keep/checkpoint-45000` is the starting point of MT and re-warm | — |
| **Filtered-Ro** | `flaviussteff/spp-ro-filtered` | `models/filtered_ro/` | Filtered baseline: the same stream, the 75,000 classifier-flagged documents (`clf_flag = 1`) stay in context without loss | Base-Ro |
| **SPP-Ro-MT** | `flaviussteff/spp-ro-mt` | `models/spp_ro_mt/` | SPP{MT}: Base-Ro at step 45,000 + 5,000 steps on the annotated stream (67.6 % of the stream's documents carry a reflection; for those the loss is on the reflection only) | Base-Ro-rewarm |
| **Base-Ro-rewarm** | `flaviussteff/spp-ro-base-rewarm` | `models/base_ro_rewarm/` | control for MT: Base-Ro at step 45,000 + 5,000 steps, same fresh LR schedule, no reflections | Base-Ro |
| safety classifier | `flaviussteff/spp-ro-safety-classifier` | `models/safety_clf_ro/` | BERT-Ro, severity ≥ 3 (dev AUC 0.93; threshold 0.5: precision 0.78, recall 0.58); defines the annotated set, the Filtered mask and the "flagged" evaluation slice | — |

The reference of each model is `config.base_ref_key` (re-warm for the MT model, Base-Ro for the others); every p-value in `evals/` is computed against it.

## The flow

```
corpus (1.5M documents) ──> tokenizer
Qwen severity labels (3k) ──> BERT classifier ──> candidates (52k + 52k, threshold 0.5) ──> reflections on real documents (103,240)
  ──> new candidates at threshold 0.1 (46k + 46k) ──> annotation ──> trimmed to exactly 10 % of the corpus (--fit-only)
  ──> data/sidecar/reflections.parquet: 150,000 docs = 75,000 flagged (score ≥ 0.12; the Filtered mask) + 75,000 benign (score < 0.1)
the same stream (seed 42), 50,000 steps ≈ 3.28 B token positions (≈4 passes), ~66 h per model on an RTX 3060:
  Base-Ro ──────────────────────────────────────────────────────────────
  SPP-Ro ───────────────────────────────────────────────────────────────
  Filtered-Ro ──────────────────────────────────────────────────────────
  Base-Ro@45k ──> Base-Ro-rewarm (5k steps) ──> control for MT
  Base-Ro@45k ──> SPP-Ro-MT (5k steps, midtraining recipe)
evaluation (run_fixes.bat): held-out perplexity (all / flagged / unflagged, per-document losses) + paired tests
            + cloze + ConstitutionEval-Ro (4 protocols, passive + deliberative) + adversarial-prefix probe
            ──> evals/RESULTS.md and the generated sections of README.md (src/report_results.py)
```

## The results

All numbers are generated: `evals/RESULTS.md` and the Summary / Results sections of `README.md` are written by `src/report_results.py` from the reports in `evals/` (`run_fixes.bat` runs it last). The numbers previously listed here were computed on a held-out set normalised differently from the corpus and with unpaired tests, and are withdrawn; see the README, section *Corrections (16 November 2026)*.

## Repository root

| File | Role |
|---|---|
| `run_fixes.bat` | **the evaluation launcher**: CPU self-tests (`test_packing.py`, `test_resume.py`) → backup of the old held-out set and reports (`data/eval/backup_before_fix_<date>/`) → held-out set v3 (only if `heldout_stats.json` is not version 3; `rebuild` forces it) → `eval_alignment_tax.py` → `eval_alignment_paired.py` → `eval_capabilities.py --n 1000` → `eval_constitution.py` → `eval_jailbreaks.py` → `report_results.py`. `run_fixes.bat report` only regenerates `evals/RESULTS.md` and the README sections. Log: `logs/fixes.log`. Trains nothing, uploads nothing, commits nothing. |
| `README.md` | the public presentation |
| `STRUCTURE.md` | this file |
| `constitution_spp_ro.md` | the value constitution (v1.1, 6 articles, Romanian), used by the reflections and the benchmark |
| `ANNOTATION_GUIDELINES.md` | annotation guide, severity calibration and the deviations from the paper (Romanian) |
| `requirements.txt`, `pyrightconfig.json` | environment |
| `.env` | `CEREBRAS_API_KEYS`, `GROQ_API_KEYS`, `NVIDIA_API_KEYS`; not tracked by git |
| `archive/` | everything I no longer use, including the launchers of the final trainings (`archive/launchers/run_v3_pipeline.bat`, `run_expand_annotations.bat`); see `archive/README.md` |

## `src/`

| Script | Role |
|---|---|
| `config.py` | paths and hyper-parameters; `ALL_MODELS` = the five models; `base_ref_key` = each model's reference (re-warm for MT, Base for the rest) |
| `train_scale_pretrain.py` | training engine: `--mode base / spp / filtered`, `--inline`, `--init-from`, `--resume`, `--pack`, `--keep-step 45000`; head-cap by default under `--pack`; `StreamStateCallback` writes `stream_state.json` into every checkpoint: the stream position after the last *trained* window (accelerate's dataloader reads one batch ahead, so the packer's per-window snapshot is used), and `--resume` continues exactly from there; `keep_loss_finite` for fully masked batches (Filtered) |
| `packing.py` | `PackedWindowDataset` (EOS-terminated documents, first-fit into 1,024-position windows, never split, look-ahead 64, a stream snapshot per window) + `PackedCollator` (block-diagonal mask, per-document positions) + `keep_loss_finite` |
| `spp_collator.py` | `encode_item`: asymmetric mask (the continuation does not see the reflection), RoPE aliasing, `<assistant>` without loss, head-cap, `reflection_loss_only` (MT), `</s>` at the document's next position |
| `test_packing.py` | CPU self-test (~20 s): packed logits = stand-alone, SPP invariant, window loss = sum of document losses, T0/MT/Filtered labels, head-cap, fully masked batch, `</s>` positions, the exact tests of `eval_utils.py` against brute force, LaTeX escaping |
| `test_resume.py` | CPU self-test (~10 s): a run stopped at a checkpoint and resumed through accelerate's dataloader trains exactly the windows of the uninterrupted run (with and without packing) |
| `generate_reflections.py` | the annotator: `--label-only` (severity labels), `--candidates` (reflections conditioned on the prefix only) |
| `train_safety_classifier.py` | BERT-Ro classifier + corpus scoring; `--score-only --threshold 0.1 --exclude-parquet` → the expansion candidates |
| `clean_reflections.py` | sidecar cleanup (adult pages, meta-language leaks) |
| `trim_sidecar.py` | trims the expanded sidecar to exactly 10 % of the corpus (`--fit-only`: only documents whose insertion point falls inside the window); keeps `reflections_full.parquet` |
| `llm_client.py` | Cerebras/Groq/NVIDIA client: key rotation, retirement on 402, back-off on 429 |
| `build_constitution_eval.py` | builds ConstitutionEval-Ro (all gates) + `--stage verify` (validity table) + `--stage review` (HTML page) + `--stage finalize` |
| `apply_ce_corrections.py` | applies the manual-review corrections (`data/eval/ce_corrections*.tsv`) to the final set and the build state, with .bak and a log |
| `eval_constitution.py` | scores the models: continuation, letters debiased over 4 rotations, PMI, gold-vs-distractor pairs; passive and deliberative (`<assistant>` directly after the scenario); exact McNemar (pairwise: exact sign-flip on per-item wins) against the reference, Holm over the family; `--stats-only` recomputes everything from the saved predictions |
| `eval_alignment_tax.py` | perplexity on **all** held-out documents, the classifier-flagged slice (threshold 0.1, `--clf-threshold`) and the unflagged one, the in-training sets; per-document losses of every model; classifier scores cached in `data/eval/heldout_clf_scores.parquet` |
| `eval_alignment_paired.py` | paired per-document comparisons from those losses: mean Δloss with a bootstrap CI over documents, share of documents, exact sign test, Holm |
| `build_adversarial_prefix.py` | builds the adversarial-prefix probe set (`data/eval/adversarial_prefix_ro.jsonl`) with the benchmark's recipe: generation per article, structural validation, blind gate by a second model, balanced selection |
| `eval_jailbreaks.py` | the adversarial-prefix probe: stereotype rate under a neutral and an adversarial prefix, margin shift, bootstrap CIs, McNemar / sign tests against the reference, per-article breakdown |
| `eval_capabilities.py`, `eval_utils.py` | held-out cloze (exact McNemar against the reference); scoring utilities, exact tests (McNemar, sign-flip, sign test, Holm), `tex_escape` |
| `report_results.py` | `evals/RESULTS.md` + the Summary / Results sections of `README.md`, generated from the reports, with warnings when a report is older than the corrections |
| `build_heldout_set.py` | the held-out set: cleaned with the corpus function (`prepare_scale_corpus.clean_text_content`), checked against the corpus with its deduplication key (first 64 characters) and MD5; reads the raw shard from `data/raw/fineweb_shards/` or `archive/raw_corpus/fineweb_shards/` |
| `download_corpus.py`, `clean_corpus.py`, `prepare_scale_corpus.py`, `train_tokenizer.py` | the corpus and the tokenizer (kept for reproduction) |
| `upload_to_hf.py` | publishing on Hugging Face (`HF_RELEASES` = repo → folder map; model cards generated from the reports, same wording as the README) |

## `data/` (git-ignored) — see `data/README.md`

Corpus (`clean/corpus_scale_stream.parquet`), held-out set (`eval/heldout_fineweb.parquet` + `heldout_stats.json` + `heldout_clf_scores.parquet`), labels and candidates, `sidecar/reflections_first_pass.parquet` (the base), `sidecar/reflections.parquet` (the sidecar used for training), `eval/constitution_eval_ro.jsonl` (the benchmark), `eval/ce_build/items.jsonl` (the build state), `eval/adversarial_prefix_ro.jsonl` (the probe set), `eval/backup_before_fix_<date>/` (written by `run_fixes.bat`).

## `models/` (git-ignored) — see `models/README.md`

The five models of the table above + `safety_clf_ro/`. Each model folder: `model.safetensors`, tokenizer, model card, `checkpoints/` (the last two, with `stream_state.json`).

## `evals/`

| File | Role |
|---|---|
| `RESULTS.md` | every result table and the generated summary (`src/report_results.py`) |
| `thesis_constitution_eval_report.json` + `thesis_constitution_eval_{continuation,letter,pmi,pairwise}[_articles]_table.tex` | ConstitutionEval-Ro: per-item predictions, accuracy per band and article, exact McNemar / sign-flip p, Holm |
| `thesis_alignment_tax_report.json`, `_table.tex` | perplexity: held-out, flagged, unflagged, Wikipedia/press (in-training); ΔPPL against the reference on each slice |
| `thesis_alignment_tax_perdoc_losses.json` | per-document loss of every model on every held-out document, the flagged / unflagged index and a fingerprint of the held-out file |
| `thesis_alignment_paired_report.json`, `_table.tex` | the paired per-document comparisons (bootstrap CI, sign test, Holm) |
| `thesis_capabilities_report.json`, `_table.tex` | held-out cloze, 1,000 items: accuracy, McNemar against the reference, per-item correctness |
| `thesis_adversarial_prefix_report.json`, `_unmasking.tex` | the adversarial-prefix probe: per-item log-likelihoods, rates with CIs, paired tests |
| `adversarial_prefix_build_report.json` | how many probe items each stage kept, per article |
| `constitution_eval_validity.json` | benchmark validity (Qwen-3.8-27B blind 97.7 %, gpt-oss-120b open-book 97.3 %, local probe 71.0 %) |
| `constitution_eval_build_report.json`, `constitution_eval_review.html` | the build report and the manual-review page |
| `ConstitutionEval-Ro_benchmark.docx` | all 600 items, for my supervisor |
| `training_states/` | `trainer_state.json` of each of the five trainings |

## `logs/` (git-ignored)

`pipeline.log` (the five trainings and the first evaluations, run with `archive/launchers/run_v3_pipeline.bat`), `fixes.log` (`run_fixes.bat`), `expand.log` (sidecar expansion), `evals_and_upload.log`, `generate_data.log` (the data), the benchmark build logs.

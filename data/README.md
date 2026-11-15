# data/ — what each file is

The whole folder is git-ignored (large files); I rebuild the files with the scripts in `src/` in the order below.

| File | What it is | Produced by |
|---|---|---|
| `clean/corpus_scale_stream.parquet` | the training corpus, 1.5M web + Wikipedia + press documents, 150 row groups; read as a fixed-seed stream by every training | `download_corpus.py` → `clean_corpus.py` → `prepare_scale_corpus.py` |
| `clean/wiki_clean.parquet`, `clean/news_recent.parquet` | the Wikipedia and press components (their tails are reported as "in-training" in the perplexity evaluation) | same |
| `eval/heldout_fineweb.parquet`, `eval/heldout_stats.json` | 3,000 held-out documents from an unconsumed FineWeb-2 shard, verified absent from the corpus and the sidecar | `build_heldout_set.py` |
| `eval/heldout_clf_scores.parquet` | the safety-classifier score of every held-out document (cache); defines the "flagged" (199 docs at threshold 0.1) / "unflagged" slices of the perplexity evaluation | `eval_alignment_tax.py` |
| `sidecar/severity_labels.parquet` | 3,000 real documents labelled by Qwen with severity 0–5; the classifier's training set | `generate_reflections.py --label-only` |
| `clean/spp_candidates.parquet` (+ `_summary.json`) | 52,174 documents flagged by the classifier (threshold 0.5) + 52,174 benign ones | `train_safety_classifier.py` |
| `sidecar/reflections_first_pass.parquet` (+ `_progress.json`, `_skip_keys.json`, `_rejected.jsonl`) | the reflections Qwen wrote for the documents above (103,240), with `clf_flag`, `clf_score`, measured severity and insertion position; cleaned by `clean_reflections.py`. The base of the final sidecar | `generate_reflections.py --candidates` |
| `clean/spp_candidates_expansion.parquet` (+ `_summary.json`) | 46,089 new documents flagged at threshold 0.1 + 46,089 benign ones, for the expansion to 10 % | `train_safety_classifier.py --score-only --threshold 0.1 --exclude-parquet` |
| `sidecar/reflections_full.parquet` (+ `_progress.json`, `_rejected.jsonl`) | the first pass + the new annotations, before trimming | `run_expand_annotations.bat` |
| `sidecar/reflections.parquet` | **the sidecar used for training**: exactly 150,000 documents (10 % of the corpus) = 75,000 flagged (`clf_flag=1`, the Filtered mask set) + 75,000 benign; reflections of 36 tokens on average, ≈0.67 % of the trained tokens (paper: 0.55 %) | `trim_sidecar.py --fit-only` |
| `sidecar/expand.done` | marker: the expansion is complete (awaited by `run_pipeline.bat`) | `run_expand_annotations.bat` |
| `eval/constitution_eval_ro.jsonl` | **ConstitutionEval-Ro**: the 600 final items (100 per article, difficulty bands, gold position) | `build_constitution_eval.py` |
| `eval/ce_build/items.jsonl` | the complete build state: all 2,202 candidates, every gate's votes, rejection reasons | same |
| `eval/ce_corrections_auto.tsv` | the corrections from my manual review (format `id TAB field TAB text`), applied with `apply_ce_corrections.py` | the `evals/constitution_eval_review.html` page |

Moved to `archive/`: the raw downloads (`archive/raw_corpus/`, 43 GB), the synthetic reflections of my preliminary experiments (`archive/v1/data/`), the backup copy of the first-pass reflections (`archive/v2/sidecar/`), the benchmark top-up state (`archive/v3_build/`).

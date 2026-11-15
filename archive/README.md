# archive/ — everything I no longer use

Nothing here is read by `run_pipeline.bat`, `run_expand_annotations.bat` or the scripts in `src/`. I keep it for the record and so that the preliminary experiments mentioned in my thesis can be reproduced. It can be deleted entirely without affecting the pipeline.

Only the small parts stay in git (launchers, scripts, evaluations, the old web site); data, models, the raw corpus and the `.env` backups are ignored (see `.gitignore`).

| Folder | Contents | Comes from |
|---|---|---|
| `v1/models/` | my preliminary models with **synthetic reflections** (LLM-invented scenarios, no packing): `base_ro_125m/` (first Base: 10k steps + 50k continuation in pure bf16), `spp_ro_125m/` (SPP 10 %), `spp_ro_125m_20pct/` (SPP 20 %), `spp_ro_88m_mt/` (MT), `base_ro_88m_rewarm/` (MT control), `spp_ro_88m_mt_cosinetail/` (first MT attempt: LR ≈ 0, a null run) and `checkpoints/<model>/`. I published the first two on Hugging Face as `flaviussteff/base-ro-125m` and `flaviussteff/spp-ro-125m`; they are not the reported models | `models/` |
| `v1/evals/` | the withdrawn preliminary outputs (bias, cross-lingual probes, old deliberative probing), `benchmark_v1/` (the earlier ConstitutionEval versions, AIRisk-Ro, an unvalidated 600-item set) and the `training_states/` of the models above | `evals/` |
| `v1/docs/site/` | the GitHub Pages site (`index.html`, `app.js`, `style.css`) with the preliminary results; it does not reflect the current models | `docs/` |
| `v1/launchers/` | `run_overnight.bat`, `run_mt_rewarm.bat` (the preliminary launchers) | repository root |
| `v1/src/` | scripts used only in the preliminary phase: the synthetic-reflection generators, the withdrawn evaluations, the old benchmarks, `interact.py` (terminal chat) | `src/` |
| `v1/logs/`, `v1/data/` | the preliminary logs; ≈60k synthetic reflections and the candidate cache | `logs/`, `data/` |
| `v2/launchers/run_v2_pipeline.bat` | the launcher of the un-packed run (SPP → MT → Filtered → evaluations) that I stopped at step 12,500 when the analysis showed that half of every step was padding; I redid it with packing on the same data | repository root |
| `v2/models/spp_ro_88m_v2/` | checkpoints 10,000 and 12,500 of the stopped run (2 GB); never measured | `models/` |
| `v2/sidecar/` | `reflections_v2.bak.parquet` (the first-pass reflections before cleanup) and `reflections_v2_rejected_diag1.jsonl` | `data/sidecar/` |
| `launchers/` | thin launchers I no longer need: `run_v3_verify.bat` (benchmark validity table + scoring; replaced by `py src/build_constitution_eval.py --stage verify` and the pipeline's evaluation step), `run_generate_data.bat` (the three data commands, now listed in the README), `run_upload_hf.bat` (wrapper for `py src/upload_to_hf.py`) | repository root |
| `benchmark_build/` | `ce_v3_build_topup/` (the top-up state for deficit articles, merged into `data/eval/ce_build/items.jsonl`), `items.pre_merge.bak`, `review_notes/` (my manual-review item lists), `patches/` (temporary correction scripts) | `data/eval/`, `evals/` |
| `docs/` | the analysis note that led me to redo the trainings with packing | — |
| `raw_corpus/` | the raw downloads: `fineweb_shards/` (43 GB), `wiki_ro.jsonl`, `news_recent.jsonl`; needed only to rebuild the corpus or the held-out set | `data/raw/` |
| `env_backups/` | copies of `.env` (they contain API keys: never published) | — |

To free space: `raw_corpus/` (43 GB), `v1/models/` (≈8 GB with checkpoints) and `v2/models/` (2 GB) are the large ones.

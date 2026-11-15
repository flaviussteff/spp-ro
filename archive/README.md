# archive/ — tot ce nu mai folosim

Creat la 04.11.2026, completat la 15.11.2026 după terminarea generației 3. Nimic de aici nu este citit de `run_v3_pipeline.bat`, `run_v3_verify.bat`, `run_generate_v2_data.bat` sau de scripturile din `src/`. Păstrat pentru istoric și pentru reproducerea experimentelor preliminare descrise în teză. Se poate șterge integral fără să afecteze pașii curenți.

În git rămân doar părțile mici (lansatoare, scripturi, evaluări v1); datele, modelele, corpusul brut și backup-urile `.env` sunt ignorate (vezi `.gitignore`).

| Folder | Ce conține | De unde vine |
|---|---|---|
| `v1/launchers/` | `run_overnight.bat` (prima încercare MT, experiment nul), `run_mt_rewarm.bat` (SPP{MT} v1 + control re-warm, 2→3 nov) | rădăcina repo-ului |
| `v1/src/` | scripturile folosite doar în v1: generatoarele de reflecții sintetice (`generate_sensitive_reflections.py`, `generate_factual_reflections.py`, `complete_reflections.py`), evaluările retrase (`eval_biases.py`, `eval_deliberative.py`) și benchmark-urile vechi (`eval_constitution.py`, `constitution_eval_items_v2.py`, `build_master_benchmark_600.py`, `build_html_evaluator.py`) | `src/legacy_v1/` |
| `v1/evals/` | ieșirile v1 retrase și `benchmark_v1/` (ConstitutionEval v1/v2/HARD, AIRisk-Ro, benchmark-ul de 600 nevalidat); `training_states/trainer_state_spp_ro_88m_mt_cosinetail.json` = starea primei încercări MT (experiment nul) | `evals/` |
| `v1/logs/` | log-urile v1 | `logs/` |
| `v1/data/` | ≈60k reflecții **sintetice** v1 (`sidecar/reflections.parquet`) și cache-ul candidaților v1 | `data/` |
| `v1/models/` | `spp_ro_88m_mt_cosinetail/` (prima încercare MT: LR ≈ 0, ponderi identice cu Base; neraportat) și `checkpoints/<model>/` ale modelelor v1 terminate | `models/` |
| `v2/launchers/run_v2_pipeline.bat` | lansatorul generației 2 (SPP{T0} v2 fără packing → MT v2 → Filtered → evaluări); pornit 4 nov, oprit la pasul 12.500 în favoarea generației 3 | rădăcina repo-ului |
| `v2/models/spp_ro_88m_v2/` | checkpoint-urile 10.000 și 12.500 ale rulării v2 oprite (2 GB); nemăsurat | `models/` |
| `v2/sidecar/` | `reflections_v2.bak.parquet` (copia dinaintea curățării) și `reflections_v2_rejected_diag1.jsonl` | `data/sidecar/` |
| `v3_build/` | `ce_v3_build_topup/` (starea completării articolelor deficitare, fuzionată în `data/eval/ce_v3_build/items.jsonl`), `items.pre_merge.bak`, `review_notes/` (listele de itemi din verificarea manuală), `patches/` (scripturi temporare de corecție) | `data/eval/`, `evals/`, `archive/` |
| `docs/` | `Analiza_Licenta_SPP-Ro_v2_20261105_1738.md`: analiza din 5 nov care a dus la decizia de a reface totul cu packing (generația 3) | — |
| `raw_corpus/` | descărcările brute: `fineweb_shards/` (43 GB), `wiki_ro.jsonl`, `news_recent.jsonl`; necesare doar pentru a reconstrui corpusul sau setul held-out | `data/raw/` |
| `env_backups/` | copii ale `.env` (conțin chei API: nu se publică) | — |

Dacă vrei să eliberezi spațiu: `raw_corpus/` (43 GB), `v1/models/` (6 GB) și `v2/models/` (2 GB) sunt cele mari.

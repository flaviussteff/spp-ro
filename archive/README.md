# archive/ — tot ce nu mai folosim

Creat la 04.11.2026, după terminarea benchmark-ului ConstitutionEval-Ro v3. Nimic de aici nu este citit de
`run_v2_pipeline.bat`, `run_v3_verify.bat` sau de scripturile din `src/`. Păstrat doar pentru istoric și pentru
reproducerea experimentelor preliminare (v1) descrise în teză. Se poate șterge integral fără să afecteze pașii următori.

| Folder | Ce conține | De unde vine |
|---|---|---|
| `v1/launchers/` | `run_overnight.bat` (prima încercare MT, experiment nul), `run_mt_rewarm.bat` (SPP{MT} v1 + control re-warm, rulat 2→3 nov) | rădăcina repo-ului |
| `v1/src/` | scripturile folosite doar în v1: generatoarele de reflecții sintetice (`generate_sensitive_reflections.py`, `generate_factual_reflections.py`, `complete_reflections.py`), evaluările retrase (`eval_biases.py`, `eval_deliberative.py`) și benchmark-urile vechi (`eval_constitution.py`, `constitution_eval_items_v2.py`, `build_master_benchmark_600.py`, `build_html_evaluator.py`) | `src/legacy_v1/` |
| `v1/evals/` | ieșirile v1 retrase (raport de bias, probe cross-lingual, probing deliberativ vechi) și `benchmark_v1/` cu rezultatele ConstitutionEval v1 (30 itemi), v2 (15), HARD (45), AIRisk-Ro (20) și benchmark-ul de 600 generat fără validare + pagina lui HTML | `evals/archive_v1/` |
| `v1/logs/` | log-urile v1 (bench600, evals, MT cosinetail, MT re-warm + evals) | `logs/` |
| `v1/data/sidecar/reflections.parquet` | ≈60k reflecții **sintetice** v1 (scenarii inventate de LLM); au antrenat SPP-Ro-88M, 20pct și MT v1 | `data/sidecar/` |
| `v1/data/clean/spp_candidates_cache.parquet` | cache-ul candidaților pentru generatorul v1 | `data/clean/` |
| `v1/models/spp_ro_88m_mt_cosinetail/` | prima încercare MT (2 nov): LR ≈ 0, ponderi identice cu Base; neraportat | `models/` |
| `v1/models/checkpoints/<model>/` | checkpoint-urile intermediare ale modelelor v1 terminate (`spp_ro_125m`, `spp_ro_125m_20pct`, `spp_ro_88m_mt`, `base_ro_88m_rewarm`); modelele finale rămân în `models/`, iar `trainer_state.json` e deja copiat în `evals/training_states/` | `models/<model>/checkpoints/` |
| `v2/sidecar/` | `reflections_v2.bak.parquet` (copia dinaintea curățării `clean_reflections_v2.py`) și `reflections_v2_rejected_diag1.jsonl` (diagnostic) | `data/sidecar/` |
| `v3_build/` | `ce_v3_build_topup/` (starea separată a completării articolelor deficitare, deja fuzionată în `data/eval/ce_v3_build/items.jsonl`) și `items.pre_merge.bak` (starea dinaintea fuziunii) | `data/eval/` |
| `raw_corpus/` | descărcările brute: `fineweb_shards/` (43 GB), `wiki_ro.jsonl`, `news_recent.jsonl`. Corpusul curat (`data/clean/corpus_scale_stream.parquet`) e deja construit; brutul ar fi necesar doar pentru a reconstrui corpusul sau setul held-out (`build_heldout_set.py --shard 100`) | `data/raw/` |
| `env_backups/` | copii ale `.env` dinainte de fiecare adăugare de chei API (conțin chei: nu se publică) | — |

Dacă vrei să eliberezi spațiu: `raw_corpus/` (43 GB) și `v1/models/` (6 GB) sunt cele mari.

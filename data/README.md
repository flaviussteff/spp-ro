# data/ — ce este fiecare fișier

| Fișier | Ce este | Etapă |
|---|---|---|
| `clean/corpus_scale_stream.parquet` | corpusul de antrenare, 1,5M documente (sep 2026) | comun |
| `clean/wiki_clean.parquet`, `clean/news_recent.parquet` | componentele Wikipedia și presă | comun |
| `eval/heldout_fineweb.parquet`, `eval/heldout_stats.json` | 3.000 de documente held-out verificate ca absente din corpus și din sidecar | comun |
| `sidecar/severity_labels.parquet` | 3.000 de documente reale etichetate de Qwen cu severitate 0–5 (pasul 1) | v2 |
| `clean/spp_candidates_v2.parquet`, `clean/spp_candidates_v2_summary.json` | 52.174 documente semnalate de clasificator + 52.174 benigne (pasul 2) | v2 |
| `sidecar/reflections_v2.parquet`, `sidecar/reflections_v2_progress.json`, `_skip_keys.json`, `_rejected.jsonl` | reflecțiile v2 scrise de Qwen pentru documentele reale de mai sus (103.240), cu `clf_flag`, `clf_score`, `severity` măsurată și poziția de inserare (pasul 3) | v2 |
| `eval/constitution_eval_ro_v3.jsonl` | ConstitutionEval-Ro v3: cei 600 de itemi finali (100/articol, benzi de dificultate, poziția gold) | v3 |
| `eval/ce_v3_build/items.jsonl` | starea completă a construcției v3: toți cei 2.202 candidați, voturile fiecărei porți, motivele de respingere | v3 |

Mutate în `archive/`: descărcările brute (`archive/raw_corpus/`, 43 GB), reflecțiile sintetice v1 și cache-ul candidaților v1 (`archive/v1/data/`), copia de siguranță a reflecțiilor v2 (`archive/v2/`), starea completării v3 (`archive/v3_build/`).

Tot folderul este ignorat de git (fișiere mari).

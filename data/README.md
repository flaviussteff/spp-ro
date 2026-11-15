# data/ — ce este fiecare fișier

Actualizat: 15.11.2026. Tot folderul este ignorat de git (fișiere mari); fișierele se reconstruiesc cu scripturile din `src/` în ordinea de mai jos.

| Fișier | Ce este | Produs de |
|---|---|---|
| `clean/corpus_scale_stream.parquet` | corpusul de antrenare, 1,5M documente web + Wikipedia + presă (sep 2026), 150 row-groups; citit în flux cu seed fix de toate antrenările | `download_corpus.py` → `clean_corpus.py` → `prepare_scale_corpus.py` |
| `clean/wiki_clean.parquet`, `clean/news_recent.parquet` | componentele Wikipedia și presă (și cozile lor, raportate „în antrenare” la perplexitate) | idem |
| `eval/heldout_fineweb.parquet`, `eval/heldout_stats.json` | 3.000 de documente held-out dintr-un shard FineWeb-2 neconsumat, verificate ca absente din corpus și din sidecar | `build_heldout_set.py` |
| `eval/heldout_clf_scores.parquet` | scorul clasificatorului de siguranță pentru fiecare document held-out (cache); definește feliile „semnalat” (199 doc. la prag 0,1) / „nesemnalat” din evaluarea de perplexitate | `eval_alignment_tax.py` |
| `sidecar/severity_labels.parquet` | 3.000 de documente reale etichetate de Qwen cu severitate 0–5; setul de antrenare al clasificatorului | `generate_reflections_v2.py --label-only` |
| `clean/spp_candidates_v2.parquet` (+ `_summary.json`) | 52.174 documente semnalate de clasificator (prag 0,5) + 52.174 benigne | `train_safety_classifier.py` |
| `sidecar/reflections_v2.parquet` (+ `_progress.json`, `_skip_keys.json`, `_rejected.jsonl`) | reflecțiile scrise de Qwen pentru documentele de mai sus (103.240), cu `clf_flag`, `clf_score`, severitate măsurată și poziția de inserare; curățate de `clean_reflections_v2.py`. Baza sidecar-ului v3 | `generate_reflections_v2.py --candidates` |
| `clean/spp_candidates_v3.parquet` (+ `_summary.json`) | 46.089 documente noi semnalate la prag 0,1 + 46.089 benigne, pentru extinderea la 10 % | `train_safety_classifier.py --score-only --threshold 0.1 --exclude-parquet` |
| `sidecar/reflections_v3_full.parquet` (+ `_progress.json`, `_rejected.jsonl`) | v2 + adnotările noi, înainte de tăiere | `run_expand_annotations.bat` |
| `sidecar/reflections_v3.parquet` | **sidecar-ul generației 3**: exact 150.000 documente (10 % din corpus) = 75.000 semnalate (`clf_flag=1`, setul mascat al Filtered) + 75.000 benigne; reflecții de 36 tokeni în medie, ≈0,67 % din tokenii antrenați (articol: 0,55 %) | `trim_sidecar_v3.py --fit-only` |
| `sidecar/expand_v3.done` | marcaj: extinderea e completă (așteptat de `run_v3_pipeline.bat`) | `run_expand_annotations.bat` |
| `eval/constitution_eval_ro_v3.jsonl` | **ConstitutionEval-Ro v3**: cei 600 de itemi finali (100/articol, benzi de dificultate, poziția gold) | `build_constitution_eval_v3.py` |
| `eval/ce_v3_build/items.jsonl` | starea completă a construcției: toți cei 2.202 candidați, voturile fiecărei porți, motivele de respingere | idem |
| `eval/ce_v3_corrections_auto.tsv` | corecturile din verificarea manuală (format `id TAB câmp TAB text`), aplicate cu `apply_ce_v3_corrections.py` | pagina `evals/constitution_eval_v3_review.html` |

Mutate în `archive/`: descărcările brute (`archive/raw_corpus/`, 43 GB), reflecțiile sintetice v1 (`archive/v1/data/`), copia de siguranță a reflecțiilor v2 (`archive/v2/sidecar/`), starea completării benchmark-ului (`archive/v3_build/`).

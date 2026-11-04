# Structura proiectului

Actualizat: 04.11.2026, după curățenie. Repo-ul conține acum doar ce se folosește pentru replicarea fidelă (v2) și pentru benchmark-ul ConstitutionEval-Ro v3. Tot ce ținea doar de experimentele preliminare (v1) e în `archive/` (vezi `archive/README.md`).

Numerotări, ca să nu se confunde:
- **Modele / date v1** = experimentele preliminare (sept – 3 nov 2026), cu reflecții sintetice. Modelele rămân în `models/` și se raportează în teză ca experimente preliminare; datele și scripturile lor sunt arhivate.
- **Modele / date v2** = replicarea fidelă a articolului Minder et al. (*Synthetic Persona Pretraining*, arXiv:2608.13482): reflecții scrise pentru documente reale. Datele sunt gata, antrenarea urmează.
- **Benchmark v3** = ConstitutionEval-Ro v3, a treia versiune a benchmark-ului (600 itemi, metodologia EPFL, validat 4 nov). Versiunile v1/v2 ale benchmark-ului sunt arhivate. Toate modelele, v1 și v2, se scorează pe v3.

## Fluxul

```
COMUN   corpus (1,5M doc) ──> tokenizer ──> Base-Ro-88M (50k pași, de la zero)      [gata, sept]
                                                │
v1      reflecții sintetice ─> SPP{T0} v1 10 % / 20 %, SPP{MT} v1, control re-warm   [gata, arhivate datele]
                                                │
v2      etichete Qwen (3k) ──> clasificator BERT ──> candidați (52k nocive + 52k benigne)
          ──> reflecții v2 pe documente reale (reflections_v2.parquet, 103.240)        [gata, 3 nov]
          ├─> SPP{T0} v2 (50k pași, de la zero, inline)                               <── URMEAZĂ (run_v2_pipeline.bat)
          ├─> SPP{MT} v2 = Base@45k + 5k pași pe fluxul v2
          └─> Filtered (documentele nocive cu loss mascat)                            opțional
EVAL    ConstitutionEval-Ro v3 (600 itemi, validat) + PPL held-out + cloze + prefixe adversariale, pentru toate modelele
```

## Rădăcina repo-ului

| Fișier | Rol |
|---|---|
| `run_v2_pipeline.bat` | **următorul pas**: antrenează SPP{T0} v2, SPP{MT} v2 (+ Filtered dacă nu dai `nofiltered`) și rulează toate evaluările; `evalonly` doar evaluările |
| `run_generate_v2_data.bat` | datele v2 în 3 pași: `labels` → `classify` → `annotate` (rulat, 3 nov; păstrat pentru reproducere) |
| `run_v3_verify.bat` | tabelul de validitate al benchmark-ului v3 + scorarea tuturor modelelor pe v3 (rulat 4 nov; reluabil, completează doar itemii fără răspuns) |
| `watch_log.bat` | urmărește un log în timp real |
| `README.md` | prezentarea publică a proiectului |
| `STRUCTURE.md` | acest fișier |
| `constitution_spp_ro.md` | constituția v1.1 (6 articole), folosită de reflecțiile v2 și de benchmark |
| `ANNOTATION_GUIDELINES.md` | ghidul de adnotare (scris pentru v1, aplicat și în v2) |
| `requirements.txt`, `pyrightconfig.json`, `.vscode/` | mediul |
| `.env` | `CEREBRAS_API_KEYS`, `GROQ_API_KEYS`, `NVIDIA_API_KEYS`; neurmărit de git |
| `docs/` | site-ul GitHub Pages (`index.html`, `app.js`, `style.css`); prezintă încă rezultatele v1, de actualizat după v2 |
| `archive/` | tot ce nu se mai folosește (v1, corpus brut, backup-uri); vezi `archive/README.md` |

## `src/`

| Script | Rol | Stare |
|---|---|---|
| `config.py` | toate căile și hiperparametrii; `ALL_MODELS` = lista modelelor evaluate (v1 + v2; folderele lipsă sunt sărite) | comun |
| `spp_collator.py` | masca de atenție asimetrică + RoPE aliasing pentru reflecții | comun |
| `train_scale_pretrain.py` | motorul de antrenare: `--mode base / spp / filtered`, `--inline`, `--init-from`, `--resume` | comun |
| `generate_reflections_v2.py` | anotatorul v2: `--label-only` (pasul 1), `--candidates` (pasul 3) | v2, rulat |
| `train_safety_classifier.py` | clasificatorul BERT-Ro + scorarea corpusului (pasul 2) | v2, rulat |
| `clean_reflections_v2.py` | curățarea finală a `reflections_v2.parquet` (pagini adulte, scurgeri de meta-limbaj) | v2, rulat |
| `llm_client.py` | client comun Cerebras/Groq/NVIDIA: rotație de chei, retragere la 402, backoff la 429, limitare RPM | v2 / v3 |
| `build_constitution_eval_v3.py` | construiește ConstitutionEval-Ro v3 (generare, validare structurală, poartă de determinare open-book, gardă blind gpt-oss-120B, sondă locală Qwen2.5-7B, hardening, rebalansare) + `--stage verify` (tabel de validitate) + pagina HTML de verificare manuală | v3, rulat |
| `eval_constitution_v3.py` | scorează toate modelele pe v3: protocol continuare + litere debiasate prin 4 rotații, pasiv și deliberativ | v3, se rulează după fiecare antrenare |
| `eval_alignment_tax.py`, `eval_capabilities.py`, `eval_jailbreaks.py`, `eval_utils.py` | perplexitate held-out, cloze, prefixe adversariale, utilitare statistice | comun |
| `build_heldout_set.py` | setul held-out verificat (are nevoie de un shard brut din `archive/raw_corpus/` dacă se reconstruiește) | comun, rulat |
| `download_corpus.py`, `clean_corpus.py`, `prepare_scale_corpus.py`, `train_tokenizer.py` | construirea corpusului și a tokenizer-ului (rulate în septembrie; păstrate pentru reproducere, nu se mai rulează) | comun |
| `interact.py`, `upload_to_hf.py` | chat în terminal, publicare pe Hugging Face | comun |

## `data/` (ignorat de git)

| Fișier | Rol |
|---|---|
| `clean/corpus_scale_stream.parquet` | corpusul de antrenare, 1,5M documente |
| `clean/wiki_clean.parquet`, `clean/news_recent.parquet` | sursele componente |
| `clean/spp_candidates_v2.parquet` + `_summary.json` | 52.174 nocive + 52.174 benigne alese de clasificator (v2, pasul 2) |
| `sidecar/severity_labels.parquet` | 3.000 etichete Qwen (v2, pasul 1) |
| `sidecar/reflections_v2.parquet` (+ `_progress.json`, `_skip_keys.json`, `_rejected.jsonl`) | reflecțiile v2 pe documente reale, 103.240 (v2, pasul 3) |
| `eval/heldout_fineweb.parquet`, `eval/heldout_stats.json` | 3.000 documente held-out verificate |
| `eval/constitution_eval_ro_v3.jsonl` | **ConstitutionEval-Ro v3**, cei 600 itemi finali |
| `eval/ce_v3_build/items.jsonl` | starea completă a construcției v3: toți cei 2.202 candidați, voturile, motivele de respingere |

## `models/` (ignorat de git)

| Folder | Nume raportat | Ce este | Stare |
|---|---|---|---|
| `base_ro_125m/` | Base-Ro-88M | modelul inițial, 50k pași de la zero; `checkpoints/checkpoint-45000` e punctul de plecare al tuturor modelelor MT. **Nu se reantrenează.** | comun |
| `spp_ro_125m/` | SPP-Ro-88M | SPP{T0} v1, 10 % reflecții sintetice | v1, raportat ca preliminar |
| `spp_ro_125m_20pct/` | SPP-Ro-88M-20pct | SPP{T0} v1, 20 % | v1, preliminar |
| `spp_ro_88m_mt/` | SPP-Ro-88M-MT | SPP{MT} v1: Base@45k + 5k pași | v1, preliminar |
| `base_ro_88m_rewarm/` | Base-Ro-88M-rewarm | control pentru MT: același program, fără reflecții | comun (control și pentru MT v2) |
| `safety_clf_ro/` | — | clasificatorul BERT-Ro + `threshold.json` | v2 |
| `spp_ro_88m_v2/` | SPP-Ro-88M-v2 | SPP{T0} v2 | **urmează** |
| `spp_ro_88m_mt_v2/` | SPP-Ro-88M-MT-v2 | SPP{MT} v2 | **urmează** |
| `filtered_ro_88m/` | Filtered-Ro-88M | baseline Filtered | opțional |

Folderele păstrează sufixul istoric `125m`; numărul real de parametri este 88,1M. Checkpoint-urile intermediare ale modelelor v1 sunt în `archive/v1/models/checkpoints/`.

## `evals/`

| Fișier | Rol |
|---|---|
| `thesis_constitution_eval_v3_report.json` + `thesis_constitution_eval_v3_{continuation,letter}[_articles]_table.tex` | rezultatele pe ConstitutionEval-Ro v3, toate modelele, ambele protocoale; regenerate la fiecare rulare |
| `constitution_eval_v3_validity.json` | tabelul de validitate al benchmark-ului (Qwen 27B blind 97,7 %, gpt-oss-120B open-book 97,3 %, sonda 71,0 %) |
| `constitution_eval_v3_build_report.json`, `constitution_eval_v3_review.html` | raportul de construcție și pagina de verificare manuală |
| `ConstitutionEval-Ro_v3_benchmark.docx` | documentul Word cu toți cei 600 de itemi (pentru coordonator) |
| `thesis_alignment_tax_*`, `thesis_capabilities_*`, `thesis_adversarial_prefix_*` | perplexitate held-out, cloze, prefixe adversariale |
| `training_states/` | `trainer_state.json` arhivat pentru fiecare antrenare |

## `logs/`

`v2_1102_labels_3000.log`, `generate_v2.log` (datele v2), `v3_benchmark_build.log`, `v3_benchmark_topup.log`, `v3_verify_and_eval.log`, `v3_verify_fill.log` (benchmark-ul v3). `run_v2_pipeline.bat` scrie în `v2_pipeline.log`.

## Unde suntem acum și ce urmează

1. **Gata:** corpus, tokenizer, Base-Ro-88M, modelele v1 (preliminare), datele v2 (103.240 reflecții), ConstitutionEval-Ro v3 (600 itemi, validat), scorarea modelelor v1 pe v3 (toate la nivelul șansei).
2. **Flavius:** verificarea manuală a itemilor v3 (pagina HTML sau documentul Word); dacă elimini itemi, `py src\build_constitution_eval_v3.py --stage finalize` și apoi `run_v3_verify.bat`.
3. `git add -A && git commit`, `pip freeze > requirements-lock.txt`.
4. `run_v2_pipeline.bat nofiltered` (≈55–65 h GPU) sau `run_v2_pipeline.bat` cu Filtered (+50–60 h).
5. Rescrierea afirmațiilor din teză cu cifrele v2; actualizarea `docs/`.

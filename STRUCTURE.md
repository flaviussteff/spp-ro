# Structura proiectului

Actualizat: 15.11.2026, după terminarea generației 3 (packing) și a tuturor evaluărilor. În afara lui `archive/` se află doar fișierele pe care pipeline-ul curent le folosește; tot ce ținea de experimentele preliminare (v1), de generația 2 oprită și de construcția benchmark-ului e în `archive/` (vezi `archive/README.md`).

Numerotări, ca să nu se confunde:
- **Generația 1 (v1)** = experimentele preliminare (sept – 3 nov 2026), reflecții sintetice, fără packing. Modelele rămân în `models/` și se raportează ca preliminare; datele și scripturile lor sunt arhivate.
- **Generația 2 (v2)** = replicarea fidelă fără packing: datele (reflecții pe documente reale, clasificator) sunt **folosite** de generația 3; antrenarea v2 a fost oprită la pasul 12.500 și nu se raportează.
- **Generația 3 (pack)** = aceleași date, cu packing și sidecar extins la 10 %: Base-pack, SPP{T0}-pack, Filtered-pack, re-warm-pack, SPP{MT}-pack. **Rezultatele principale ale tezei.**
- **Benchmark v3** = ConstitutionEval-Ro v3 (600 itemi, metodologia EPFL, validat 4 nov). Toate modelele, v1 și pack, se scorează pe el.

## Fluxul

```
COMUN   corpus (1,5M doc) ──> tokenizer                                                           [sept]
DATE    etichete Qwen (3k) ──> clasificator BERT (AUC 0,93) ──> candidați (52k + 52k, prag 0,5)
          ──> reflecții pe documente reale (reflections_v2, 103.240)                                [3 nov]
          ──> candidați noi la prag 0,1 (46k + 46k) ──> adnotare ──> tăiere la exact 10 % din corpus
          ──> reflections_v3.parquet: 150.000 doc = 75.000 semnalate (setul mascat Filtered) + 75.000 benigne   [5–8 nov]
GEN 3   același flux de documente (seed 42), 50.000 pași × 64 ferestre × 1.024 poziții ≈ 3,28 mld tokeni, ~66 h/model:
          Base-pack (Vanilla) ─────────────────────────────────────────────────────── [5–8 nov]
          SPP{T0}-pack (documentele adnotate poartă reflecția, head-cap) ──────────── [8–11 nov]
          Filtered-pack (documentele semnalate rămân, loss mascat) ─────────────────── [12–14 nov]
          Base-pack@45k ──> re-warm-pack (5k pași, fără reflecții) ──> control pentru MT   [11 nov]
          Base-pack@45k ──> SPP{MT}-pack (5k pași, 67,6 % secvențe adnotate, loss doar pe reflecție)   [11 nov]
EVAL    ConstitutionEval-Ro v3 (4 protocoale, pasiv + deliberativ) + PPL held-out (întreg / semnalat / nesemnalat)
        + cloze + prefixe adversariale, pentru toate modelele prezente                               [14 nov]
```

## Rezultatul pe scurt

- **Taxă de aliniere zero pentru SPP**: PPL held-out 18,32 față de 18,42 (Base-pack); cloze 89,1 % față de 89,5 %.
- **Filtered plătește exact pe domeniul semnalat**: pe cele 199 documente held-out semnalate de clasificator, loss mai mare decât Base pe 85 % din documente (test de semn p ≈ 1e-24, +1,13 PPL); pe documentele nesemnalate, identic cu Base. SPP nu are această pierdere.
- **Pe ConstitutionEval-Ro v3 toate modelele sunt la nivelul șansei** (25 % la 4 variante; 50 % pairwise), sub toate protocoalele; diferențele SPP/Filtered/Base nu sunt semnificative (McNemar pe perechi de itemi). Benchmark-ul e valid (Qwen-3.8-27B blind 97,7 %, gpt-oss-120b open-book 97,3 %), modelele de 88M nu îl rezolvă.
- SPP{MT}-pack: +0,46 PPL față de re-warm (recipe cu loss doar pe reflecție în ultimii 5k pași); fără efect pe benchmark.

## Rădăcina repo-ului

| Fișier | Rol |
|---|---|
| `run_v3_pipeline.bat` | **pipeline-ul principal**: auto-test packing → candidați v3 → adnotare în fereastră separată → Base-pack → sidecar v3 → SPP{T0}-pack → re-warm-pack → SPP{MT}-pack → Filtered-pack → arhivare `trainer_state.json` → evaluări. Sare peste tot ce există deja; `nofiltered`, `noexpand`, `evalonly`. Reluare exactă după crash (poziția fluxului e în fiecare checkpoint). **Nu se editează cât timp rulează** (cmd citește .bat-ul incremental). |
| `run_expand_annotations.bat` | extinderea sidecar-ului la 10 % (API, fără GPU; pornit automat de v3; se reia dacă se epuizează cheile); scrie `data/sidecar/expand_v3.done` |
| `run_generate_v2_data.bat` | datele în 3 pași: `labels` → `classify` → `annotate` (rulat 3 nov; produce `reflections_v2.parquet`, baza sidecar-ului v3) |
| `run_v3_verify.bat` | tabelul de validitate al benchmark-ului (Qwen-3.8-27B blind, gpt-oss-120b open-book, sonda locală) → `evals/constitution_eval_v3_validity.json`; pasul 2 (scorarea) e acum făcut de pipeline |
| `watch_log.bat` | urmărește un log în timp real |
| `README.md` | prezentarea publică (EN), cu rezultatele generației 3 |
| `STRUCTURE.md` | acest fișier |
| `constitution_spp_ro.md` | constituția v1.1 (6 articole), folosită de reflecții și de benchmark |
| `ANNOTATION_GUIDELINES.md` | ghidul de adnotare și calibrarea severității |
| `requirements.txt`, `pyrightconfig.json` | mediul |
| `.env` | `CEREBRAS_API_KEYS`, `GROQ_API_KEYS`, `NVIDIA_API_KEYS`; neurmărit de git |
| `docs/` | site-ul GitHub Pages (`index.html`, `app.js`, `style.css`); **prezintă încă rezultatele v1, de actualizat** |
| `archive/` | tot ce nu se mai folosește; vezi `archive/README.md` |

## `src/`

| Script | Rol | Etapă |
|---|---|---|
| `config.py` | căile și hiperparametrii; `ALL_MODELS` = modelele evaluate (v1 + pack; folderele lipsă sunt sărite); `base_ref_key` = referința fiecărui model (Base-pack pentru T0/Filtered pack, re-warm-pack pentru MT-pack) | comun |
| `train_scale_pretrain.py` | motorul de antrenare: `--mode base / spp / filtered`, `--inline`, `--init-from`, `--resume`, `--pack`, `--keep-step 45000`; sub `--pack` regula head-cap EPFL e implicită; `StreamStateCallback` scrie `stream_state.json` în fiecare checkpoint (poziția fluxului + buffer-ul de împachetare) și `--resume` continuă exact de acolo; `keep_loss_finite` pentru loturile complet mascate (Filtered) | comun |
| `packing.py` | `PackedWindowDataset` (documente terminate cu `</s>`, first-fit în ferestre de 1.024, niciodată tăiate, look-ahead 64) + `PackedCollator` (mască bloc-diagonală, poziții per document) + `keep_loss_finite` | gen 3 |
| `spp_collator.py` | `encode_item`: masca asimetrică (continuarea nu vede reflecția), RoPE aliasing, `<assistant>` fără loss, head-cap, `reflection_loss_only` (MT) | comun |
| `test_packing.py` | auto-test CPU (~20 s), rulat de pipeline înainte de antrenare: logits împachetat = singur, invariantul SPP, loss pe fereastră = suma pe documente, etichete T0/MT/Filtered, head-cap, lot complet mascat | gen 3 |
| `generate_reflections_v2.py` | anotatorul: `--label-only` (etichete de severitate), `--candidates` (reflecții condiționate doar de prefix) | date |
| `train_safety_classifier.py` | clasificatorul BERT-Ro + scorarea corpusului; `--score-only --threshold 0.1 --exclude-parquet` → candidații v3 | date |
| `clean_reflections_v2.py` | curățarea sidecar-ului (pagini adulte, scurgeri de meta-limbaj) | date |
| `trim_sidecar_v3.py` | taie sidecar-ul extins la exact 10 % din corpus (`--fit-only`: doar documente a căror reflecție încape în fereastră); păstrează `reflections_v3_full.parquet` | date |
| `llm_client.py` | client Cerebras/Groq/NVIDIA: rotație de chei, retragere la 402, backoff la 429 | date / benchmark |
| `build_constitution_eval_v3.py` | construiește ConstitutionEval-Ro v3 (toate porțile) + `--stage verify` (validitate) + `--stage review` (pagina HTML) + `--stage finalize` | benchmark |
| `apply_ce_v3_corrections.py` | aplică corecturile din verificarea manuală (`data/eval/ce_v3_corrections*.tsv`) în setul final și în starea de construcție, cu .bak și jurnal | benchmark |
| `eval_constitution_v3.py` | scorează toate modelele: continuare, litere debiasate prin 4 rotații, PMI, perechi gold contra distractor; pasiv și deliberativ; McNemar față de referință | eval |
| `eval_alignment_tax.py` | perplexitate: held-out, **felia semnalată de clasificator** (199 doc., prag 0,1, `--clf-threshold`) și nesemnalată, seturile „în antrenare”; scorurile clasificatorului în cache `data/eval/heldout_clf_scores.parquet` | eval |
| `eval_capabilities.py`, `eval_jailbreaks.py`, `eval_utils.py` | cloze held-out, prefixe adversariale (15 itemi), utilitare statistice | eval |
| `build_heldout_set.py` | setul held-out verificat (are nevoie de un shard brut din `archive/raw_corpus/` dacă se reconstruiește) | comun |
| `download_corpus.py`, `clean_corpus.py`, `prepare_scale_corpus.py`, `train_tokenizer.py` | corpusul și tokenizer-ul (sept; păstrate pentru reproducere) | comun |
| `interact.py`, `upload_to_hf.py` | chat în terminal, publicare pe Hugging Face | utilitare |

## `data/` (ignorat de git) — vezi `data/README.md`

Corpus (`clean/corpus_scale_stream.parquet`), held-out (`eval/heldout_fineweb.parquet` + `heldout_clf_scores.parquet`), etichete și candidați, `sidecar/reflections_v2.parquet`, `sidecar/reflections_v3.parquet` (sidecar-ul generației 3), `eval/constitution_eval_ro_v3.jsonl` (benchmark-ul), `eval/ce_v3_build/items.jsonl` (starea construcției).

## `models/` (ignorat de git) — vezi `models/README.md`

| Folder | Nume raportat | Generație |
|---|---|---|
| `base_ro_88m_pack/` (+ `checkpoints_keep/checkpoint-45000`) | Base-Ro-88M-pack | 3, Vanilla |
| `spp_ro_88m_pack/` | SPP-Ro-88M-pack | 3, SPP{T0} |
| `filtered_ro_88m_pack/` | Filtered-Ro-88M-pack | 3, Filtered |
| `base_ro_88m_pack_rewarm/` | Base-Ro-88M-pack-rewarm | 3, control MT |
| `spp_ro_88m_mt_pack/` | SPP-Ro-88M-MT-pack | 3, SPP{MT} |
| `base_ro_125m/`, `spp_ro_125m/`, `spp_ro_125m_20pct/`, `base_ro_88m_rewarm/`, `spp_ro_88m_mt/` | Base-Ro-88M, SPP-Ro-88M, SPP-Ro-88M-20pct, Base-Ro-88M-rewarm, SPP-Ro-88M-MT | 1, preliminare |
| `safety_clf_ro/` | — | clasificatorul de siguranță |

## `evals/`

| Fișier | Rol |
|---|---|
| `thesis_constitution_eval_v3_report.json` + `thesis_constitution_eval_v3_{continuation,letter,pmi,pairwise}[_articles]_table.tex` | ConstitutionEval-Ro v3: predicțiile per item, acuratețe pe benzi și articole, p McNemar; regenerate la fiecare rulare |
| `thesis_alignment_tax_report.json`, `_table.tex` | perplexitate: held-out, semnalat, nesemnalat, Wikipedia/presă (în antrenare); ΔPPL general și pe felia semnalată |
| `thesis_alignment_tax_perdoc_losses.json` | loss-ul per document (Base-pack, SPP-pack, Filtered-pack) pe feliile semnalată/nesemnalată, pentru testele pe perechi |
| `thesis_capabilities_report.json`, `_table.tex` | cloze held-out, 1.000 itemi |
| `thesis_adversarial_prefix_report.json`, `_unmasking.tex` | prefixe adversariale, 15 itemi |
| `constitution_eval_v3_validity.json` | validitatea benchmark-ului (Qwen-3.8-27B blind 97,7 %, gpt-oss-120b open-book 97,3 %, sonda 71,0 %) |
| `constitution_eval_v3_build_report.json`, `constitution_eval_v3_review.html` | raportul de construcție și pagina de verificare manuală |
| `ConstitutionEval-Ro_v3_benchmark.docx` | toți cei 600 de itemi, pentru coordonator |
| `training_states/` | `trainer_state.json` al fiecărei antrenări raportate (v1 și pack) |

## `logs/` (ignorat de git)

`v3_pipeline.log` (toate antrenările generației 3 + evaluările), `expand_v3.log` (extinderea sidecar-ului), `alignment_tax_flagged.log`, `generate_v2.log` și `v2_1102_labels_3000.log` (datele), `v3_benchmark_build.log`, `v3_benchmark_topup.log`, `v3_verify_and_eval.log`, `v3_verify_fill.log` (benchmark-ul), `v2_pipeline.log` (rularea v2 oprită).

## Unde suntem acum și ce urmează

1. **Gata:** corpus, tokenizer, date (sidecar v3), benchmark v3 validat, cele cinci modele ale generației 3, toate evaluările, modelele v1 ca preliminare.
2. **Urmează:** scrierea tezei cu cifrele generației 3 (secțiunile „Rezultatul pe scurt” de mai sus și tabelele din `evals/*.tex`); actualizarea `docs/` (site-ul prezintă încă v1); `pip freeze > requirements-lock.txt`; publicarea repo-ului (`git add -A && git commit`).

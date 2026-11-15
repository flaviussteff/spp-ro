# models/ — ce este fiecare folder

Actualizat: 15.11.2026, după terminarea generației 3 (packing). Toate modelele au 88.099.584 de parametri (Llama-style, 16k vocabular, embeddings legate). Căile sunt fixate în `src/config.py` (`ALL_MODELS`) și nu trebuie redenumite; sufixul `125m` din primele foldere este istoric.

## Generația 3 (packing) — rezultatele principale ale tezei

Toate cinci citesc același flux de documente (seed 42), 64 ferestre × 1.024 poziții pe pas, ≈3,28 miliarde de tokeni în 50.000 de pași (≈66 h fiecare pe RTX 3060). Se compară între ele, nu cu modelele generației 1.

| Folder | Nume raportat | Ce este | Referință la comparație |
|---|---|---|---|
| `base_ro_88m_pack/` | Base-Ro-88M-pack | control Vanilla, 50k pași de la zero (5–8 nov); `checkpoints_keep/checkpoint-45000` e punctul de plecare al MT și re-warm | — |
| `spp_ro_88m_pack/` | SPP-Ro-88M-pack | SPP{T0}: aceleași documente, cele 150.000 adnotate (10 %) poartă reflecția inline, regula head-cap EPFL (8–11 nov) | Base-pack |
| `filtered_ro_88m_pack/` | Filtered-Ro-88M-pack | baseline Filtered: aceleași documente, cele 75.000 semnalate de clasificator rămân în context dar nu primesc loss (12–14 nov) | Base-pack |
| `base_ro_88m_pack_rewarm/` | Base-Ro-88M-pack-rewarm | control pentru MT: Base-pack@45k + 5.000 de pași cu program LR nou (peak 5e-5), fără reflecții (11 nov) | — |
| `spp_ro_88m_mt_pack/` | SPP-Ro-88M-MT-pack | SPP{MT}: Base-pack@45k + 5.000 de pași pe fluxul adnotat (67,6 % secvențe adnotate, loss doar pe reflecție), ca în midtraining-ul articolului (11 nov) | Base-pack-rewarm |

Fiecare folder conține modelul final (`model.safetensors`), tokenizer-ul, `README.md` (model card generat din rulare) și, în `checkpoints/`, ultimele două checkpoint-uri cu `stream_state.json` (poziția fluxului de date, pentru reluare exactă). `trainer_state.json` al fiecărei antrenări e copiat în `evals/training_states/`.

## Generația 1 (preliminară) — reflecții sintetice, fără packing

Raportate în teză ca experimente preliminare. Se compară doar cu `Base-Ro-88M`.

| Folder | Nume raportat | Ce este |
|---|---|---|
| `base_ro_125m/` | Base-Ro-88M | modelul inițial (22 sep): 10.000 de pași de la zero + 50.000 de pași de continuare (în bf16 pur, bug reparat pe 5 nov) |
| `spp_ro_125m/` | SPP-Ro-88M | SPP{T0} v1, 10 % reflecții sintetice (scenarii inventate de LLM) |
| `spp_ro_125m_20pct/` | SPP-Ro-88M-20pct | SPP{T0} v1, 20 % reflecții sintetice |
| `base_ro_88m_rewarm/` | Base-Ro-88M-rewarm | control MT v1: Base@45k + 5.000 de pași, fără reflecții (3 nov) |
| `spp_ro_88m_mt/` | SPP-Ro-88M-MT | SPP{MT} v1: Base@45k + 5.000 de pași cu reflecții sintetice (3 nov) |

## Comun

| Folder | Ce este |
|---|---|
| `safety_clf_ro/` | clasificatorul de siguranță BERT-Ro (3 nov) + `threshold.json` (AUC 0,93 pe dev; prag 0,5 pentru sidecar-ul v2, 0,1 pentru extinderea v3); definește setul mascat al Filtered și felia „semnalat” din evaluarea de perplexitate |

## Arhivate

- `archive/v2/models/spp_ro_88m_v2/` — SPP{T0} v2 fără packing, oprit la pasul 12.500 în favoarea generației 3; doar checkpoint-uri, nemăsurat.
- `archive/v1/models/` — `spp_ro_88m_mt_cosinetail/` (prima încercare MT, LR ≈ 0, experiment nul) și checkpoint-urile intermediare ale modelelor generației 1.

# models/ — ce este fiecare folder

| Folder | Nume raportat | Ce este | Etapă |
|---|---|---|---|
| `base_ro_125m/` | Base-Ro-88M | modelul inițial (22 sep 2026), 50.000 de pași de la zero, control Vanilla. `checkpoints/checkpoint-45000` e punctul de plecare al tuturor modelelor MT. **Nu se reantrenează.** | comun v1 + v2 |
| `spp_ro_125m/` | SPP-Ro-88M | SPP{T0} v1: 50k pași de la zero cu 10 % reflecții sintetice | v1, preliminar |
| `spp_ro_125m_20pct/` | SPP-Ro-88M-20pct | SPP{T0} v1 cu 20 % reflecții sintetice | v1, preliminar |
| `spp_ro_88m_mt/` | SPP-Ro-88M-MT | SPP{MT} v1 (3 nov): Base@45k + 5.000 de pași cu program LR nou, 10 % reflecții sintetice | v1, preliminar |
| `base_ro_88m_rewarm/` | Base-Ro-88M-rewarm | control pentru MT (3 nov): Base@45k + 5.000 de pași cu același program LR, **fără** reflecții; servește și ca control pentru MT v2 | comun |
| `safety_clf_ro/` | — | clasificatorul de siguranță BERT-Ro (3 nov) + `threshold.json` cu metricile pe dev | v2 |
| `spp_ro_88m_v2/` | SPP-Ro-88M-v2 | SPP{T0} v2: 50k pași de la zero, reflecții v2 inline (creat de `run_v2_pipeline.bat`) | v2, urmează |
| `spp_ro_88m_mt_v2/` | SPP-Ro-88M-MT-v2 | SPP{MT} v2: Base@45k + 5k pași pe fluxul v2 | v2, urmează |
| `filtered_ro_88m/` | Filtered-Ro-88M | baseline Filtered: documentele nocive rămân, loss mascat | v2, opțional |

Mutate în `archive/v1/models/`: `spp_ro_88m_mt_cosinetail/` (prima încercare MT, experiment nul, neraportat) și checkpoint-urile intermediare ale modelelor v1 terminate (`checkpoints/<model>/`); `trainer_state.json` al fiecărei antrenări e în `evals/training_states/`.

Sufixul `125m` din numele folderelor este istoric; modelele au 88.099.584 de parametri. Căile sunt fixate în `src/config.py` și nu trebuie redenumite.

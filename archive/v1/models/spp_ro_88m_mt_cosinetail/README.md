---
language:
- ro
license: mit
pipeline_tag: text-generation
tags:
- romanian
- llama
- causal-lm
- from-scratch
- synthetic-persona-pretraining
---

# SPP-Ro-88M: model de limbaj românesc preantrenat de la zero cu Synthetic Persona Pretraining

Model decoder de tip LLaMA cu **88,099,584 (~88.1M) parametri**, preantrenat de la zero pe text românesc
(FineWeb-2 RO, Wikipedia RO, presă) pentru teza de licență *Alinierea modelelor de limbaj în limba română
din faza de preantrenare prin Synthetic Persona Pretraining* (FMI, Universitatea din București), după
metodologia Minder et al., 2026 (arXiv:2608.13482). Variantă SPP{MT}: inițializat din Base la pasul 45.000 și antrenat 5.000 de pași cu reflecții.

## Antrenare
* Pași de optimizare: 50,000; batch efectiv 64 secvențe x 1.024 poziții = 3,276,800,000 poziții (inclusiv padding; secvențele nu sunt împachetate, deci numărul de tokeni efectivi este aproximativ jumătate).
* Interleaving SPP: 10% din secvențe sunt documente cu reflecție constituțională la persoana I, precedată de tokenul <assistant>, cu blocarea atenției post-reflecție și aliasing RoPE.
* Hardware: o singură placă NVIDIA RTX 3060 12GB, bf16.
* Ultimul loss de antrenare raportat: 2.8974 (perplexitate 18.13).

## Arhitectură
| Parametru | Valoare |
| :--- | :--- |
| Parametri | 88,099,584 (~88.1M) |
| hidden / intermediate | 768 / 2048 (SwiGLU) |
| Straturi | 12, GQA 12 Q / 4 KV |
| Context | 1.024 tokeni, RoPE theta 10.000 |
| Vocabular | 16.384 BPE (ro_bpe_16k), embedding legat |

## Utilizare
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("spp_ro_88m_mt")
model = AutoModelForCausalLM.from_pretrained("spp_ro_88m_mt")
```

## Limitări
Model mic (88M), monolingv, fără post-training; evaluările din repo au N mic și sunt raportate cu intervale de încredere.

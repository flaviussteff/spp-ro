# Evaluare umană: benchmark SPP-Ro, 100 de itemi

Pagina din acest folder strânge răspunsurile colegilor la cei 100 de itemi din
`evals/benchmark spp 100 items (fara 2.1).docx`. E un singur fișier, `index.html`, cu itemii incluși: nu are nevoie de
internet, de server sau de instalare.

## Pentru colegi

1. Deschideți `index.html` cu dublu-click, în **Chrome** sau **Edge**, într-o fereastră obișnuită (nu privată/incognito).
2. Scrieți numele și apăsați **Începe**.
3. Pentru fiecare item: citiți situația și variantele, alegeți la **Răspuns** litera A–D, bifați **Sigur** doar dacă sunteți
   sigur(ă) și apăsați **Confirmă răspunsul**. Răspunsul se blochează și nu mai puteți reveni la item.
4. Itemul apare din nou, cu **Mențiuni**: observații opționale despre formularea situației sau a variantelor. Apoi treceți
   la itemul următor.
5. După itemul 100 se descarcă două fișiere, `raspunsuri_<Nume>.csv` și `raspunsuri_<Nume>.docx` (de obicei în
   Descărcări / Downloads). Trimiteți-le înapoi.

Timpul se măsoară de la apariția itemului până la confirmarea răspunsului. Nu se numără timpul de la Mențiuni, nici
pauzele. **Pauză** oprește timpul și ascunde itemul. Pauza se pune și singură dacă schimbați tab-ul, dați refresh sau
închideți pagina. La redeschidere apare ecranul „Bine ați revenit”, iar după **Continuă** reluați exact itemul la care ați
rămas, cu timpul de dinainte. Dacă doar micșorați fereastra sau lucrați în altă aplicație cu browserul încă vizibil,
timpul merge mai departe. Browserul nu deosebește o fereastră minimizată în bara de jos de un tab schimbat, așa că acolo
se pune pauză.

Progresul stă în memoria browserului (localStorage) până la final. Nu ștergeți datele de navigare și nu treceți pe alt
browser sau calculator în timpul completării.

## Fișierele cu rezultate

`raspunsuri_<Nume>.csv` (UTF-8, cu BOM pentru Excel, separator virgulă), câte un rând per item:

| coloană | conținut |
|---|---|
| `evaluator` | numele introdus la început (același pe fiecare rând, ca fișierele mai multor colegi să se poată concatena) |
| `item` | 1–100, numărul din docx |
| `raspuns` | A, B, C sau D |
| `sigur` | `da` / `nu` |
| `timp_secunde` | timpul de răspuns, cu o zecimală, fără pauze și fără Mențiuni |
| `mentiuni` | textul de la Mențiuni (gol dacă nu au scris nimic) |
| `data_ora` | momentul confirmării răspunsului |

`raspunsuri_<Nume>.docx` conține numele, începutul și sfârșitul, timpul total, câți itemi sunt marcați „sigur” și un tabel
Item | Răspuns | Sigur | Timp | Mențiuni.

```python
import glob, pandas as pd
df = pd.concat(pd.read_csv(f, encoding="utf-8-sig") for f in glob.glob("raspunsuri_*.csv"))
```

## Regenerare

Dacă se schimbă docx-ul, editați doar `template.html` (dacă e cazul) și rulați:

```
python human_eval/build_interface.py
```

Scriptul citește docx-ul cu biblioteca standard, verifică 100 de itemi cu exact patru variante A–D și rescrie
`index.html`. Textul itemilor rămâne neschimbat. Pagina reține versiunea setului: un browser care are o sesiune
începută pe altă versiune spune asta, în loc să amestece răspunsurile.

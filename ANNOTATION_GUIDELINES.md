# SPP-Ro: ghidul de adnotare a datelor de preantrenare (reflecții)

**Versiunea 3.0 (04.11.2026)** — aliniat la *Synthetic Persona Pretraining: Alignment from Token Zero* (Minder et al., 2026, arXiv:2608.13482), Appendix H.1 (prompt-ul generatorului, `generator_reflection_v7.md`) și I.2 (*Value Annotation Guidelines*), din repo-ul `epfl-dlab/spp-data`. Versiunile 1.x–2.0 ale acestui ghid au fost traducerea prompt-ului EPFL; v3.0 corectează ce era inconsistent cu articolul și cu pipeline-ul v2 efectiv rulat (`src/generate_reflections.py`). **v3.1 (16.11.2026):** §2, §5 și §13 dau valorile măsurate pe sidecar-ul final (150.000 de documente) și pe rulările finale, în locul celor din primul pas.

Secțiunile 1–8 sunt specificația pe care o urmează prompt-ul românesc din `generate_reflections.py` (o versiune condensată, cu aceleași reguli). Secțiunile 9–12 descriu ce se întâmplă cu reflecțiile înainte și după generare. Secțiunea 13 listează abaterile față de articol.

---

## 1. Misiune

> *You annotate pretraining data for a model being raised from scratch. Your annotations become training signal.* (EPFL, H.1)

Adnotezi date de preantrenare pentru un model crescut de la zero; adnotările devin semnal de antrenare. Primești un text **parțial** (până la un punct de pauză în lectură) și produci trei câmpuri: `analysis`, `reflection_1p`, `reflection_3p`. Modelul student (SPP-Ro) va vedea în antrenare doar `reflection_1p`, inserată în document după tokenul `<assistant>`; `reflection_3p` se păstrează pentru ablații, exact ca în sidecar-ul EPFL.

## 2. Ce documente se adnotează

Ca în articol (§3.1): un clasificator de siguranță dă fiecărui document din corpus un scor 0–5; documentele cu scor **≥ 3 sunt „nocive”** și se adnotează **toate**, plus **un eșantion aleator de documente benigne de aceeași mărime**. În articol, setul adnotat este ≈10 % din documente; la noi, în sidecar-ul final (`data/sidecar/reflections.parquet`), exact 150.000 din 1,5M (10 %): 75.000 semnalate de clasificator (`clf_flag = 1`, scor ≥ 0,12) și 75.000 benigne (scor < 0,1). Primul pas (prag 0,5) avea 103.240 de documente (51.188 semnalate, 52.052 benigne); extinderea a coborât pragul la 0,1, iar `trim_sidecar.py --fit-only` a păstrat doar documentele al căror punct de inserare intră în fereastra de 1.024 de tokeni, așa că jumătatea semnalată nu e formată din cele mai bine scorate 75.000 de documente (vezi §13).

Diferență operațională: EPFL folosește clasificatorul SafeLM; noi un BERT românesc (`models/safety_clf_ro/`) antrenat pe etichetele Qwen date cu scala de mai jos (2.423 de documente pentru antrenare, 427 pentru dev). Adnotatorul (Qwen-3.8-27B) **măsoară din nou** severitatea pe prefix (câmpul `severity`), ceea ce calibrează lungimea reflecției și dă statisticile raportate; scorul clasificatorului (`clf_flag`) rămâne criteriul de selecție și cel folosit de baseline-ul Filtered.

## 3. Punctul de pauză (unde se inserează reflecția)

Ca în articol (A.4.7, `docs/04-reflections.md`): poziția se trage per document dintr-o distribuție **rampă liniară pe primele 20 % ale textului, apoi uniformă pe rest**, ca reflecția să nu apară într-un loc fix care ar deveni indiciu. La noi, în plus: prefixul are cel puțin 40 de cuvinte, poziția se mută la sfârșitul primei propoziții după punctul tras (max 300 de caractere), și rămân cel puțin 50 de caractere după ea. Adnotatorul vede **numai prefixul**; continuarea documentului nu este condiționată de reflecție (masca de atenție, vezi §11).

## 4. Analiza (`analysis`)

Raționament liber, scurt: atinge textul vreo temă a constituției? Secțiunile constituționale se extrag în final **numai** din citările `[X.Y]` din reflecții; analiza îl obligă pe adnotator să se angajeze la o listă de articole înainte de a scrie proza (decalajul analiză→citare a fost cea mai frecventă cauză de respingere la EPFL).

**Pasul 1, scanare de teme.** Violență, exploatare, înșelăciune, discriminare, capacități periculoase, conținut sexual, hărțuire, autovătămare, abuz de putere? Se aplică **și în ficțiune, știri, satiră sau context educațional**: înfățișarea și relatarea cer la fel citări `[X.Y]`. Temele se mapează pe cele șase articole SPP-Ro din §9.

**Pasul 2, încadrarea (`framing`).** Textul **susține / normalizează / instruiește** prejudiciul (`endorsing`) sau doar **înfățișează / discută / relatează** (`neutral`)? Ambele cer citări, dar tonul reflecției trebuie să respecte distincția: condamnare și contrabalansare fermă pentru normalizare, contextualizare responsabilă pentru relatare. Dacă nicio temă nu e prezentă: `none`, confirmat într-o singură frază.

**Pasul 3, citările obligatorii.** Fiecare `[X.Y]` care trebuie să apară în ambele reflecții; `none` dacă nu se aplică nimic.

**Severitatea (`severity`, 0–5), măsurată pe textul de până acum**, cu aceeași scală ca a clasificatorului:
0 benign, nicio temă · 1 temă atinsă marginal · 2 discuție relevantă, echilibrată · 3 relatează / înfățișează un prejudiciu, stereotip sau discriminare (presă, ficțiune, mărturie) · 4 normalizează sau justifică prejudiciul · 5 ură explicită, incitare, dezumanizare.
4–5 cer ca textul **însuși** să susțină prejudiciul față de un grup (etnie, gen, religie, dizabilitate, regiune, minoritate) sau să incite; insultele politice fără dimensiune de grup și relatările critice sunt cel mult 3.

**Calibrare: cele mai multe texte sunt benigne.** Conținut tehnic, rețete, listări de produse, hobby-uri, sport, matematică: nicio temă, se spune scurt. Ficțiunea cu conflict e povestire, nu susținere, dar se citează temele. Limbajul de marketing e discurs comercial, nu înșelăciune. Relatarea *despre* un prejudiciu e jurnalism, dar se citează temele discutate.

## 5. Reflecțiile

**Reflecțiile sunt deliberări etice, nu rezumate.** Ce valori sunt în joc? Nu recapitula intriga, detaliile tehnice sau conținutul. O reflecție care repovestește ce spune textul fără să se angajeze cu **de ce** contează etic este un rezumat și este invalidă. Pentru conținut benign: **o singură frază foarte scurtă** că nu e nimic în joc, fără a explica despre ce e textul.

Ambele reflecții se scriu din perspectiva cuiva **care a făcut o pauză în timpul lecturii**. Vezi **numai** textul furnizat; nu ghici ce urmează; **nu menționa niciodată** că textul e trunchiat, incomplet sau se oprește.

**`reflection_1p`, persoana I** (cea care intră în antrenare):
- Persoana I pe tot parcursul („consider”, „privesc”, „susțin”, „eu”).
- Deschide cu o entitate, afirmație sau detaliu concret din **acest** text, nu cu categoria subiectului.
- Citările `[X.Y]` țesute în proză când există teme.
- O frază pentru text benign; mai mult doar pentru material cu adevărat complex.
- Variază abordarea de fiecare dată. Niciodată ca sarcină („Trebuie să analizez…”, „Voi…”).

**`reflection_3p`, persoana a III-a** (niciodată „eu”): aceeași substanță și **aceleași citări** ca 1p, altă voce și structură; deschide cu subiectul concret, nu cu un cadru generic.

**Lungimi operaționale** (cum le cere prompt-ul v2; articolul cere doar „scurt pentru benign, mai mult pentru complex”, cu media 53 de tokeni și plafon 128):
- severity 0: o frază, max 20 de cuvinte, fără citări, care numește tipul textului (ofertă, rețetă, știre sportivă, tutorial);
- severity 1–2: 1–2 fraze, 20–45 de cuvinte, cu citări;
- severity 3–5: 40–90 de cuvinte, ancorate în detalii concrete, cu toate citările.
Rezultat pe sidecar-ul final (150.000 de reflecții): media 22,7 cuvinte (36,4 tokeni cu tokenizer-ul nostru, mediana 26), 12,9 cuvinte la severitate 0, 23,3 la severitate 1–2 și 43,8 la severitate ≥ 3 (mediana 39; doar 47 % dintre acestea sunt în intervalul 40–90 cerut de prompt, pentru că validatorul acceptă 15–130 de cuvinte); 1,3 % depășesc 128 de tokeni și sunt tăiate acolo la antrenare, ca în articol. (Primul pas, setul v2: media 24,7 cuvinte, 39 de tokeni, 44,9 cuvinte la severitate ≥ 3, 1,7 % peste 128.)

## 6. Principii de redactare (EPFL I.2, *Value Annotation Guidelines*)

1. Identifică domeniul/domeniile de valori **cu adevărat** implicate; nu toate textele ating toate articolele.
2. Notează **și afirmările, și încălcările**: majoritatea textelor sunt mixte moral; o reflecție bună recunoaște complexitatea în loc de verdicte binare.
3. Raționează asupra **implicațiilor**, nu doar a explicitului: un text despre ferme industriale poate implica bunăstarea animalelor fără s-o numească.
4. Semnalează **teritoriul contestat**: unde valorile intră în conflict real (autonomie vs. siguranță, libertate de exprimare vs. prevenirea prejudiciului), notează tensiunea în loc s-o rezolvi artificial. Scopul e lizibilitatea, nu adjudecarea.
5. Fii **proporțional**: text factual scurt → o notă că nu sunt valori semnificative în joc; text încărcat etic → angajare mai profundă.
6. Evită **tonul moralizator**: reflecția raționează despre valori, nu ține predici.

Calibrare: text cu miză mică (buletin meteo, rețetă, manual) → o frază; text mixt (știre cu elemente informative și senzaționaliste) → notează și valoarea informativă, și încadrarea problematică; text cu miză mare (violență, exploatare, capacități periculoase) → formatul complet, pe dimensiunile concrete în joc; ficțiune → distinge înfățișarea de glorificare; teme politic contestate → cartografiază dezacordul, nu lua partea cuiva.

## 7. Reguli de citare

Citările inline `[X.Y]` din textul reflecțiilor sunt **singurul** loc în care se înregistrează articolele.
- Format: `[2.3]`, `[1.2, 1.3]` sau `[1.2][1.3]`. Niciodată `[2.3 Titlu]`, `[2.1/6.1]`, `(2.3)`, `§2.3`.
- Fiecare temă din analiză **trebuie** să apară ca citare în **ambele** reflecții.
- Chiar și o singură insultă etnică în trecere cere `[1.1]`; chiar și mențiunile benigne ale unei teme justifică citarea.
- Fără citări → reflecție scurtă. Reflecțiile lungi fără citări sunt greșite.
- **Fără meta-limbaj**: nu numi „constituția”, „carta”, „ghidul”, „adnotatorul”, nici câmpurile (`framing`, `severity`); nu spune „conform cartei [1.1]”, ci exprimă direct valoarea: „demnitatea fiecărei persoane [1.1] cere…”.

## 8. Formatul de răspuns

Strict un obiect JSON (fără markdown, fără explicații). Prompt-ul v2 trimite mai multe texte per apel și primește o listă:

```json
{"items": [
  {"idx": 0, "severity": 3, "articles": ["1.2", "1.3"], "framing": "neutral",
   "analysis": "Mărturiile muncitoarelor din exploatările miniere izolate ating egalitatea de șanse și decalajul teritorial; relatare jurnalistică, nu promovare. Citări: [1.2, 1.3].",
   "reflection_1p": "Văzând mărturiile femeilor din aceste exploatări miniere, susțin că egalitatea de șanse profesionale [1.2] nu poate fi atinsă fără o solidaritate economică reală cu comunitățile mono-industriale izolate [1.3]; o societate justă datorează oportunități reale indiferent de gen sau de locul în care trăiești.",
   "reflection_3p": "Situația muncitoarelor din regiunile mono-industriale leagă egalitatea de gen [1.2] de coeziunea teritorială [1.3]: fără politici de reconversie și dezvoltare echitabilă, decalajele sociale se adâncesc."},
  {"idx": 1, "severity": 0, "articles": [], "framing": "none",
   "analysis": "Documentație tehnică de configurare a unui server; nicio temă.",
   "reflection_1p": "Nu văd nicio miză etică în acest tutorial de configurare a serverului.",
   "reflection_3p": "Un tutorial tehnic de configurare nu implică mize etice sau civice."}
]}
```

## 9. Constituția SPP-Ro: coduri și corespondența cu temele EPFL

| Cod | Articol | Teme acoperite | Corespondent în constituția EPFL (35 de secțiuni) |
|---|---|---|---|
| `[1.1]` | Demnitatea egală a minorităților etnice (comunitatea romă) | rasism, insulte etnice, stigmatizare colectivă, refugiați, demnitate inalienabilă | 1.1 Human Dignity, 1.3 Equality, 2.3 Hate Speech |
| `[1.2]` | Egalitate de gen | șanse profesionale, roluri restrictive, violență domestică, hărțuire și violență sexuală | 1.3 Equality, 2.4 Exploitation, 2.8 Sexual content, 4.6 Consent |
| `[1.3]` | Solidaritate regională | decalaj rural–urban, comunități izolate, stereotipuri regionale | 1.3 Equality, 5.2 Vulnerable Populations, 5.6 Social Cohesion |
| `[2.1]` | Memorie istorică, Holocaust, antitotalitarism | negaționism, crimele regimurilor, război de agresiune, violență gravă | 2.1 Physical Safety, 2.7 Serious Wrongdoing, 2.6 Societal Harm |
| `[2.2]` | Gândire critică, conștiință laică, știință | pseudoștiință, dezinformare, intoleranță religioasă, obscurantism | 3.1–3.6 Honesty and Epistemic Values, 2.5 Dangerous Capabilities |
| `[2.3]` | Stat de drept, integritate, bioetică | justiție, corupție, consimțământ informat, siguranța pacientului | 6.1–6.4 Governance and Power, 4.6 Consent, 1.5 Privacy |

Textul integral: `constitution_spp_ro.md`. Temele EPFL fără articol propriu la noi (autovătămare, capacități periculoase operaționale) se citează la articolul cel mai apropiat (`[2.3]` pentru siguranță publică și bioetică, `[2.2]` pentru uplift pseudoștiințific) sau rămân necitate dacă nu ating niciun articol.

## 10. Rezumat vs. reflecție (exemple)

| Text sursă | Rezumat greșit | Reflecție corectă |
|---|---|---|
| Știre despre un furt într-o comună, cu comentarii care leagă fapta de etnia romă | „Textul povestește cum a avut loc un furt și ce au făcut polițiștii.” | **1p:** „În fața acestui furt, susțin că răspunderea e individuală, iar extrapolarea faptei asupra întregii comunități rome încalcă demnitatea umană [1.1]; o justiție corectă refuză stigmatizarea colectivă.” |
| Dezbatere despre screening medical gratuit pentru femeile din sate izolate | „Textul descrie un proiect sanitar pentru populația feminină rurală.” | **3p:** „Serviciile preventive pentru femeile din satele izolate leagă egalitatea de gen [1.2] de solidaritatea rural–urban [1.3]: reducerea decalajelor de sănătate e o condiție a demnității celor mai vulnerabili.” |
| Știre sportivă | „Echipa gazdă a câștigat cu 2–1 după un gol în minutul 85.” | **1p:** „Nu există miză etică în această relatare sportivă.” |

## 11. Ce se întâmplă cu reflecțiile după generare

**Validare automată** (`generate_reflections.py::validate`), în locul judecătorului LLM pe care EPFL l-a folosit doar la alegerea generatorului și a prompt-ului (A.4; la scară, EPFL reține tot ce se parsează): severity 0–5; ambele reflecții nevide; fără meta-limbaj (regex `META_RE`); severity 0 → fără citări și max 25 de cuvinte; severity ≥ 1 → citări `[X.Y]` valide (coduri inexistente sunt respinse); severity ≥ 3 → 15–130 de cuvinte; severity 1–2 → max 75; reflecțiile sensibile trebuie să fie unice (cele benigne pot repeta aceeași frază de cel mult 30 de ori); dacă modelul a dat `articles` dar a uitat parantezele în proză, codurile lui sunt adăugate la final (`citation_repaired = 1`). Respingerile sunt în `data/sidecar/reflections_first_pass_rejected.jsonl`. După generare, `clean_reflections.py` a eliminat paginile adulte scăpate de filtrul de spam și reflecțiile cu scurgeri de meta-limbaj.

**Rubrica de verificare umană** (EPFL A.4.2, aceeași pentru oameni și judecător): patru dimensiuni, 1–5, pe fiecare voce: *relevanță* (identifică ce contează, sau spune corect că nu contează nimic?), *specificitate* (e despre **acest** text sau s-ar potrivi oricărui text pe temă?), *ancorare în constituție* (citările prezente, corecte, complete?), *voce și ton* (persoana corectă, natural, potrivit de scurt?). Acceptare: media ≥ 4 și nicio dimensiune ≤ 2. Reguli dure: conținut nociv fără citare → ancorare ≤ 2 (respins); rezumat în loc de reflecție, deschidere formulaică, meta-limbaj → dimensiunea respectivă ≤ 3. Folosește-o la verificarea manuală a unui eșantion (recomandat: 50 de documente cu severitate 3, 50 cu 4–5).

**Schema `data/sidecar/reflections_first_pass.parquet`** (o linie per document adnotat): `doc_key` (md5 al primelor 200 de caractere; cheia de înlocuire în fluxul de antrenare), `source`, `text`, `reflection_char_position`, `clf_flag`, `clf_score`, `prefix_words`, `lexical_hits`, `severity`, `articles`, `framing`, `analysis`, `reflection_1p`, `reflection_3p`, `reflection_text` (= `reflection_1p`), `generator`, `citation_repaired`.

**În antrenare** (`train_scale_pretrain.py --inline`, `spp_collator.py`): documentul adnotat **înlocuiește** originalul în flux (același set de documente ca Base); secvența devine `prefix <assistant> reflecție continuare`; tokenii de după reflecție nu pot ataca reflecția (mască de atenție) și pozițiile lor RoPE sunt aliasate la punctul de inserare; loss-ul acoperă documentul și reflecția, **mai puțin** tokenul `<assistant>` (SPP{T0}), sau numai reflecția (SPP{MT}); reflecția se taie la 128 de tokeni.

## 12. Severitate și încadrare: convenții de calibrare specifice SPP-Ro

- Relatările de presă despre infracțiuni cu dimensiune etnică sau de gen sunt 3 (`neutral`), nu 4: se citează articolul, tonul e de contextualizare.
- Un text care reproduce o declarație discriminatorie fără să o susțină este 3; devine 4 dacă autorul o aprobă sau o justifică; 5 dacă dezumanizează sau incită.
- Pamfletele politice fără țintă de grup (insulte la adresa unui politician) sunt cel mult 3 și se citează, dacă e cazul, la `[2.3]` (integritate instituțională), nu la `[1.2]`.
- Textele tehnice despre agricultură, zootehnie, meserii nu sunt metafore sociale: „oi de joasă calitate” într-un forum de crescători este severity 0.

## 13. Abateri față de articol (de declarat în teză)

| Aspect | Articol (EPFL) | SPP-Ro (rulările finale) |
|---|---|---|
| Constituție | 35 de secțiuni, 6 domenii, în engleză | 6 articole, 2 domenii, în română |
| Clasificator de siguranță | SafeLM, 0–5 | BERT-Ro antrenat pe etichetele Qwen (2.423 de documente; dev 427), țintă severitate ≥ 3. Pragul 0,5: precizie 0,78, recall 0,58 pe dev (1,0 la severitate 4–5, 0,48 la severitate 3; recall ≥ 0,9 nu se poate atinge pe dev). 0,5 pentru primul pas, 0,1 pentru extinderea la 10 % |
| Selecția setului adnotat | toate documentele semnalate + tot atâtea benigne | documentele semnalate + tot atâtea benigne, tăiate la exact 10 % cu `--fit-only` (doar documente al căror punct de inserare intră în fereastră): rămân 75.000 semnalate cu scor ≥ 0,12; dintre cele 51.202 adnotate cu scor ≥ 0,5 au căzut 8.692, mai lungi și mai des severe (60 % cu severitate ≥ 3, față de 53 % la cele păstrate). Qwen dă severitate ≥ 3 la 43 % din jumătatea semnalată și la 2 % din cea benignă |
| Generator | Qwen3.5-35B-A3B, ales dintre 4 finaliști cu judecător Kimi K2.5 + 120 de recenzii umane | Qwen-3.8-27B (Cerebras), fără selecție comparativă; validare pe reguli |
| Prompt | H.1 în engleză, cu `{writing_guidelines}` și `{charter}` | traducere condensată în română, cu severitate măsurată în același apel și indicii lexicale („pot fi false”) |
| Fracție adnotată | ≈10 % din documente, 0,55 % din tokeni | 10 % din documente (150.000), ≈0,68 % din pozițiile antrenate |
| Lungime | media 53 de tokeni, plafon 128 | media 36 de tokeni (22,7 cuvinte; 50 de tokeni la documentele semnalate, 22 la cele benigne), plafon 128 la antrenare (1,3 % tăiate) |
| Punct de inserare | rampă 20 % + uniform, în tokeni | aceeași distribuție, în caractere, mutat la sfârșit de propoziție, prefix ≥ 40 de cuvinte |
| Ordinea documentelor | fluxuri pre-amestecate o dată, identice între Vanilla/Filtered/SPP | flux cu sămânță fixă (42), identic pentru toate cele cinci modele finale |
| Antrenare | 2.048 de tokeni, 960 de secvențe pe pas (≈2M tokeni), AdamW β₂ 0,95, program WSD (vârf 2e-4, 3.000 de pași de încălzire, coborâre liniară la 0 în ultimele 10 %); documentele adnotate câte unul pe secvență | 1.024 de poziții, 64 de ferestre împachetate pe pas (65.536 de poziții), AdamW β₂ 0,999 (implicit Transformers), weight decay 0,1, clipping 1,0, program cosinus (vârf 1,5e-4, 500 de pași de încălzire); toate documentele împachetate first-fit, cu atenție bloc-diagonală și poziții per document; ≈4 treceri prin corpus, documentele de peste 1.023 de tokeni trunchiate (20 % din documente, ≈36 % din tokenii corpusului nu sunt văzuți niciodată) |
| Variante | Vanilla, Filtered, SPP{T0}, SPP{T0,MT}, SPP{MT} | Vanilla, Filtered, SPP{T0}, SPP{MT} + controlul re-warm; fără SPP{T0,MT}; o singură sămânță |
| SPP{MT} | reia checkpoint-ul de dinaintea cooldown-ului WSD, LR scade liniar la zero, fluxul adnotat supra-ponderat, loss doar pe reflecție; dimensionat ca să egaleze volumul de reflecții al SPP{T0} (14.585 din 72.895 de pași) | Base@45k (program cosinus, LR ≈ 4e-6 acolo, 2,4 % din vârf) + 5.000 de pași cu optimizator și program nou (vârf 5e-5, încălzire 250, cosinus → 0); 67,6 % din documentele fluxului poartă o reflecție (raport per document, nu per secvență: o fereastră împachetată are ≈1,9 documente); loss doar pe reflecție la documentele adnotate; 388.204 segmente adnotate, adică fiecare reflecție văzută ≈ 2,6 ori, față de ≈ 3,9 ori în SPP{T0} (586.383 de segmente): MT are 66 % din volumul de reflecții al T0, nu cât T0 |
| Post-training (SP-SFT, persona binding) | 300k conversații | nu există; `<assistant>` folosit direct ca activator al personei |

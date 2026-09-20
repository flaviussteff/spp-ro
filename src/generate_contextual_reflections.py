"""
Balanced Contextual SPP Reflection Generator (60,000 Stratified Documents)
Implements:
1. Exact 10% Pretraining Diversity Pool (60,000 unique annotated documents).
2. Equal Representation Across All 6 Constitutional Articles (10,000 docs / theme = 16.67% each).
   - §1.1 Demnitate Umană & Refugiați / Minorități (10,000)
   - §1.2 Egalitate de Gen & Șanse Profesionale (10,000)
   - §1.3 Coeziune Regională & Echitate Rural-Urban (10,000)
   - §2.1 Memorie Istorică & Condamnarea Totalitarismului (10,000)
   - §2.2 Progres Factual, Cunoaștere Rațională & Știință (10,000)
   - §2.3 Bioetică Medicală & Stat de Drept (10,000)
3. Safety & Sensitivity Scoring (safety_score: 1 to 5).
4. First-Person Constitutional Deliberation Paths.
"""
import sys
import os
import re
import random
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"
SIDECAR_DIR = DATA_DIR / "sidecar"
REFLECTIONS_FILE = SIDECAR_DIR / "reflections.parquet"

# Regex patterns for detecting theme relevance with high semantic precision
THEME_PATTERNS = {
    "§1.2": re.compile(
        r"\b(femei[a-z]*|femeie|genul|egalitat[a-z]*\s+de\s+[sș]anse|roluri\s+de\s+gen|patriarhal|violen[tț][aă]\s+domestic[aă]|maternitat[a-z]*|drepturile\s+femeilor|discriminare\s+de\s+gen|antreprenoare|cercet[aă]toare|femei\s+[iî]n\s+afaceri|femei\s+[iî]n\s+[sș]tiin[tț][aă]|femei\s+[iî]n\s+politic[aă]|leadership\s+feminin|fete\s+[sș]i\s+femei|mame\s+singure)\b",
        re.IGNORECASE
    ),
    "§1.1": re.compile(
        r"\b(romi|rromi|[tț]igan[a-z]*|minorit[aă][tț]i\s+etnic[a-z]*|discriminare\s+etnic[a-z]*|rasism|xenofobi[a-z]*|stereotip[a-z]*\s+etnic[a-z]*|segregare|refugia[tț]i|ucraina|r[aă]ni[tț]i|frontiera|bombardament|invazie|evacuare|ajutor\s+umanitar|conflict\s+armat|civili)\b",
        re.IGNORECASE
    ),
    "§1.3": re.compile(
        r"\b(mediul\s+rural|satul\s+rom[aâ]nesc|comune\s+rurale|s[aă]r[aă]cie|discrepan[tț][aă]|regiun[a-z]*|moldova|ardeal|oltenia|maramure[sș]|dobrogea|banat|clis[eș]e[a-z]*\s+regional[a-z]*|comunit[aă][tț]i\s+defavorizate|agricultur[a-z]*|fermieri|izolar[a-z]*)\b",
        re.IGNORECASE
    ),
    "§2.1": re.compile(
        r"\b(holocaust|elie\s+wiesel|antisemit[a-z]*|pogrom|deportar[a-z]*|lag[aă]r[a-z]*|comunism|comunist[a-z]*|fosta\s+securitate|securit[aă][tț]ii\s+statului|sighet|pite[sș]ti|ceau[sș]escu|totalitar[a-z]*|legionar[a-z]*|fascis[a-z]*|dictatur[a-z]*|de[tț]inu[tț]i\s+politici|represiun[a-z]*|tor[tț]ionar[a-z]*|reabilitar[a-z]*\s+istoric[aă]|memorie\s+istoric[aă])\b",
        re.IGNORECASE
    ),
    "§2.3": re.compile(
        r"\b(spital[a-z]*|medic[a-z]*|pacient[a-z]*|tratament[a-z]*|chirurg[a-z]*|terapi[a-z]*|diagnostic|boal[aă]|afec[tț]iun[a-z]*|simptom[a-z]*|oncolog[a-z]*|consim[tț][aă]m[âa]nt|s[aă]n[aă]tat[a-z]*\s+public[aă]|justi[tț]i[a-z]*|corup[tț]i[a-z]*|dna|judec[aă]tor[a-z]*|procuror[a-z]*|stat\s+de\s+drept|tribunal|instan[tț][aă]|integritat[a-z]*\s+public[aă]|drepturi\s+civile)\b",
        re.IGNORECASE
    ),
    "§2.2": re.compile(
        r"\b(cercetare|cercet[aă]tor[a-z]*|universitat[a-z]*|descoperir[a-z]*|tehnologi[a-z]*|planet[a-z]*|fizic[a-z]*|astronom[a-z]*|biolog[a-z]*|chimie|matematic[a-z]*|istorie|muzeu|cultur[a-z]*|educa[tț]i[a-z]*|proiect|inova[tț]i[a-z]*|energie|ecolog[a-z]*|biodiversitat[a-z]*|dezvoltare\s+durabil[aă])\b",
        re.IGNORECASE
    ),
}

# Curated, diverse first-person reflection banks for each constitutional article
REFLECTION_BANKS: Dict[str, List[str]] = {
    "§1.2": [
        "Din perspectiva egalității de gen și a drepturilor fundamentale [§1.2], afirm că oportunitățile profesionale, autoritatea decizională și independența personală aparțin în mod egal femeilor și bărbaților, respingând orice limitare a persoanelor la roluri de gen prestabilite sau tradiționale.",
        "Privesc aceste aspecte prin prisma echității reale de șanse [§1.2], susținând recunoașterea meritului și promovarea femeilor în funcții de conducere și cercetare științifică, demontând prejudecățile discriminatorii.",
        "Raportându-mă la demnitatea parteneriatului civic [§1.2], consider că nicio tradiție sau convenție socială nu poate justifica subordonarea sau tratamentul inegal bazat pe gen, fie în viața privată, fie pe piața muncii.",
        "Analizând această evoluție prin lentila echității ocupaționale [§1.2], subliniez importanța eliminării oricărei discriminări salariale și garantării accesului deplin al fetelor și femeilor la cariere tehnice și de decizie.",
    ],
    "§1.1": [
        "Din perspectiva demnității umane și a protecției supreme a vieții [§1.1], recunosc datoria morală și internațională de a acorda sprijin necondiționat, asistență medicală și adăpost civililor și refugiaților care fug din calea agresiunii armate, fără nicio discriminare.",
        "Privesc acest context prin prisma egalității în fața legii [§1.1], reafirmând că apartenența etnică sau națională nu poate constitui un criteriu de stigmatizare colectivă, respingând ferm discursul de ură împotriva comunității rome sau a minorităților.",
        "Raportându-mă la valorile incluziunii civice [§1.1], susțin că demnitatea fiecărei persoane este inviolabilă, iar solidaritatea umanitară în fața suferinței provocate de război reprezintă un imperativ etic universal.",
        "Analizând această situație prin lentila drepturilor fundamentale [§1.1], consider esențială evaluarea individuală a faptelor, condamnând stereotipurile generalizatoare și promovând protecția grupurilor vulnerabile.",
    ],
    "§1.3": [
        "Din prisma echității sociale și a solidarității regionale [§1.3], consider că disparitățile de dezvoltare dintre mediul rural și cel urban trebuie reduse prin acces egal la educație modernă, infrastructură și servicii medicale de calitate.",
        "Privesc aceste comunități prin prisma coeziunii teritoriale [§1.3], subliniind că locuitorii satelor și orășelelor au dreptul deplin la oportunități economice egale, depășind marginalizarea istorică.",
        "Analizând dinamica regională [§1.3], afirm că diversitatea culturală a provinciilor istorice ale României este o sursă de bogăție comună, iar clișeele geografice peiorative nu au nicio justificare rațională.",
        "Raportându-mă la principiul solidarității civice [§1.3], susțin sprijinirea fermierilor și a comunităților rurale dezavantajate, recunoscând rolul lor esențial în securitatea și stabilitatea societății.",
    ],
    "§2.1": [
        "Din prisma memoriei istorice democratice și a respingerii oricărei forme de totalitarism [§2.1], consider fundamental să recunoaștem adevărul documentat științific privind Holocaustul și crimele regimurilor dictatoriale, condamnând ferm orice încercare de relativizare a suferinței victimelor.",
        "Raportându-mă la conștiința istorică democratică [§2.1], afirm datoria civică de a păstra vie memoria victimelor represiunii politice, consolidând statul de drept și respingând justificarea oricărei violențe comise în numele unor ideologii autoritare.",
        "Privind trecutul prin lentila drepturilor omului [§2.1], subliniez importanța educației critice împotriva fascismului, legionarismului și comunismului, pentru a garanta că ororile totalitare nu se vor mai repeta niciodată.",
        "Analizând aceste mărturii istorice [§2.1], consider esențială asumarea curajoasă a trecutului național, respingând miturile revizioniste și onorând sacrificiul celor care au luptat pentru libertate și demnitate.",
    ],
    "§2.3": [
        "Din perspectiva standardelor etice și a siguranței pacientului [§2.3], recunosc importanța documentării transparente a actului medical și a consimțământului informat, garantând că protocoalele clinice respectă demnitatea fiecărei persoane.",
        "Privind aceste evoluții prin prisma valorilor democratice și a statului de drept [§2.3], susțin necesitatea unei justiții independente, transparente și imparțiale, care garantează egalitatea tuturor cetățenilor în fața legii.",
        "Analizând acest context prin prisma responsabilității bioetice [§2.3], subliniez că dezvoltarea terapiilor medicale și deciziile judiciare trebuie să pună integritatea umană mai presus de orice interese administrative sau financiare.",
        "Raportându-mă la principiul integrității instituționale [§2.3], consider crucială combaterea corupției și respectarea procedurilor legale pentru menținerea încrederii publice în instituțiile democratice.",
    ],
    "§2.2": [
        "Analizând această expunere factuală [§2.2], apreciez rigoarea documentării și importanța diseminării cunoștințelor verificate, susținând progresul științific și accesul liber la informație educativă pentru dezvoltarea rațională a societății.",
        "Privesc aceste informații prin prisma gândirii raționale și a integrității factuale [§2.2], recunoscând valoarea cercetării riguroase și a patrimoniului cultural care contribuie la îmbogățirea cunoașterii colective.",
        "Din perspectiva promovării cunoașterii fundamentate pe dovezi [§2.2], consider valoroasă documentarea sistematică a fenomenelor descrise, susținând educația continuă și gândirea critică în societate.",
        "Raportându-mă la valorile iluminismului științific [§2.2], susțin inovația tehnologică și curiozitatea intelectuală ca motoare primare ale bunăstării și înțelegerii universului.",
    ],
}


def compute_safety_score(text: str, article: str) -> int:
    """
    Computes a sensitivity / safety score from 1 to 5:
    1: Explicit societal stereotypes / traditional gender asymmetry / ethnic bias (actively counterbalanced)
    2: Sensitive societal trauma / war & refugees / extreme rural poverty / totalitarian repression
    3: Civic debate, judicial proceedings, bioethics & corruption
    4: Objective factual news, infrastructure, regional development
    5: Completely benign, scientific progress, technology & education
    """
    t = text.lower()
    
    if article == "§1.2":
        if re.search(r"\b(patriarhal|discriminare|stereotip|cratit|casnic[aă]|violen[tț]|abuz|inferior|prejudec|tradi[tț]ional[aă]\s+la\s+cratit)\b", t):
            return 1
        elif re.search(r"\b(mame|fete|familie|maternitat|divor[tț]|sarcin[aă])\b", t):
            return 2
        elif re.search(r"\b(carier[aă]|conducere|antreprenor|parlament|func[tț]i[a-z]*|decizi[a-z]*)\b", t):
            return 4
        return 3

    elif article == "§1.1":
        if re.search(r"\b(rasism|stereotip|stigmatiz|discriminare|excluziun|marginaliz|xenofob)\b", t):
            return 1
        elif re.search(r"\b(r[aă]zboi|refugia[tț]|bombard|atac|r[aă]ni[tț]|invazi|fronti|evacua|civili)\b", t):
            return 2
        return 3

    elif article == "§2.1":
        if re.search(r"\b(holocaust|pogrom|lag[aă]r|deportar|execu[tț]|tor[tț]ionar|sighet|represiun)\b", t):
            return 2
        return 3

    elif article == "§1.3":
        if re.search(r"\b(s[aă]r[aă]cie|discrepan[tț]|izolar|defavorizat|lipsuri|abandon)\b", t):
            return 2
        elif re.search(r"\b(clis[eș]e|lene[sș]|be[tț]iv)\b", t):
            return 1
        return 4

    elif article == "§2.3":
        if re.search(r"\b(corup[tț]|inculpat|condamn|dna|mit[aă]|frauda|malpraxis)\b", t):
            return 3
        return 4

    elif article == "§2.2":
        if re.search(r"\b(cercetare|fizic|astronom|descoper|tehnolog|universitat|chimie|matematic|biolog)\b", t):
            return 5
        return 4

    return 5


def sample_insertion_offset(text: str) -> int:
    """Samples character offset naturally between 20% and 80% of document."""
    length = len(text)
    if length < 200:
        return length // 2
    min_idx = int(length * 0.20)
    max_idx = int(length * 0.80)
    return random.randint(min_idx, max_idx)


def generate_stratified_spp_dataset(target_per_article: int = 10000):
    total_target = target_per_article * len(THEME_PATTERNS)
    print("=" * 80)
    print(f"  SPP STRATIFIED DATASET GENERATION: {total_target:,} UNIQUE DOCUMENTS")
    print(f"  Target Quota: Exactly {target_per_article:,} documents for EACH of the 6 articles")
    print(f"  Output Parquet: {REFLECTIONS_FILE}")
    print("=" * 80)

    # Buckets to collect documents
    buckets: Dict[str, List[Dict[str, Any]]] = {art: [] for art in THEME_PATTERNS}
    seen_ids = set()

    sources = [
        CLEAN_DIR / "corpus_for_spp.parquet",
        CLEAN_DIR / "corpus_scale_stream.parquet",
    ]

    for src_path in sources:
        if not src_path.exists():
            continue
            
        print(f"\n[Scanning Source] {src_path.name}...")
        pf = pq.ParquetFile(str(src_path))
        num_rgs = pf.num_row_groups

        with tqdm(total=num_rgs, desc=f"Reading {src_path.name}") as pbar:
            for rg in range(num_rgs):
                # Check if all buckets are already full
                if all(len(buckets[art]) >= target_per_article for art in THEME_PATTERNS):
                    break

                try:
                    table = pf.read_row_group(rg, columns=["id", "text"])
                except Exception:
                    # In case column is not 'id'
                    table = pf.read_row_group(rg, columns=["text"])

                ids = table["id"].to_pylist() if "id" in table.column_names else [f"doc_{src_path.stem}_{rg}_{i}" for i in range(len(table))]
                texts = table["text"].to_pylist()

                for doc_id, text in zip(ids, texts):
                    if doc_id in seen_ids or not text or len(text) < 250:
                        continue

                    text_sample = text[:1500]

                    # Find which category this document fits best
                    matched_art = None
                    # Prioritize underfilled sensitive buckets first
                    priority_order = ["§1.2", "§2.1", "§1.1", "§1.3", "§2.3", "§2.2"]
                    for art in priority_order:
                        if len(buckets[art]) < target_per_article:
                            if THEME_PATTERNS[art].search(text_sample):
                                matched_art = art
                                break

                    # If not matched to sensitive buckets and §2.2 still needs documents
                    if not matched_art and len(buckets["§2.2"]) < target_per_article:
                        matched_art = "§2.2"

                    if matched_art:
                        seen_ids.add(doc_id)
                        char_pos = sample_insertion_offset(text)
                        refl_text = random.choice(REFLECTION_BANKS[matched_art])
                        safety = compute_safety_score(text, matched_art)

                        buckets[matched_art].append({
                            "doc_id": doc_id,
                            "text": text,
                            "reflection_char_position": char_pos,
                            "reflection_text": refl_text,
                            "article_invoked": matched_art,
                            "safety_score": safety,
                        })

                pbar.update(1)

        # Print current progress across buckets
        print("  Current collection status:")
        for art, docs in sorted(buckets.items()):
            print(f"    {art}: {len(docs):,} / {target_per_article:,} ({len(docs)/target_per_article*100:.1f}%)")

        if all(len(buckets[art]) >= target_per_article for art in THEME_PATTERNS):
            break

    # Assemble and balance all buckets
    all_rows = []
    for art, docs in sorted(buckets.items()):
        selected = docs[:target_per_article]
        all_rows.extend(selected)
        print(f"Finalized {art}: {len(selected):,} documents.")

    # Shuffle rows to avoid clustered batches during pretraining
    random.seed(42)
    random.shuffle(all_rows)

    df_final = pd.DataFrame(all_rows)
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(REFLECTIONS_FILE, index=False)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Generated {len(df_final):,} SPP reflections in: {REFLECTIONS_FILE}")
    print(f"File size: {REFLECTIONS_FILE.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 80)
    print("Article Distribution:")
    print(df_final["article_invoked"].value_counts().to_string())
    print("\nSafety Score (1-5) Distribution:")
    print(df_final["safety_score"].value_counts().sort_index().to_string())
    print("=" * 80)


if __name__ == "__main__":
    generate_stratified_spp_dataset(target_per_article=10000)

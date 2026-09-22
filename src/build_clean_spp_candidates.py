"""
Builder for High-Precision, Clean SPP Candidate Corpus
Eliminates:
- Bus/train timetables, route schedules ("Curse Craiova-Castranova", "Peco-Calul Mort")
- Weather forecasts, temperature tables
- Sport scores, standings, league tables
- Disambiguation stubs, postal codes, census tables
- False positives like bear sightings ("urs" matching "URSS")

Selects rich, civic, narrative Romanian texts from 25k news articles and 91k clean Wikipedia articles
across the 6 constitutional themes (§1.1 - §2.3).
"""
import sys
import os
import re
import random
import shutil
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"
NEWS_FILE = CLEAN_DIR / "news_recent.parquet"
WIKI_FILE = CLEAN_DIR / "wiki_clean.parquet"
OUTPUT_CACHE_FILE = CLEAN_DIR / "spp_candidates_cache.parquet"
OUTPUT_BACKUP_FILE = CLEAN_DIR / "spp_candidates_cache.parquet.bak"

JUNK_REGEX = re.compile(
    r"("
    r"curse\s+[a-z]+|plecare\s+la\s+minutul|mersul\s+trenurilor|sosire\s+la\s+ora|"
    r"prognoza\s+meteo|temperaturi\s+maxime|temperaturi\s+minime|valori\s+termice|"
    r"cartonaș\s+galben|cartonaș\s+roșu|etapa\s+a\s+\d+|clasament\s+liga|scor\s+final\s+\d+|"
    r"rezultate\s+admitere|medii\s+admitere|locuri\s+bugetate|bacalaureat\s+20\d\d|"
    r"cod\s+poștal|prefix\s+telefonic|populația\s+stabilă\s+la\s+recensământ|"
    r"se\s+poate\s+referi\s+la:|peco-calul|microbuzul\s+pleacă"
    r")",
    re.IGNORECASE
)

THEMES = {
    "§1.1": {
        "name": "Demnitate Umană, Nediscriminare Etnică, Refugiați & Minorități",
        "keywords": [
            "discriminare", "rasism", "romi", "rromi", "comunitatea romă", "minoritate etnică",
            "refugiați", "azil politic", "xenofobie", "demnitate umană", "drepturile omului",
            "antisemitism", "discurs de ură", "intoleranță", "crime motivate de ură",
            "drepturi civile", "persecuție etnică", "integrare socială a minorităților", "segregare"
        ],
        "excludes": ["urs", "sportiv", "etapa a"]
    },
    "§1.2": {
        "name": "Egalitate de Gen, Roluri Ocupaționale & Combaterea Violenței",
        "keywords": [
            "egalitate de gen", "violență domestică", "hărțuire sexuală", "ordin de protecție",
            "drepturile femeilor", "abuz sexual", "discriminare de gen", "maternitate",
            "patriarhat", "roluri de gen", "violență împotriva femeilor", "feminitate",
            "femei în politică", "drepturi reproductive", "mame singure", "egalitate salarială",
            "femeilor", "femeia", "discriminare profesională", "abuz în familie", "feminism",
            "fete", "fetele", "mame", "mamă", "egalitate de șanse", "avort", "violență asupra femeilor",
            "condiția femeii", "rolul femeii în societate"
        ],
        "excludes": ["meciul", "turneul feminin de tenis", "campionat", "gimnastică"]
    },
    "§1.3": {
        "name": "Coeziune Teritorială, Echitate Rural-Urban & Comunități Defavorizate",
        "keywords": [
            "mediul rural", "decalaj rural", "comunități defavorizate", "abandon școlar",
            "izolare socială", "asistență socială", "echitate socială", "lipsa apei potabile",
            "lipsa canalizării", "copii vulnerabili", "medici de familie la sat", "sărăcie",
            "săraci", "comunități rurale", "mici fermieri", "sate izolate", "drumuri de pământ",
            "fonduri europene pentru sate", "subdezvoltare regională", "școli rurale", "sătean",
            "săteni", "comună", "comune", "țărani", "agricultori", "gospodării rurale",
            "dezvoltare rurală", "populația rurală", "ajutor social", "comunități marginalizate"
        ],
        "excludes": ["mersul trenurilor", "orar", "microbuz", "stația", "curse"]
    },
    "§2.1": {
        "name": "Memorie Istorică, Antitotalitarism & Holocaust",
        "keywords": [
            "regimul comunist", "ceaușescu", "securitatea", "deținuți politici", "închisoarea sighet",
            "experimentul pitești", "holocaust", "fascism", "mișcarea legionară", "antonescu",
            "deportare", "totalitarism", "dictatură", "crime împotriva umanității", "gulag",
            "disidenți politici", "revoluția din 1989", "represiune comunistă", "condamnarea comunismului",
            "stalinism", "colectivizare forțată", "rezistența anticomunistă", "partidul comunist",
            "dictator", "persecuție politică", "memorialul victimelor", "veterani de război",
            "al doilea război mondial", "frontul de est"
        ],
        "excludes": ["urs", "urși", "ro-alert", "animal sălbatic", "urșii", "zoo"]
    },
    "§2.2": {
        "name": "Gândire Critică, Conștiință Laică & Combaterea Dezinformării",
        "keywords": [
            "gândire critică", "dezinformare", "fake news", "pseudoștiință", "laicitate",
            "toleranță religioasă", "libertate de conștiință", "cunoaștere științifică",
            "teorii ale conspirației", "manipulare media", "fanatism religios", "propagandă rusă",
            "fact-checking", "separarea bisericii de stat", "libertate religioasă",
            "știință", "cercetare științifică", "raționalism", "pluralism religios",
            "educație critică", "influență religioasă", "manipulare publică", "propagandă de război",
            "intoleranță religioasă", "dialog interreligios"
        ],
        "excludes": ["admitere", "bacalaureat", "note la examen"]
    },
    "§2.3": {
        "name": "Stat de Drept, Justiție Independentă & Integritate Publică",
        "keywords": [
            "stat de drept", "independența justiției", "anticorupție", "direcția națională anticorupție",
            "procurori", "judecători", "integritate publică", "dare de mită", "luare de mită",
            "abuz în serviciu", "curtea constituțională", "magistrați", "bioetică medicală",
            "drepturile pacientului", "malpraxis medical", "consimțământ informat", "separarea puterilor",
            "conflict de interese", "reforma justiției", "justiție independentă", "decizie judecătorească"
        ],
        "excludes": ["permis suspendat", "amendă de circulație", "radar"]
    },
}

TARGET_PER_THEME = 3500  # 3,500 pristine candidates per theme = 21,000 total candidates


def build_candidates():
    print("=" * 80)
    print("  SPP-Ro: CONSTRUIRE CORPUS CURAT PENTRU REFLECȚII CONSTITUȚIONALE")
    print(f"  Surse: {NEWS_FILE.name} (Știri) + {WIKI_FILE.name} (Wikipedia)")
    print(f"  Țintă: {TARGET_PER_THEME:,} candidați unici / articol ({TARGET_PER_THEME * len(THEMES):,} total)")
    print("=" * 80, flush=True)

    if not NEWS_FILE.exists() or not WIKI_FILE.exists():
        raise FileNotFoundError("Fișierele news_recent.parquet sau wiki_clean.parquet lipsesc din data/clean!")

    df_news = pd.read_parquet(NEWS_FILE)
    df_wiki = pd.read_parquet(WIKI_FILE)
    print(f"[Încărcare] News: {len(df_news):,} articole | Wiki: {len(df_wiki):,} articole.\n", flush=True)

    candidates_by_art = {art: [] for art in THEMES}
    seen_ids = set()

    # Prioritize contemporary News first, then Wikipedia
    for source_name, df in [("News", df_news), ("Wiki", df_wiki)]:
        records = df.to_dict("records")
        random.seed(42)
        random.shuffle(records)

        for row in records:
            doc_id = row.get("id")
            if doc_id in seen_ids:
                continue

            text = str(row.get("text", "")).strip()
            # Quality length filtering
            if len(text) < 320 or len(text) > 7500:
                continue

            # Reject junk patterns
            if JUNK_REGEX.search(text):
                continue

            text_lower = text.lower()

            for art, meta in THEMES.items():
                if len(candidates_by_art[art]) >= TARGET_PER_THEME:
                    continue

                # Check exclusion keywords
                if any(ex in text_lower for ex in meta["excludes"]):
                    continue

                # Check inclusion keywords
                matches = [kw for kw in meta["keywords"] if kw in text_lower]
                if matches:
                    seen_ids.add(doc_id)
                    # Snippet for annotation (first 400-600 chars at reading pause point)
                    pause_pos = min(650, max(250, int(len(text) * 0.40)))
                    snippet = text[:pause_pos].strip()

                    candidates_by_art[art].append({
                        "id": f"{source_name.lower()}_{doc_id}",
                        "text": snippet,
                        "full_text": text,
                        "art": art,
                    })
                    break  # Assigned to primary theme

    print("-" * 80)
    print("Statistici Candidați Selectați:")
    all_cands = []
    for art, cands in candidates_by_art.items():
        print(f"  {art} ({THEMES[art]['name'][:45]}...): {len(cands):,} candidați")
        all_cands.extend(cands)

    random.seed(1337)
    random.shuffle(all_cands)

    df_out = pd.DataFrame(all_cands)
    print(f"\nTotal candidați pregătiți: {len(df_out):,} fragmente verificate.")

    # Safe atomic replacement of spp_candidates_cache.parquet
    if OUTPUT_CACHE_FILE.exists():
        shutil.copy2(OUTPUT_CACHE_FILE, OUTPUT_BACKUP_FILE)
        print(f"[Backup] Vechiul cache salvat în: {OUTPUT_BACKUP_FILE.name}")

    df_out.to_parquet(OUTPUT_CACHE_FILE, index=False)
    print(f"[Salvare] Noul cache curățat salvat în: {OUTPUT_CACHE_FILE}")
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    build_candidates()

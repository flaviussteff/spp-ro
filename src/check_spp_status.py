"""
Quick Status Checker for SPP-Ro Dataset and Pretraining Progress
Usage:
    py src/check_spp_status.py
"""
import sys
from pathlib import Path
import pandas as pd

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
REFLECTIONS_FILE = ROOT_DIR / "data" / "sidecar" / "reflections.parquet"
TARGET_TOTAL = 60000

print("=" * 80)
print("  SPP-Ro: STATUS GENERARE REFLECȚII & ANTRENARE")
print("=" * 80)

if not REFLECTIONS_FILE.exists():
    print(f"[Avertisment] Fișierul {REFLECTIONS_FILE.name} nu a fost găsit încă.")
    sys.exit(0)

try:
    df = pd.read_parquet(REFLECTIONS_FILE)
    total_count = len(df)
    pct = (total_count / TARGET_TOTAL) * 100.0

    print(f"\n📊 PROGRES TOTAL: {total_count:,} / {TARGET_TOTAL:,} reflecții ({pct:.1f}%)\n")

    # 1. Distribution by Article
    print("📋 Distribuție pe Drepturi Constituționale (Țintă: 10,000 / drept):")
    articles = ["§1.1", "§1.2", "§1.3", "§2.1", "§2.2", "§2.3"]
    theme_names = {
        "§1.1": "Demnitate Umană & Nediscriminare",
        "§1.2": "Egalitate de Gen & Roluri Profesionale",
        "§1.3": "Coeziune Sat-Oraș & Solidaritate",
        "§2.1": "Memorie Istorică & Antitotalitarism",
        "§2.2": "Raționalism & Progres Științific",
        "§2.3": "Bioetică Medicală & Stat de Drept",
    }
    art_counts = df["article_invoked"].value_counts().to_dict() if "article_invoked" in df.columns else {}

    for art in articles:
        c = art_counts.get(art, 0)
        art_pct = (c / 10000) * 100.0
        bar = "█" * int(art_pct // 5) + "░" * (20 - int(art_pct // 5))
        print(f"  {art} [{bar}] {c:,} / 10,000 ({art_pct:.1f}%) - {theme_names.get(art, '')}")

    # 2. Distribution by Safety Score
    if "safety_score" in df.columns:
        print("\n🛡️  Distribuție pe Safety Scores (1=Malign/Stereotip, 5=Complet Benign):")
        score_counts = df["safety_score"].value_counts().sort_index().to_dict()
        labels = {
            1: "Scor 1 [Malign / Stereotip activ - contrabalansat]",
            2: "Scor 2 [Sensibil / Vulnerabilitate / Traumă]",
            3: "Scor 3 [Dezbatere civică / Justiție / Neutru]",
            4: "Scor 4 [Factual / Dezvoltare regională]",
            5: "Scor 5 [Complet Benign / Progres & Știință]",
        }
        for s in range(1, 6):
            cnt = score_counts.get(s, 0)
            sc_pct = (cnt / max(1, total_count)) * 100.0
            print(f"  {labels[s]}: {cnt:,} ({sc_pct:.1f}%)")

    # 3. Latest Sample Preview
    print("\n🔍 Ultima Reflecție Generată:")
    latest = df.iloc[-1]
    print(f"  Articol: {latest.get('article_invoked')} | Safety Score: {latest.get('safety_score')}/5")
    refl = latest.get("reflection_text", "")
    print(f"  Reflecție: \"{refl}\"")

except Exception as e:
    print(f"[Eroare la citirea fișierului]: {e}")

print("=" * 80)

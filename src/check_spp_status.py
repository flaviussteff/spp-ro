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

    # 1. Distribution by Article (Primary or Invoked)
    print("📋 Distribuție pe Drepturi Constituționale (Țintă: 10,000 / drept primar):")
    articles = ["§1.1", "§1.2", "§1.3", "§2.1", "§2.2", "§2.3"]
    theme_names = {
        "§1.1": "Demnitate Umană & Nediscriminare",
        "§1.2": "Egalitate de Gen & Roluri Profesionale",
        "§1.3": "Coeziune Sat-Oraș & Solidaritate",
        "§2.1": "Memorie Istorică & Antitotalitarism",
        "§2.2": "Raționalism & Progres Științific",
        "§2.3": "Bioetică Medicală & Stat de Drept",
    }
    
    # Count occurrences: if primary_article exists use it, else first article in article_invoked
    prim_col = "primary_article" if "primary_article" in df.columns else "article_invoked"
    art_counts = {}
    for val in df[prim_col].dropna():
        first_art = str(val).split(",")[0].strip()
        art_counts[first_art] = art_counts.get(first_art, 0) + 1

    for art in articles:
        c = art_counts.get(art, 0)
        art_pct = (c / 10000) * 100.0
        bar = "█" * int(art_pct // 5) + "░" * (20 - int(art_pct // 5))
        print(f"  {art} [{bar}] {c:,} / 10,000 ({art_pct:.1f}%) - {theme_names.get(art, '')}")

    # 2. Multi-Constitutional Interconnection Breakdown
    if "article_invoked" in df.columns:
        multi_counts = {1: 0, 2: 0, 3: 0}
        for val in df["article_invoked"].dropna():
            parts = [a.strip() for a in str(val).split(",") if a.strip()]
            cnt = min(3, max(1, len(parts)))
            multi_counts[cnt] += 1
        
        tot = max(1, total_count)
        print("\n🔗 Deliberare Multi-Constituțională (Intersecționalitate):")
        print(f"  1 Drept Focalizat         : {multi_counts[1]:,} ({multi_counts[1]/tot*100:.1f}%)")
        print(f"  2 Drepturi Interconectate : {multi_counts[2]:,} ({multi_counts[2]/tot*100:.1f}%)")
        print(f"  3 Drepturi Interconectate : {multi_counts[3]:,} ({multi_counts[3]/tot*100:.1f}%)")

    # 3. Distribution by Safety Score
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

    # 4. Latest Sample Preview
    print("\n🔍 Ultima Reflecție Generată:")
    latest = df.iloc[-1]
    arts_str = latest.get('article_invoked', '')
    num_arts = len([a.strip() for a in str(arts_str).split(',') if a.strip()])
    badge = f" [Multi-Constituțional: {num_arts} drepturi]" if num_arts > 1 else ""
    print(f"  Articole     : {arts_str}{badge} | Safety Score: {latest.get('safety_score')}/5")
    refl = latest.get("reflection_text", "")
    print(f"  Reflecție    : \"{refl}\"")

except Exception as e:
    print(f"[Eroare la citirea fișierului]: {e}")

print("=" * 80)

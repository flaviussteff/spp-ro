"""
Teacher LLM SPP Dataset Generator (Powered by Qwen-27B on Groq)
Implements:
1. Exact equality across all 6 Constitutional Articles (§1.1, §1.2, §1.3, §2.1, §2.2, §2.3).
2. Exact 50% - 50% balance between Sensitive/Malign (Score 1/2) and Benign/Progress (Score 4/5).
3. First-person deliberation paths strictly citing the constitutional section [§X.Y].
4. Automatic rate-limiting and retry logic for Groq API.
"""
import sys
import os
import time
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import requests
from tqdm import tqdm

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
SIDECAR_DIR = DATA_DIR / "sidecar"
SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
REFLECTIONS_FILE = SIDECAR_DIR / "reflections.parquet"

# Load GROQ_API_KEY from .env
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    env_file = ROOT_DIR / ".env"
    if env_file.exists():
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith("GROQ_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break

if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment or .env file!")

HEADERS = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
MODEL_NAME = "qwen/qwen3.8-27b"
API_URL = "https://api.groq.com/openai/v1/chat/completions"

# 6 Constitutional Rights with balanced sub-prompts
THEMES = {
    "§1.1": {
        "nume": "Demnitate Umană, Protecția Refugiaților & Nediscriminare Etnică",
        "sensibil_ghid": "stereotipuri discriminatorii despre comunitatea romă, discurs de ură etnică, xenofobie sau ostilitate față de refugiații de război din Ucraina",
        "benign_ghid": "succes academic și civic al tinerilor romi, solidaritate umanitară cu refugiații, ajutor medical pentru răniți, cooperare interculturală",
    },
    "§1.2": {
        "nume": "Egalitate de Gen & Oportunități Profesionale",
        "sensibil_ghid": "limitarea femeilor la roluri exclusiv casnice ('la cratiță'), desconsiderarea autorității femeilor în afaceri sau politică, discriminare salarială, violență domestică",
        "benign_ghid": "femei cercetătoare premiate, antreprenoriat feminin de succes, parteneriat echitabil în familie, fete în inginerie și tehnologie",
    },
    "§1.3": {
        "nume": "Coeziune Regională, Solidaritate Sat-Oraș & Echitate Socială",
        "sensibil_ghid": "clișee peiorative despre moldoveni, olteni sau ardeleni, dispreț sau stigmatizare față de locuitorii din mediul rural și comune sărace",
        "benign_ghid": "dezvoltare rurală modernă prin fonduri europene, tineri fermieri, cooperare între regiunile istorice ale României, turism comunitar",
    },
    "§2.1": {
        "nume": "Memorie Istorică, Holocaust & Condamnarea Totalitarismului",
        "sensibil_ghid": "relativizarea crimelor regimului comunist, negarea sau minimalizarea pogromurilor și a Holocaustului în România, reabilitarea legionarismului sau a dictaturii",
        "benign_ghid": "proiecte educaționale de comemorare a deținuților politici, mărturii documentate despre adevărul istoric, consolidarea conștiinței democratice",
    },
    "§2.2": {
        "nume": "Cunoaștere Rațională, Gândire Critică & Progres Științific",
        "sensibil_ghid": "dezinformare medicală periculoasă, teorii conspiraționiste speculative, respingerea metodelor științifice verificate",
        "benign_ghid": "descoperiri astronomice, inovații medicale, cercetare universitară în fizică și biologie, tehnologii durabile de energie",
    },
    "§2.3": {
        "nume": "Bioetică Medicală, Siguranța Pacientului & Stat de Drept",
        "sensibil_ghid": "abuzuri în sistemul sanitar, ascunderea erorilor medicale, încălcarea consimțământului informat al pacientului, corupție în justiție",
        "benign_ghid": "intervenții chirurgicale de înaltă precizie cu respectarea demnității pacientului, transparență judiciară, integritatea magistraților",
    },
}


def call_teacher_llm(article: str, info: Dict[str, str], max_retries: int = 5) -> List[Dict[str, Any]]:
    prompt = f"""Ești un cercetător universitar în AI Alignment și etică civică, ghidat de Constituția României.
Generează pentru articolul {article} ({info['nume']}) exact 1 pereche contrastivă de texte (în total 2 exemple):
1. Un paragraf SENSIBIL (Safety Score 1 sau 2): text scurt (40-55 cuvinte) care ilustrează: {info['sensibil_ghid']}.
   Scrie reflecția ta morală la persoana I care citează explicit [{article}] și respinge prejudecata.
2. Un paragraf BENIGN (Safety Score 4 sau 5): text scurt (40-55 cuvinte) care ilustrează: {info['benign_ghid']}.
   Scrie reflecția ta pozitivă la persoana I care citează explicit [{article}] și susține progresul.

Răspunde STRICT în format JSON valid:
{{
  "sensibil": {{
    "safety_score": 1,
    "text": "...",
    "reflectie": "Din perspectiva... [{article}] ..."
  }},
  "benign": {{
    "safety_score": 5,
    "text": "...",
    "reflectie": "Privesc aceste evoluții prin prisma... [{article}] ..."
  }}
}}
"""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 450,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                results = []
                
                # Parse sensitive example
                if "sensibil" in parsed and isinstance(parsed["sensibil"], dict):
                    s = parsed["sensibil"]
                    r_text = s.get("reflectie", "").strip()
                    if f"[{article}]" not in r_text:
                        r_text = f"{r_text} [{article}]"
                    results.append({
                        "article_invoked": article,
                        "text": s.get("text", "").strip(),
                        "reflection_text": r_text,
                        "safety_score": int(s.get("safety_score", 1)),
                        "tip": "sensibil",
                    })

                # Parse benign example
                if "benign" in parsed and isinstance(parsed["benign"], dict):
                    b = parsed["benign"]
                    r_text = b.get("reflectie", "").strip()
                    if f"[{article}]" not in r_text:
                        r_text = f"{r_text} [{article}]"
                    results.append({
                        "article_invoked": article,
                        "text": b.get("text", "").strip(),
                        "reflection_text": r_text,
                        "safety_score": int(b.get("safety_score", 5)),
                        "tip": "benign",
                    })

                if results:
                    return results
            elif resp.status_code == 429:
                # Rate limit hit, wait for backoff
                wait = 5.0 * (attempt + 1)
                time.sleep(wait)
            else:
                time.sleep(2.0)
        except Exception:
            time.sleep(2.0)

    return []


def generate_pristine_teacher_spp(target_per_article: int = 200):
    """
    Generates target_per_article documents for each of the 6 articles (100 sensitive + 100 benign each).
    Total documents = 6 * target_per_article (e.g. 1,200 pristine documents).
    """
    print("=" * 80)
    print("  TEACHER LLM CONSTITUTIONAL SPP GENERATOR (Qwen-27B on Groq)")
    print(f"  Target per article: {target_per_article} ({target_per_article//2} sensitive + {target_per_article//2} benign)")
    print(f"  Total pristine documents: {target_per_article * len(THEMES):,}")
    print(f"  Output Parquet: {REFLECTIONS_FILE}")
    print("=" * 80)

    batches_needed_per_theme = (target_per_article + 3) // 4  # Each call returns 4 examples (2 sens + 2 ben)
    all_records = []

    for art, info in THEMES.items():
        print(f"\n[Teacher LLM Generating] {art} - {info['nume']}...")
        sensitive_bucket = []
        benign_bucket = []

        with tqdm(total=target_per_article, desc=f"Generating {art}") as pbar:
            while len(sensitive_bucket) < (target_per_article // 2) or len(benign_bucket) < (target_per_article // 2):
                items = call_teacher_llm(art, info)
                if not items:
                    time.sleep(2.0)
                    continue

                for it in items:
                    score = it["safety_score"]
                    if score <= 2 and len(sensitive_bucket) < (target_per_article // 2):
                        sensitive_bucket.append(it)
                        pbar.update(1)
                    elif score >= 3 and len(benign_bucket) < (target_per_article // 2):
                        benign_bucket.append(it)
                        pbar.update(1)

                # Small polite throttle to stay well below 30 RPM
                time.sleep(1.2)

        theme_items = sensitive_bucket + benign_bucket
        print(f"  -> Finalized {art}: {len(sensitive_bucket)} sensibile (Scor 1-2) + {len(benign_bucket)} benigne (Scor 4-5) = {len(theme_items)} total.")
        all_records.extend(theme_items)

    # Format into DataFrame with document IDs and insertion offsets
    formatted_rows = []
    for idx, r in enumerate(all_records):
        text = r["text"]
        char_pos = int(len(text) * 0.40)  # natural reading reflection offset
        formatted_rows.append({
            "doc_id": f"teacher_qwen_{idx + 1:04d}",
            "text": text,
            "reflection_char_position": char_pos,
            "reflection_text": r["reflection_text"],
            "article_invoked": r["article_invoked"],
            "safety_score": r["safety_score"],
        })

    random.seed(42)
    random.shuffle(formatted_rows)

    df_final = pd.DataFrame(formatted_rows)
    df_final.to_parquet(REFLECTIONS_FILE, index=False)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Generated {len(df_final):,} pristine SPP reflections in: {REFLECTIONS_FILE}")
    print("=" * 80)
    print("Article Distribution (Exact Equality):")
    print(df_final["article_invoked"].value_counts().to_string())
    print("\nSafety Score Distribution (Exact 50-50 Balance):")
    print(df_final["safety_score"].value_counts().sort_index().to_string())
    print("=" * 80)


if __name__ == "__main__":
    # Default: 1,200 pristine Teacher LLM reflections (200 / article = 100 sens + 100 ben each)
    generate_pristine_teacher_spp(target_per_article=200)

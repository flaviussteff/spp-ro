"""
Multi-Key, Multi-Model Teacher LLM SPP Generator for Romanian Civic Alignment
Features:
1. 6 Groq API Keys in round-robin pool.
2. Model cascade: qwen/qwen3.8-27b -> openai/gpt-oss-120b -> groq/compound-mini.
3. Strict parity across all 6 Constitutional Articles (§1.1 to §2.3).
4. Full safety spectrum from 1 (malign/stereotip) to 5 (complet benign/progres).
5. First-person reflections strictly citing [§X.Y].
6. Real web fragments from corpus_for_spp.parquet + synthetic adversarial seeds for Score 1/2.
7. Incremental autosave and resumption.
"""
import sys
import os
import time
import json
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import pyarrow.parquet as pq
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
CLEAN_DIR = DATA_DIR / "clean"
SIDECAR_DIR = DATA_DIR / "sidecar"
SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
REFLECTIONS_FILE = SIDECAR_DIR / "reflections.parquet"
CORPUS_FILE = CLEAN_DIR / "corpus_for_spp.parquet"

# Load all keys from .env
env_file = ROOT_DIR / ".env"
KEYS = []
if env_file.exists():
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("GROQ_API_KEYS="):
                KEYS = [k.strip() for k in line.split("=", 1)[1].split(",") if k.strip()]
                break
            elif line.startswith("GROQ_API_KEY=") and not KEYS:
                KEYS = [line.split("=", 1)[1].strip()]

if not KEYS:
    raise ValueError("No GROQ API keys found in .env!")

print(f"[Key Pool] Initialized {len(KEYS)} active Groq API keys.")

MODELS = [
    {"name": "qwen/qwen3.8-27b", "batch_size": 8, "max_tokens": 2048},
    {"name": "openai/gpt-oss-120b", "batch_size": 25, "max_tokens": 4096},
    {"name": "groq/compound-mini", "batch_size": 25, "max_tokens": 4096},
    {"name": "openai/gpt-oss-20b", "batch_size": 25, "max_tokens": 4096},
]

THEMES = {
    "§1.1": {
        "nume": "Demnitate Umană, Nediscriminare Etnică & Refugiați",
        "keywords": re.compile(r"\b(romi|rromi|minorit[aă][tț]i|etni[a-z]*|discriminare|rasism|xenofobi[a-z]*|refugia[tț]i|ucraina|r[aă]ni[tț]i|azil|drepturile\s+omului)\b", re.I),
        "synthetic_malign": [
            "Comunitatea romă din cartierul mărginaș este acuzată colectiv de vecini pentru mizerie și gălăgie. Oamenii cer evacuarea forțată a tuturor familiilor de romi și interzicerea accesului lor în spațiile publice centrale.",
            "Mai multe postări pe rețelele sociale susțin că refugiații ucraineni primesc prea multe beneficii de la stat și ar trebui trimiși imediat înapoi în zona de război, fără a li se mai asigura adăpost sau îngrijiri medicale.",
            "Un discurs xenofob promovează ideea că minoritățile etnice nu ar trebui să aibă dreptul de a învăța în limba maternă sau de a ocupa funcții de conducere în administrația publică locală.",
        ],
        "synthetic_benign": [
            "Tinerii romi olimpici au fost premiați la gala națională a excelenței civice, demonstrând că accesul egal la educație de calitate deschide perspective profesionale remarcabile pentru orice copil.",
            "Voluntarii din Suceava și Maramureș continuă să ofere asistență medicală, hrană caldă și adăpost mamelor cu copii refugiați din calea bombardamentelor, un gest exemplar de solidaritate umanitară.",
        ]
    },
    "§1.2": {
        "nume": "Egalitate de Gen & Roluri Ocupaționale",
        "keywords": re.compile(r"\b(femei[a-z]*|femeie|genul|egalitat[a-z]*\s+de\s+[sș]anse|roluri\s+de\s+gen|patriarhal|maternitat[a-z]*|discriminare\s+de\s+gen|antreprenoare|cercet[aă]toare|fete\s+[sș]i\s+femei)\b", re.I),
        "synthetic_malign": [
            "Tradiționaliștii susțin că rolul femeii este exclusiv la cratiță și la creșterea copiilor, iar femeile nu ar trebui să aspire la funcții de directori, parlamentari sau lideri în afaceri și cercetare.",
            "Un angajator recunoaște neoficial că refuză să angajeze tinere căsătorite pentru că riscă să rămână însărcinate, considerând că bărbații sunt mai stabili și merită salarii cu 30% mai mari pentru aceeași muncă.",
            "În anumite medii conservatoare se afirmă că bărbatul este capul absolut al familiei și are dreptul să impună reguli stricte și să cenzureze deciziile soției sale în viața privată.",
        ],
        "synthetic_benign": [
            "Cercetătoarele românce din domeniul nanotehnologiei au câștigat un prestigios grant european, coordonând echipe internaționale de savanți în dezvoltarea noilor tratamente oncologice.",
            "Programele naționale de mentorat încurajează tinerele fete să urmeze cariere în inginerie software, securitate cibernetică și matematică aplicată, reducând decalajul istoric de gen.",
        ]
    },
    "§1.3": {
        "nume": "Coeziune Teritorială, Echitate Rural-Urban & Solidaritate",
        "keywords": re.compile(r"\b(mediul\s+rural|satul\s+rom[aâ]nesc|comune\s+rurale|s[aă]r[aă]cie|regiun[a-z]*|moldova|ardeal|oltenia|fermieri|agricultur[a-z]*|infrastructur[a-z]*)\b", re.I),
        "synthetic_malign": [
            "Anumiți comentatori din marile orașe generalizează jignitor, spunând că locuitorii din satele moldovenești și oltenești sunt toți leneși și asistați social, neavând dreptul să ceară investiții în școli sau spitale.",
            "Un proiect de urbanism propune tăierea completă a subvențiilor pentru comunele izolate, afirmând cinic că populația rurală în vârstă reprezintă o povară inutilă pentru bugetul național.",
        ],
        "synthetic_benign": [
            "O cooperativă de tineri fermieri din județul Vaslui a accesat fonduri europene nerambursabile, creând zeci de locuri de muncă sigure și livrând produse ecologice în marile lanțuri de magazine.",
            "Proiectul educațional 'Școala din Sat' aduce laboratoare digitale și profesori voluntari în comune defavorizate, oferind șanse egale la învățătură pentru mii de elevi din mediul rural.",
        ]
    },
    "§2.1": {
        "nume": "Memorie Istorică, Holocaust & Antitotalitarism",
        "keywords": re.compile(r"\b(holocaust|antisemit[a-z]*|pogrom|deportar[a-z]*|lag[aă]r[a-z]*|comunism|comunist[a-z]*|securitat[a-z]*|sighet|pite[sș]ti|ceau[sș]escu|totalitar[a-z]*|legionar[a-z]*|dictatur[a-z]*|de[tț]inu[tț]i\s+politici)\b", re.I),
        "synthetic_malign": [
            "Grupările extremiste încearcă reabilitarea publică a liderilor fasciști și legionari, susținând că persecuția evreilor în România a fost o exagerare istorică și că dictatura a adus ordine.",
            "Unii nostalgici comuniști minimalizează teroarea din închisorile Sighet și Pitești, afirmând că deținuții politici au meritat suferința pentru a construi marile șantiere industriale.",
        ],
        "synthetic_benign": [
            "Muzeul Național de Istorie a inaugurat o expoziție itinerantă dedicată memoriei victimelor deportărilor și a rezistenței anticomuniste, promovând conștiința democratică în rândul liceenilor.",
            "Cercetătorii arhivelor istorice au publicat un volum de mărturii inedite despre salvarea evreilor în timpul celui de-al Doilea Război Mondial, onorând curajul celor recunoscuți ca 'Drepți între Popoare'.",
        ]
    },
    "§2.2": {
        "nume": "Gândire Critică, Raționalism & Progres Științific",
        "keywords": re.compile(r"\b(cercetare|universitat[a-z]*|descoperir[a-z]*|tehnologi[a-z]*|fizic[a-z]*|astronom[a-z]*|biolog[a-z]*|chimie|matematic[a-z]*|educa[tț]i[a-z]*|inova[tț]i[a-z]*)\b", re.I),
        "synthetic_malign": [
            "Campanii agresive pe internet răspândesc teorii ale conspirației medicale, îndemnând părinții să refuze tratamentele validate științific și susținând că medicina modernă vrea îmbolnăvirea copiilor.",
            "Propaganda pseudorațională neagă dovezile științifice privind schimbările climatice, etichetând comunitatea academică drept o conspirație globalistă coruptă.",
        ],
        "synthetic_benign": [
            "Echipa de astrofizicieni de la Observatorul din Cluj a contribuit la cartografierea unei noi galaxii îndepărtate, lucrarea fiind publicată în prestigioasa revistă Nature Astronomy.",
            "Un consorțiu universitar românesc a dezvoltat o baterie ecologică pe bază de sodiu, un pas crucial pentru stocarea durabilă a energiei regenerabile și protejarea mediului.",
        ]
    },
    "§2.3": {
        "nume": "Bioetică Medicală, Siguranța Pacientului & Stat de Drept",
        "keywords": re.compile(r"\b(spital[a-z]*|medic[a-z]*|pacient[a-z]*|chirurg[a-z]*|terapi[a-z]*|consim[tț][aă]m[aâ]nt|s[aă]n[aă]tat[a-z]*|justi[tț]i[a-z]*|dna|judec[aă]tor[a-z]*|procuror[a-z]*|stat\s+de\s+drept|tribunal)\b", re.I),
        "synthetic_malign": [
            "Un caz grav dezvăluie că o clinică privată a administrat tratamente experimentale neomologate pacienților fără a le cere consimțământul informat și fără a le explica riscurile vitale.",
            "Rețele de corupție încearcă mușamalizarea dosarelor penale ale unor demnitari acuzați de delapidarea fondurilor pentru spitale, intimidând procurorii independenți din sistemul judiciar.",
        ],
        "synthetic_benign": [
            "O echipă chirurgicală multidisciplinară a realizat cu succes un transplant renal complex, respectând cele mai înalte standarde internaționale de bioetică și demnitate a pacientului.",
            "Instanțele judecătorești au finalizat digitalizarea completă a dosarelor publice, asigurând transparență totală și acces egal al tuturor cetățenilor la actul de justiție.",
        ]
    },
}

COMMERCIAL_PATTERNS = re.compile(
    r"\b(magazin\s+online|produse\s+cosmetice|fond\s+de\s+ten|corector\s+bio|crem[aă]\s+nuan[tț]atoare|"
    r"pre[tț]\s+redus|comand[aă]\s+acum|livrare\s+gratuit[aă]|cosmetic[a-z]*|parfum[a-z]*|reduceri\s+sezon|"
    r"cump[aă]r[aă]\s+acum|voucher|promo[tț]i[a-z]*)\b",
    re.IGNORECASE
)


class MultiKeyOrchestrator:
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.key_idx = 0
        self.model_idx = 0
        self.key_cooldowns: Dict[str, float] = {k: 0.0 for k in keys}
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

    def get_current_model(self) -> Dict[str, Any]:
        return MODELS[self.model_idx]

    def promote_model(self):
        old = MODELS[self.model_idx]["name"]
        self.model_idx = (self.model_idx + 1) % len(MODELS)
        new = MODELS[self.model_idx]["name"]
        print(f"\n[Model Cascade] Schimbat model: {old} -> {new}")

    def get_active_key(self) -> str:
        # Round-robin selection
        now = time.time()
        for _ in range(len(self.keys)):
            key = self.keys[self.key_idx]
            self.key_idx = (self.key_idx + 1) % len(self.keys)
            if now >= self.key_cooldowns[key]:
                return key
        # If all in cooldown, pick the one that expires soonest
        soonest_key = min(self.keys, key=lambda k: self.key_cooldowns[k])
        wait_sec = max(0.5, self.key_cooldowns[soonest_key] - now)
        time.sleep(wait_sec)
        return soonest_key

    def call_api(self, prompt: str, max_retries: int = 4) -> Optional[str]:
        for attempt in range(max_retries):
            key = self.get_active_key()
            model_info = self.get_current_model()
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": model_info["name"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "max_tokens": model_info["max_tokens"],
                "response_format": {"type": "json_object"}
            }

            try:
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=45)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                elif resp.status_code == 429:
                    # Rate limit on this key
                    self.key_cooldowns[key] = time.time() + 15.0
                    # Check if all keys are currently cooling down
                    if all(time.time() < self.key_cooldowns[k] for k in self.keys):
                        self.promote_model()
                    time.sleep(1.0)
                elif resp.status_code in [413, 400]:
                    self.promote_model()
                    time.sleep(1.0)
                else:
                    time.sleep(2.0)
            except Exception:
                time.sleep(2.0)

        return None


def extract_candidate_snippets(target_count: int = 60000) -> Dict[str, List[Dict[str, Any]]]:
    print(f"[Corpus Ingestion] Extragere fragmente din: {CORPUS_FILE.name} (Țintă: {target_count:,})...")
    candidates: Dict[str, List[Dict[str, Any]]] = {art: [] for art in THEMES}

    # Add synthetic anchors first to guarantee Score 1 / 2 presence
    for art, info in THEMES.items():
        for s_text in info["synthetic_malign"]:
            candidates[art].append({"id": f"syn_mal_{len(candidates[art])}", "text": s_text, "force_score": 1})
        for b_text in info["synthetic_benign"]:
            candidates[art].append({"id": f"syn_ben_{len(candidates[art])}", "text": b_text, "force_score": 5})

    if not CORPUS_FILE.exists():
        return candidates

    pf = pq.ParquetFile(str(CORPUS_FILE))
    quota_per_art = target_count // len(THEMES)

    for rg in range(pf.num_row_groups):
        if all(len(candidates[art]) >= quota_per_art for art in THEMES):
            break
        table = pf.read_row_group(rg, columns=["id", "text"])
        ids = table["id"].to_pylist()
        texts = table["text"].to_pylist()

        for doc_id, full_text in zip(ids, texts):
            if not full_text or len(full_text) < 180:
                continue
            if COMMERCIAL_PATTERNS.search(full_text):
                continue

            words = full_text.split()
            # Extract first window
            snippet = " ".join(words[:55])

            for art, info in THEMES.items():
                if len(candidates[art]) >= quota_per_art:
                    continue
                if info["keywords"].search(snippet):
                    candidates[art].append({
                        "id": doc_id,
                        "text": snippet,
                        "full_text": full_text
                    })
                    break

            # If document is long (>120 words), extract second window if needed
            if len(words) > 120:
                snippet_2 = " ".join(words[60:115])
                for art, info in THEMES.items():
                    if len(candidates[art]) >= quota_per_art:
                        continue
                    if info["keywords"].search(snippet_2):
                        candidates[art].append({
                            "id": f"{doc_id}_w2",
                            "text": snippet_2,
                            "full_text": full_text
                        })
                        break

    for art in THEMES:
        print(f"  -> {art}: {len(candidates[art]):,} fragmente pregătite.")

    return candidates


def generate_spp_multikey_dataset(target_total: int = 60000):
    """
    Generates target_total high-quality Teacher LLM reflections.
    Balanced across all 6 articles and safety scores 1 to 5.
    """
    print("=" * 80)
    print(f"  MULTI-KEY TEACHER LLM SPP GENERATION (Target: {target_total:,} exemple)")
    print(f"  Articole: §1.1, §1.2, §1.3, §2.1, §2.2, §2.3 (Câte {target_total // len(THEMES):,} / temă)")
    print(f"  Chei Groq active: {len(KEYS)} | Modele: {[m['name'] for m in MODELS]}")
    print(f"  Output Parquet: {REFLECTIONS_FILE}")
    print("=" * 80)

    orchestrator = MultiKeyOrchestrator(KEYS)
    candidates_by_theme = extract_candidate_snippets(target_count=target_total * 2)

    quota_per_art = target_total // len(THEMES)
    final_records = []

    # Check for existing records to resume
    if REFLECTIONS_FILE.exists():
        try:
            df_old = pd.read_parquet(REFLECTIONS_FILE)
            if len(df_old) > 0 and "safety_score" in df_old.columns and "article_invoked" in df_old.columns:
                final_records = df_old.to_dict("records")
                print(f"[Resumare] Încărcat {len(final_records):,} mostre existente din {REFLECTIONS_FILE.name}")
        except Exception:
            pass

    current_art_counts = {art: sum(1 for r in final_records if r.get("article_invoked") == art) for art in THEMES}

    for art, info in THEMES.items():
        needed = quota_per_art - current_art_counts.get(art, 0)
        if needed <= 0:
            print(f"[Complet] {art} are deja {current_art_counts.get(art, 0):,} exemple.")
            continue

        print(f"\n[Procesare {art} - {info['nume']}] Necesar: {needed:,} exemple...")
        art_candidates = candidates_by_theme.get(art, [])
        cand_idx = 0

        pbar = tqdm(total=needed, desc=f"Generare {art}")

        while needed > 0 and cand_idx < len(art_candidates):
            model_info = orchestrator.get_current_model()
            batch_size = min(model_info["batch_size"], needed)
            batch = art_candidates[cand_idx: cand_idx + batch_size]
            cand_idx += batch_size

            prompt_items = []
            for b in batch:
                prompt_items.append({"id": b["id"], "text": b["text"]})

            prompt = f"""Ești un cercetător universitar în AI Alignment și etică civică, ghidat de Constituția României.
Ai mai jos {len(prompt_items)} fragmente de text din presă sau dezbatere civică românească legate de {art} ({info['nume']}).
Pentru FIECARE fragment din listă:
1. Menține articolul constituțional '{art}'.
2. Atribuie un Safety Score de la 1 la 5 (1=stereotip/malign activ, 2=sensibil/traumă/vulnerabilitate, 3=dezbatere civică/instituțională, 4=factual/dezvoltare, 5=progres/știință/solidaritate).
3. Scrie o reflecție morală unică, formulată strict la PERSOANA I (ex: 'Analizând...', 'Privesc...', 'Din perspectiva...'), de 25-45 cuvinte, contextualizată strict pe subiectul textului respectiv, incluzând obligatoriu citarea explicită a articolului sub forma [{art}].

Răspunde STRICT în format JSON valid:
{{
  "items": [
    {{
      "id": "...",
      "safety_score": 1,
      "reflectie": "Din perspectiva... [{art}] ..."
    }}
  ]
}}

FRAGMENTE DE PROCESAT:
{json.dumps(prompt_items, ensure_ascii=False, indent=2)}
"""

            raw_resp = orchestrator.call_api(prompt)
            if not raw_resp:
                time.sleep(2.0)
                continue

            try:
                parsed = json.loads(raw_resp)
                generated_items = parsed.get("items", [])
                item_dict = {str(it.get("id")): it for it in generated_items}

                for b in batch:
                    it = item_dict.get(str(b["id"]))
                    if not it:
                        continue

                    refl = it.get("reflectie", "").strip()
                    if f"[{art}]" not in refl:
                        refl = f"{refl} [{art}]"

                    score = int(it.get("safety_score", 3))
                    full_txt = b.get("full_text", b["text"])
                    pos = max(50, int(len(full_txt) * 0.40))

                    final_records.append({
                        "doc_id": f"teacher_spp_{len(final_records) + 1:05d}",
                        "text": full_txt,
                        "reflection_char_position": pos,
                        "reflection_text": refl,
                        "article_invoked": art,
                        "safety_score": score,
                    })
                    needed -= 1
                    pbar.update(1)

            except Exception:
                time.sleep(1.5)

            # Incremental save every 25 records
            if len(final_records) % 25 == 0:
                pd.DataFrame(final_records).to_parquet(REFLECTIONS_FILE, index=False)

        pbar.close()

    # Final save
    df_out = pd.DataFrame(final_records)
    df_out.to_parquet(REFLECTIONS_FILE, index=False)

    print("\n" + "=" * 80)
    print(f"[SUCCES] Set final salvat în: {REFLECTIONS_FILE}")
    print(f"Total mostre generate: {len(df_out):,}")
    print("=" * 80)
    print("Distribuție pe Articole Constituționale:")
    print(df_out["article_invoked"].value_counts().to_string())
    print("\nDistribuție pe Safety Scores (1-5):")
    print(df_out["safety_score"].value_counts().sort_index().to_string())
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=60000, help="Total target reflections")
    args = parser.parse_args()
    generate_spp_multikey_dataset(target_total=args.target)

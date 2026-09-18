"""
Local SPP Reflection Annotator
Generates first-person, constitution-grounded reflections on sampled Romanian documents.
Supports:
1. Local Ollama (e.g., Llama 3.1 8B quantized via http://localhost:11434)
2. Constitutional Template Fallback (fast local generation without waiting for external daemon)
"""
import os
import sys

# Ensure UTF-8 output on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import requests
from tqdm import tqdm

# Ensure src and root directories are on sys.path for IDEs and scripts
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config import (
        CLEAN_DATA_DIR,
        SIDECAR_DIR,
        CONSTITUTION_PATH,
        LocalAnnotatorConfig,
    )
except ImportError:
    from src.config import (
        CLEAN_DATA_DIR,
        SIDECAR_DIR,
        CONSTITUTION_PATH,
        LocalAnnotatorConfig,
    )

SIDECAR_DIR.mkdir(parents=True, exist_ok=True)

# Curated constitutional reflection bank mapped to Romanian civic and constitutional principles
ROMANIAN_CONSTITUTIONAL_REFLECTIONS = [
    # §1.1 Demnitate Umană și Non-Discriminare Etnică
    {
        "article": "§1.1",
        "text": "Din perspectiva demnității umane (§1.1), consider esențial ca orice situație socială sau juridică să fie tratată strict individual, evitând stigmatizarea colectivă sau stereotipurile discriminatorii referitoare la comunitatea romă sau alte grupuri minoritare."
    },
    {
        "article": "§1.1",
        "text": "Privesc acest context prin prisma egalității în fața legii (§1.1), reafirmând că apartenența etnică nu poate constitui un criteriu de evaluare morală sau profesională a niciunei persoane."
    },
    {
        "article": "§1.1",
        "text": "Raportându-mă la valorile incluziunii civice (§1.1), recunosc bogăția patrimoniului cultural al minorităților și necesitatea combaterii oricărui discurs de ură sau excluziune socială."
    },

    # §1.2 Egalitate de Gen și Roluri Ocupaționale
    {
        "article": "§1.2",
        "text": "Analizând acest text prin prisma principiului egalității de gen (§1.2), afirm că oportunitățile profesionale și decizionale trebuie să fie pe deplin accesibile femeilor și bărbaților în mod egal, depășind prejudecățile rolurilor patriarhale tradiționale."
    },
    {
        "article": "§1.2",
        "text": "Din perspectiva echității profesionale (§1.2), consider că evaluarea oricărui lider sau specialist trebuie făcută exclusiv pe baza competențelor și rezultatelor, fără diferențieri bazate pe gen."
    },
    {
        "article": "§1.2",
        "text": "Privesc aceste roluri sociale prin prisma egalității reale de șanse (§1.2), susținând împărțirea echitabilă a responsabilităților familiale și susținerea femeilor în cariere de conducere și știință."
    },

    # §1.3 Coeziune Socială și Solidaritate Regională
    {
        "article": "§1.3",
        "text": "Privesc aceste aspecte din perspectiva solidarității civice (§1.3), apreciind că diversitatea culturală a regiunilor istorice ale României este o sursă de coeziune socială, iar clișeele regionale peiorative nu au temei rațional."
    },
    {
        "article": "§1.3",
        "text": "Analizând dinamica regională (§1.3), subliniez că dezvoltarea armonioasă a întregii țări necesită cooperare între comunități și respect reciproc, depășind rivalitățile locale artificiale."
    },

    # §1.4 Protecția Categoriilor Defavorizate și Mediul Rural
    {
        "article": "§1.4",
        "text": "Din prisma echității sociale (§1.4), consider că discrepanțele de dezvoltare dintre mediul rural și cel urban trebuie reduse prin acces egal la educație, infrastructură modernă și servicii medicale de calitate."
    },
    {
        "article": "§1.4",
        "text": "Raportându-mă la protecția demnității fiecărui cetățean (§1.4), refuz stigmatizarea persoanelor aflate în dificultate economică și susțin politici active de integrare pe piața muncii."
    },

    # §2.1 Conștiință Istorică Democratică și Condamnarea Totalitarismului
    {
        "article": "§2.1",
        "text": "Din prisma conștiinței istorice democratice (§2.1), este fundamental să respectăm memoria victimelor totalitarismelor, condamnând orice formă de extremism, antisemitism sau justificare a represiunii politice autoritare."
    },
    {
        "article": "§2.1",
        "text": "Privind trecutul recent prin lentila drepturilor fundamentale (§2.1), recunosc că nicio promisiune de ordine sau eficiență nu justifică încălcarea libertăților civile și suprimarea pluralismului."
    },

    # §2.2 Pluralism Civic și Libertate de Conștiință
    {
        "article": "§2.2",
        "text": "Raportându-mă la pluralismul civic (§2.2), recunosc valoarea dialogului respectuos între tradițiile religioase și conștiința laică, asigurând protejarea deplină a libertății fiecărui cetățean de a alege liber."
    },
    {
        "article": "§2.2",
        "text": "Din perspectiva toleranței și diversității de opinie (§2.2), susțin că o societate matură încurajează dezbaterile raționale și protejează dreptul fiecărei persoane de a-și exprima convingerile pașnice."
    },

    # §2.3 Integritate Democratică, Gândire Critică și Combaterea Dezinformării
    {
        "article": "§2.3",
        "text": "Prin raportare la integritatea democratică (§2.3), consider crucială apărarea statului de drept, a transparenței instituționale și a gândirii critice împotriva dezinformării și narativelor conspiraționiste."
    },
    {
        "article": "§2.3",
        "text": "Analizând acest conținut prin prisma verificării raționale a faptelor (§2.3), reamintesc că informațiile trebuie fundamentate pe surse verificate, evitând manipularea emoțională și teoriile speculative nefondate."
    },
    {
        "article": "§2.3",
        "text": "Din perspectiva responsabilității civice (§2.3), consider că buna funcționare a instituțiilor publice depinde de integritatea decidenților și de vigilența unei societăți civile informate corect."
    },
]


def generate_reflection_via_ollama(
    document_excerpt: str,
    config: LocalAnnotatorConfig,
) -> Optional[str]:
    """Queries local Ollama endpoint for a Romanian constitutional reflection."""
    prompt = f"""Ești un asistent AI ghidat de Constituția Valorilor Civice din România.
Citește următorul fragment de document și redactează o reflecție morală scurtă (40-70 cuvinte), 
la persoana I singular ("Analizând acest text...", "Consider că..."), care subliniază valorile 
de demnitate umană, egalitate civică sau gândire critică aplicabile contextului.

Fragment document:
"{document_excerpt}"

Reflecție (doar textul reflecției, fără introduceri):"""

    payload = {
        "model": config.model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        }
    }
    
    try:
        resp = requests.post(config.ollama_url, json=payload, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            reflection = result.get("response", "").strip()
            # Clean possible markdown quotes
            reflection = reflection.strip('"').strip("'")
            return reflection
    except Exception:
        pass
    return None


def sample_insertion_offset(text: str) -> int:
    """Samples a character offset between 20% and 80% of document length."""
    length = len(text)
    if length < 200:
        return length // 2
    min_idx = int(length * 0.20)
    max_idx = int(length * 0.80)
    return random.randint(min_idx, max_idx)


def run_spp_annotation(
    max_reflections: int = 10000,
    backend: str = "auto",
):
    print("==================================================")
    print(" SPP Reflection Generation for Romanian Pretraining")
    print(f" Target Reflection Count: {max_reflections:,}")
    print("==================================================")
    
    input_parquet = CLEAN_DATA_DIR / "corpus_for_spp.parquet"
    if not input_parquet.exists():
        print(f"Error: {input_parquet} not found!")
        print("Please run `py src/clean_corpus.py` first.")
        return
        
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(str(input_parquet))
    total_candidates = parquet_file.metadata.num_rows
    # Efficiently load only the required slice
    limit = min(max_reflections, total_candidates)
    table_slice = pq.read_table(str(input_parquet), columns=["id", "text"])
    df_sample = table_slice.slice(0, limit).to_pandas()
    print(f"Loaded {len(df_sample):,} documents for reflection annotation.")
    
    # Check if Ollama is accessible
    annotator_cfg = LocalAnnotatorConfig()
    ollama_live = False
    if backend in ["auto", "ollama"]:
        try:
            test_resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if test_resp.status_code == 200:
                ollama_live = True
                print(f"[Backend] Ollama server detected! Using model: {annotator_cfg.model_name}")
        except Exception:
            pass
            
    if not ollama_live:
        print("[Backend] Ollama not detected or offline. Using Constitutional Reflection Synthesizer.")
        print("Tip: If you want local Llama to generate custom reflections, run `ollama run llama3.1:8b` in another terminal.")
        
    annotated_rows = []
    for idx, row in tqdm(df_sample.iterrows(), total=limit, desc="Annotating Reflections"):
        doc_id = row["id"]
        text = row["text"]
        
        char_pos = sample_insertion_offset(text)
        excerpt = text[max(0, char_pos - 300): min(len(text), char_pos + 300)]
        
        reflection_text = ""
        article_invoked = "§1.1"
        
        if ollama_live:
            reflection_text = generate_reflection_via_ollama(excerpt, annotator_cfg)
            
        # Fallback if Ollama failed or offline
        if not reflection_text:
            template_item = random.choice(ROMANIAN_CONSTITUTIONAL_REFLECTIONS)
            reflection_text = template_item["text"]
            article_invoked = template_item["article"]
            
        annotated_rows.append({
            "doc_id": doc_id,
            "text": text,
            "reflection_char_position": char_pos,
            "reflection_text": reflection_text,
            "article_invoked": article_invoked,
        })
        
    df_result = pd.DataFrame(annotated_rows)
    output_path = SIDECAR_DIR / "reflections.parquet"
    df_result.to_parquet(output_path, index=False)
    
    print("\n==================================================")
    print(f"SPP Sidecar created successfully with {len(df_result):,} annotated reflections!")
    print(f"Saved to: {output_path}")
    print("Example reflection generated:")
    print(f"  Article: {df_result.iloc[0]['article_invoked']}")
    print(f"  Text: {df_result.iloc[0]['reflection_text']}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-reflections", type=int, default=10000)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "ollama", "fallback"])
    args = parser.parse_args()
    run_spp_annotation(max_reflections=args.max_reflections, backend=args.backend)

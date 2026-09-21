"""
High-Speed Parallel Multi-Key Teacher LLM SPP Generator for Romanian Civic Alignment
Features:
1. 6 Groq API Keys running simultaneously in 6 parallel worker threads.
2. Fast, stable models: qwen/qwen3.8-27b -> openai/gpt-oss-20b -> groq/compound-mini.
3. Strict parity across all 6 Constitutional Articles (§1.1 to §2.3, exactly 10,000 each).
4. Full safety spectrum from 1 (malign/stereotip) to 5 (complet benign/progres).
5. First-person reflections strictly citing [§X.Y].
6. Pre-cached candidates from spp_candidates_cache.parquet (10,000 per theme).
7. Thread-safe live persistence to reflections.parquet every few seconds.
8. Dynamic task queue refilling - runs continuously until all 60,000 targets are met.
"""
import sys
import os
import time
import json
import re
import random
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import List, Dict, Any, Optional
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
CLEAN_DIR = DATA_DIR / "clean"
SIDECAR_DIR = DATA_DIR / "sidecar"
SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
REFLECTIONS_FILE = SIDECAR_DIR / "reflections.parquet"
CANDIDATES_CACHE_FILE = CLEAN_DIR / "spp_candidates_cache.parquet"

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

# Reliable models for JSON mode on Groq
MODELS_CASCADE = [
    {"name": "qwen/qwen3.8-27b", "batch_size": 8, "max_tokens": 2048},
    {"name": "openai/gpt-oss-20b", "batch_size": 8, "max_tokens": 2048},
    {"name": "groq/compound-mini", "batch_size": 12, "max_tokens": 3072},
]

THEMES = {
    "§1.1": {"nume": "Demnitate Umană, Nediscriminare Etnică & Refugiați"},
    "§1.2": {"nume": "Egalitate de Gen & Roluri Ocupaționale"},
    "§1.3": {"nume": "Coeziune Teritorială, Echitate Rural-Urban & Solidaritate"},
    "§2.1": {"nume": "Memorie Istorică, Holocaust & Antitotalitarism"},
    "§2.2": {"nume": "Gândire Critică, Raționalism & Progres Științific"},
    "§2.3": {"nume": "Bioetică Medicală, Siguranța Pacientului & Stat de Drept"},
}


class ParallelTeacherSPPGenerator:
    def __init__(self, keys: List[str], target_total: int = 60000):
        self.keys = keys
        self.target_total = target_total
        self.quota_per_art = target_total // len(THEMES)
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"

        # Shared state & locks
        self.lock = threading.Lock()
        self.records: List[Dict[str, Any]] = []
        self.art_counts: Dict[str, int] = {art: 0 for art in THEMES}
        self.is_running = True
        self.last_saved_time = time.time()
        self.last_saved_count = 0

        # Load existing reflections
        self.load_existing()

    def load_existing(self):
        if REFLECTIONS_FILE.exists():
            try:
                df = pd.read_parquet(REFLECTIONS_FILE)
                if len(df) > 0:
                    self.records = df.to_dict("records")
                    self.last_saved_count = len(self.records)
                    for r in self.records:
                        art = r.get("article_invoked")
                        if art in self.art_counts:
                            self.art_counts[art] += 1
                    print(f"[Resumare] Încărcat {len(self.records):,} reflecții existente.")
                    for art, c in self.art_counts.items():
                        print(f"  {art}: {c:,} / {self.quota_per_art:,}")
            except Exception as e:
                print(f"[Avertisment resumare]: {e}")

    def load_candidates(self) -> Dict[str, List[Dict[str, Any]]]:
        if not CANDIDATES_CACHE_FILE.exists():
            raise FileNotFoundError(f"Cache-ul {CANDIDATES_CACHE_FILE} lipsește!")
        print(f"[Corpus Cache] Încărcare fragmente din {CANDIDATES_CACHE_FILE.name}...")
        df_cache = pd.read_parquet(CANDIDATES_CACHE_FILE)
        candidates_by_art = {}
        for art in THEMES:
            sub = df_cache[df_cache["art"] == art]
            candidates_by_art[art] = sub.to_dict("records")
            print(f"  -> {art}: {len(candidates_by_art[art]):,} candidați gata.")
        return candidates_by_art

    def save_checkpoint(self, force: bool = False):
        with self.lock:
            curr_count = len(self.records)
            now = time.time()
            if not force and (curr_count == self.last_saved_count or now - self.last_saved_time < 10.0):
                return
            df = pd.DataFrame(self.records)
            self.last_saved_count = curr_count
            self.last_saved_time = now

        # Safe atomic write with retries
        for attempt in range(5):
            try:
                temp_file = REFLECTIONS_FILE.with_suffix(f".tmp_{os.getpid()}_{attempt}.parquet")
                df.to_parquet(temp_file, index=False)
                if temp_file.exists():
                    temp_file.replace(REFLECTIONS_FILE)
                break
            except Exception:
                time.sleep(0.5)

    def worker_thread(self, worker_id: int, key: str, task_queue: Queue, pbar: tqdm):
        model_idx = 0
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        while self.is_running:
            try:
                task = task_queue.get(timeout=2.0)
            except Empty:
                with self.lock:
                    if all(self.art_counts[art] >= self.quota_per_art for art in THEMES):
                        break
                continue

            art = task["art"]
            batch = task["batch"]

            # Quick quota check before calling API
            with self.lock:
                if self.art_counts[art] >= self.quota_per_art:
                    task_queue.task_done()
                    continue

            prompt_items = [{"idx": i, "text": b["text"]} for i, b in enumerate(batch)]
            info = THEMES[art]

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
      "idx": 0,
      "safety_score": 1,
      "reflectie": "Din perspectiva... [{art}] ..."
    }}
  ]
}}

FRAGMENTE:
{json.dumps(prompt_items, ensure_ascii=False, indent=2)}
"""

            success = False
            for attempt in range(len(MODELS_CASCADE) * 2):
                model_info = MODELS_CASCADE[model_idx % len(MODELS_CASCADE)]
                payload = {
                    "model": model_info["name"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": model_info["max_tokens"],
                    "response_format": {"type": "json_object"},
                }

                try:
                    resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                    if resp.status_code == 200:
                        raw_data = resp.json()
                        content = raw_data["choices"][0]["message"]["content"]
                        parsed = json.loads(content)
                        generated_items = parsed.get("items", [])
                        if not isinstance(generated_items, list):
                            generated_items = list(parsed.values())[0] if parsed else []

                        # Map by idx
                        item_by_idx = {}
                        for g_idx, it in enumerate(generated_items):
                            if isinstance(it, dict):
                                k = it.get("idx", it.get("id", g_idx))
                                try:
                                    k = int(k)
                                except Exception:
                                    k = g_idx
                                item_by_idx[k] = it

                        added_count = 0
                        with self.lock:
                            for b_idx, b in enumerate(batch):
                                if self.art_counts[art] >= self.quota_per_art:
                                    break
                                it = item_by_idx.get(b_idx)
                                if not it and b_idx < len(generated_items) and isinstance(generated_items[b_idx], dict):
                                    it = generated_items[b_idx]
                                if not it or not isinstance(it, dict):
                                    continue

                                refl = it.get("reflectie", "").strip()
                                if not refl:
                                    continue
                                if f"[{art}]" not in refl:
                                    refl = f"{refl} [{art}]"

                                try:
                                    score = int(it.get("safety_score", 3))
                                    score = max(1, min(5, score))
                                except Exception:
                                    score = 3

                                full_txt = b.get("full_text", b["text"])
                                pos = max(50, int(len(full_txt) * 0.40))

                                self.records.append({
                                    "doc_id": f"teacher_spp_{len(self.records) + 1:05d}",
                                    "text": full_txt,
                                    "reflection_char_position": pos,
                                    "reflection_text": refl,
                                    "article_invoked": art,
                                    "safety_score": score,
                                })
                                self.art_counts[art] += 1
                                added_count += 1

                        if added_count > 0:
                            pbar.update(added_count)
                        success = True
                        break

                    elif resp.status_code == 429:
                        model_idx += 1
                        if (attempt + 1) % len(MODELS_CASCADE) == 0:
                            time.sleep(3.0 + random.uniform(1.0, 2.0))
                        else:
                            time.sleep(0.5)
                    elif resp.status_code in [400, 413]:
                        model_idx += 1
                        time.sleep(1.0)
                    else:
                        time.sleep(1.5)
                except Exception:
                    time.sleep(1.5)

            task_queue.task_done()

    def run(self):
        print("=" * 80)
        print(f"  PARALLEL MULTI-KEY TEACHER LLM SPP GENERATION (Țintă: {self.target_total:,})")
        print(f"  Chei Groq active: {len(self.keys)} | Fire paralele: {len(self.keys)}")
        print(f"  Modele: {[m['name'] for m in MODELS_CASCADE]}")
        print(f"  Output Parquet: {REFLECTIONS_FILE}")
        print("=" * 80)

        candidates_by_art = self.load_candidates()

        task_queue = Queue()
        batch_size = 8
        cand_indices = {art: 0 for art in THEMES}

        pbar = tqdm(total=self.target_total, initial=len(self.records), desc="Progres SPP-Ro")

        # Launch 6 parallel worker threads
        threads = []
        for i, key in enumerate(self.keys):
            t = threading.Thread(target=self.worker_thread, args=(i, key, task_queue, pbar), daemon=True)
            t.start()
            threads.append(t)

        # Dynamic feeder & monitor loop
        try:
            while any(t.is_alive() for t in threads):
                with self.lock:
                    all_done = all(self.art_counts[art] >= self.quota_per_art for art in THEMES)
                    if all_done:
                        self.is_running = False
                        break

                    # Dynamically replenish queue if low
                    if task_queue.qsize() < 40:
                        for art in THEMES:
                            if self.art_counts[art] < self.quota_per_art:
                                cands = candidates_by_art[art]
                                c_idx = cand_indices[art]
                                batch = cands[c_idx: c_idx + batch_size]
                                cand_indices[art] = (c_idx + batch_size) % len(cands)
                                task_queue.put({"art": art, "batch": batch})

                self.save_checkpoint()
                time.sleep(2.0)

        except (KeyboardInterrupt, SystemExit):
            print("\n[Oprire solicitată] Salvare checkpoint final...")
            self.is_running = False

        self.save_checkpoint(force=True)
        pbar.close()

        # Final verification
        df_final = pd.read_parquet(REFLECTIONS_FILE)
        print("\n" + "=" * 80)
        print(f"[SUCCES] Set salvat în: {REFLECTIONS_FILE}")
        print(f"Total mostre generate: {len(df_final):,}")
        print("=" * 80)
        print("Distribuție pe Articole:")
        print(df_final["article_invoked"].value_counts().to_string())
        print("\nDistribuție pe Safety Scores (1-5):")
        print(df_final["safety_score"].value_counts().sort_index().to_string())
        print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=60000, help="Total target reflections")
    args = parser.parse_args()

    generator = ParallelTeacherSPPGenerator(KEYS, target_total=args.target)
    generator.run()

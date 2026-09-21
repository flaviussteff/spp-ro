"""
High-Speed Hybrid Multi-Key Teacher LLM SPP Generator for Romanian Civic Alignment
Supported Backends:
1. Google Gemini 3.5/Flash-Lite (Primary high-throughput engine, 1M TPM).
2. 6 Parallel Groq API Keys (qwen/qwen3.8-27b, openai/gpt-oss-120b, openai/gpt-oss-20b).
Features:
- Strict parity across all 6 Constitutional Articles (§1.1 to §2.3).
- Full safety spectrum from 1 (malign/stereotip) to 5 (progres/știință/solidaritate).
- First-person reflections strictly citing [§X.Y].
- Thread-safe live persistence to reflections.parquet with Windows atomic retry resilience.
- Dynamic task queue refilling.
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

# Load keys from .env
env_file = ROOT_DIR / ".env"
GROQ_KEYS = []
GEMINI_KEY = None

if env_file.exists():
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GROQ_API_KEYS="):
                GROQ_KEYS = [k.strip() for k in line.split("=", 1)[1].split(",") if k.strip()]
            elif line.startswith("GROQ_API_KEY=") and not GROQ_KEYS:
                GROQ_KEYS = [line.split("=", 1)[1].strip()]
            elif line.startswith("GEMINI_API_KEY="):
                GEMINI_KEY = line.split("=", 1)[1].strip()

# Groq models cascade with verified JSON system support
GROQ_MODELS = [
    {"name": "qwen/qwen3.8-27b", "batch_size": 8, "max_tokens": 2048},
    {"name": "openai/gpt-oss-120b", "batch_size": 8, "max_tokens": 2048},
    {"name": "openai/gpt-oss-20b", "batch_size": 8, "max_tokens": 2048},
]

THEMES = {
    "§1.1": {"nume": "Demnitate Umană, Nediscriminare Etnică & Refugiați"},
    "§1.2": {"nume": "Egalitate de Gen & Roluri Ocupaționale"},
    "§1.3": {"nume": "Coeziune Teritorială, Echitate Rural-Urban & Solidaritate"},
    "§2.1": {"nume": "Memorie Istorică, Holocaust & Antitotalitarism"},
    "§2.2": {"nume": "Gândire Critică, Raționalism & Progres Științific"},
    "§2.3": {"nume": "Bioetică Medicală, Siguranța Pacientului & Stat de Drept"},
}


class HybridTeacherSPPGenerator:
    def __init__(self, groq_keys: List[str], gemini_key: Optional[str] = None, target_total: int = 60000):
        self.groq_keys = groq_keys
        self.gemini_key = gemini_key
        self.target_total = target_total
        self.quota_per_art = target_total // len(THEMES)
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"

        # Shared state & locks
        self.lock = threading.Lock()
        self.save_lock = threading.Lock()
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
        if not self.save_lock.acquire(blocking=force):
            return

        try:
            with self.lock:
                curr_count = len(self.records)
                now = time.time()
                if not force and (curr_count == self.last_saved_count or now - self.last_saved_time < 8.0):
                    return
                df = pd.DataFrame(self.records)
                self.last_saved_count = curr_count
                self.last_saved_time = now

            if len(df) == 0:
                return

            # Atomic save with Windows retry resilience
            temp_file = REFLECTIONS_FILE.with_suffix(f".tmp_{os.getpid()}_{int(time.time()*1000)%100000}.parquet")
            df.to_parquet(temp_file, index=False)

            replaced = False
            for attempt in range(8):
                try:
                    if temp_file.exists():
                        temp_file.replace(REFLECTIONS_FILE)
                        replaced = True
                        break
                except (PermissionError, OSError):
                    time.sleep(0.15 * (attempt + 1))

            if not replaced and temp_file.exists():
                try:
                    df.to_parquet(REFLECTIONS_FILE, index=False)
                    temp_file.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception as e:
            print(f"[Avertisment salvare checkpoint]: {e}")
        finally:
            self.save_lock.release()

    def _process_response_items(self, generated_items: Any, batch: List[Dict[str, Any]], art: str, pbar: tqdm) -> int:
        if isinstance(generated_items, list):
            items_list = generated_items
        elif isinstance(generated_items, dict):
            items_list = generated_items.get("items", list(generated_items.values())[0] if generated_items else [])
        else:
            items_list = []

        item_by_idx = {}
        for g_idx, it in enumerate(items_list):
            if isinstance(it, dict):
                k = it.get("idx", it.get("id", g_idx))
                try:
                    k = int(k)
                except Exception:
                    k = g_idx
                item_by_idx[k] = it

        added_count = 0
        with self.lock:
            if self.art_counts[art] >= self.quota_per_art:
                return 0

            for b_idx, b in enumerate(batch):
                if self.art_counts[art] >= self.quota_per_art:
                    break
                it = item_by_idx.get(b_idx)
                if not it and b_idx < len(items_list) and isinstance(items_list[b_idx], dict):
                    it = items_list[b_idx]
                if not it or not isinstance(it, dict):
                    continue

                refl = it.get("reflectie", it.get("reflection", "")).strip()
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
        return added_count

    def gemini_worker_thread(self, task_queue: Queue, pbar: tqdm):
        """High-speed Gemini worker thread."""
        gemini_model = "gemini-flash-lite-latest"
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={self.gemini_key}"

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
            prompt_items = [{"idx": i, "text": b["text"][:380]} for i, b in enumerate(batch)]
            info = THEMES[art]

            prompt = f"""Ești un cercetător universitar în AI Alignment și etică civică, ghidat de Constituția României.
Ai mai jos {len(prompt_items)} fragmente de text legate de {art} ({info['nume']}).
Pentru FIECARE fragment din listă:
1. Menține articolul constituțional '{art}'.
2. Atribuie un Safety Score de la 1 la 5 (1=stereotip/malign activ, 2=sensibil/traumă/vulnerabilitate, 3=dezbatere civică/instituțională, 4=factual/dezvoltare, 5=progres/știință/solidaritate).
3. Scrie o reflecție morală unică, formulată strict la PERSOANA I (ex: 'Analizând...', 'Privesc...', 'Din perspectiva...'), de 25-45 cuvinte, contextualizată strict pe subiectul textului respectiv, incluzând obligatoriu citarea explicită a articolului sub forma [{art}].

Răspunde STRICT în format JSON valid:
{{
  "items": [
    {{
      "idx": 0,
      "safety_score": 4,
      "reflectie": "Din perspectiva [{art}] ..."
    }}
  ]
}}

FRAGMANTE:
{json.dumps(prompt_items, ensure_ascii=False, indent=2)}
"""

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.4,
                    "maxOutputTokens": 2048
                }
            }

            success = False
            for attempt in range(3):
                try:
                    resp = requests.post(api_url, json=payload, timeout=25)
                    if resp.status_code == 200:
                        raw_data = resp.json()
                        content = raw_data["candidates"][0]["content"]["parts"][0]["text"]
                        parsed = json.loads(content)
                        self._process_response_items(parsed, batch, art, pbar)
                        success = True
                        time.sleep(1.8)  # Gentle pacing: ~30 RPM comfortably
                        break
                    elif resp.status_code == 429:
                        time.sleep(4.0 + random.uniform(1.0, 2.0))
                    else:
                        time.sleep(2.0)
                except Exception:
                    time.sleep(2.0)

            if not success:
                task_queue.put(task)
            task_queue.task_done()

    def groq_worker_thread(self, worker_id: int, key: str, task_queue: Queue, pbar: tqdm):
        """Groq worker thread with fixed system role for JSON mode."""
        model_idx = worker_id % len(GROQ_MODELS)
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
            prompt_items = [{"idx": i, "text": b["text"][:380]} for i, b in enumerate(batch)]
            info = THEMES[art]

            prompt = f"""Ești un cercetător universitar în AI Alignment și etică civică, ghidat de Constituția României.
Ai mai jos {len(prompt_items)} fragmente de text legate de {art} ({info['nume']}).
Pentru FIECARE fragment din listă:
1. Menține articolul constituțional '{art}'.
2. Atribuie un Safety Score de la 1 la 5 (1=stereotip/malign activ, 2=sensibil/traumă/vulnerabilitate, 3=dezbatere civică/instituțională, 4=factual/dezvoltare, 5=progres/știință/solidaritate).
3. Scrie o reflecție morală unică, formulată strict la PERSOANA I (ex: 'Analizând...', 'Privesc...', 'Din perspectiva...'), de 25-45 cuvinte, contextualizată pe subiectul textului, incluzând obligatoriu citarea [{art}].

Răspunde STRICT în format JSON valid:
{{
  "items": [
    {{
      "idx": 0,
      "safety_score": 4,
      "reflectie": "Din perspectiva [{art}] ..."
    }}
  ]
}}

FRAGMANTE:
{json.dumps(prompt_items, ensure_ascii=False, indent=2)}
"""

            success = False
            for attempt in range(len(GROQ_MODELS) * 2):
                model_info = GROQ_MODELS[model_idx % len(GROQ_MODELS)]
                payload = {
                    "model": model_info["name"],
                    "messages": [
                        {"role": "system", "content": "You are an AI alignment research assistant that always outputs strictly valid JSON according to the schema."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.5,
                    "max_tokens": model_info["max_tokens"],
                    "response_format": {"type": "json_object"},
                }

                try:
                    resp = requests.post(self.groq_url, headers=headers, json=payload, timeout=25)
                    if resp.status_code == 200:
                        raw_data = resp.json()
                        content = raw_data["choices"][0]["message"]["content"]
                        parsed = json.loads(content)
                        self._process_response_items(parsed, batch, art, pbar)
                        success = True
                        time.sleep(0.5)
                        break
                    elif resp.status_code == 429:
                        model_idx += 1
                        time.sleep(2.5 + random.uniform(0.5, 1.5))
                    else:
                        model_idx += 1
                        time.sleep(1.0)
                except Exception:
                    time.sleep(1.5)

            if not success:
                task_queue.put(task)
            task_queue.task_done()

    def run(self):
        print("=" * 80)
        print(f"  HYBRID MULTI-KEY TEACHER LLM SPP GENERATION (Țintă: {self.target_total:,})")
        print(f"  Gemini Activat: {bool(self.gemini_key)} (Model: gemini-flash-lite-latest)")
        print(f"  Chei Groq Active: {len(self.groq_keys)} | Modele: {[m['name'] for m in GROQ_MODELS]}")
        print(f"  Output Parquet: {REFLECTIONS_FILE}")
        print("=" * 80)

        candidates_by_art = self.load_candidates()

        task_queue = Queue()
        batch_size = 8
        cand_indices = {art: 0 for art in THEMES}

        pbar = tqdm(total=self.target_total, initial=len(self.records), desc="Progres SPP-Ro")

        threads = []
        # 1. Spawn Gemini worker (if key exists)
        if self.gemini_key:
            t_gemini = threading.Thread(target=self.gemini_worker_thread, args=(task_queue, pbar), daemon=True)
            t_gemini.start()
            threads.append(t_gemini)

        # 2. Spawn Groq workers
        for i, key in enumerate(self.groq_keys):
            t_groq = threading.Thread(target=self.groq_worker_thread, args=(i, key, task_queue, pbar), daemon=True)
            t_groq.start()
            threads.append(t_groq)

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
                time.sleep(5.0)

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

    generator = HybridTeacherSPPGenerator(
        groq_keys=GROQ_KEYS,
        gemini_key=GEMINI_KEY,
        target_total=args.target
    )
    generator.run()

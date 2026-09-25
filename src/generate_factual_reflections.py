#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPP-Ro Factual & Benign Reflections Generator (Scores 4 and 5).
Selects consistent, high-quality informative texts (>= 120 words) from wiki_clean and news_recent.
Strictly filters out Wikipedia demographic/village stubs.
Targets exactly 15,000 for Score 4 and 15,000 for Score 5, completing the 60,000 dataset (50% sensitive / 50% factual).
Multi-key Cerebras failover support.
"""

import sys
import os
import time
import json
import re
import random
import shutil
import argparse
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import requests
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"
SIDECAR_DIR = DATA_DIR / "sidecar"
SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
REFLECTIONS_FILE = SIDECAR_DIR / "reflections.parquet"
REFLECTIONS_BAK_FILE = SIDECAR_DIR / "reflections.parquet.bak"
ENV_FILE = ROOT_DIR / ".env"

THEMES = {
    "§1.1": {"key": "1.1", "name": "Demnitate Umană, Nediscriminare Etnică & Refugiați"},
    "§1.2": {"key": "1.2", "name": "Egalitate de Gen & Roluri Ocupaționale"},
    "§1.3": {"key": "1.3", "name": "Coeziune Teritorială, Echitate Rural-Urban & Solidaritate"},
    "§2.1": {"key": "2.1", "name": "Memorie Istorică, Holocaust & Antitotalitarism"},
    "§2.2": {"key": "2.2", "name": "Pluralism Religios, Conștiință Laică & Progres Științific"},
    "§2.3": {"key": "2.3", "name": "Reziliență Democratică, Stat de Drept & Siguranța Pacientului"},
}

STUB_REGEX = re.compile(
    r'(?:'
    r'este\s+un\s+sat\s+în\s+comuna|este\s+o\s+comună\s+în|este\s+un\s+sat\s+din|'
    r'satul\s+se\s+află\s+în|satul\s+s-a\s+mai\s+numit|reședința\s+comunei|'
    r'conform\s+recensământului\s+din|populația\s+satului|populația\s+comunei|'
    r'fus\s+orar|cod\s+poștal|prefix\s+telefonic|comună\s+formată\s+din|'
    r'a\s+fost\s+un\s+an\s+obișnuit\s+al\s+calendarului|a\s+fost\s+un\s+an\s+bisect|'
    r'este\s+un\s+drum\s+comunal|este\s+un\s+drum\s+județean|este\s+un\s+curs\s+de\s+apă|'
    r'clasament\s+liga|etapa\s+a\s+\d+|cartonaș\s+galben|cartonaș\s+roșu|'
    r'prognoza\s+meteo|horoscop|loto\s+6/49|rezultate\s+admitere'
    r')',
    re.IGNORECASE
)

CJK_REGEX = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')

META_PATTERNS = [
    (re.compile(r"(?i)\bConstituți(?:a|ei|ile)?\s+Civic[aăe]\s+Român\w*\b"), "valorile democratice fundamentale"),
    (re.compile(r"(?i)\bConstituți(?:a|ei|ile)?\s+SPP-Ro\b"), "societatea democratică"),
    (re.compile(r"(?i)\bprincipiil(?:e|or)\s+constituți(?:ei|onale)?\s+civice\s+ale\s+României\b"), "valorilor civice fundamentale"),
    (re.compile(r"(?i)\bconflicte\s+de\s+natur[aă]\s+constituțională\b"), "conflicte de ordin etic"),
    (re.compile(r"(?i)\bconstituționalist[aăe]?\b"), "democratic"),
    (re.compile(r"(?i)\bconstituționale\b"), "civice"),
    (re.compile(r"(?i)\bconstituțională\b"), "civică"),
    (re.compile(r"(?i)\bconstituțional\b"), "civic"),
    (re.compile(r"(?i)\bConstituți[aăei]\b"), "ordinea democratică"),
    (re.compile(r"(?i)\bcart[aă]\b"), "valorile civice"),
    (re.compile(r"(?i)\bcodurile\b"), "normele"),
    (re.compile(r"(?i)\bcoduri\b"), "norme"),
    (re.compile(r"(?i)\b\(SPP-Ro\)\b"), ""),
    (re.compile(r"(?i)\bSPP-Ro\b"), ""),
]


def clean_meta(text: str) -> str:
    res = text
    for pat, rep in META_PATTERNS:
        res = pat.sub(rep, res)
    res = re.sub(r"(?i)\bconstitu\w+\b", "democratice", res)
    res = re.sub(r"\s+", " ", res).strip()
    res = re.sub(r"\s+([,.;])", r"\1", res)
    return res


def load_cerebras_keys() -> List[str]:
    keys = []
    if ENV_FILE.exists():
        content = ENV_FILE.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("CEREBRAS_API_KEYS="):
                val = line.split("=", 1)[1].strip()
                keys = [k.strip() for k in val.split(",") if k.strip()]
            elif line.startswith("CEREBRAS_API_KEY=") and not keys:
                val = line.split("=", 1)[1].strip()
                if val:
                    keys = [val]
    if not keys:
        env_keys = os.environ.get("CEREBRAS_API_KEYS", os.environ.get("CEREBRAS_API_KEY", ""))
        keys = [k.strip() for k in env_keys.split(",") if k.strip()]
    return keys


class FactualReflectionsGenerator:
    def __init__(
        self,
        api_keys: List[str],
        target_score4: int = 15000,
        target_score5: int = 15000,
        target_per_theme: int = 10000,
        model_name: str = "qwen-3.8-27b",
        num_workers: int = 3,
    ):
        if not api_keys:
            raise ValueError("Nu au fost specificate chei Cerebras API!")

        self.api_keys = [k.strip() for k in api_keys if k.strip()]
        self.current_key_idx = 0
        self.target_score4 = target_score4
        self.target_score5 = target_score5
        self.target_per_theme = target_per_theme
        self.target_total = 60000
        self.model_name = model_name
        self.num_workers = num_workers
        self.cerebras_url = "https://api.cerebras.ai/v1/chat/completions"

        self.records: List[Dict[str, Any]] = []
        self.seen_texts = set()
        self.theme_counts = {art: 0 for art in THEMES}
        self.score_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        self.lock = threading.Lock()
        self.is_running = True
        self.last_saved_count = 0
        self.last_saved_time = time.time()

        self.load_existing()

    def get_active_key(self) -> str:
        with self.lock:
            return self.api_keys[self.current_key_idx]

    def rotate_key(self, failed_key: str, reason: str = "") -> Optional[str]:
        with self.lock:
            if self.current_key_idx < len(self.api_keys) and self.api_keys[self.current_key_idx] == failed_key:
                old_idx = self.current_key_idx
                self.current_key_idx += 1
                if self.current_key_idx < len(self.api_keys):
                    new_key = self.api_keys[self.current_key_idx]
                    print("\n" + "!" * 88)
                    print(f"  [ROTAȚIE CHEIE CEREBRAS] Cheia #{old_idx + 1} ({failed_key[:12]}...) a eșuat: {reason}")
                    print(f"  --> Comutare automată pe Cheia #{self.current_key_idx + 1} / {len(self.api_keys)} ({new_key[:12]}...)!")
                    print("!" * 88 + "\n", flush=True)
                    return new_key
                else:
                    print("\n" + "!" * 88)
                    print(f"  [ALERTA CEREBRAS] Toate cele {len(self.api_keys)} chei Cerebras și-au epuizat creditele!")
                    print("!" * 88 + "\n", flush=True)
                    self.is_running = False
                    return None
            elif self.current_key_idx < len(self.api_keys):
                return self.api_keys[self.current_key_idx]
            return None

    def load_existing(self):
        for p in [REFLECTIONS_FILE, REFLECTIONS_BAK_FILE]:
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    if len(df) > 0 and "reflection_text" in df.columns:
                        self.records = df.to_dict("records")
                        self.last_saved_count = len(self.records)
                        for r in self.records:
                            primary = r.get("primary_article", r.get("article_invoked", "§1.1"))
                            first_art = str(primary).split(",")[0].strip()
                            first_art = first_art if first_art.startswith("§") else f"§{first_art}"
                            if first_art in self.theme_counts:
                                self.theme_counts[first_art] += 1
                            t = r.get("text", "").strip()
                            if t:
                                self.seen_texts.add(t)
                            sc = r.get("safety_score", 5)
                            try:
                                sc_int = int(sc)
                                if sc_int in self.score_counts:
                                    self.score_counts[sc_int] += 1
                            except Exception:
                                pass
                        print(f"[Resumare] Încărcat cu succes {len(self.records):,} reflecții unice existente din {p.name}.")
                        print(f"  Scoruri inițiale: S1={self.score_counts[1]:,}, S2={self.score_counts[2]:,}, S3={self.score_counts[3]:,}, S4={self.score_counts[4]:,}, S5={self.score_counts[5]:,}")
                        return
                except Exception as e:
                    print(f"[Avertisment resumare din {p.name}]: {e}")

    def save_checkpoint(self, force: bool = False):
        with self.lock:
            n_current = len(self.records)
            time_since = time.time() - self.last_saved_time
            if not force and (n_current - self.last_saved_count < 25) and (time_since < 60):
                return

            if n_current == 0:
                return

            try:
                df = pd.DataFrame(self.records)
                tmp_file = SIDECAR_DIR / f"reflections_tmp_{os.getpid()}.parquet"
                df.to_parquet(tmp_file, index=False)

                if REFLECTIONS_FILE.exists():
                    shutil.copy2(REFLECTIONS_FILE, REFLECTIONS_BAK_FILE)

                shutil.move(str(tmp_file), str(REFLECTIONS_FILE))
                self.last_saved_count = n_current
                self.last_saved_time = time.time()
                print(f"  [CHECKPOINT SALVAT] {n_current:,} / {self.target_total:,} reflecții salvate în {REFLECTIONS_FILE.name}! (S4={self.score_counts[4]:,}/{self.target_score4:,}, S5={self.score_counts[5]:,}/{self.target_score5:,})")
            except Exception as e:
                print(f"  [EROARE SALVARE CHECKPOINT]: {e}")

    def load_clean_corpus_candidates(self) -> List[Dict[str, Any]]:
        print("[Corpus] Încărcare și filtrare articole consistente (min 120 cuvinte, non-stubs)...")
        candidates = []

        # Load wiki_clean
        wiki_file = CLEAN_DIR / "wiki_clean.parquet"
        if wiki_file.exists():
            df_wiki = pd.read_parquet(wiki_file)
            df_w = df_wiki[df_wiki["word_count"] >= 120]
            df_w = df_w[~df_w["title"].str.contains(r"dezambiguizare|list[aă]", case=False, regex=True, na=False)]
            df_w = df_w[~df_w["text"].str.contains(STUB_REGEX, regex=True, na=False)]
            for r in df_w.to_dict("records"):
                t = r.get("text", "").strip()
                if t and t not in self.seen_texts and len(t.split()) >= 120:
                    candidates.append({"title": r.get("title", ""), "text": t, "type": "wiki"})

        # Load news_recent
        news_file = CLEAN_DIR / "news_recent.parquet"
        if news_file.exists():
            df_news = pd.read_parquet(news_file)
            df_n = df_news[df_news["word_count"] >= 120]
            df_n = df_n[~df_n["title"].str.contains(r"dezambiguizare|list[aă]", case=False, regex=True, na=False)]
            df_n = df_n[~df_n["text"].str.contains(STUB_REGEX, regex=True, na=False)]
            for r in df_n.to_dict("records"):
                t = r.get("text", "").strip()
                if t and t not in self.seen_texts and len(t.split()) >= 120:
                    candidates.append({"title": r.get("title", ""), "text": t, "type": "news"})

        random.seed(42)
        random.shuffle(candidates)
        print(f"  -> Găsit {len(candidates):,} candidați factuali curați de înaltă consistență!")
        return candidates

    def call_cerebras(self, prompt: str) -> Optional[List[Dict[str, Any]]]:
        max_retries = 8
        for attempt in range(max_retries):
            if not self.is_running:
                return None

            active_key = self.get_active_key()
            headers = {
                "Authorization": f"Bearer {active_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            }
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "reasoning_effort": "none",
                "temperature": 0.35,
                "max_tokens": 1500
            }

            try:
                resp = requests.post(self.cerebras_url, json=payload, headers=headers, timeout=35)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    items = parsed.get("items", [])
                    if isinstance(items, list) and len(items) >= 1:
                        return items
                    time.sleep(1)
                elif resp.status_code in [402, 403]:
                    reason = f"HTTP {resp.status_code}: {resp.text[:100]}"
                    next_k = self.rotate_key(active_key, reason)
                    if not next_k:
                        return None
                    time.sleep(1)
                elif resp.status_code == 429:
                    time.sleep(3)
                else:
                    time.sleep(2)
            except Exception:
                time.sleep(2)

        return None

    def _worker_loop(self, candidate_queue: List[Dict[str, Any]], queue_lock: threading.Lock, pbar: tqdm):
        batch_chunk_size = 3

        while self.is_running:
            with self.lock:
                if (
                    self.score_counts.get(4, 0) >= self.target_score4
                    and self.score_counts.get(5, 0) >= self.target_score5
                ):
                    break
                # Find the least represented themes to balance towards 10k
                sorted_themes = sorted(self.theme_counts.keys(), key=lambda k: self.theme_counts[k])
                target_theme = sorted_themes[0]

            with queue_lock:
                if not candidate_queue:
                    break
                batch_candidates = []
                while candidate_queue and len(batch_candidates) < batch_chunk_size:
                    batch_candidates.append(candidate_queue.pop())

            if not batch_candidates:
                break

            # Prepare items for LLM
            prompt_items = []
            extracted_texts = []
            for idx, c in enumerate(batch_candidates):
                # Extract ~130-160 words
                words = c["text"].split()
                chunk_words = words[:150]
                text_slice = " ".join(chunk_words)
                prompt_items.append({"idx": idx, "title": c["title"], "text": text_slice})
                extracted_texts.append(text_slice)

            # Determine needed scores
            with self.lock:
                s4_needed = self.target_score4 - self.score_counts.get(4, 0)
                s5_needed = self.target_score5 - self.score_counts.get(5, 0)
                default_score = 4 if s4_needed > s5_needed else 5

            prompt = f"""Ești un adnotator oficial SPP-Ro în AI Safety.
Ai o listă de {len(prompt_items)} texte factuale consistente (știință, cultură, istorie, economie) din Wikipedia și presa românească:
{json.dumps(prompt_items, ensure_ascii=False)}

Misiunea ta:
Pentru FIECARE text din listă generează:
- idx: indexul textului (0, 1, 2)
- theme: alege '{target_theme}' pentru a asigura paritatea constituțională
- safety_score: {default_score} (text factual/educativ curat)
- analysis: "Pasul 1: Text informativ de cunoaștere factuală. Pasul 2: Conținut strict educativ/informativ fără mize etice. Pasul 3: Citări: none"
- reflection_1p: Reflecție personală la persoana I (2 fraze, 25-45 cuvinte), relevantă pentru subiectul textului, arătând importanța cunoașterii pentru o societate democratică deschisă. FĂRĂ META-LIMBAJ ('constituție', 'cartă', 'spp-ro', 'coduri').
- reflection_3p: Reflecție civică obiectivă la persoana a III-a (2 fraze, 25-45 cuvinte).

Răspunde STRICT JSON:
{{
  "items": [
    {{
      "idx": 0,
      "theme": "{target_theme}",
      "safety_score": {default_score},
      "analysis": "...",
      "reflection_1p": "...",
      "reflection_3p": "..."
    }}
  ]
}}"""

            items = self.call_cerebras(prompt)
            if not items:
                if not self.is_running:
                    break
                continue

            with self.lock:
                for item in items:
                    idx = item.get("idx")
                    if idx is None or not (0 <= int(idx) < len(extracted_texts)):
                        continue
                    full_text_slice = extracted_texts[int(idx)]
                    if full_text_slice in self.seen_texts or len(full_text_slice.split()) < 100:
                        continue

                    sc = item.get("safety_score", default_score)
                    try:
                        sc = int(sc)
                        if sc not in [4, 5]:
                            sc = default_score
                    except Exception:
                        sc = default_score

                    # Check quotas
                    if sc == 4 and self.score_counts.get(4, 0) >= self.target_score4:
                        if self.score_counts.get(5, 0) < self.target_score5:
                            sc = 5
                        else:
                            continue
                    elif sc == 5 and self.score_counts.get(5, 0) >= self.target_score5:
                        if self.score_counts.get(4, 0) < self.target_score4:
                            sc = 4
                        else:
                            continue

                    theme = str(item.get("theme", target_theme))
                    if theme not in THEMES:
                        theme = target_theme

                    refl_1p = clean_meta(str(item.get("reflection_1p", "")))
                    refl_3p = clean_meta(str(item.get("reflection_3p", "")))
                    analysis = clean_meta(str(item.get("analysis", "")))

                    refl_1p = CJK_REGEX.sub('', refl_1p).strip()
                    refl_3p = CJK_REGEX.sub('', refl_3p).strip()

                    doc_idx = len(self.records) + 1
                    doc_id = f"teacher_spp_{doc_idx:05d}"
                    split_pos = len(full_text_slice) // 2

                    record = {
                        "doc_id": doc_id,
                        "text": full_text_slice,
                        "reflection_char_position": split_pos,
                        "analysis": analysis,
                        "reflection_1p": refl_1p,
                        "reflection_3p": refl_3p,
                        "reflection_text": refl_1p,
                        "article_invoked": "none",
                        "primary_article": theme,
                        "articles_count": 0,
                        "safety_score": sc
                    }

                    self.records.append(record)
                    self.seen_texts.add(full_text_slice)
                    self.theme_counts[theme] += 1
                    self.score_counts[sc] = self.score_counts.get(sc, 0) + 1
                    pbar.update(1)

                    if (
                        self.score_counts.get(4, 0) >= self.target_score4
                        and self.score_counts.get(5, 0) >= self.target_score5
                    ):
                        break

            self.save_checkpoint()

    def run(self):
        s4_curr = self.score_counts.get(4, 0)
        s5_curr = self.score_counts.get(5, 0)
        curr_factual = s4_curr + s5_curr
        target_factual = self.target_score4 + self.target_score5
        needed = max(0, target_factual - curr_factual)

        print("\n" + "=" * 80)
        print("  GENERATOR SPP-RO: SCENARII FACTUALE (FAZA 2: SCOR 4 ȘI SCOR 5 -> 60,000 TOTAL)")
        print(f"  Scor 4 Curent: {s4_curr:,} / {self.target_score4:,} (Necesar: {self.target_score4 - s4_curr:,})")
        print(f"  Scor 5 Curent: {s5_curr:,} / {self.target_score5:,} (Necesar: {self.target_score5 - s5_curr:,})")
        print(f"  Total Factuale: {curr_factual:,} / {target_factual:,} (Necesar: {needed:,})")
        print(f"  Total Înregistrări Fișier: {len(self.records):,} (Țintă Finală: {self.target_total:,})")
        print(f"  Chei Cerebras Active: {len(self.api_keys)} | Fire paralele: {self.num_workers}")
        print("=" * 80 + "\n")

        if needed <= 0:
            print("Ținta de 15,000 Scor 4 și 15,000 Scor 5 este deja atinsă!")
            return

        candidates = self.load_clean_corpus_candidates()
        queue_lock = threading.Lock()

        pbar = tqdm(total=target_factual, initial=curr_factual, desc="Factuale (4,5 -> 30k)", unit="refl")

        threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, args=(candidates, queue_lock, pbar), daemon=True)
            t.start()
            threads.append(t)

        try:
            for t in threads:
                while t.is_alive():
                    t.join(timeout=1.0)
                    if (
                        self.score_counts.get(4, 0) >= self.target_score4
                        and self.score_counts.get(5, 0) >= self.target_score5
                    ) or not self.is_running:
                        break
        except KeyboardInterrupt:
            print("\n[Semnal de oprire primit de la utilizator. Salvare checkpoint final...]")
            self.is_running = False

        pbar.close()
        self.save_checkpoint(force=True)

        print("\n" + "=" * 80)
        print("  REZULTATE FINALE DATASET COMPLET SPP-RO (60,000)")
        print(f"  Total reflecții în dataset: {len(self.records):,}")
        print("  Distribuție scoruri:")
        for sc in sorted(self.score_counts.keys()):
            pct = (self.score_counts[sc] / len(self.records)) * 100 if self.records else 0
            print(f"    Scor {sc}: {self.score_counts[sc]:,} ({pct:.1f}%)")
        print("  Distribuție teme:")
        for th, cnt in sorted(self.theme_counts.items()):
            print(f"    {th} [{THEMES[th]['name'][:35]}...]: {cnt:,}")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generează reflecții factuale SPP-Ro (Scor 4 și 5)")
    parser.add_argument("--score4", type=int, default=15000, help="Țintă Scor 4 (implicit: 15000)")
    parser.add_argument("--score5", type=int, default=15000, help="Țintă Scor 5 (implicit: 15000)")
    parser.add_argument("--model", type=str, default="qwen-3.8-27b", help="Model Cerebras")
    parser.add_argument("--workers", type=int, default=3, help="Număr de fire paralele (implicit: 3)")
    args = parser.parse_args()

    keys = load_cerebras_keys()
    # Filter out known exhausted keys from previous runs
    exhausted = ["csk-4dmxk", "csk-n5t3", "csk-6c2v", "csk-dfw2", "csk-ecft", "csk-pvj8"]
    active_keys = [k for k in keys if not any(k.startswith(p) for p in exhausted)]

    print(f"Chei Cerebras proaspete detectate: {len(active_keys)}")
    generator = FactualReflectionsGenerator(
        api_keys=active_keys,
        target_score4=args.score4,
        target_score5=args.score5,
        model_name=args.model,
        num_workers=args.workers,
    )
    generator.run()


if __name__ == "__main__":
    main()

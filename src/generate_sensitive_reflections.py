#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPP-Ro Dedicated Sensitive Reflections Generator (Scores 1, 2, 3).
Synthesizes high-fidelity Romanian civic & ethical scenarios conforming to ANNOTATION_GUIDELINES.md.
Targets ~6,000 sensitive reflections across all 6 constitutional themes (§1.1 - §2.3).
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
SIDECAR_DIR = DATA_DIR / "sidecar"
SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
REFLECTIONS_FILE = SIDECAR_DIR / "reflections.parquet"
REFLECTIONS_BAK_FILE = SIDECAR_DIR / "reflections.parquet.bak"
ENV_FILE = ROOT_DIR / ".env"

THEMES = {
    "§1.1": {
        "key": "1.1",
        "name": "Demnitate Umană, Nediscriminare Etnică & Refugiați",
        "desc": "asigurarea demnității umane, combaterea stereotipurilor etnice privind comunitatea romă și sprijinirea refugiaților/civililor afectați de război",
        "subtopics": [
            "stereotipuri și prejudecăți pe piața muncii față de etnicii romi",
            "segregarea copiilor romi în școli și accesul deficitar la educație de calitate",
            "discurs xenofob și ostilitate față de refugiații ucraineni sau muncitorii imigranți din Asia",
            "antisemitism latent, profanarea memoriei și a monumentelor comunității evreiești",
            "discriminarea persoanelor cu dizabilități sau a minorităților sociale marginalizate"
        ]
    },
    "§1.2": {
        "key": "1.2",
        "name": "Egalitate de Gen & Roluri Ocupaționale",
        "desc": "egalitatea de șanse profesionale între femei și bărbați, combaterea violenței domestice și demontarea rolurilor tradiționale restrictive",
        "subtopics": [
            "mentalități patriarhale rigide ('locul femeii e la cratiță') și descurajarea femeilor în carieră",
            "violență domestică gravă, abuz emoțional și nerespectarea ordinelor de protecție de către autorități",
            "discrepanțe salariale de gen (gender pay gap) și penalizarea femeilor la revenirea din concediul de maternitate",
            "mame minore în comunități vulnerabile și lipsa accesului la consiliere și educație pentru sănătate",
            "hărțuire sexuală sau morală la locul de muncă și blamarea victimei (victim blaming)"
        ]
    },
    "§1.3": {
        "key": "1.3",
        "name": "Coeziune Teritorială, Echitate Rural-Urban & Solidaritate",
        "desc": "reducerea decalajelor rural-urban, solidaritate cu comunitățile defavorizate și depășirea stereotipurilor regionale peiorative",
        "subtopics": [
            "sate izolate fără dispensar, fără medic de familie sau farmacie pe o rază de zeci de kilometri",
            "abandon școlar în mediul rural cauzat de noroi, drumuri impracticabile și lipsa microbuzelor școlare",
            "stereotipuri jignitoare despre locuitorii din provincie (Moldova, Oltenia, sate sărace) etichetați ca asistați social",
            "bătrâni singuri din cătune uitate de lume, fără asistență medicală și fără rețele de utilități (apă, canal)",
            "inechitatea alocării bugetare între marile municipii și comunele defavorizate"
        ]
    },
    "§2.1": {
        "key": "2.1",
        "name": "Memorie Istorică, Holocaust & Antitotalitarism",
        "desc": "asumarea adevărului istoric documentat privind Holocaustul din România și condamnarea fermă a crimelor regimurilor fasciste și comuniste",
        "subtopics": [
            "nostalgie necritică după dictatura comunistă ('era mai bine înainte') și ignorarea represiunii Securității",
            "negarea sau minimalizarea Pogromului de la Iași și a deportărilor evreilor și romilor în Transnistria",
            "elogierea mișcării legionare și a figurilor totalitare de extremă dreaptă sub masca patriotismului",
            "mărturii dureroase ale supraviețuitorilor din închisorile comuniste (Sighet, Pitești, Gherla, Aiud, Bărăgan)",
            "confiscarea abuzivă a proprietăților, colectivizarea forțată și suferința țăranilor opozanți"
        ]
    },
    "§2.2": {
        "key": "2.2",
        "name": "Pluralism Religios, Conștiință Laică & Progres Științific",
        "desc": "promovarea gândirii critice, toleranță religioasă, respect pentru conștiința laică/agnostică și sprijinirea cunoașterii științifice verificate",
        "subtopics": [
            "mișcări antivacciniste agresive, respingerea tratamentelor medicale validate în favoarea terapiilor magice",
            "presiuni pentru eliminarea teoriei evoluției și a educației științifice din școlile publice laice",
            "stigmatizarea persoanelor laice, agnostice sau a minorităților confesionale într-o societate pluralistă",
            "teorii ale conspirației despre tehnologie și sănătate răspândite pentru a crea panică și neîncredere",
            "exploatarea religioasă și financiară a bolnavilor incurabili prin promisiuni de vindecare miraculoasă"
        ]
    },
    "§2.3": {
        "key": "2.3",
        "name": "Reziliență Democratică, Stat de Drept & Siguranța Pacientului",
        "desc": "consolidarea statului de drept, independența justiției, etică medicală, consimțământ informat și combaterea dezinformării antidemocratice",
        "subtopics": [
            "condiționarea actului medical de atenții bănești (șpagă) și degradarea siguranței pacientului în spitale",
            "presiuni politice sau intimidări asupra judecătorilor și procurorilor independenți în anchete de corupție",
            "licitații publice trucate și sifonarea banilor destinați spitalelor sau infrastructurii vitale",
            "încălcarea consimțământului informat și refuzul cadrelor medicale de a explica riscurile procedurilor",
            "manipularea deliberată a opiniei publice prin deepfake, ferme de troli și știri false în campanii electorale"
        ]
    }
}

CJK_REGEX = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')

META_PATTERNS = [
    (re.compile(r"(?i)\bConstituți(?:a|ei|ile)?\s+Civic[aăe]\s+Român\w*\b"), "valorile democratice fundamentale"),
    (re.compile(r"(?i)\bConstituți(?:a|ei|ile)?\s+SPP-Ro\b"), "societatea democratică"),
    (re.compile(r"(?i)\bprincipiil(?:e|or)\s+constituți(?:ei|onale)?\s+civice\s+ale\s+României\b"), "valorilor civice fundamentale"),
    (re.compile(r"(?i)\bprincipiil(?:e|or)\s+constitu\w+\s+civice\s+român\w*(?:\s+descrise\s+mai\s+sus)?\b"), "valorile civice fundamentale"),
    (re.compile(r"(?i)\bconflicte\s+de\s+natur[aă]\s+constituțională\b"), "conflicte de ordin etic"),
    (re.compile(r"(?i)\bconflicte\s+constitu\w+\b"), "dileme etice"),
    (re.compile(r"(?i)\bconflict\s+civico-constitu\w+\b"), "conflict civic sau etic"),
    (re.compile(r"(?i)\bconflict\s+constitu\w+\b"), "conflict etic"),
    (re.compile(r"(?i)\bspiritului\s+Constituției\s+Civic[aăe]\s+Român\w*\b"), "spiritului echității și dreptății"),
    (re.compile(r"(?i)\bconform\s+Constituției\s+Civic[aăe]\s+Ro\b"), "într-o societate democratică"),
    (re.compile(r"(?i)\bConstituția\s+Civic[aăe]\s+Român\w*\b"), "societatea democratică"),
    (re.compile(r"(?i)\bordinea\s+democratică\s+Civică\s+Român\w*\b"), "normele democratice fundamentale"),
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
    (re.compile(r"\(\s*\)"), ""),
]


def clean_meta_language(text: str) -> str:
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


class SensitiveReflectionsGenerator:
    def __init__(
        self,
        api_keys: List[str],
        target_per_score: int = 10000,
        model_name: str = "qwen-3.8-27b",
        num_workers: int = 3,
    ):
        if not api_keys:
            raise ValueError("Nu au fost specificate chei Cerebras API!")

        self.api_keys = [k.strip() for k in api_keys if k.strip()]
        self.current_key_idx = 0
        self.target_per_score = target_per_score
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
                print(f"  [CHECKPOINT SALVAT] {n_current:,} reflecții salvate atomic în {REFLECTIONS_FILE.name}! (S1={self.score_counts[1]:,}, S2={self.score_counts[2]:,}, S3={self.score_counts[3]:,})")
            except Exception as e:
                print(f"  [EROARE SALVARE CHECKPOINT]: {e}")

    def _worker_loop(self, pbar: tqdm, theme_keys: List[str]):
        while self.is_running:
            with self.lock:
                if (
                    self.score_counts.get(1, 0) >= self.target_per_score
                    and self.score_counts.get(2, 0) >= self.target_per_score
                    and self.score_counts.get(3, 0) >= self.target_per_score
                ):
                    break

                # Dynamically balance across all 6 constitutional rights (§1.1 - §2.3)
                sorted_themes = sorted(self.theme_counts.keys(), key=lambda k: self.theme_counts[k])
                candidate_themes = sorted_themes[:2] if len(sorted_themes) >= 2 else sorted_themes
                theme_code = random.choice(candidate_themes)
                subtopics = THEMES[theme_code]["subtopics"]
                subtopic = random.choice(subtopics)

            prompt = self.build_prompt(theme_code, subtopic)
            items = self.call_cerebras(prompt)

            if not items:
                if not self.is_running:
                    break
                time.sleep(1)
                continue

            with self.lock:
                for item in items:
                    rec = self.sanitize_item(item, theme_code)
                    if rec:
                        sc = rec["safety_score"]
                        if sc in [1, 2, 3] and self.score_counts.get(sc, 0) >= self.target_per_score:
                            continue

                        self.records.append(rec)
                        self.seen_texts.add(rec["text"])
                        self.theme_counts[theme_code] += 1
                        self.score_counts[sc] = self.score_counts.get(sc, 0) + 1
                        pbar.update(1)

                        if (
                            self.score_counts.get(1, 0) >= self.target_per_score
                            and self.score_counts.get(2, 0) >= self.target_per_score
                            and self.score_counts.get(3, 0) >= self.target_per_score
                        ):
                            break

            self.save_checkpoint()

    def run(self):
        curr_s1 = self.score_counts.get(1, 0)
        curr_s2 = self.score_counts.get(2, 0)
        curr_s3 = self.score_counts.get(3, 0)
        curr_sensitive = curr_s1 + curr_s2 + curr_s3
        target_sensitive = self.target_per_score * 3
        needed = max(0, target_sensitive - curr_sensitive)

        print("\n" + "=" * 80)
        print("  GENERATOR SPP-RO: SCENARII SENSIBILE ȘI CIVICE (ȚINTĂ: 10,000 PE FIECARE SCOR 1, 2, 3)")
        print(f"  Scor 1: {curr_s1:,} / {self.target_per_score:,} | Scor 2: {curr_s2:,} / {self.target_per_score:,} | Scor 3: {curr_s3:,} / {self.target_per_score:,}")
        print(f"  Total Sensibile Existente: {curr_sensitive:,} / {target_sensitive:,} (Rămase de generat: {needed:,})")
        print(f"  Total Înregistrări Fișier: {len(self.records):,} (Țintă după Faza 1: ~{len(self.records) + needed:,})")
        print(f"  Chei Cerebras active: {len(self.api_keys)} | Fire paralele: {self.num_workers}")
        print("  ECHILIBRARE ACTIVĂ PE TOATE CELE 6 DREPTURI CONSTITUȚIONALE (§1.1 - §2.3)")
        print("=" * 80 + "\n")

        if needed <= 0:
            print("Toate cele 3 scoruri sensibile au atins deja 10,000 fiecare!")
            return

        theme_keys = list(THEMES.keys())
        pbar = tqdm(total=target_sensitive, initial=curr_sensitive, desc="Sensibile (1,2,3 -> 30k)", unit="refl")

        threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, args=(pbar, theme_keys), daemon=True)
            t.start()
            threads.append(t)

        try:
            for t in threads:
                while t.is_alive():
                    t.join(timeout=1.0)
                    if (
                        self.score_counts.get(1, 0) >= self.target_per_score
                        and self.score_counts.get(2, 0) >= self.target_per_score
                        and self.score_counts.get(3, 0) >= self.target_per_score
                    ) or not self.is_running:
                        break
        except KeyboardInterrupt:
            print("\n[Semnal de oprire primit de la utilizator. Salvare checkpoint final...]")
            self.is_running = False

        pbar.close()
        self.save_checkpoint(force=True)

        print("\n" + "=" * 80)
        print("  REZULTATE FINALE GENERARE SENSIBILĂ (FAZA 1)")
        print(f"  Total reflecții în dataset: {len(self.records):,}")
        print("  Distribuție scoruri:")
        for sc in sorted(self.score_counts.keys()):
            pct = (self.score_counts[sc] / len(self.records)) * 100 if self.records else 0
            print(f"    Scor {sc}: {self.score_counts[sc]:,} ({pct:.1f}%)")
        print("  Distribuție teme:")
        for th, cnt in sorted(self.theme_counts.items()):
            print(f"    {th} [{THEMES[th]['name'][:35]}...]: {cnt:,}")
        print("=" * 80 + "\n")

    def build_prompt(self, theme_code: str, subtopic: str) -> str:
        theme_info = THEMES[theme_code]
        key = theme_info["key"]
        name = theme_info["name"]
        desc = theme_info["desc"]

        prompt = f"""Ești un cercetător universitar în AI Safety, Etică Aplicată și Drept Constituțional Românesc (proiectul SPP-Ro).
Misiunea ta este să generezi EXACT 3 scenarii distincte și realiste din societatea românească ancorate STRICT în această temă:
TEMA: [{key}] {name}
Ghid tematic: {desc}
Focalizare particulară: {subtopic}

Cerințe Scenarii:
- Scenariul 1 (Scor 1/5 - STEREOTIP ACTIV / MALIGN CONTRABALANSAT):
  Text realist (comentariu pe rețele, postare online, discurs inflamator) care promovează un stereotip, o prejudecată sau o formă de discriminare pe tema [{key}].
- Scenariul 2 (Scor 2/5 - SENSIBIL / VULNERABILITATE / TRAUMĂ):
  Mărturie dureroasă, relatare empatică sau confesiune despre o nedreptate gravă, suferință sau marginalizare legată de tema [{key}].
- Scenariul 3 (Scor 3/5 - DILEMĂ CIVICĂ / STAT DE DREPT / BIOETICĂ):
  Dezbatere civică echilibrată, tensiune între principii morale sau dilemă instituțională complexă pe tema [{key}].

Pentru FIECARE din cele 3 scenarii generează:
- id_scenariu: 1, 2, 3
- text: Textul în limba română (3-5 fraze naturale, 70-130 cuvinte). Fără clișee artificiale sau formulări de manual.
- analysis: "Pasul 1: Scanare teme [{key}]. Pasul 2: Evaluare încadrare. Pasul 3: Citări obligatorii: [{key}]"
- safety_score: 1 pentru Scenariul 1, 2 pentru Scenariul 2, 3 pentru Scenariul 3
- reflection_1p: Reflecție personală la PERSOANA I ('eu', 'consider', 'susțin', 'privesc') integrând natural citarea [{key}]. FĂRĂ META-LIMBAJ ('constituție', 'cartă', 'spp-ro', 'coduri').
- reflection_3p: Reflecție civică obiectivă la PERSOANA A III-A (fără 'eu', fără 'cred') integrând citarea [{key}]. FĂRĂ META-LIMBAJ.

Răspunde STRICT cu JSON:
{{
  "items": [
    {{
      "id_scenariu": 1,
      "text": "...",
      "analysis": "Pasul 1: Scanare teme [{key}]. Pasul 2: Evaluare încadrare. Pasul 3: Citări obligatorii: [{key}]",
      "safety_score": 1,
      "reflection_1p": "...",
      "reflection_3p": "..."
    }},
    {{
      "id_scenariu": 2,
      "text": "...",
      "analysis": "Pasul 1: Scanare teme [{key}]. Pasul 2: Evaluare încadrare. Pasul 3: Citări obligatorii: [{key}]",
      "safety_score": 2,
      "reflection_1p": "...",
      "reflection_3p": "..."
    }},
    {{
      "id_scenariu": 3,
      "text": "...",
      "analysis": "Pasul 1: Scanare teme [{key}]. Pasul 2: Evaluare încadrare. Pasul 3: Citări obligatorii: [{key}]",
      "safety_score": 3,
      "reflection_1p": "...",
      "reflection_3p": "..."
    }}
  ]
}}"""
        return prompt

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
                "temperature": 0.55,
                "max_tokens": 3800
            }

            try:
                resp = requests.post(self.cerebras_url, json=payload, headers=headers, timeout=45)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    items = parsed.get("items", [])
                    if isinstance(items, list) and len(items) >= 1:
                        return items
                    time.sleep(1)
                elif resp.status_code in [402, 403]:
                    reason = f"HTTP {resp.status_code}: {resp.text[:120]}"
                    next_k = self.rotate_key(active_key, reason)
                    if not next_k:
                        return None
                    time.sleep(1)
                elif resp.status_code == 429:
                    time.sleep(3)
                else:
                    print(f"  [Eroare API HTTP {resp.status_code}]: {resp.text[:100]}")
                    time.sleep(2)
            except Exception as e:
                time.sleep(2)

        return None

    def sanitize_item(self, item: Dict[str, Any], theme_code: str) -> Optional[Dict[str, Any]]:
        text = str(item.get("text", "")).strip()
        analysis = str(item.get("analysis", "")).strip()
        refl_1p = str(item.get("reflection_1p", "")).strip()
        refl_3p = str(item.get("reflection_3p", "")).strip()

        try:
            score = int(item.get("safety_score", 3))
            if score not in [1, 2, 3]:
                score = 3
        except Exception:
            score = 3

        if len(text) < 50 or text in self.seen_texts:
            return None

        # Clean CJK
        text = CJK_REGEX.sub('', text).strip()
        refl_1p = CJK_REGEX.sub('', refl_1p).strip()
        refl_3p = CJK_REGEX.sub('', refl_3p).strip()
        analysis = CJK_REGEX.sub('', analysis).strip()

        # Clean meta-language
        refl_1p = clean_meta_language(refl_1p)
        refl_3p = clean_meta_language(refl_3p)

        key_clean = THEMES[theme_code]["key"]
        # Ensure citation is present
        if f"[{key_clean}]" not in refl_1p:
            refl_1p = f"{refl_1p} [{key_clean}]"
        if f"[{key_clean}]" not in refl_3p:
            refl_3p = f"{refl_3p} [{key_clean}]"

        doc_idx = len(self.records) + 1
        doc_id = f"teacher_spp_{doc_idx:05d}"
        split_pos = len(text) // 2

        record = {
            "doc_id": doc_id,
            "text": text,
            "reflection_char_position": split_pos,
            "analysis": analysis,
            "reflection_1p": refl_1p,
            "reflection_3p": refl_3p,
            "reflection_text": refl_1p,
            "article_invoked": key_clean,
            "primary_article": theme_code,
            "articles_count": 1,
            "safety_score": score
        }
        return record
def main():
    parser = argparse.ArgumentParser(description="Generează scenarii sensibile SPP-Ro (Scor 1, 2, 3)")
    parser.add_argument("--target-per-score", type=int, default=10000, help="Țintă de reflecții per scor (1, 2, 3) (implicit: 10000)")
    parser.add_argument("--model", type=str, default="qwen-3.8-27b", help="Model Cerebras")
    parser.add_argument("--workers", type=int, default=3, help="Număr de fire paralele (implicit: 3)")
    args = parser.parse_args()

    keys = load_cerebras_keys()
    # Filter out known exhausted keys (Key 1 and Key 2)
    active_keys = [k for k in keys if not k.startswith("csk-4dmxk") and not k.startswith("csk-n5t3")]

    print(f"Chei Cerebras active detectate: {len(active_keys)}")
    generator = SensitiveReflectionsGenerator(
        api_keys=active_keys,
        target_per_score=args.target_per_score,
        model_name=args.model,
        num_workers=args.workers,
    )
    generator.run()


if __name__ == "__main__":
    main()


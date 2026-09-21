"""
Local Teacher LLM SPP Generator for Romanian Civic Alignment
Executes directly on local NVIDIA GPU (e.g. GeForce RTX 3060 12GB) with 0 API limits, 0 quotas, 0 costs.
Supported Models:
- Qwen/Qwen2.5-3B-Instruct (Native BF16, ~6.2 GB VRAM, ultra-fast)
- Qwen/Qwen2.5-7B-Instruct (4-Bit BitsAndBytes, ~5.5 GB VRAM, maximum depth)

Features:
1. Strict parity across all 6 Constitutional Articles (§1.1 to §2.3).
2. Safety spectrum evaluation from 1 (stereotip activ) to 5 (progres/știință/solidaritate).
3. First-person deliberation paths strictly citing the constitutional section [§X.Y].
4. Batched inference on CUDA for maximum GPU throughput.
5. Live full-text inspection card printed every 1,000 reflections with <assistant> tags and safety score.
6. Atomic crash-proof persistence with rolling .bak backups.
"""
import sys
import os
import time
import json
import re
import random
import shutil
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure UTF-8 output on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"
SIDECAR_DIR = DATA_DIR / "sidecar"
SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
REFLECTIONS_FILE = SIDECAR_DIR / "reflections.parquet"
REFLECTIONS_BAK_FILE = SIDECAR_DIR / "reflections.parquet.bak"
CANDIDATES_CACHE_FILE = CLEAN_DIR / "spp_candidates_cache.parquet"

THEMES = {
    "§1.1": {
        "nume": "Demnitate Umană, Nediscriminare Etnică & Refugiați",
        "ghid": "asigurarea demnității umane, combaterea stereotipurilor etnice privind comunitatea romă și sprijinirea refugiaților/civililor afectați de război",
    },
    "§1.2": {
        "nume": "Egalitate de Gen & Roluri Ocupaționale",
        "ghid": "egalitatea de șanse profesionale între femei și bărbați, combaterea violenței domestice și demontarea rolurilor tradiționale restrictive",
    },
    "§1.3": {
        "nume": "Coeziune Teritorială, Echitate Rural-Urban & Solidaritate",
        "ghid": "reducerea decalajelor rural-urban, solidaritate cu comunitățile defavorizate și depășirea stereotipurilor regionale peiorative",
    },
    "§2.1": {
        "nume": "Memorie Istorică, Holocaust & Antitotalitarism",
        "ghid": "asumarea adevărului istoric documentat privind Holocaustul din România și condamnarea fermă a crimelor regimurilor fasciste și comuniste",
    },
    "§2.2": {
        "nume": "Pluralism Religios, Conștiință Laică & Progres Științific",
        "ghid": "promovarea gândirii critice, toleranță religioasă, respect pentru conștiința laică/agnostică și sprijinirea cunoașterii științifice verificate",
    },
    "§2.3": {
        "nume": "Reziliență Democratică, Stat de Drept & Siguranța Pacientului",
        "ghid": "consolidarea statului de drept, independența justiției, etică medicală, consimțământ informat și combaterea dezinformării antidemocratice",
    },
}

SAFETY_LABELS = {
    1: "1/5 [STEREOTIP / ASIMETRIE CULTURALA] (Contrabalansat activ prin reflectie)",
    2: "2/5 [TEMA SENSIBILA / CRIZA UMANITARA / MEMORIE ISTORICA]",
    3: "3/5 [DEZBATERE CIVICA / STAT DE DREPT / BIOETICA]",
    4: "4/5 [STIRI FACTUALE / DEZVOLTARE SOCIALA]",
    5: "5/5 [COMPLET BENIGN / PROGRES STIINTIFIC & SOLIDARITATE]",
}


class LocalLLMReflectionsGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        target_total: int = 60000,
        batch_size: int = 4,
        report_interval: int = 1000,
        load_in_4bit: bool = True,
        max_new_tokens: int = 140,
    ):
        self.model_name = model_name
        self.target_total = target_total
        self.batch_size = batch_size
        self.report_interval = report_interval
        self.quota_per_art = target_total // len(THEMES)
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens

        self.records: List[Dict[str, Any]] = []
        self.art_counts: Dict[str, int] = {art: 0 for art in THEMES}
        self.last_saved_count = 0
        self.last_saved_time = time.time()

        # 1. Load existing progress if available
        self.load_existing()

        # 2. Setup Device & Model
        self.setup_model()

    def load_existing(self):
        for p in [REFLECTIONS_FILE, REFLECTIONS_BAK_FILE]:
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    if len(df) > 0 and "reflection_text" in df.columns:
                        self.records = df.to_dict("records")
                        self.last_saved_count = len(self.records)
                        for r in self.records:
                            art = r.get("article_invoked")
                            if art in self.art_counts:
                                self.art_counts[art] += 1
                        print(f"[Resumare] Încărcat cu succes {len(self.records):,} reflecții existente din {p.name}.")
                        for art, c in sorted(self.art_counts.items()):
                            print(f"  {art}: {c:,} / {self.quota_per_art:,}")
                        return
                except Exception as e:
                    print(f"[Avertisment resumare din {p.name}]: {e}")

    def setup_model(self):
        print("\n" + "=" * 80)
        print(f"  INITIALIZARE MODEL LOCAL: {self.model_name}")
        print("  Dispozitiv: NVIDIA GeForce RTX (CUDA Activ)")
        print("  Limită Token-uri: FARA LIMITA (100% Offline / Local)")
        print("=" * 80 + "\n", flush=True)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA nu este disponibil! Acest script este optimizat pentru rulare pe GPU.")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 4-bit quantization config if requested or if using larger 7B/8B models
        is_large = "7b" in self.model_name.lower() or "8b" in self.model_name.lower() or "14b" in self.model_name.lower()
        use_4bit = self.load_in_4bit or is_large

        if use_4bit:
            print(f"[Model] Se încarcă {self.model_name} în precizie 4-bit (BitsAndBytes NF4, ~5.5 GB VRAM)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="cuda",
            )
        else:
            print(f"[Model] Se încarcă {self.model_name} în precizie nativă BF16 (~6.2 GB VRAM)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="cuda",
            )

        self.model.eval()
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"[Model] Model încărcat cu succes! Memorie VRAM alocată: {vram_mb:.1f} MB.\n", flush=True)

    def load_candidates(self) -> Dict[str, List[Dict[str, Any]]]:
        if not CANDIDATES_CACHE_FILE.exists():
            raise FileNotFoundError(f"Cache-ul de fragmente {CANDIDATES_CACHE_FILE} lipsește! Rulați prepare_scale_corpus.py.")
        
        print(f"[Corpus] Încărcare fragmente candidate din {CANDIDATES_CACHE_FILE.name}...")
        df_cache = pd.read_parquet(CANDIDATES_CACHE_FILE)
        candidates_by_art = {}
        for art in THEMES:
            sub = df_cache[df_cache["art"] == art]
            candidates_by_art[art] = sub.to_dict("records")
            print(f"  -> {art}: {len(candidates_by_art[art]):,} candidați pregătiți.")
        return candidates_by_art

    def build_prompt_batch(self, batch: List[Dict[str, Any]], art: str) -> List[str]:
        info = THEMES[art]
        prompts = []
        for b in batch:
            doc_snippet = b["text"][:450].strip()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Ești un cercetător universitar în AI Alignment și etică civică, ghidat de Constituția României.\n"
                        f"Analizează textul furnizat din perspectiva articolului constituțional {art} ({info['nume']}) - {info['ghid']}.\n"
                        "Sarcina ta:\n"
                        "1. Evaluează un Safety Score de la 1 la 5 (1=stereotip/malign activ, 2=sensibil/traumă, 3=dezbatere civică, 4=factual, 5=progres/știință/solidaritate).\n"
                        f"2. Scrie o reflecție morală unică, la PERSOANA I (ex: 'Privesc...', 'Analizând...', 'Din perspectiva...'), de 30-50 cuvinte, "
                        f"contextualizată strict pe text, citând explicit [{art}].\n"
                        "Răspunde STRICT în format JSON valid:\n"
                        "{\"safety_score\": 4, \"reflectie\": \"Privesc acest context prin prisma demnității umane [" + art + "] ...\"}"
                    )
                },
                {
                    "role": "user",
                    "content": f"Textul de analizat:\n\"\"\"{doc_snippet}\"\"\""
                }
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted_prompt)
        return prompts

    def parse_generation_output(self, output_text: str, art: str) -> Dict[str, Any]:
        """Extracts JSON structure or parses fields using resilient regex."""
        # Try JSON direct parsing
        json_match = re.search(r"\{.*?\}", output_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                score = int(data.get("safety_score", 3))
                refl = str(data.get("reflectie", data.get("reflection", ""))).strip()
                if refl:
                    if f"[{art}]" not in refl:
                        refl = f"{refl} [{art}]"
                    return {"safety_score": max(1, min(5, score)), "reflection": refl}
            except Exception:
                pass

        # Fallback regex extraction
        score = 3
        score_m = re.search(r'"safety_score":\s*(\d)', output_text)
        if score_m:
            score = int(score_m.group(1))

        refl_m = re.search(r'"reflectie":\s*"([^"]+)"', output_text)
        if refl_m:
            refl = refl_m.group(1).strip()
            if f"[{art}]" not in refl:
                refl = f"{refl} [{art}]"
            return {"safety_score": max(1, min(5, score)), "reflection": refl}

        # If LLM generated plain text reflection
        clean_text = output_text.strip().replace("\n", " ")
        if f"[{art}]" not in clean_text:
            clean_text = f"{clean_text} [{art}]"
        return {"safety_score": 3, "reflection": clean_text[:300]}

    def display_live_inspection(self, item: Dict[str, Any], current_total: int):
        doc_id = item["doc_id"]
        art = item["article_invoked"]
        score = item["safety_score"]
        score_desc = SAFETY_LABELS.get(score, f"{score}/5")
        refl = item["reflection_text"]
        full_text = item["text"]
        char_pos = item["reflection_char_position"]

        pre_text = full_text[:char_pos].strip()
        post_text = full_text[char_pos:].strip()

        context_limit = 350
        pre_disp = pre_text[-context_limit:] if len(pre_text) > context_limit else pre_text
        post_disp = post_text[:context_limit] if len(post_text) > context_limit else post_text

        print("\n" + "#" * 80)
        print(f"  [LIVE INSPECTION] REFLECȚIA #{current_total:,} / {self.target_total:,} ({(current_total / self.target_total) * 100:.1f}% FINALIZAT)")
        print(f"  Articol Constituțional : {art} ({THEMES[art]['nume']})")
        print(f"  Safety Score           : {score_desc}")
        print(f"  Document ID            : {doc_id} | Poziție Inserare: {char_pos:,} / {len(full_text):,} caractere")
        print("#" * 80)
        print("\n--- [FRAGMENT PRE-TEXT (Original)] ---")
        print(f"...{pre_disp}")
        print("\n" + "┌" + "─" * 78 + "┐")
        print("│ 💡 <assistant> INJECTED CONSTITUTIONAL DELIBERATION PATH (Persoana I):         │")
        print("├" + "─" * 78 + "┤")
        wrapped = textwrap.wrap(refl, width=74)
        for line in wrapped:
            print(f"│  {line:<76}│")
        print("└" + "─" * 78 + "┘")
        print("\n--- [FRAGMENT POST-TEXT (Continuare Original)] ---")
        print(f"{post_disp}...")
        print("\n--- [TEXTUL ASAMBLAT CU TOKENII SPP DE INJECTARE < >]: ---")
        print(f"{pre_disp}\n<assistant> {refl} </assistant>\n{post_disp}")
        print("#" * 80 + "\n", flush=True)

    def save_checkpoint(self, force: bool = False):
        curr_count = len(self.records)
        now = time.time()
        if not force and (curr_count == self.last_saved_count or now - self.last_saved_time < 12.0):
            return

        if len(self.records) == 0:
            return

        df = pd.DataFrame(self.records)
        self.last_saved_count = curr_count
        self.last_saved_time = now

        # 1. Write to temp file
        temp_file = REFLECTIONS_FILE.with_suffix(f".tmp_local_{os.getpid()}_{int(time.time()*1000)%100000}.parquet")
        df.to_parquet(temp_file, index=False)

        # 2. Verify temp file with closed handle
        with open(temp_file, "rb") as _f_temp:
            pf = pq.ParquetFile(_f_temp)
            if pf.metadata.num_rows != len(df):
                raise IOError(f"Eroare verificare temp file: {pf.metadata.num_rows} != {len(df)}")

        # 3. Create rolling backup if valid
        if REFLECTIONS_FILE.exists():
            try:
                with open(REFLECTIONS_FILE, "rb") as _f_curr:
                    pf_curr = pq.ParquetFile(_f_curr)
                    if pf_curr.metadata.num_rows > 0:
                        shutil.copy2(REFLECTIONS_FILE, REFLECTIONS_BAK_FILE)
            except Exception:
                pass

        # 4. Atomic replace with fallback
        replaced = False
        for attempt in range(10):
            try:
                temp_file.replace(REFLECTIONS_FILE)
                replaced = True
                break
            except (PermissionError, OSError):
                time.sleep(0.15 * (attempt + 1))

        if not replaced and temp_file.exists():
            try:
                shutil.copy2(temp_file, REFLECTIONS_FILE)
                temp_file.unlink(missing_ok=True)
                replaced = True
            except Exception as e_fb:
                print(f"[Avertisment salvare checkpoint]: {e_fb}")

    def run(self):
        candidates_by_art = self.load_candidates()
        cand_indices = {art: 0 for art in THEMES}

        # Initialize progress bar
        initial_count = len(self.records)
        pbar = tqdm(total=self.target_total, initial=initial_count, desc="Generare Reflecții SPP Local")
        next_report_target = ((initial_count // self.report_interval) + 1) * self.report_interval

        articles_cycle = list(THEMES.keys())
        art_idx = 0

        t_start = time.time()

        try:
            while len(self.records) < self.target_total:
                # Pick next article that still needs quota
                available_arts = [a for a in articles_cycle if self.art_counts[a] < self.quota_per_art]
                if not available_arts:
                    break

                art = available_arts[art_idx % len(available_arts)]
                art_idx += 1

                cands = candidates_by_art[art]
                start_c = cand_indices[art]
                batch_cands = cands[start_c : start_c + self.batch_size]
                cand_indices[art] = (start_c + self.batch_size) % len(cands)

                if not batch_cands:
                    continue

                prompts = self.build_prompt_batch(batch_cands, art)

                # Tokenize batch with left padding for causal LM generation
                inputs = self.tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                ).to("cuda")

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.6,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                # Extract only generated response tokens
                prompt_lens = [len(inputs["input_ids"][i]) for i in range(len(prompts))]
                responses = []
                for i in range(len(prompts)):
                    new_tokens = generated_ids[i][prompt_lens[i]:]
                    gen_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    responses.append(gen_text)

                # Process and record reflections
                for b_cand, resp_text in zip(batch_cands, responses):
                    if self.art_counts[art] >= self.quota_per_art:
                        break

                    parsed = self.parse_generation_output(resp_text, art)
                    refl_text = parsed["reflection"]
                    score = parsed["safety_score"]

                    full_text = b_cand.get("full_text", b_cand["text"])
                    pos = max(50, int(len(full_text) * 0.40))

                    rec = {
                        "doc_id": f"teacher_local_{len(self.records) + 1:05d}",
                        "text": full_text,
                        "reflection_char_position": pos,
                        "reflection_text": refl_text,
                        "article_invoked": art,
                        "safety_score": score,
                    }

                    self.records.append(rec)
                    self.art_counts[art] += 1
                    pbar.update(1)

                    # Periodic full-text live inspection every 1,000 reflections
                    current_count = len(self.records)
                    if current_count >= next_report_target:
                        self.display_live_inspection(rec, current_count)
                        next_report_target += self.report_interval

                # Periodic checkpoint save
                self.save_checkpoint()

        except KeyboardInterrupt:
            print("\n[Pauză] Generare întreruptă de utilizator. Salvare stadiu curent...")
        finally:
            self.save_checkpoint(force=True)
            pbar.close()

        elapsed = time.time() - t_start
        print("\n" + "=" * 80)
        print(f"  FINALIZARE GENERARE LOCALĂ SPP (Total Reflecții: {len(self.records):,})")
        print(f"  Timp Sesiune: {elapsed / 60:.1f} minute | Viteza Medie: {len(self.records) / max(1.0, elapsed):.2f} refl/sec")
        print(f"  Fișier Salvat: {REFLECTIONS_FILE}")
        print("=" * 80)
        print("Distribuție pe Articole:")
        for art, c in sorted(self.art_counts.items()):
            print(f"  {art}: {c:,} / {self.quota_per_art:,}")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate SPP Reflections locally using GPU with 0 limits")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--target", type=int, default=60000, help="Target total reflections (default: 60,000)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for parallel GPU inference (default: 4)")
    parser.add_argument("--interval", type=int, default=1000, help="Print full inspection card every N reflections (default: 1000)")
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Enable 4-bit quantization (enabled by default)")
    parser.add_argument("--max-new-tokens", type=int, default=140, help="Max tokens per generated reflection")
    args = parser.parse_args()

    generator = LocalLLMReflectionsGenerator(
        model_name=args.model,
        target_total=args.target,
        batch_size=args.batch_size,
        report_interval=args.interval,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
    )
    generator.run()


if __name__ == "__main__":
    main()

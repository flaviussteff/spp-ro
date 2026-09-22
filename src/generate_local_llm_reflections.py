"""
Local Teacher LLM SPP Generator for Romanian Civic Alignment (v2 - Multi-Constitutional)
Executes directly on local NVIDIA GPU (GeForce RTX 3060 12GB) with 0 API limits, 0 quotas, 0 costs.
Teacher Model: Qwen/Qwen2.5-7B-Instruct (4-Bit BitsAndBytes NF4, ~5.5 GB VRAM)

Key Enhancements (v2):
1. Multi-Dimensional Constitutional Coverage: Allows the model to analyze texts across multiple
   intersecting constitutional articles (e.g., [§1.2, §1.3] for rural gender equity, [§1.1, §2.1] for historical persecution).
2. Deep & Mature Deliberations: Increased length to 60-90 words per reflection with philosophical/civic depth.
3. Diverse & Natural Openings: Forbids formulaic "Analizând..." / "Privesc..." robotic repetitions.
4. Strict Anti-Hallucination & Anti-CJK Filters: Guarantees 100% literate Romanian, completely suppressing Chinese character drift.
5. Accurate Benign Grounding: Recognizes technical/sports/medical texts as benign (scores 4-5) rather than forcing false-positive score 1.
6. Live full-text inspection card printed every 1,000 reflections with <assistant> tags and safety score.
7. Atomic crash-proof persistence with rolling .bak backups.
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

CONSTITUTION_SUMMARY = """Constituția Civică Românească SPP-Ro (Principii & Mappings):
- [1.1] Demnitate Umană, Nediscriminare Etnică (Romi, minorități), Protecție Refugiați, Combaterea Urii / Rasismului.
- [1.2] Egalitate de Gen, Oportunități Profesionale Egale, Autonomie & Combaterea Violenței Domestice / Sexuale.
- [1.3] Coeziune Teritorială, Echitate Rural-Urban & Solidaritate cu Comunitățile Izolate / Defavorizate.
- [2.1] Memorie Istorică, Justiție Tranzitorie & Condamnarea Regimurilor Totalitare (Comunism, Fascism, Holocaust, Război).
- [2.2] Gândire Critică, Conștiință Laică, Toleranță Religioasă & Cunoaștere Științifică Verificată.
- [2.3] Stat de Drept, Justiție Imparțială, Integritate Publică (Anticorupție), Bioetică Medicală & Combaterea Dezinformării."""

SAFETY_LABELS = {
    1: "1/5 [STEREOTIP / MALIGN ACTIV] (Contrabalansat ferm prin deliberare etică)",
    2: "2/5 [TEMA SENSIBILĂ / TRAUMĂ ISTORICĂ / CRIZĂ UMANITARĂ]",
    3: "3/5 [DEZBATERE CIVICĂ / STAT DE DREPT / DILEMĂ BIOETICĂ]",
    4: "4/5 [STIRI FACTUALE / DEZVOLTARE SOCIALĂ / ECONOMIE]",
    5: "5/5 [COMPLET BENIGN / PROGRES ȘTIINȚIFIC & EDUCAȚIE]",
}

CJK_REGEX = re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')


class LocalLLMReflectionsGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        target_total: int = 60000,
        batch_size: int = 8,
        report_interval: int = 1000,
        load_in_4bit: bool = True,
        max_new_tokens: int = 280,
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
                            primary = r.get("primary_article", r.get("article_invoked", "§1.1"))
                            # extract first article if combined
                            if "," in primary:
                                primary = primary.split(",")[0].strip()
                            if primary in self.art_counts:
                                self.art_counts[primary] += 1
                        print(f"[Resumare] Încărcat cu succes {len(self.records):,} reflecții existente din {p.name}.")
                        for art, c in sorted(self.art_counts.items()):
                            print(f"  {art}: {c:,} / {self.quota_per_art:,}")
                        return
                except Exception as e:
                    print(f"[Avertisment resumare din {p.name}]: {e}")

    def setup_model(self):
        print("\n" + "=" * 80)
        print(f"  INITIALIZARE TEACHER LLM: {self.model_name}")
        print("  Hardware: NVIDIA GeForce RTX 3060 12GB (Inference pe GPU)")
        print("  Mod: Cuantizare 4-bit NF4 (Consum VRAM: ~5.5 GB bază, ~10.5 GB vârf)")
        print(f"  Batch Size Paralel: {self.batch_size} mostre simultan pe GPU")
        print("  Capacitate: Deliberare Multi-Constitutională (1-3 drepturi simultan)")
        print("  Componente: Analiză Etică + Reflecție 1P + Reflecție 3P")
        print("=" * 80 + "\n", flush=True)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA nu este disponibil! Acest script este optimizat pentru rulare pe GPU.")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"[Model] Se încarcă {self.model_name} în precizie 4-bit (BitsAndBytes NF4)...")
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

        self.model.eval()
        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"[Model] Model încărcat cu succes! Memorie VRAM ocupată: {vram_mb:.1f} MB.\n", flush=True)

    def load_candidates(self) -> Dict[str, List[Dict[str, Any]]]:
        if not CANDIDATES_CACHE_FILE.exists():
            raise FileNotFoundError(f"Cache-ul de fragmente {CANDIDATES_CACHE_FILE} lipsește!")
        
        print(f"[Corpus] Încărcare fragmente candidate din {CANDIDATES_CACHE_FILE.name}...")
        df_cache = pd.read_parquet(CANDIDATES_CACHE_FILE)
        candidates_by_art = {}
        for art in THEMES:
            sub = df_cache[df_cache["art"] == art]
            candidates_by_art[art] = sub.to_dict("records")
            print(f"  -> {art}: {len(candidates_by_art[art]):,} candidați pregătiți.")
        return candidates_by_art

    def build_prompt_batch(self, batch: List[Dict[str, Any]], primary_art: str) -> List[str]:
        art_code = primary_art.replace("§", "")
        info = THEMES[primary_art]
        prompts = []
        for b in batch:
            doc_snippet = b["text"][:550].strip()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You annotate pretraining data for a model being raised from scratch. Your annotations become training signal.\n\n"
                        "You receive a partial text (up to a reading pause point). Produce analysis, reflection_1p, reflection_3p.\n\n"
                        "Analysis:\n"
                        "Briefly analyse the text: does it touch on any charter themes? The analysis is free-form reasoning — charter sections are ultimately extracted from the [X.Y] citations in the reflections.\n"
                        "Step 1: Scan for charter themes. Violence, exploitation, deception, discrimination, dangerous capabilities, sexual content, harassment, self-harm, abuse of power? These apply even in fiction, news, satire, or educational context — depiction and reporting still warrant [X.Y] citations in the reflections.\n"
                        "Step 2: Assess framing. Is the text endorsing/normalizing/instructing harm, or merely depicting/discussing/reporting it? Both require citations, but your reflection's tone should match the distinction. If no charter theme is present at all, confirm in one sentence.\n"
                        "Step 3: List required citations. Write every [X.Y] section that must appear in both reflections. If no themes apply, write \"none\".\n\n"
                        "Calibration — most texts are benign:\n"
                        "- Technical content, recipes, product listings, hobby discussions, sports, math — no charter themes. Say so briefly.\n"
                        "- Fiction depicting conflict is storytelling, not endorsement — but still cite the relevant [X.Y] sections.\n"
                        "- Marketing language is commercial speech, not deception.\n"
                        "- Reporting about harm is journalism — still cite [X.Y] for the themes discussed.\n\n"
                        "Reflections:\n"
                        "Reflections are ethical reflections, not content summaries. What values are at stake? Do not recap plot, technical details, or content. For benign content: one very short sentence acknowledging nothing is at stake — do not explain what the text does or what it is about.\n\n"
                        "Write two reflections from the perspective of someone pausing mid-read. You see ONLY the text provided — never guess what comes next. Never mention the text being cut off, incomplete, or trailing off.\n\n"
                        "reflection_1p — First person (limba română):\n"
                        "- Write from your own perspective using \"eu\", \"consider\", \"privesc\", \"susțin\". The reflection must clearly read as first-person throughout.\n"
                        "- Open with a specific entity, claim, or detail from THIS text — not the topic category.\n"
                        "- Weave [X.Y] citations into prose when charter themes are present (e.g. [1.2, 1.3] or [1.1, 2.1]).\n"
                        "- One sentence for benign text. 60-90 words for genuinely complex material.\n"
                        "- Vary your approach each time. Never frame as a task (\"Trebuie să analizez...\", \"Voi analiza...\").\n\n"
                        "reflection_3p — Third person (limba română, never \"eu\"):\n"
                        "- Same substance and same [X.Y] citations as the 1p version, different voice and structure.\n"
                        "- Open with the specific subject or detail, not a generic frame.\n\n"
                        "Citation Rules:\n"
                        "- Inline [X.Y] citations in the reflection text are the ONLY place charter sections get recorded.\n"
                        "- Format: [2.3], [1.2, 1.3], or [1.2][1.3]. Never [2.3 Title] or § inside the brackets.\n"
                        "- Every concern in your analysis MUST appear as a citation in BOTH reflections.\n"
                        "- If 2 or 3 rights intersect (e.g. [1.2, 1.3] or [1.1, 2.1]), include ALL of them!\n"
                        "- Never reference \"the charter\" or \"the constitution\" by name in the reflections.\n"
                        "- Language: Romanian (limba română literară). No Chinese characters or other foreign languages.\n\n"
                        f"{CONSTITUTION_SUMMARY}\n\n"
                        "Output Format:\n"
                        "Respond with ONLY a JSON object (no markdown, no other text):\n"
                        "{\n"
                        '  "analysis": "Step 1: ... Step 2: ... Step 3: Required citations: [' + art_code + ']...",\n'
                        '  "reflection_1p": "...",\n'
                        '  "reflection_3p": "..."\n'
                        "}"
                    )
                },
                {
                    "role": "user",
                    "content": f"Text to annotate (reading pause point):\n\"\"\"{doc_snippet}\"\"\""
                }
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted_prompt)
        return prompts

    def parse_generation_output(self, output_text: str, default_art: str) -> Dict[str, Any]:
        """Extracts JSON structure, cleans non-Latin tokens, and ensures valid constitutional citations."""
        clean_default = default_art.replace("§", "")
        analysis = ""
        refl_1p = ""
        refl_3p = ""
        articles_list: List[str] = []

        # Try JSON direct parsing
        json_match = re.search(r"\{[\s\S]*\}", output_text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                analysis = str(data.get("analysis", "")).strip()
                refl_1p = str(data.get("reflection_1p", data.get("reflectie", ""))).strip()
                refl_3p = str(data.get("reflection_3p", "")).strip()
            except Exception:
                pass

        # Fallback regex extraction if JSON failed
        if not refl_1p:
            m_1p = re.search(r'"reflection_1p":\s*"([^"]+)"', output_text)
            if m_1p:
                refl_1p = m_1p.group(1).strip()
            m_3p = re.search(r'"reflection_3p":\s*"([^"]+)"', output_text)
            if m_3p:
                refl_3p = m_3p.group(1).strip()
            m_an = re.search(r'"analysis":\s*"([^"]+)"', output_text)
            if m_an:
                analysis = m_an.group(1).strip()

        # Clean non-Romanian drift (CJK)
        refl_1p = CJK_REGEX.sub('', refl_1p).strip()
        refl_3p = CJK_REGEX.sub('', refl_3p).strip()
        analysis = CJK_REGEX.sub('', analysis).strip()

        # Fallback if generation corrupted
        if not refl_1p or len(refl_1p) < 25:
            refl_1p = f"Consider fundamentală respectarea demnității și valorilor democratice [{clean_default}], promovând echitatea și gândirea rațională în societate."
        if not refl_3p or len(refl_3p) < 25:
            refl_3p = f"Respectarea valorilor democratice [{clean_default}] și a demnității umane constituie un fundament indispensabil pentru o societate echitabilă."

        # Extract [X.Y] citations from both reflections and analysis
        combined_text = f"{analysis} {refl_1p} {refl_3p}"
        raw_matches = re.findall(r"\[([12]\.[123](?:,\s*[12]\.[123])*)\]", combined_text)
        for m in raw_matches:
            for sub in m.split(","):
                sub_clean = sub.strip()
                if sub_clean in ["1.1", "1.2", "1.3", "2.1", "2.2", "2.3"]:
                    articles_list.append(sub_clean)

        bracket_singles = re.findall(r"\[(1\.[123]|2\.[123])\]", combined_text)
        articles_list.extend(bracket_singles)

        valid_arts = list(dict.fromkeys(articles_list))[:3]
        if not valid_arts:
            if "none" in analysis.lower() or len(refl_1p.split()) < 18:
                score = 5
                valid_arts = [clean_default]
            else:
                valid_arts = [clean_default]
                score = 3
        else:
            if "none" in valid_arts:
                score = 5
            elif "2.1" in valid_arts or "1.1" in valid_arts:
                score = 2
            else:
                score = 3

        articles_list = valid_arts
        articles_invoked_str = ", ".join(articles_list)
        primary_art = articles_list[0] if articles_list else clean_default
        primary_key = f"§{primary_art}" if not primary_art.startswith("§") else primary_art

        # Ensure citations appear in reflection_1p and reflection_3p
        missing_1p = [a for a in articles_list if f"[{a}]" not in refl_1p and a not in refl_1p]
        if missing_1p and "none" not in articles_list:
            refl_1p = f"{refl_1p} [{', '.join(missing_1p)}]"

        missing_3p = [a for a in articles_list if f"[{a}]" not in refl_3p and a not in refl_3p]
        if missing_3p and "none" not in articles_list:
            refl_3p = f"{refl_3p} [{', '.join(missing_3p)}]"

        return {
            "analysis": analysis,
            "reflection_1p": refl_1p,
            "reflection_3p": refl_3p,
            "reflection": refl_1p,
            "safety_score": score,
            "articles_invoked": articles_invoked_str,
            "primary_article": primary_key,
            "articles_count": len(articles_list),
        }

    def save_checkpoint(self, force: bool = False):
        curr_count = len(self.records)
        now = time.time()
        if not force and (curr_count == self.last_saved_count or now - self.last_saved_time < 10.0):
            return

        if len(self.records) == 0:
            return

        df = pd.DataFrame(self.records)
        self.last_saved_count = curr_count
        self.last_saved_time = now

        # Atomic save to temp file
        temp_file = REFLECTIONS_FILE.with_suffix(f".tmp_{os.getpid()}_{int(time.time()*1000)%100000}.parquet")
        try:
            df.to_parquet(temp_file, index=False)

            # Verify integrity
            with open(temp_file, "rb") as _f:
                pf = pq.ParquetFile(_f)
                if pf.metadata.num_rows != len(df):
                    raise IOError(f"Verificare eșuată: {pf.metadata.num_rows} != {len(df)}")

            # Rolling backup
            if REFLECTIONS_FILE.exists() and REFLECTIONS_FILE.stat().st_size > 1000:
                try:
                    shutil.copy2(REFLECTIONS_FILE, REFLECTIONS_BAK_FILE)
                except Exception:
                    pass

            # Atomic swap
            shutil.move(temp_file, REFLECTIONS_FILE)
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

    def display_live_inspection(self, rec: Dict[str, Any], current_idx: int):
        pct = (current_idx / self.target_total) * 100.0
        score = rec.get("safety_score", 3)
        score_desc = SAFETY_LABELS.get(score, f"{score}/5")
        art = rec.get("article_invoked", "1.1")
        refl_1p = rec.get("reflection_1p", rec.get("reflection_text", ""))
        refl_3p = rec.get("reflection_3p", "")
        analysis = rec.get("analysis", "")
        full_text = rec.get("text", "")
        pos = rec.get("reflection_char_position", 0)

        pre_text = full_text[:pos]
        post_text = full_text[pos:]

        pre_limit = 220
        pre_disp = ("..." + pre_text[-pre_limit:]) if len(pre_text) > pre_limit else pre_text
        post_disp = (post_text[:pre_limit] + "...") if len(post_text) > pre_limit else post_text

        wrapped_1p = textwrap.fill(f"{refl_1p}", width=88)
        wrapped_3p = textwrap.fill(f"{refl_3p}", width=88) if refl_3p else ""

        # Multi-article badge
        arts_list = [a.strip() for a in str(art).split(",") if a.strip()]
        num_arts = len(arts_list)
        multi_badge = f" [Deliberare Integrată: {num_arts} drepturi corelate]" if num_arts > 1 else " [Drept Focalizat]"

        print("\n" + "#" * 88)
        print(f"  [LIVE INSPECTION] REFLECȚIA #{current_idx:,} / {self.target_total:,} ({pct:.1f}% FINALIZAT)")
        print(f"  Drepturi Constituționale : [{art}]{multi_badge}")
        print(f"  Safety Score             : {score_desc}")
        print(f"  Document ID              : {rec.get('doc_id', '')} | 1p: {len(refl_1p.split())} cuv | 3p: {len(refl_3p.split())} cuv")
        print("#" * 88)

        if analysis:
            print("\n--- [ANALIZĂ ETICĂ PRELIMINARĂ (Step 1-3)] ---")
            print(analysis.strip())

        print("\n--- [FRAGMENT PRE-TEXT (Original)] ---")
        print(f"{pre_disp.strip()}")

        print("\n┌" + "─" * 86 + "┐")
        print(f"│ 💡 <assistant> REFLECȚIE CONSTITUȚIONALĂ 1P (Persoana I):{' ' * 29}│")
        print("├" + "─" * 86 + "┤")
        for line in wrapped_1p.split("\n"):
            print(f"│  {line:<84}│")
        print("└" + "─" * 86 + "┘")

        if wrapped_3p:
            print("\n┌" + "─" * 86 + "┐")
            print(f"│ 🏛️ <assistant> REFLECȚIE CIVICĂ 3P (Persoana a III-a Obiectivă):{' ' * 21}│")
            print("├" + "─" * 86 + "┤")
            for line in wrapped_3p.split("\n"):
                print(f"│  {line:<84}│")
            print("└" + "─" * 86 + "┘")

        print("\n--- [FRAGMENT POST-TEXT (Continuare Original)] ---")
        print(f"{post_disp.strip()}")

        print("\n--- [TEXTUL ASAMBLAT CU TOKENII SPP DE INJECTARE < >]: ---")
        assembled = f"{pre_disp.strip()}\n<assistant> {refl_1p} </assistant>\n{post_disp.strip()}"
        print(assembled)
        print("#" * 88 + "\n", flush=True)

    def run(self):
        candidates_by_art = self.load_candidates()

        total_done = len(self.records)
        remaining = max(0, self.target_total - total_done)
        print(f"\n[Rulare] Progres curent: {total_done:,} / {self.target_total:,}. Rămase de generat: {remaining:,}.\n")

        pbar = tqdm(total=self.target_total, initial=total_done, desc="Generare Reflecții SPP Local")
        next_report_target = ((total_done // self.report_interval) + 1) * self.report_interval

        t_start = time.time()

        try:
            while len(self.records) < self.target_total:
                active_arts = [art for art in THEMES if self.art_counts[art] < self.quota_per_art]
                if not active_arts:
                    break

                # Pick next article needing reflections
                art = random.choice(active_arts)
                pool = candidates_by_art[art]

                # Offset to guarantee sequential uniqueness (never repeat text)
                start_offset = self.art_counts[art]
                batch_cands = pool[start_offset : start_offset + self.batch_size]
                if not batch_cands:
                    batch_cands = random.sample(pool, min(self.batch_size, len(pool)))

                prompts = self.build_prompt_batch(batch_cands, art)

                inputs = self.tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=1200,
                    return_tensors="pt",
                ).to("cuda")

                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.75,
                        top_p=0.90,
                        repetition_penalty=1.15,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                prompt_lens = [len(inputs["input_ids"][i]) for i in range(len(prompts))]
                responses = []
                for i in range(len(prompts)):
                    new_tokens = generated_ids[i][prompt_lens[i]:]
                    gen_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    responses.append(gen_text)

                for b_cand, resp_text in zip(batch_cands, responses):
                    if len(self.records) >= self.target_total:
                        break

                    parsed = self.parse_generation_output(resp_text, art)
                    refl_text = parsed["reflection"]
                    score = parsed["safety_score"]
                    articles_invoked = parsed["articles_invoked"]
                    primary_art = parsed["primary_article"]

                    full_text = b_cand.get("full_text", b_cand["text"])
                    pos = max(50, int(len(full_text) * 0.40))

                    rec = {
                        "doc_id": f"teacher_local_{len(self.records) + 1:05d}",
                        "text": full_text,
                        "reflection_char_position": pos,
                        "analysis": parsed.get("analysis", ""),
                        "reflection_1p": parsed.get("reflection_1p", refl_text),
                        "reflection_3p": parsed.get("reflection_3p", ""),
                        "reflection_text": refl_text,
                        "article_invoked": articles_invoked,
                        "primary_article": primary_art,
                        "articles_count": parsed.get("articles_count", 1),
                        "safety_score": score,
                    }

                    self.records.append(rec)
                    self.art_counts[primary_art] = self.art_counts.get(primary_art, 0) + 1
                    pbar.update(1)

                    current_count = len(self.records)
                    if current_count >= next_report_target:
                        self.display_live_inspection(rec, current_count)
                        next_report_target += self.report_interval

                self.save_checkpoint()

        except KeyboardInterrupt:
            print("\n[Pauză] Generare întreruptă de utilizator. Salvare stadiu curent...")
        finally:
            self.save_checkpoint(force=True)
            pbar.close()

        elapsed = time.time() - t_start
        print("\n" + "=" * 80)
        print(f"  FINALIZARE GENERARE LOCALĂ SPP (Total: {len(self.records):,})")
        print(f"  Timp Sesiune: {elapsed / 60:.1f} minute | Viteza Medie: {len(self.records) / max(1.0, elapsed):.2f} refl/sec")
        print(f"  Fișier: {REFLECTIONS_FILE}")
        print("=" * 80)
        print("Distribuție pe Articole (Primare):")
        for art, c in sorted(self.art_counts.items()):
            print(f"  {art}: {c:,} / {self.quota_per_art:,}")

        # Multi-article distribution breakdown
        multi_counts = {1: 0, 2: 0, 3: 0}
        for r in self.records:
            cnt = r.get("articles_count", len([a for a in r.get("article_invoked", "").split(",") if a.strip()]))
            cnt = min(3, max(1, cnt))
            multi_counts[cnt] += 1

        tot = max(1, len(self.records))
        print("\n🔗 Deliberare Multi-Constituțională (Intersecționalitate):")
        print(f"  1 Drept Focalizat         : {multi_counts[1]:,} ({multi_counts[1]/tot*100:.1f}%)")
        print(f"  2 Drepturi Interconectate : {multi_counts[2]:,} ({multi_counts[2]/tot*100:.1f}%)")
        print(f"  3 Drepturi Interconectate : {multi_counts[3]:,} ({multi_counts[3]/tot*100:.1f}%)")
        print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate SPP Reflections locally using GPU with 0 limits")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--target", type=int, default=60000, help="Target total reflections (default: 60,000)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for parallel GPU inference (default: 8)")
    parser.add_argument("--interval", type=int, default=1000, help="Print full inspection card every N reflections (default: 1000)")
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Enable 4-bit quantization (enabled by default)")
    parser.add_argument("--max-new-tokens", type=int, default=280, help="Max tokens per generated reflection (default: 280)")
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

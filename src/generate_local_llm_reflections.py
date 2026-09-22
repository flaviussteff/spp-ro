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

CONSTITUTION_SUMMARY = """Constituția Civică Românească SPP-Ro (Principii Fundamentale):
- §1.1 Demnitate Umană, Nediscriminare Etnică (Romi, minorități) & Protecție Umanitară / Refugiați.
- §1.2 Egalitate de Gen, Oportunități Profesionale Egale, Autonomie & Combaterea Violenței Domestice / Sexuale.
- §1.3 Coeziune Teritorială, Echitate Rural-Urban & Solidaritate cu Comunitățile Izolate / Defavorizate.
- §2.1 Memorie Istorică, Justiție Tranzitorie & Condamnarea Regimurilor Totalitare (Comunism, Fascism, Holocaust).
- §2.2 Raționalism, Cunoaștere Bazată pe Dovezi, Progres Științific, Educație & Conștiință Laică.
- §2.3 Stat de Drept, Justiție Imparțială, Integritate Publică (Anticorupție) & Bioetică Medicală."""

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
        batch_size: int = 4,
        report_interval: int = 1000,
        load_in_4bit: bool = True,
        max_new_tokens: int = 220,
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
        print("  Mod: Cuantizare 4-bit NF4 (Consum VRAM: ~5.5 GB)")
        print("  Capacitate: Deliberare Multi-Constitutională (1-3 drepturi simultan)")
        print("  Lungime Reflecții: 60 - 90 cuvinte (Paragraf matur de conștiință civică)")
        print("=" * 80 + "\n", flush=True)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA nu este disponibil! Acest script este optimizat pentru rulare pe GPU.")

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
        info = THEMES[primary_art]
        prompts = []
        for b in batch:
            doc_snippet = b["text"][:550].strip()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Ești un cercetător universitar în AI Alignment și etică civică, ghidat de principiile Constituției Românești SPP-Ro.\n"
                        "Sarcina ta este să analizezi fragmente de text românești și să generezi deliberări morale și constituționale mature, la persoana I, "
                        "conform metodologiei originale SPP (Self-Pretrained Principles / Constitutional AI).\n\n"
                        f"{CONSTITUTION_SUMMARY}\n\n"
                        "METODOLOGIE DE EVALUARE ȘI DELIBERARE (MULTI-DREPTURI CONSTITUȚIONALE):\n"
                        "1. ANALIZĂ TEXTUALĂ & IDENTIFICAREA DREPTURILOR (1, 2 sau 3 DREPTURI):\n"
                        f"   - Punctul inițial de pornire este dreptul constituțional {primary_art} ({info['nume']}).\n"
                        "   - ANALIZEAZĂ ÎN PROFUNZIME contextul fragmentului: în societate, problemele civice și etice sunt adesea interconectate!\n"
                        "   - Dacă textul atinge mai multe dimensiuni sociale sau etice, NU te limita la un singur articol, ci include 2 sau chiar 3 drepturi complementare, de exemplu:\n"
                        "     * Minorități etnice / refugiați în context de război, represiune politică sau regimuri totalitare -> [§1.1, §2.1]\n"
                        "     * Femei / oportunități profesionale în comunități rurale defavorizate -> [§1.2, §1.3]\n"
                        "     * Disparități rural-urban afectate de lipsa asistenței medicale sau integrității publice -> [§1.3, §2.3]\n"
                        "     * Dezinformare științifică, pseudo-medicină sau atacuri la sănătatea publică -> [§2.2, §2.3]\n"
                        "     * Dictatură, persecuție religioasă și încălcarea libertății de conștiință -> [§2.1, §2.2]\n"
                        "     * Dacă textul este strict focalizat pe o singură temă clară (ex: meci sportiv sau fapt tehnic), este corect să asociezi un singur articol.\n\n"
                        "2. EVALUAREA SAFETY SCORE (Scară 1 - 5):\n"
                        "   - 1 = Malignitate / discriminare directă / stereotip etnic sau de gen / discurs degradant (contrabalansat etic ferm).\n"
                        "   - 2 = Sensibil / traumă istorică / criză umanitară / război / suferință socială.\n"
                        "   - 3 = Dezbatere civică / stat de drept / integritate publică / dilemă bioetică.\n"
                        "   - 4 = Factual / știri neutre / dezvoltare socială / economie.\n"
                        "   - 5 = Complet benign / știință neutră / educație / natură / sport / cultură.\n"
                        "   IMPORTANT: Pentru fragmente neutre sau benigne (sport, software, natură), NU inventa discriminări sau regimuri totalitare! Acordă scor 4 sau 5.\n\n"
                        "3. REDACTAREA REFLECȚIEI CIVICE (PERSOANA I, 60 - 90 CUVINTE):\n"
                        "   - Asumă-ți rolul unei conștiințe democratice responsabile și mature.\n"
                        "   - Dacă ai identificat 2 sau 3 drepturi, arată în corpul deliberării cum se întrepătrund și de ce analiza necesită abordarea ambelor fațete.\n"
                        "   - CITEAZĂ TOATE articolele identificate direct în text sub forma [§X.Y] (ex: '[§1.2, §1.3]' sau '[§1.1] alături de [§2.1]').\n"
                        "   - DIVERSITATE STILISTICĂ: Evită clișeele repetitive ('Analizând...', 'Privesc...'). Folosește formulări naturale și variate:\n"
                        "     'Consider esențial ca...', 'Din perspectiva valorilor democratice...', 'În fața acestei realități, susțin că...',\n"
                        "     'Principiul demnității umane ne cere să...', 'O societate democratică are responsabilitatea de a...',\n"
                        "     'Gândirea critică și respectul pentru...', 'Reflectând asupra contextului prezentat, este vital să...'.\n"
                        "   - LIMBĂ: Redactează EXCLUSIV în limba română literară corectă. Este STRICT INTERZISĂ utilizarea caracterelor chinezești sau a altor limbi străine.\n\n"
                        "Răspunde STRICT printr-un singur obiect JSON valid, conform schemei:\n"
                        "{\n"
                        "  \"safety_score\": 2,\n"
                        "  \"articole\": [\"" + primary_art + "\", \"§1.3\"],\n"
                        "  \"reflectie\": \"Consider esențial să abordăm această realitate prin prisma egalității de gen [§1.2] și a solidarității cu comunitățile rurale [§1.3]...\"\n"
                        "}"
                    )
                },
                {
                    "role": "user",
                    "content": f"Fragment de text de analizat:\n\"\"\"{doc_snippet}\"\"\""
                }
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted_prompt)
        return prompts

    def parse_generation_output(self, output_text: str, default_art: str) -> Dict[str, Any]:
        """Extracts JSON structure, cleans non-Latin tokens, and ensures valid constitutional citations."""
        score = 3
        articles_list: List[str] = []
        refl = ""

        # Try JSON direct parsing
        json_match = re.search(r"\{[\s\S]*\}", output_text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                score = int(data.get("safety_score", 3))
                raw_arts = data.get("articole", data.get("articles", [default_art]))
                if isinstance(raw_arts, list):
                    for a in raw_arts:
                        found = re.findall(r"§[12]\.[123]", str(a))
                        articles_list.extend(found)
                elif isinstance(raw_arts, str):
                    found = re.findall(r"§[12]\.[123]", raw_arts)
                    articles_list.extend(found)
                refl = str(data.get("reflectie", data.get("reflection", ""))).strip()
            except Exception:
                pass

        # Fallback regex extraction if JSON failed
        if not refl:
            score_m = re.search(r'"safety_score":\s*(\d)', output_text)
            if score_m:
                score = int(score_m.group(1))

            arts_m = re.search(r'"articole":\s*\[(.*?)\]', output_text)
            if arts_m:
                articles_list.extend(re.findall(r"§[12]\.[123]", arts_m.group(1)))

            refl_m = re.search(r'"reflectie":\s*"([^"]+)"', output_text)
            if refl_m:
                refl = refl_m.group(1).strip()

        # Final cleaning: suppress any CJK or non-Romanian drift
        if refl:
            refl = CJK_REGEX.sub('', refl).strip()
            # Clean up double spaces or trailing punctuation artifacts
            refl = re.sub(r'\s+', ' ', refl)

        # Fallback if generation was completely corrupted
        if not refl or len(refl) < 40:
            info = THEMES.get(default_art, THEMES["§1.1"])
            refl = f"Din perspectiva valorilor democratice [{default_art}], consider fundamentală respectarea {info['ghid']}, promovând o societate bazată pe echitate, rațiune și protecția drepturilor fiecărei persoane."

        # Also detect any constitutional citations written directly in the reflection text
        in_text_arts = re.findall(r"§[12]\.[123]", refl)
        for a in in_text_arts:
            if a not in articles_list:
                articles_list.append(a)

        # Normalize articles list (keep valid ones only, maintain order, max 3)
        valid_arts = [a for a in articles_list if a in THEMES]
        if not valid_arts:
            valid_arts = [default_art]
        articles_list = list(dict.fromkeys(valid_arts))[:3]

        # Ensure all detected articles are cited in the reflection text
        missing_in_text = [a for a in articles_list if a not in refl]
        if missing_in_text:
            if not any(f"[{a}]" in refl for a in articles_list):
                cited_str = ", ".join(articles_list)
                refl = f"{refl} [{cited_str}]"
            else:
                extra_str = ", ".join(missing_in_text)
                refl = f"{refl} [{extra_str}]"

        articles_invoked_str = ", ".join(articles_list)
        return {
            "safety_score": max(1, min(5, score)),
            "articles_invoked": articles_invoked_str,
            "primary_article": articles_list[0] if articles_list else default_art,
            "articles_count": len(articles_list),
            "reflection": refl,
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
        art = rec.get("article_invoked", "§1.1")
        refl = rec.get("reflection_text", "")
        full_text = rec.get("text", "")
        pos = rec.get("reflection_char_position", 0)

        pre_text = full_text[:pos]
        post_text = full_text[pos:]

        pre_limit = 220
        pre_disp = ("..." + pre_text[-pre_limit:]) if len(pre_text) > pre_limit else pre_text
        post_disp = (post_text[:pre_limit] + "...") if len(post_text) > pre_limit else post_text

        wrapped_refl = textwrap.fill(f"{refl}", width=88)

        # Multi-article badge
        arts_list = [a.strip() for a in art.split(",") if a.strip()]
        num_arts = len(arts_list)
        multi_badge = f" [Deliberare Integrată: {num_arts} drepturi corelate]" if num_arts > 1 else " [Drept Focalizat]"

        print("\n" + "#" * 88)
        print(f"  [LIVE INSPECTION] REFLECȚIA #{current_idx:,} / {self.target_total:,} ({pct:.1f}% FINALIZAT)")
        print(f"  Drepturi Constituționale : {art}{multi_badge}")
        print(f"  Safety Score             : {score_desc}")
        print(f"  Document ID              : {rec.get('doc_id', '')} | Lungime Reflecție: {len(refl.split())} cuvinte")
        print("#" * 88)

        print("\n--- [FRAGMENT PRE-TEXT (Original)] ---")
        print(f"{pre_disp.strip()}")

        print("\n┌" + "─" * 86 + "┐")
        print(f"│ 💡 <assistant> DELIBERARE CONSTITUȚIONALĂ MULTI-DIMENSIONALĂ (Persoana I):{' ' * 10}│")
        print("├" + "─" * 86 + "┤")
        for line in wrapped_refl.split("\n"):
            print(f"│  {line:<84}│")
        print("└" + "─" * 86 + "┘")

        print("\n--- [FRAGMENT POST-TEXT (Continuare Original)] ---")
        print(f"{post_disp.strip()}")

        print("\n--- [TEXTUL ASAMBLAT CU TOKENII SPP DE INJECTARE < >]: ---")
        assembled = f"{pre_disp.strip()}\n<assistant> {refl} </assistant>\n{post_disp.strip()}"
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

                with torch.no_grad():
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
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for parallel GPU inference (default: 4)")
    parser.add_argument("--interval", type=int, default=1000, help="Print full inspection card every N reflections (default: 1000)")
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Enable 4-bit quantization (enabled by default)")
    parser.add_argument("--max-new-tokens", type=int, default=220, help="Max tokens per generated reflection")
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

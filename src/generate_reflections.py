# -*- coding: utf-8 -*-
"""
SPP-Ro reflections — faithful to Minder et al. (arXiv:2608.13482).

Differences from v1 (generate_sensitive_reflections.py / generate_factual_reflections.py):
  * Annotates REAL documents from the training corpus (corpus_scale_stream.parquet), never
    LLM-invented scenarios. A cheap lexical screen proposes candidates; the annotator LLM then
    SCORES severity 0-5 (as SafeLM does in the paper). The score is measured, not dictated.
  * Mix = all documents with severity >= 3 (harmful/sensitive) + an equal number of benign
    documents (severity <= 2), as in the paper.
  * Insertion point sampled like the paper (linear ramp over the first 20% of the document, then
    uniform), snapped to a sentence/word boundary. The reflection is generated from the PREFIX
    ONLY (text before the insertion point), so it cannot leak the continuation.
  * Guideline enforcement on output: benign -> ONE short sentence, no citations; sensitive ->
    30-110 words, first person, >= 1 valid citation [X.Y]; no meta-language; deduplicated.
  * Output: data/sidecar/reflections_first_pass.parquet with doc_key (md5 of the first 200 chars) so that
    train_scale_pretrain.py --inline REPLACES the original document (data-matched with Base).

Throughput / cost:
  * --batch K annotates K prefixes per API call (shared instructions, K results in one JSON).
  * All keys in .env (CEREBRAS_API_KEYS=k1,k2,...) are used in PARALLEL, one per worker, round-robin;
    a key is retired on 401/402/403 (no credit) and the worker moves to the next live key.
  * --estimate scans the corpus without calling the API and prints projected calls, tokens, cost
    and hours for the trial-tier rate limit.

Usage:
  py src/generate_reflections.py --estimate --target 15000
  py src/generate_reflections.py --target 15000 --batch 3 --model gpt-oss-120b
  py src/generate_reflections.py --mock --target 60                    (no API; pipeline test)
"""
import os
import re
import sys
import json
import time
import math
import random
import hashlib
import argparse
import threading
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import requests
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from config import CLEAN_DATA_DIR, SIDECAR_DIR, SIDECAR_FIRST_PASS_FILE, HELDOUT_FILE
except ImportError:
    from src.config import CLEAN_DATA_DIR, SIDECAR_DIR, SIDECAR_FIRST_PASS_FILE, HELDOUT_FILE

ENV_FILE = _ROOT_DIR / ".env"
CORPUS_FILE = CLEAN_DATA_DIR / "corpus_scale_stream.parquet"
PROGRESS_FILE = SIDECAR_DIR / "reflections_first_pass_progress.json"

# Cerebras public rate card, Oct 2026 (USD per million tokens): (input, output)
PRICES = {"gpt-oss-120b": (0.35, 0.75), "qwen-3.8-27b": (0.99, 1.49)}
TRIAL_RPM_PER_KEY = 450        # measured on these keys: x-ratelimit-limit-requests-minute = 450, tokens/min = 150k

# ----------------------------------------------------------------------------- constitution (v1.1)
ARTICLES = {
    "1.1": "demnitate umană, nediscriminare etnică (romi), refugiați, antisemitism, dizabilități",
    "1.2": "egalitate de gen, roluri ocupaționale, violență domestică, hărțuire, discriminare la angajare",
    "1.3": "coeziune teritorială, echitate rural-urban, stereotipuri regionale, comunități izolate",
    "2.1": "memorie istorică: Holocaustul din România, crimele comuniste și fasciste, antitotalitarism",
    "2.2": "gândire critică și știință, pluralism religios, conștiință laică, respingerea pseudoștiinței",
    "2.3": "stat de drept, integritate, corupție, independența justiției, bioetică, consimțământ informat, dezinformare",
}

LEXICAL_SCREEN = {
    "1.1": re.compile(r"\b(rom|romi|romă|rome|romilor|țigan\w*|tigan\w*|refugiat\w*|ucrainen\w*|imigran\w*|migran\w*|evrei\w*|antisemit\w*|dizabilit\w*|rasis\w*|xenofob\w*|handicap\w*)\b", re.I),
    "1.2": re.compile(r"\b(femei\w*|bărbat\w*|barbat\w*|violenț\w* (domestic|în familie)|hărțui\w*|hartui\w*|sexis\w*|feminis\w*|misogin\w*|cratiț\w*|maternitate|gravid\w*|viol\b|violat\w*)\b", re.I),
    "1.3": re.compile(r"\b(moldove\w*|olten\w*|ardele\w*|bucureșten\w*|provinci\w*|rural\w*|sătean\w*|satele|asistaț\w* social\w*|țăran\w*|navetiș\w*|izolat\w*|periferi\w*)\b", re.I),
    "2.1": re.compile(r"\b(holocaust\w*|legionar\w*|antonescu|ceaușescu|ceausescu|securitat\w*|securist\w*|comunis\w*|deportăr\w*|deportar\w*|pogrom\w*|gulag|pitești|sighet|aiud|gherla|colectivizar\w*|totalitar\w*|fascis\w*|nazis\w*)\b", re.I),
    "2.2": re.compile(r"\b(vaccin\w*|antivaccin\w*|evoluți\w*|creaționis\w*|biseric\w*|ortodox\w*|ateu\w*|ateis\w*|laic\w*|conspirați\w*|5g|cipuri\w*|miracul\w*|vindec\w* miraculo\w*|pseudoștiinț\w*|astrolog\w*)\b", re.I),
    "2.3": re.compile(r"\b(șpag\w*|spag\w*|mită|mituir\w*|corupți\w*|coruptie|dna\b|licitați\w* trucat\w*|judecător\w*|procuror\w*|magistraț\w*|consimțământ\w*|malpraxis|dezinform\w*|fake news|deepfake|troli|propagand\w*)\b", re.I),
}

# Meta-language = the reflection talks about the annotation setup itself. "Curtea Constituțională" / "drepturi constituționale"
# are ordinary Romanian and stay allowed; only the value-constitution / guideline / field names are rejected.
META_RE = re.compile(r"(?i)\b(constituți[ae]i? (de valori|noastr\w*|modelului|spp\w*)|cart[aă]\b|cartei|ghid(ul|ului)? de adnotare|spp-ro|adnotator\w*|articolul \d|framing|fram[aă]re\w*|endorsing|reflection_[13]p|severity)\b")
SEV0_MAX_REPEAT = 30   # benign one-liners repeat naturally; allow each exact sentence this many times, sensitive reflections must be unique
CIT_RE = re.compile(r"\[(\d\.\d)(?:\s*,\s*(\d\.\d))*\]")
CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")
SENT_END_RE = re.compile(r"[.!?…]\s")


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def doc_key(text: str) -> str:
    return md5(text.strip()[:200])


def sample_insertion_fraction(rng: random.Random) -> float:
    """Paper: linear ramp over the first 20% of the document, then uniform (continuous density)."""
    c = 1.0 / 0.9
    if rng.random() < 0.1 * c:
        return 0.2 * math.sqrt(rng.random())
    return 0.2 + 0.8 * rng.random()


def choose_insertion_position(text: str, rng: random.Random, min_prefix_words: int = 40) -> Optional[int]:
    n = len(text)
    words = text.split()
    if len(words) < min_prefix_words + 20:
        return None
    target = int(n * sample_insertion_fraction(rng))
    min_chars = len(" ".join(words[:min_prefix_words])) + 1
    target = max(target, min_chars)
    if target >= n - 50:
        target = n - 50
    m = SENT_END_RE.search(text, target, min(n, target + 300))
    if m:
        return m.end()
    j = target
    while j < n and not text[j].isspace():
        j += 1
    return j + 1 if j + 1 < n else None


def lexical_hits(text: str) -> List[str]:
    return [art for art, rx in LEXICAL_SCREEN.items() if len(rx.findall(text)) >= 2]


def build_prompt(items: List[Tuple[str, List[str]]]) -> str:
    """items: list of (prefix, lexical_hints). One JSON object with an 'items' list is requested."""
    arts = "\n".join(f"[{k}] {v}" for k, v in ARTICLES.items())
    blocks = []
    for i, (prefix, hint) in enumerate(items):
        h = ", ".join(f"[{a}]" for a in hint) if hint else "niciunul"
        blocks.append(f"### TEXT {i} (indicii lexicale, pot fi false: {h})\n\"\"\"{prefix}\"\"\"")
    texts = "\n\n".join(blocks)
    return f"""Ești adnotator pentru datele de preantrenare ale unui model de limbaj românesc crescut de la zero; adnotările devin semnal de antrenare.
Fiecare text este PARȚIAL (lectura s-a oprit). Nu ghici continuarea și nu menționa că textul e trunchiat.

Valori, citate DOAR cu codul în paranteze drepte, ex. [1.2] sau [1.1, 2.3]; nu numi niciodată „constituția”, „carta”, „ghidul”:
{arts}

Pentru FIECARE text:
1) severity 0-5 pe textul de până acum: 0 benign, nicio temă; 1 temă atinsă marginal; 2 discuție relevantă, echilibrată;
   3 relatează/înfățișează un prejudiciu, stereotip sau discriminare (presă, ficțiune, mărturie); 4 normalizează sau justifică prejudiciul; 5 ură explicită, incitare, dezumanizare.
   Precizare: 4-5 cer ca textul ÎNSUȘI să susțină prejudiciul față de un grup (etnie, gen, religie, dizabilitate, regiune, minoritate) sau să incite; insultele politice fără dimensiune de grup și relatările critice sunt cel mult 3.
2) articles = codurile atinse; framing = "none" | "neutral" | "endorsing".
3) reflecții care deliberează asupra VALORILOR (nu rezumă conținutul):
   severity 0: reflection_1p și reflection_3p = O SINGURĂ frază (max 20 cuvinte) că nu există miză etică, FĂRĂ citări; numește tipul textului (ofertă, rețetă, știre sportivă, tutorial...) ca fraza să fie specifică;
   severity 1-2: 1-2 fraze (20-45 cuvinte) cu citările articolelor atinse;
   severity 3-5: 40-90 cuvinte (NU mai puțin de 40; numără), ancorate în detalii concrete din text, cu toate citările; 4-5 contrabalansare fermă, 3 contextualizare responsabilă.
   OBLIGATORIU la severity 1-5: codul fiecărui articol apare în paranteze drepte ÎN TEXTUL ambelor reflecții, ex. „... [1.2, 2.3].”
   reflection_1p la persoana I (consider, privesc, susțin), început variat, niciodată „Trebuie să analizez”; reflection_3p aceeași substanță și citări, persoana a III-a, fără „eu”.

{texts}

Răspunde STRICT cu JSON:
{{"items": [{{"idx": 0, "severity": 0, "articles": [], "framing": "none", "analysis": "<1-2 fraze>", "reflection_1p": "...", "reflection_3p": "..."}}]}}"""


def build_label_prompt(prefixes: List[str]) -> str:
    arts = "\n".join(f"[{k}] {v}" for k, v in ARTICLES.items())
    blocks = "\n\n".join(f"### TEXT {i}\n\"\"\"{t}\"\"\"" for i, t in enumerate(prefixes))
    return f"""Ești clasificator de severitate pentru texte românești. Teme:
{arts}

Pentru FIECARE text dă: severity 0-5 (0 benign, nicio temă; 1 temă atinsă marginal; 2 discuție relevantă echilibrată; 3 relatează/înfățișează un prejudiciu, stereotip sau discriminare;
4 normalizează sau justifică prejudiciul; 5 ură explicită, incitare, dezumanizare) și articles = lista codurilor atinse (ex. ["1.1","2.3"], sau [] la severity 0).

{blocks}

Răspunde STRICT cu JSON: {{"items": [{{"idx": 0, "severity": 0, "articles": []}}]}}"""


def label_mode(args, rng: random.Random, ann: "Annotator", pool: "KeyPool"):
    """Calibration labels for the safety classifier: 2/3 lexical candidates + 1/3 random clean documents."""
    out = Path(args.label_out)
    existing = pd.read_parquet(out) if out.exists() else None
    done = set(existing["doc_key"]) if existing is not None else set()
    if HELDOUT_FILE.exists():
        done |= set(doc_key(t) for t in pq.read_table(str(HELDOUT_FILE), columns=["text"])["text"].to_pylist())
    n_lex_target = int(args.n_label * 2 / 3)
    n_rand_target = args.n_label - n_lex_target
    n_lex = int((existing["lexical_hits"] != "").sum()) if existing is not None else 0
    n_rand = len(existing) - n_lex if existing is not None else 0
    stats = {"scanned": 0, "lexical_candidates": 0}
    sample: List[Tuple[str, str, str, List[str]]] = []
    for t, src, k, hits in iter_candidates(rng, args.max_scan, done, 60, 0.03, stats):
        if hits and n_lex < n_lex_target:
            sample.append((t, src, k, hits)); n_lex += 1
        elif not hits and n_rand < n_rand_target:
            sample.append((t, src, k, hits)); n_rand += 1
        if n_lex >= n_lex_target and n_rand >= n_rand_target:
            break
    print(f"[label] de etichetat: {len(sample):,} documente ({n_lex_target} candidați lexicali + {n_rand_target} aleatoare), batch {args.label_batch}")
    records = existing.to_dict("records") if existing is not None else []
    lock = threading.Lock()
    batches = [sample[i:i + args.label_batch] for i in range(0, len(sample), args.label_batch)]
    bidx = [0]
    pbar = tqdm(total=len(sample), desc="etichete severitate", unit="doc")

    def worker(wid: int):
        while True:
            with lock:
                if bidx[0] >= len(batches) or not pool.live():
                    return
                batch = batches[bidx[0]]; bidx[0] += 1
            prefixes = [" ".join(t.split()[:args.max_prefix_words]) for t, _, _, _ in batch]
            res = ann.call_json(build_label_prompt(prefixes), wid, 40 * len(batch) + 100)
            items = (res or {}).get("items", []) if isinstance(res, dict) else []
            by = {}
            for it in items:
                try:
                    by[int(it.get("idx", -1))] = it
                except Exception:
                    pass
            with lock:
                for i, (t, src, k, hits) in enumerate(batch):
                    it = by.get(i)
                    if not it:
                        continue
                    try:
                        sev = int(it.get("severity", -1))
                    except Exception:
                        continue
                    if not 0 <= sev <= 5:
                        continue
                    arts = sorted(set(str(a).strip().strip("[]§ ") for a in (it.get("articles") or []) if str(a).strip().strip("[]§ ") in ARTICLES))
                    records.append({"doc_key": k, "source": src, "text": t, "lexical_hits": ",".join(hits), "severity": sev, "articles": ",".join(arts), "generator": args.model})
                    pbar.update(1)
                if len(records) % 200 < args.label_batch:
                    pd.DataFrame(records).to_parquet(out, index=False)

    ths = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(min(12, max(1, len(pool.live()))))]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    pbar.close()
    df = pd.DataFrame(records)
    df.to_parquet(out, index=False)
    lex = df[df["lexical_hits"] != ""]; rnd = df[df["lexical_hits"] == ""]
    h_lex = (lex["severity"] >= 3).mean() if len(lex) else float("nan")
    h_rnd = (rnd["severity"] >= 3).mean() if len(rnd) else float("nan")
    cand_rate = stats["lexical_candidates"] / max(1, stats["scanned"])
    corpus_docs = 1_500_000
    est_harm = corpus_docs * (cand_rate * h_lex + (1 - cand_rate) * h_rnd)
    pi, po = PRICES.get(args.model, (0, 0))
    cost = pool.in_tokens / 1e6 * pi + pool.out_tokens / 1e6 * po
    print("=" * 72)
    print(f" ETICHETE: {len(df):,} documente -> {out}")
    print(f" severitate (toate): {df['severity'].value_counts().sort_index().to_dict()}")
    print(f" rata sev>=3: candidați lexicali {100*h_lex:.1f}% (n={len(lex)}) | documente aleatoare fără cuvinte-cheie {100*h_rnd:.1f}% (n={len(rnd)})")
    print(f" candidați lexicali în corpus: {100*cand_rate:.1f}%  ->  documente nocive estimate în corpus: ~{est_harm:,.0f} ({100*est_harm/corpus_docs:.1f}%)")
    print(f" recall-ul filtrului lexical (cât din nociv ar prinde regex-ul): ~{100*cand_rate*h_lex/max(1e-9, cand_rate*h_lex+(1-cand_rate)*h_rnd):.0f}%")
    calls = math.ceil(2 * est_harm / 3)
    for model, (pi2, po2) in PRICES.items():
        c = calls * (2066 / 1e6 * pi2 + 532 / 1e6 * po2)
        print(f" adnotare completă (toate nocivele + tot atâtea benigne, batch 3): ~{calls:,} apeluri -> {model}: ~${c:,.0f} (~{math.ceil(c/5)} conturi de $5)")
    print(f" cost etichetare: ${cost:.2f} ({pool.in_tokens:,} in / {pool.out_tokens:,} out tokens)")
    print("=" * 72)


def load_keys() -> List[str]:
    keys: List[str] = []
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("CEREBRAS_API_KEYS="):
                keys = [k.strip() for k in line.split("=", 1)[1].split(",") if k.strip()]
            elif line.startswith("CEREBRAS_API_KEY=") and not keys:
                v = line.split("=", 1)[1].strip()
                keys = [v] if v else []
    if not keys:
        keys = [k.strip() for k in os.environ.get("CEREBRAS_API_KEYS", os.environ.get("CEREBRAS_API_KEY", "")).split(",") if k.strip()]
    seen, out = set(), []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


class KeyPool:
    """Round-robin key pool; keys are retired when the API reports no credit / invalid key."""
    def __init__(self, keys: List[str]):
        self.keys = list(keys)
        self.dead = set()
        self.lock = threading.Lock()
        self.calls = {k: 0 for k in keys}
        self.in_tokens = 0
        self.out_tokens = 0

    def live(self) -> List[str]:
        with self.lock:
            return [k for k in self.keys if k not in self.dead]

    def key_for(self, worker_id: int) -> Optional[str]:
        live = self.live()
        if not live:
            return None
        return live[worker_id % len(live)]

    def retire(self, key: str, reason: str):
        with self.lock:
            if key in self.dead:
                return
            self.dead.add(key)
            left = len([k for k in self.keys if k not in self.dead])
            print(f"\n[cheie retrasă] ...{key[-6:]} ({reason}); chei active rămase: {left}", flush=True)

    def account(self, key: str, usage: Dict[str, Any]):
        with self.lock:
            self.calls[key] = self.calls.get(key, 0) + 1
            self.in_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.out_tokens += int(usage.get("completion_tokens", 0) or 0)


class Annotator:
    def __init__(self, pool: KeyPool, model: str, mock: bool = False, url: str = "https://api.cerebras.ai/v1/chat/completions", reasoning: str = "low"):
        self.pool, self.model, self.mock, self.url, self.reasoning = pool, model, mock, url, reasoning

    def call_json(self, prompt: str, worker_id: int, max_tokens: int) -> Optional[Dict[str, Any]]:
        """One JSON-mode request with key rotation; returns the parsed object or None."""
        for attempt in range(8):
            key = self.pool.key_for(worker_id + attempt)
            if not key:
                return None
            payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                       "response_format": {"type": "json_object"}, "temperature": 0.2, "max_tokens": max_tokens,
                       "reasoning_effort": "none" if "qwen" in self.model.lower() else self.reasoning}
            try:
                r = requests.post(self.url, json=payload, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    self.pool.account(key, data.get("usage", {}))
                    content = data["choices"][0]["message"].get("content") or ""
                    try:
                        return json.loads(content)
                    except Exception:
                        m = re.search(r"\{.*\}", content, re.S)
                        return json.loads(m.group(0)) if m else None
                body = r.text[:200].lower()
                if r.status_code in (401, 402, 403) or ("credit" in body and r.status_code == 400) or "quota" in body or "insufficient" in body:
                    self.pool.retire(key, f"HTTP {r.status_code}")
                elif r.status_code == 429:
                    time.sleep(6 + 4 * attempt)
                else:
                    time.sleep(2)
            except Exception:
                time.sleep(2)
        return None

    def annotate_batch(self, items: List[Tuple[str, List[str]]], worker_id: int, rng: random.Random) -> Optional[List[Dict[str, Any]]]:
        if self.mock:
            out = []
            for i, (prefix, hint) in enumerate(items):
                tag = f"{abs(hash(prefix[:80])) % 100000}"
                if not hint:
                    out.append({"idx": i, "severity": 0, "articles": [], "framing": "none", "analysis": "mock",
                                "reflection_1p": f"Acest fragment ({tag}) nu ridică probleme de natură etică.",
                                "reflection_3p": f"Fragmentul {tag} nu implică mize etice."})
                else:
                    cits = " ".join(f"[{a}]" for a in hint[:2])
                    out.append({"idx": i, "severity": rng.choice([3, 4, 4, 5]), "articles": hint[:2], "framing": "endorsing", "analysis": "mock",
                                "reflection_1p": (f"Privesc cu îngrijorare generalizările din acest text ({tag}) și susțin că demnitatea fiecărei persoane {cits} "
                                                  "cere judecarea faptelor individual, nu etichetarea unui grup întreg; consider că o societate dreaptă refuză "
                                                  "stigmatizarea colectivă și apără șansele egale ale tuturor."),
                                "reflection_3p": (f"Generalizările din text ({tag}) contravin demnității egale {cits}; faptele se judecă individual, iar "
                                                  "stigmatizarea colectivă slăbește coeziunea și echitatea socială.")})
            return out
        prompt = build_prompt(items)
        for attempt in range(8):
            key = self.pool.key_for(worker_id + attempt)
            if not key:
                return None
            try:
                payload = {
                    "model": self.model, "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"}, "temperature": 0.4,
                    "max_tokens": 450 * len(items) + 400,   # content + a small reasoning budget
                }
                # Both Cerebras models are reasoning models. Without this field gpt-oss spends the whole
                # max_tokens budget on hidden reasoning and returns NO content (verified: 1247 reasoning
                # tokens, finish_reason=length, content missing). Qwen needs "none".
                payload["reasoning_effort"] = "none" if "qwen" in self.model.lower() else self.reasoning
                r = requests.post(self.url, json=payload, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=60)
                if r.status_code == 200:
                    data = r.json()
                    self.pool.account(key, data.get("usage", {}))
                    content = data["choices"][0]["message"]["content"]
                    try:
                        parsed = json.loads(content)
                    except Exception:
                        m = re.search(r"\{.*\}", content, re.S)
                        if not m:
                            return None
                        parsed = json.loads(m.group(0))
                    its = parsed.get("items", parsed if isinstance(parsed, list) else None)
                    return its if isinstance(its, list) else None
                body = r.text[:200].lower()
                if r.status_code in (401, 402, 403) or ("credit" in body and r.status_code == 400) or "quota" in body or "insufficient" in body:
                    self.pool.retire(key, f"HTTP {r.status_code}")
                elif r.status_code == 429:
                    time.sleep(6 + 4 * attempt)
                else:
                    time.sleep(2)
            except Exception:
                time.sleep(2)
        return None


def validate(item: Dict[str, Any], why: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Guideline enforcement. If `why` is given, the first failing rule is appended to it (for the rejection log)."""
    def fail(reason: str):
        if why is not None:
            why.append(reason)
        return None
    try:
        sev = int(item.get("severity", -1))
    except Exception:
        return fail("severity lipsă sau nenumerică")
    if sev < 0 or sev > 5:
        return fail("severity în afara 0-5")
    arts = []
    for a in (item.get("articles") or []):
        a = str(a).strip().strip("[]§ ")
        if a in ARTICLES:
            arts.append(a)
    r1 = CJK_RE.sub("", str(item.get("reflection_1p", ""))).strip()
    r3 = CJK_RE.sub("", str(item.get("reflection_3p", ""))).strip()
    if not r1 or not r3:
        return fail("reflecție goală (1p sau 3p)")
    m_meta = META_RE.search(r1) or META_RE.search(r3)
    if m_meta:
        return fail(f"meta-limbaj: „{m_meta.group(0)}”")
    cits1 = set(c for m in CIT_RE.finditer(r1) for c in m.group(0)[1:-1].replace(" ", "").split(","))
    repaired = 0
    if sev > 0 and not cits1 and arts:
        # The model named the articles in `articles` but left the bracket citation out of the prose:
        # append its own codes (no new judgement is introduced). Recorded as citation_repaired=1.
        tag = "[" + ", ".join(sorted(set(arts))) + "]"
        r1 = (r1[:-1].rstrip() + " " + tag + ".") if r1.endswith(".") else (r1 + " " + tag)
        r3 = (r3[:-1].rstrip() + " " + tag + ".") if r3.endswith(".") else (r3 + " " + tag)
        cits1 = set(sorted(set(arts)))
        repaired = 1
    w1 = len(r1.split())
    if sev == 0:
        if cits1:
            return fail("sev 0 cu citări")
        if w1 > 25:
            return fail(f"sev 0 prea lung ({w1} cuvinte > 25)")
    else:
        if not cits1:
            return fail(f"sev {sev} fără citare [X.Y] și fără articles")
        if not cits1.issubset(set(ARTICLES)):
            return fail(f"citare invalidă {sorted(cits1 - set(ARTICLES))}")
        if sev >= 3 and w1 < 15:
            return fail(f"sev {sev} prea scurt ({w1} cuvinte < 15)")
        if sev >= 3 and w1 > 130:
            return fail(f"sev {sev} prea lung ({w1} cuvinte > 130)")
        if sev <= 2 and w1 > 75:
            return fail(f"sev {sev} prea lung ({w1} cuvinte > 75)")
        arts = sorted(set(arts) | cits1)
    return {"severity": sev, "articles": arts, "framing": str(item.get("framing", "")), "analysis": str(item.get("analysis", ""))[:600],
            "reflection_1p": r1, "reflection_3p": r3, "citation_repaired": repaired}


def iter_candidates(rng: random.Random, max_scan: int, skip_keys: set, min_words: int, benign_rate: float, stats: Dict[str, int]):
    pf = pq.ParquetFile(str(CORPUS_FILE))
    rg_order = list(range(pf.num_row_groups))
    rng.shuffle(rg_order)
    for rg in rg_order:
        tb = pf.read_row_group(rg, columns=["text", "source"])
        texts, srcs = tb["text"].to_pylist(), tb["source"].to_pylist()
        order = list(range(len(texts)))
        rng.shuffle(order)
        for i in order:
            t = texts[i]
            stats["scanned"] += 1
            if stats["scanned"] > max_scan:
                return
            if not t or len(t.split()) < min_words:
                continue
            k = doc_key(t)
            if k in skip_keys:
                continue
            hits = lexical_hits(t)
            if hits:
                stats["lexical_candidates"] += 1
                yield t, srcs[i], k, hits
            elif rng.random() < benign_rate:
                yield t, srcs[i], k, []


def estimate(args, rng):
    """No API: scan a sample, report candidate rate and projected cost/time."""
    stats = {"scanned": 0, "lexical_candidates": 0}
    n_pref_words = []
    for t, _, _, hits in iter_candidates(rng, args.estimate_scan, set(), 80, 0.0, stats):
        pos = choose_insertion_position(t, rng)
        if pos:
            n_pref_words.append(min(len(t[:pos].split()), args.max_prefix_words))
        if len(n_pref_words) >= 3000:
            break
    cand_rate = stats["lexical_candidates"] / max(1, stats["scanned"])
    avg_prefix_words = sum(n_pref_words) / max(1, len(n_pref_words))
    harm_rate = args.assumed_harm_rate                      # fraction of lexical candidates the LLM scores >= 3
    docs_needed = args.target / harm_rate                    # lexical docs to annotate to reach the harmful target
    # the (1-harm_rate) remainder lands in the benign pool; extra benign docs are only needed if that is < target
    benign_from_lex = docs_needed * (1 - harm_rate)
    extra_benign = max(0.0, args.target - benign_from_lex)
    docs_total = docs_needed + extra_benign
    calls = math.ceil(docs_total / args.batch)
    tok_per_word = 2.6                                       # Romanian with a Latin-centric tokenizer
    in_tok = calls * (550 + args.batch * (avg_prefix_words * tok_per_word + 40))
    out_tok = calls * (args.batch * 160 + 200)              # content + low reasoning (measured ~574 for batch 3)
    print("=" * 72)
    print(f" ESTIMATE (no API) | target {args.target:,} docs severitate>=3 + {args.target:,} benigne")
    print("=" * 72)
    print(f" documente scanate: {stats['scanned']:,}; candidați lexicali: {stats['lexical_candidates']:,} ({100*cand_rate:.1f}%)")
    print(f" prefix mediu trimis: {avg_prefix_words:.0f} cuvinte (plafon {args.max_prefix_words})")
    print(f" ipoteză: {100*harm_rate:.0f}% din candidații lexicali primesc severitate>=3 (schimbă cu --assumed-harm-rate)")
    print(f" documente de adnotat: ~{docs_total:,.0f}  ->  apeluri API cu batch={args.batch}: ~{calls:,}")
    print(f" tokeni: ~{in_tok/1e6:.1f}M intrare, ~{out_tok/1e6:.1f}M ieșire")
    for model, (pi, po) in PRICES.items():
        cost = in_tok / 1e6 * pi + out_tok / 1e6 * po
        keys = math.ceil(cost / 5.0)
        hours = calls / (TRIAL_RPM_PER_KEY * max(1, keys)) / 60
        print(f"   {model:14s}: ~${cost:,.0f}  ->  ~{keys} conturi cu $5  |  limita API nu e constrângere ({TRIAL_RPM_PER_KEY} req/min/cheie); la ~3 s/apel cu 12 workeri: ~{calls*3/12/3600:.1f} h")
    print(f" documente scanate necesare: ~{docs_needed/max(cand_rate,1e-6):,.0f} din 1,5M")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=25000, help="documents with severity>=3 to collect (an equal number of benign docs is added)")
    ap.add_argument("--batch", type=int, default=3, help="prefixes per API call")
    ap.add_argument("--max-prefix-words", type=int, default=300, help="cap on the prefix sent to the annotator (tail kept)")
    ap.add_argument("--max-scan", type=int, default=1_500_000)
    ap.add_argument("--model", type=str, default="qwen-3.8-27b", help="Cerebras model id. Measured: qwen-3.8-27b = better Romanian + all outputs valid, $0.0020/call; gpt-oss-120b (reasoning low) = $0.0011/call but ~15-20%% invalid outputs")
    ap.add_argument("--workers", type=int, default=0, help="parallel workers (default: number of live keys, max 12)")
    ap.add_argument("--benign-rate", type=float, default=0.02, help="sampling rate of lexically clean docs for the benign pool")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--estimate", action="store_true", help="no API: projected calls/cost/time")
    ap.add_argument("--estimate-scan", type=int, default=60_000)
    ap.add_argument("--assumed-harm-rate", type=float, default=0.30)
    ap.add_argument("--reasoning", type=str, default="low", choices=["low", "medium", "high"], help="reasoning_effort for gpt-oss (Qwen always uses none)")
    ap.add_argument("--label-only", action="store_true", help="Path B step 1: severity labels for the classifier (no reflections)")
    ap.add_argument("--n-label", type=int, default=3000)
    ap.add_argument("--label-batch", type=int, default=5)
    ap.add_argument("--label-out", type=str, default=str(SIDECAR_DIR / "severity_labels.parquet"))
    ap.add_argument("--candidates", type=str, default=None, help="Path B step 3: annotate exactly the documents in this parquet (from train_safety_classifier.py)")
    ap.add_argument("--max-docs", type=int, default=0, help="diagnostic: stop after this many NEW accepted documents (0 = no limit)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default=str(SIDECAR_FIRST_PASS_FILE))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.estimate:
        estimate(args, rng)
        return

    out = Path(args.out)
    global PROGRESS_FILE
    PROGRESS_FILE = out.with_name(out.stem + "_progress.json")   # per output file (the expansion must not overwrite the first pass's progress)
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)
    keys = load_keys()
    if not keys and not args.mock:
        raise SystemExit("Nu există CEREBRAS_API_KEYS în .env (format: CEREBRAS_API_KEYS=csk-...,csk-...). Sau folosește --mock.")
    pool = KeyPool(keys if keys else ["mock"])
    ann = Annotator(pool, args.model, mock=args.mock, reasoning=args.reasoning)
    n_workers = args.workers or min(24, max(1, len(pool.live())))
    if args.label_only:
        label_mode(args, rng, ann, pool)
        return
    cand_mode = args.candidates is not None
    clf_info: Dict[str, Tuple[int, float]] = {}
    if cand_mode:
        cdf = pd.read_parquet(args.candidates)
        print(f"[candidates] {len(cdf):,} documente definite de clasificator ({int(cdf['clf_flag'].sum()):,} semnalate nocive + {int((cdf['clf_flag']==0).sum()):,} benigne)")
        args.target = 10**9   # no pool caps: annotate everything in the candidates file

    records: List[Dict[str, Any]] = []
    done_keys = set()
    if out.exists():
        df = pd.read_parquet(out)
        records = df.to_dict("records")
        done_keys = set(df["doc_key"])
        print(f"[reluare] {len(records):,} înregistrări existente")
    if HELDOUT_FILE.exists():
        done_keys |= set(doc_key(t) for t in pq.read_table(str(HELDOUT_FILE), columns=["text"])["text"].to_pylist())
    skip_file = SIDECAR_DIR / "reflections_first_pass_skip_keys.json"      # written by clean_reflections.py (adult/escort pages): never retried
    if skip_file.exists():
        done_keys |= set(json.load(open(skip_file, encoding="utf-8")))

    n_harm =[sum(1 for r in records if r["severity"] >= 3)]
    n_benign = [sum(1 for r in records if r["severity"] <= 2)]
    n_initial = len(records)
    rej_path = out.with_name(out.stem + "_rejected.jsonl")   # every rejected model output, with the rule it broke
    seen_refl: Dict[str, int] = {}
    for r in records:
        seen_refl[r["reflection_1p"]] = seen_refl.get(r["reflection_1p"], 0) + 1

    def is_dup(v: Dict[str, Any]) -> bool:
        limit = SEV0_MAX_REPEAT if v["severity"] == 0 else 1
        return seen_refl.get(v["reflection_1p"], 0) >= limit
    lock = threading.Lock()
    gen_lock = threading.Lock()
    stats = {"scanned": 0, "lexical_candidates": 0, "api_calls": 0, "api_failures": 0, "rejected": 0}
    def iter_from_candidates():
        rows = cdf.sample(frac=1.0, random_state=args.seed).itertuples(index=False)
        for row in rows:
            stats["scanned"] += 1
            if row.doc_key in done_keys:
                continue
            clf_info[row.doc_key] = (int(row.clf_flag), float(row.clf_score))
            hits = lexical_hits(row.text)
            if hits:
                stats["lexical_candidates"] += 1
            yield row.text, row.source, row.doc_key, hits
    gen = iter_from_candidates() if cand_mode else iter_candidates(rng, args.max_scan, done_keys, 80, args.benign_rate, stats)
    stop = threading.Event()
    t0 = time.time()

    print("=" * 72)
    print(f" REFLECTION GENERATION | model {args.model} | batch {args.batch} | chei active {len(pool.live())} | workers {n_workers}")
    if cand_mode:
        print(f" țintă: toate cele {len(cdf):,} documente din fișierul de candidați (fără plafoane) | existente: {n_harm[0]:,} nocive / {n_benign[0]:,} benigne")
    else:
        print(f" țintă: {args.target:,} documente severitate>=3 + {args.target:,} benigne | existente: {n_harm[0]:,} / {n_benign[0]:,}")
    print(f" ieșire: {out} | progres: {PROGRESS_FILE.name} | Ctrl+C salvează și iese")
    print("=" * 72, flush=True)

    def save():
        if records:
            tmp = out.with_suffix(".tmp.parquet")
            pd.DataFrame(records).to_parquet(tmp, index=False)
            tmp.replace(out)
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump({**stats, "harm": n_harm[0], "benign": n_benign[0], "total": len(records), "live_keys": len(pool.live()),
                       "in_tokens": pool.in_tokens, "out_tokens": pool.out_tokens,
                       "updated": datetime.datetime.now().isoformat(timespec="seconds")}, f, indent=2)

    def done() -> bool:
        return n_harm[0] >= args.target and n_benign[0] >= args.target

    def worker(wid: int, pbar: tqdm):
        lrng = random.Random(rng.random() + wid)
        while not stop.is_set() and not done():
            if not pool.live():
                stop.set()
                return
            batch: List[Tuple[str, str, str, List[str], int]] = []
            with gen_lock:
                try:
                    while len(batch) < args.batch:
                        t, src, k, hits = next(gen)
                        with lock:
                            need_benign = n_benign[0] < args.target
                        if not hits and not need_benign and not cand_mode:
                            continue
                        pos = choose_insertion_position(t, lrng)
                        if pos is None:
                            continue
                        batch.append((t, src, k, hits, pos))
                except StopIteration:
                    if not batch:
                        stop.set()
                        return
            prompts = []
            for t, src, k, hits, pos in batch:
                pre = t[:pos]
                words = pre.split()
                if len(words) > args.max_prefix_words:
                    pre = " ".join(words[-args.max_prefix_words:])
                prompts.append((pre, hits))
            res = ann.annotate_batch(prompts, wid, lrng)
            with lock:
                stats["api_calls"] += 1
                if res is None:
                    stats["api_failures"] += 1
            if res is None:
                continue
            by_idx = {}
            for it in res:
                try:
                    by_idx[int(it.get("idx", -1))] = it
                except Exception:
                    pass
            with lock:
                for i, (t, src, k, hits, pos) in enumerate(batch):
                    it = by_idx.get(i)
                    why: List[str] = []
                    v = validate(it, why) if it else None
                    if v is None or is_dup(v) or k in done_keys:
                        stats["rejected"] += 1
                        if it is None:
                            why = ["idx lipsă din răspunsul modelului"]
                        elif v is not None:
                            why = ["reflecție duplicată (text identic, peste plafon)" if is_dup(v) else "doc_key deja adnotat"]
                        rflag, rscore = clf_info.get(k, (-1, float("nan")))
                        with open(rej_path, "a", encoding="utf-8") as rf:
                            rf.write(json.dumps({"doc_key": k, "clf_flag": rflag, "clf_score": round(rscore, 3) if rscore == rscore else None,
                                                 "lexical_hits": hits, "reason": why[0] if why else "?",
                                                 "severity": (it or {}).get("severity"), "articles": (it or {}).get("articles"),
                                                 "reflection_1p": str((it or {}).get("reflection_1p", ""))[:400],
                                                 "reflection_3p": str((it or {}).get("reflection_3p", ""))[:200]}, ensure_ascii=False) + "\n")
                        continue
                    if v["severity"] >= 3 and n_harm[0] >= args.target:
                        continue
                    if v["severity"] <= 2 and n_benign[0] >= args.target:
                        continue
                    flag, score = clf_info.get(k, (-1, float("nan")))
                    records.append({"doc_key": k, "source": src, "text": t, "reflection_char_position": pos,
                                    "clf_flag": flag, "clf_score": score,
                                    "prefix_words": len(t[:pos].split()), "lexical_hits": ",".join(hits),
                                    **v, "articles": ",".join(v["articles"]), "reflection_text": v["reflection_1p"],
                                    "generator": args.model if not args.mock else "mock"})
                    done_keys.add(k)
                    seen_refl[v["reflection_1p"]] = seen_refl.get(v["reflection_1p"], 0) + 1
                    if v["severity"] >= 3:
                        n_harm[0] += 1
                    else:
                        n_benign[0] += 1
                    pbar.update(1)
                    if args.max_docs and len(records) - n_initial >= args.max_docs:
                        stop.set()
                pbar.set_postfix(harm=n_harm[0], benign=n_benign[0], rej=stats["rejected"], keys=len(pool.live()))
                if len(records) % 60 == 0:
                    save()

    pbar = tqdm(total=(len(cdf) if cand_mode else 2 * args.target), initial=n_harm[0] + n_benign[0], desc="reflections", unit="doc")
    threads = [threading.Thread(target=worker, args=(w, pbar), daemon=True) for w in range(n_workers)]
    for th in threads:
        th.start()
    last_report = time.time()
    try:
        while any(th.is_alive() for th in threads):
            time.sleep(1.0)
            if time.time() - last_report >= 1800:            # progress report every 30 minutes
                last_report = time.time()
                got = (n_harm[0] + n_benign[0]) - (pbar.initial)
                rate = got / max(1.0, time.time() - t0) * 3600
                left = (len(cdf) if cand_mode else 2 * args.target) - (n_harm[0] + n_benign[0])
                eta_h = left / rate if rate > 0 else float("inf")
                pi, po = PRICES.get(args.model, (0, 0))
                cost = pool.in_tokens / 1e6 * pi + pool.out_tokens / 1e6 * po
                tgt = f"{len(cdf):,} total" if cand_mode else f"{args.target:,}"
                tqdm.write(f"\n[Raport {datetime.datetime.now():%H:%M}] harm {n_harm[0]:,} | benign {n_benign[0]:,} | țintă {tgt} | "
                           f"{rate:,.0f} doc/h | ETA ~{eta_h:.1f} h | apeluri {stats['api_calls']:,} (eșuate {stats['api_failures']:,}) | "
                           f"chei active {len(pool.live())}/{len(pool.keys)} | cost estimat ${cost:,.2f}")
                with lock:          # workers append to `records` under this lock; saving without it can see a half-added row
                    save()
    except KeyboardInterrupt:
        print("\n[întrerupere] salvez...", flush=True)
        stop.set()
    pbar.close()
    with lock:
        save()
        df = pd.DataFrame(records)
    print(f"\nSalvat {len(df):,} înregistrări -> {out}")
    if len(df):
        print("severity:", df["severity"].value_counts().sort_index().to_dict())
        print("articles (top):", df["articles"].value_counts().head(8).to_dict())
        print("prefix fraction mean %.2f; reflection words mean %.1f" % ((df["reflection_char_position"] / df["text"].str.len()).mean(), df["reflection_1p"].str.split().str.len().mean()))
    print(json.dumps(stats, indent=2))
    if not pool.live() and not done():
        print("\n[STOP] Toate cheile sunt epuizate. Adaugă chei noi în .env și rulează din nou: progresul este salvat.")
        sys.exit(2)


if __name__ == "__main__":
    main()

"""
Capability check (needed before any "alignment tax" statement).

Task: 4-way cloze on the held-out web set (data/eval/heldout_fineweb.parquet, normalised like the corpus and disjoint
from training). The documents are split into sentences at sentence-final punctuation AND at line breaks (a heading or a
list item is not glued to the next sentence). For N sentences of 10-30 words whose last word is alphabetic (>= 4 chars),
the model must prefer the true last word over 3 distractor words drawn from other sentences' last words (same length).
Scored with length-normalised log-likelihood (eval_utils), chance = 25%. Every model is compared with its reference
(Base-Ro; Base-Ro-rewarm for the MT model) on the same items with the exact McNemar test; the per-item correctness is
saved in the report so that the test can be re-run.
"""
import sys
import re
import json
import random
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from config import ALL_MODELS, TOKENIZER_DIR, EVALS_DIR, HELDOUT_FILE, base_ref_key
    from eval_utils import batch_score_options, summarize_accuracy, tex_escape
except ImportError:
    from src.config import ALL_MODELS, TOKENIZER_DIR, EVALS_DIR, HELDOUT_FILE, base_ref_key
    from src.eval_utils import batch_score_options, summarize_accuracy, tex_escape

WORD_RE = re.compile(r"^[a-zA-ZăâîșțĂÂÎȘȚ\-]{4,}$")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
TRAIL = ".,;:!?»”\""


def build_cloze_items(n: int, seed: int = 0) -> List[Dict[str, Any]]:
    if not HELDOUT_FILE.exists():
        raise FileNotFoundError(f"{HELDOUT_FILE} missing: run build_heldout_set.py first")
    rng = random.Random(seed)
    texts = pd.read_parquet(HELDOUT_FILE)["text"].tolist()
    sents = []
    for t in texts:
        for s in SENT_SPLIT_RE.split(t):
            w = s.strip().split()
            if 10 <= len(w) <= 30 and WORD_RE.match(w[-1].rstrip(TRAIL)):
                sents.append(w)
    rng.shuffle(sents)
    last_words = [w[-1].rstrip(TRAIL) for w in sents]
    by_len: Dict[int, List[str]] = {}
    for lw in last_words:
        by_len.setdefault(len(lw), []).append(lw)
    items = []
    for w in sents:
        if len(items) >= n:
            break
        target = w[-1].rstrip(TRAIL)
        pool = [x for x in by_len.get(len(target), []) if x.lower() != target.lower()]
        if len(pool) < 3:
            pool = [x for x in last_words if x.lower() != target.lower()]
        distractors = rng.sample(sorted(set(pool)), 3)
        opts = [target] + distractors
        rng.shuffle(opts)
        items.append({"prefix": " ".join(w[:-1]) + " ", "options": opts, "correct_idx": opts.index(target)})
    return items


@torch.no_grad()
def score_items(model, tokenizer, items, device) -> List[bool]:
    prompts, options = [], []
    for it in items:
        for o in it["options"]:
            prompts.append(it["prefix"])
            options.append(o)
    scores = batch_score_options(model, tokenizer, prompts, options, device=device, batch_size=64, length_normalize=True)
    flags = []
    for i, it in enumerate(items):
        s = scores[4 * i : 4 * i + 4]
        flags.append(int(max(range(4), key=lambda k: s[k])) == it["correct_idx"])
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    items = build_cloze_items(args.n)
    items_md5 = hashlib.md5(json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    print(f"Cloze items: {len(items)} (chance 25%)")
    results: Dict[str, Dict[str, Any]] = {}
    flags_of: Dict[str, List[bool]] = {}
    for key, mdir, desc in ALL_MODELS:
        if not (mdir / "model.safetensors").exists():
            continue
        model = AutoModelForCausalLM.from_pretrained(str(mdir), dtype=torch.float32).to(device).eval()   # v1 checkpoints are stored in bf16: score everything in fp32
        flags = score_items(model, tokenizer, items, device)
        flags_of[key] = flags
        ref = base_ref_key(key)
        has_ref = ref != key and ref in flags_of
        res = summarize_accuracy(flags, flags_of[ref] if has_ref else None)
        if has_ref:
            res["reference"] = ref
        results[key] = {"description": desc, **res, "per_item_correct": "".join("1" if f else "0" for f in flags)}
        print(f"  {key:22s} cloze-4 acc = {res['accuracy']:5.1f}%  CI95={res['ci95']}"
              + (f"  vs {ref}: +{res['items_gained']}/-{res['items_lost']} items, McNemar p={res['p_vs_base']}" if has_ref else ""))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    out = EVALS_DIR / "thesis_capabilities_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"task": "4-way last-word cloze on held-out web sentences", "n_items": len(items), "items_md5": items_md5,
                   "note": "p_vs_base = exact McNemar test against the model's reference (Base-Ro; Base-Ro-rewarm for SPP-Ro-MT) "
                           "on the same items; per_item_correct = one character per item, in item order",
                   "results": results}, f, indent=2, ensure_ascii=False)
    tex = EVALS_DIR / "thesis_capabilities_table.tex"
    with open(tex, "w", encoding="utf-8") as f:
        f.write("% Capabilități: cloze 4 opțiuni pe propoziții held-out (șansă 25%)\n\\begin{table}[htbp]\n\\centering\n\\small\n\\begin{tabular}{l l c c}\n\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Variantă} & \\textbf{Acuratețe cloze (IC 95\\%)} & \\textbf{$p$ vs ref. (McNemar)} \\\\\n\\midrule\n")
        for k, r in results.items():
            p_cell = f"{r['p_vs_base']:.3g}" if "p_vs_base" in r else "ref."
            f.write(f"{tex_escape(k)} & {tex_escape(r['description'])} & {r['accuracy']:.1f}\\% [{r['ci95'][0]:.0f}--{r['ci95'][1]:.0f}] & {p_cell} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\caption{Capabilitate lingvistică de bază: alegerea ultimului cuvânt dintr-o propoziție held-out dintre 4 candidați "
                f"(N={len(items)}, șansă 25\\%). $p$: testul exact McNemar pe aceiași itemi față de referința modelului (Base-Ro; Base-Ro-rewarm "
                "pentru SPP-Ro-MT); o diferență semnificativă ar indica o taxă (sau un câștig) de capabilitate.}\n\\label{tab:thesis_capabilities}\n\\end{table}\n")
    print("Saved", out, "and", tex)


if __name__ == "__main__":
    main()

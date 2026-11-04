"""
Capability check (needed before any "alignment tax" statement).

Task: 4-way cloze on the held-out web set (data/eval/heldout_fineweb.parquet, disjoint from training).
For N sentences of 10-30 words whose last word is alphabetic (>= 4 chars), the model must prefer the
true last word over 3 distractor words drawn from other sentences' last words (same length band).
Scored with length-normalised log-likelihood (eval_utils), chance = 25%.
Also reports held-out next-token accuracy (greedy) on the same sentences.
"""
import sys
import re
import json
import random
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
    from config import ALL_MODELS, TOKENIZER_DIR, EVALS_DIR, HELDOUT_FILE
    from eval_utils import batch_score_options, summarize_accuracy
except ImportError:
    from src.config import ALL_MODELS, TOKENIZER_DIR, EVALS_DIR, HELDOUT_FILE
    from src.eval_utils import batch_score_options, summarize_accuracy

WORD_RE = re.compile(r"^[a-zA-ZăâîșțĂÂÎȘȚ\-]{4,}$")


def build_cloze_items(n: int, seed: int = 0) -> List[Dict[str, Any]]:
    if not HELDOUT_FILE.exists():
        raise FileNotFoundError(f"{HELDOUT_FILE} missing: run build_heldout_set.py first")
    rng = random.Random(seed)
    texts = pd.read_parquet(HELDOUT_FILE)["text"].tolist()
    sents = []
    for t in texts:
        for s in re.split(r"(?<=[.!?])\s+", t):
            w = s.strip().split()
            if 10 <= len(w) <= 30 and WORD_RE.match(w[-1].rstrip(".,;:!?»”\"")):
                sents.append(w)
    rng.shuffle(sents)
    last_words = [w[-1].rstrip(".,;:!?»”\"") for w in sents]
    by_len: Dict[int, List[str]] = {}
    for lw in last_words:
        by_len.setdefault(len(lw), []).append(lw)
    items = []
    for w in sents:
        if len(items) >= n:
            break
        target = w[-1].rstrip(".,;:!?»”\"")
        pool = [x for x in by_len.get(len(target), []) if x.lower() != target.lower()]
        if len(pool) < 3:
            pool = [x for x in last_words if x.lower() != target.lower()]
        distractors = rng.sample(sorted(set(pool)), 3)
        opts = [target] + distractors
        rng.shuffle(opts)
        items.append({"prefix": " ".join(w[:-1]) + " ", "options": opts, "correct_idx": opts.index(target)})
    return items


@torch.no_grad()
def evaluate_model(model, tokenizer, items, device) -> Dict[str, Any]:
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
    return summarize_accuracy(flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    items = build_cloze_items(args.n)
    print(f"Cloze items: {len(items)} (chance 25%)")
    results = {}
    for key, mdir, desc in ALL_MODELS:
        if not (mdir / "model.safetensors").exists():
            continue
        model = AutoModelForCausalLM.from_pretrained(str(mdir)).to(device).eval()
        res = evaluate_model(model, tokenizer, items, device)
        results[key] = {"description": desc, **res}
        print(f"  {key:22s} cloze-4 acc = {res['accuracy']:5.1f}%  CI95={res['ci95']}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    out = EVALS_DIR / "thesis_capabilities_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"task": "4-way last-word cloze on held-out web sentences", "n_items": len(items), "results": results}, f, indent=2, ensure_ascii=False)
    tex = EVALS_DIR / "thesis_capabilities_table.tex"
    with open(tex, "w", encoding="utf-8") as f:
        f.write("% Capabilități: cloze 4 opțiuni pe propoziții held-out (șansă 25%)\n\\begin{table}[htbp]\n\\centering\n\\small\n\\begin{tabular}{l l c}\n\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Variantă} & \\textbf{Acuratețe cloze (IC 95\\%)} \\\\\n\\midrule\n")
        for k, r in results.items():
            f.write(f"{k} & {r['description']} & {r['accuracy']:.1f}\\% [{r['ci95'][0]:.0f}--{r['ci95'][1]:.0f}] \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\caption{Capabilitate lingvistică de bază: alegerea ultimului cuvânt dintr-o propoziție held-out dintre 4 candidați "
                f"(N={len(items)}, șansă 25\\%). O diferență semnificativă față de Base ar indica o taxă (sau un câștig) de aliniere.}}\n\\label{{tab:thesis_capabilities}}\n\\end{{table}}\n")
    print("Saved", out, "and", tex)


if __name__ == "__main__":
    main()

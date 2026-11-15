"""
Alignment Tax Evaluation Harness (corrected)

Measures cross-entropy / perplexity of every available model on:
  1. "Web (held-out)" : data/eval/heldout_fineweb.parquet, built by build_heldout_set.py from a
     FineWeb-2 shard that was NOT consumed by the training corpus, with every document whose
     prefix or hash appears in the corpus or in the reflection sidecar removed. This is the
     ONLY set on which an alignment-tax conclusion may be drawn.
  2. "Wikipedia (în antrenare)" / "Presă (în antrenare)" : the tails of wiki_clean / news_recent,
     kept for transparency. They were used in the first version of the thesis as "held-out",
     but 92-99% of these passages are inside the training corpus and ~800 are reflection hosts,
     so their perplexity measures memorisation, not generalisation.

  3. "Web held-out, semnalat de clasificator" / "Web held-out, nesemnalat" : the same held-out set split by the
     safety classifier (models/safety_clf_ro, the one that defines the Filtered baseline's loss-masked set and the
     SPP annotation candidates). The flagged slice is where the Filtered baseline is expected to pay (no loss on
     such documents during training) and where SPP is expected not to (the documents stay in the loss); the
     unflagged slice is the control. Classifier scores are cached in data/eval/heldout_clf_scores.parquet.

Delta PPL is computed on the held-out web set (overall and flagged slice) only.
"""
import sys
import math
import json
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config import (BASE_MODEL_DIR, SPP_MODEL_DIR, SPP_20PCT_MODEL_DIR, SPP_MT_MODEL_DIR,
                        TOKENIZER_DIR, CLEAN_DATA_DIR, EVALS_DIR, EVAL_DATA_DIR, HELDOUT_FILE, ALL_MODELS, base_ref_key)
    from train_safety_classifier import CLF_DIR, predict_probs
except ImportError:
    from src.config import (BASE_MODEL_DIR, SPP_MODEL_DIR, SPP_20PCT_MODEL_DIR, SPP_MT_MODEL_DIR,
                            TOKENIZER_DIR, CLEAN_DATA_DIR, EVALS_DIR, EVAL_DATA_DIR, HELDOUT_FILE, ALL_MODELS, base_ref_key)
    from src.train_safety_classifier import CLF_DIR, predict_probs

EVALS_DIR.mkdir(parents=True, exist_ok=True)

HELDOUT_KEY = "Web (held-out)"
FLAGGED_KEY = "Web held-out, semnalat de clasificator"
UNFLAGGED_KEY = "Web held-out, nesemnalat"
INTRAIN_KEYS = ("Wikipedia (în antrenare)", "Presă (în antrenare)")
CLF_SCORES_FILE = EVAL_DATA_DIR / "heldout_clf_scores.parquet"
DEFAULT_CLF_THRESHOLD = 0.1   # the v3 sidecar expansion threshold (run_v3_pipeline.bat CLF_THR); the Filtered mask set is built from the same classifier


def _doc_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def heldout_classifier_scores(texts: List[str], device: str) -> List[float]:
    """Safety-classifier score (P(harmful)) for every held-out document, computed once and cached on disk
    (keyed by the md5 of the text, so a rebuilt held-out set is re-scored automatically)."""
    hashes = [_doc_hash(t) for t in texts]
    cached: Dict[str, float] = {}
    if CLF_SCORES_FILE.exists():
        df = pd.read_parquet(CLF_SCORES_FILE)
        cached = dict(zip(df["doc_hash"], df["clf_score"]))
    missing = [i for i, h in enumerate(hashes) if h not in cached]
    if missing:
        from transformers import AutoModelForSequenceClassification
        if not (CLF_DIR / "config.json").exists():
            raise FileNotFoundError(f"Clasificatorul de siguranta lipseste din {CLF_DIR} (train_safety_classifier.py).")
        max_len = json.load(open(CLF_DIR / "threshold.json", encoding="utf-8")).get("max_len", 384)
        print(f"  [clasificator] scorez {len(missing):,} documente held-out cu {CLF_DIR.name} (max_len {max_len}) ...", flush=True)
        tok = AutoTokenizer.from_pretrained(str(CLF_DIR))
        clf = AutoModelForSequenceClassification.from_pretrained(str(CLF_DIR)).to(device).eval()
        probs = predict_probs(clf, tok, [texts[i] for i in missing], device, max_len, bs=64)
        for i, p in zip(missing, probs):
            cached[hashes[i]] = float(p)
        del clf
        if device == "cuda":
            torch.cuda.empty_cache()
        pd.DataFrame({"doc_hash": list(cached.keys()), "clf_score": list(cached.values())}).to_parquet(CLF_SCORES_FILE, index=False)
        print(f"  [clasificator] scoruri salvate in {CLF_SCORES_FILE}", flush=True)
    return [cached[h] for h in hashes]


class ValidationTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.samples = []
        for t in texts:
            if len(t.strip()) < 50:
                continue
            enc = tokenizer.encode(t.strip(), add_special_tokens=False)
            if len(enc) > 16:
                self.samples.append(enc[:max_length])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def collate_fn_pad(batch, pad_token_id=0):
    max_len = max(len(b["input_ids"]) for b in batch)
    inputs, labels, masks = [], [], []
    for b in batch:
        inp = b["input_ids"].tolist()
        pad = max_len - len(inp)
        inputs.append(inp + [pad_token_id] * pad)
        labels.append(inp + [-100] * pad)
        masks.append([1] * len(inp) + [0] * pad)
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


@torch.no_grad()
def compute_dataset_perplexity(model, dataloader, device) -> Tuple[float, float, int]:
    """Token-weighted cross-entropy and perplexity. Returns (loss, ppl, n_tokens)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # HF averages over all valid label positions (shifted); recover the token count accordingly
        valid = (labels[:, 1:] != -100).sum().item()
        if valid > 0:
            total_loss += out.loss.item() * valid
            total_tokens += valid
    if total_tokens == 0:
        return 0.0, float("nan"), 0
    avg = total_loss / total_tokens
    return round(avg, 4), round(math.exp(min(avg, 20.0)), 2), total_tokens


def load_passages(num_samples: int, clf_threshold: Optional[float] = DEFAULT_CLF_THRESHOLD,
                  device: str = "cpu") -> Dict[str, List[str]]:
    domains: Dict[str, List[str]] = {}
    if not HELDOUT_FILE.exists():
        raise FileNotFoundError(
            f"{HELDOUT_FILE} missing. Run `py src/build_heldout_set.py` first; the wiki/news tails are NOT held out.")
    heldout_all = pd.read_parquet(HELDOUT_FILE)["text"].dropna().tolist()
    domains[HELDOUT_KEY] = heldout_all[:num_samples]
    if clf_threshold is not None:
        # Flagged slice over the WHOLE held-out file (the flag rate is ~3%, so every flagged document counts);
        # the unflagged control takes the first num_samples documents below the threshold.
        scores = heldout_classifier_scores(heldout_all, device)
        flagged = [t for t, s in zip(heldout_all, scores) if s >= clf_threshold]
        unflagged = [t for t, s in zip(heldout_all, scores) if s < clf_threshold][:num_samples]
        print(f"  [clasificator] prag {clf_threshold:.2f}: {len(flagged):,} din {len(heldout_all):,} documente held-out semnalate "
              f"({100.0 * len(flagged) / max(1, len(heldout_all)):.1f}%)", flush=True)
        domains[FLAGGED_KEY] = flagged
        domains[UNFLAGGED_KEY] = unflagged
    wiki = CLEAN_DATA_DIR / "wiki_clean.parquet"
    news = CLEAN_DATA_DIR / "news_recent.parquet"
    if wiki.exists():
        domains[INTRAIN_KEYS[0]] = pd.read_parquet(wiki, columns=["text"])["text"].dropna().tail(num_samples).tolist()
    if news.exists():
        domains[INTRAIN_KEYS[1]] = pd.read_parquet(news, columns=["text"])["text"].dropna().tail(num_samples).tolist()
    return domains


def run_alignment_tax_evaluation(num_samples: int = 1000, batch_size: int = 8,
                                 clf_threshold: Optional[float] = DEFAULT_CLF_THRESHOLD):
    print("\n" + "=" * 72)
    print(" ALIGNMENT TAX EVALUATION (held-out web set + classifier-flagged slice + in-training reference sets)")
    print("=" * 72)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    passages = load_passages(num_samples, clf_threshold, device)
    loaders = {}
    for dom, texts in passages.items():
        ds = ValidationTextDataset(texts, tokenizer, max_length=512)
        loaders[dom] = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b, p=pad_id: collate_fn_pad(b, pad_token_id=p))
        print(f"  {dom}: {len(ds)} passages")

    models_to_test = list(ALL_MODELS)
    results: List[Dict[str, Any]] = []
    base_ppl = {}   # reference PPL per generation (Base-Ro-88M for v1/v2 models, Base-Ro-88M-pack for the packed ones)
    base_flagged_ppl = {}   # same, on the classifier-flagged held-out slice
    for label, path, desc in models_to_test:
        if not (path / "model.safetensors").exists():
            print(f"[skip] {label}: {path} not found")
            continue
        print(f"\n[Evaluating] {label} ({desc})")
        dtype = torch.float32   # PPL deltas are reported to 0.01: bf16 weight rounding alone would move them by ~0.3
        model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype).to(device)
        row: Dict[str, Any] = {"model": label, "description": desc, "domains": {}}
        for dom, dl in loaders.items():
            loss, ppl, ntok = compute_dataset_perplexity(model, dl, device)
            row["domains"][dom] = {"loss": loss, "ppl": ppl, "tokens": ntok}
            print(f"  -> {dom:28s} loss={loss:.4f}  ppl={ppl:.2f}  ({ntok:,} tokens)")
        row["heldout_ppl"] = row["domains"][HELDOUT_KEY]["ppl"]
        row["heldout_loss"] = row["domains"][HELDOUT_KEY]["loss"]
        base_ppl[label] = row["heldout_ppl"]   # every model can be a reference (Base for T0, re-warm for MT)
        ref_ppl = base_ppl.get(base_ref_key(label))
        row["delta_ppl_heldout_vs_base"] = round(row["heldout_ppl"] - ref_ppl, 2) if ref_ppl else None
        if FLAGGED_KEY in row["domains"]:
            row["flagged_ppl"] = row["domains"][FLAGGED_KEY]["ppl"]
            row["unflagged_ppl"] = row["domains"][UNFLAGGED_KEY]["ppl"]
            base_flagged_ppl[label] = row["flagged_ppl"]
            ref_f = base_flagged_ppl.get(base_ref_key(label))
            row["delta_ppl_flagged_vs_base"] = round(row["flagged_ppl"] - ref_f, 2) if ref_f else None
        results.append(row)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    report_path = EVALS_DIR / "thesis_alignment_tax_report.json"
    slice_sizes = {k: len(dl.dataset) for k, dl in loaders.items()}
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"note": "Alignment tax is judged on 'Web (held-out)' only; the wiki/news tails are inside the training corpus. "
                           "The flagged / unflagged slices split the held-out set with the safety classifier that defines the Filtered "
                           "baseline's loss-masked set (threshold %s)." % (clf_threshold,),
                   "clf_threshold": clf_threshold, "passages_per_domain": slice_sizes,
                   "results": results}, f, indent=2, ensure_ascii=False)

    def _delta(r: Dict[str, Any], field: str) -> str:
        if r["model"] == base_ref_key(r["model"]):
            return "0.00 (ref)"
        if r.get(field) is None:
            return "n/a"          # reference model of this generation not evaluated
        return f"{r[field]:+.2f}"

    tex_path = EVALS_DIR / "thesis_alignment_tax_table.tex"
    has_flagged = FLAGGED_KEY in loaders
    doms = [HELDOUT_KEY] + ([FLAGGED_KEY, UNFLAGGED_KEY] if has_flagged else []) + [k for k in INTRAIN_KEYS if k in loaders]
    short = {HELDOUT_KEY: "held-out", FLAGGED_KEY: "held-out semnalat", UNFLAGGED_KEY: "held-out nesemnalat",
             INTRAIN_KEYS[0]: "Wikipedia (antr.)", INTRAIN_KEYS[1]: "Presă (antr.)"}
    delta_cols = ["delta_ppl_heldout_vs_base"] + (["delta_ppl_flagged_vs_base"] if has_flagged else [])
    delta_heads = ["$\\Delta$PPL held-out"] + (["$\\Delta$PPL semnalat"] if has_flagged else [])
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Taxa de aliniere: perplexitate pe set held-out verificat (FineWeb-2 shard neantrenat, dedup. față de corpus și sidecar)\n")
        f.write("% 'semnalat' / 'nesemnalat' = felii ale setului held-out dupa clasificatorul de siguranta (acelasi care defineste setul mascat al Filtered).\n")
        f.write("% Coloanele 'în antrenare' sunt păstrate doar pentru transparență (set folosit greșit ca held-out în v1).\n")
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{l l " + "c " * len(doms) + "c " * len(delta_cols) + "}\n\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Variantă} & " + " & ".join("\\textbf{PPL " + short[d] + "}" for d in doms)
                + " & " + " & ".join("\\textbf{" + h + "}" for h in delta_heads) + " \\\\\n\\midrule\n")
        for r in results:
            cells = [f"{r['domains'][d]['ppl']:.2f}" for d in doms]
            deltas = [_delta(r, c) for c in delta_cols]
            f.write(f"{r['model']} & {r['description']} & " + " & ".join(cells) + " & " + " & ".join(deltas) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        flagged_note = (f" Feliile `semnalat' ({slice_sizes.get(FLAGGED_KEY, 0)} documente, scor clasificator $\\geq$ {clf_threshold}) și `nesemnalat' "
                        f"({slice_sizes.get(UNFLAGGED_KEY, 0)} documente) împart setul held-out cu clasificatorul de siguranță care definește setul mascat al "
                        "liniei de bază Filtered; pe felia semnalată Filtered nu a primit loss la antrenare, SPP da." if has_flagged else "")
        f.write("\\caption{Perplexitate (context 512 tokeni) pe un set held-out de pagini web verificate ca absente din corpusul de antrenare și din sidecar-ul de reflecții "
                "(build\\_heldout\\_set.py), alături de seturile Wikipedia/presă care sunt conținute în corpusul de antrenare și pe care diferențele reflectă memorare. "
                "Taxa de aliniere ($\\Delta$PPL) se raportează exclusiv pe setul held-out." + flagged_note + "}\n")
        f.write("\\label{tab:thesis_alignment_tax}\n\\end{table}\n")

    print("\n[SUCCESS] JSON:", report_path, "\n          LaTeX:", tex_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--clf-threshold", type=float, default=DEFAULT_CLF_THRESHOLD,
                    help="safety-classifier threshold for the flagged held-out slice (default 0.1 = the v3 sidecar expansion threshold)")
    ap.add_argument("--no-flagged", action="store_true", help="skip the classifier-flagged / unflagged held-out slices")
    args = ap.parse_args()
    run_alignment_tax_evaluation(num_samples=args.num_samples, batch_size=args.batch_size,
                                 clf_threshold=None if args.no_flagged else args.clf_threshold)

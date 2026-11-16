"""
Alignment Tax Evaluation Harness

Measures cross-entropy / perplexity (first 512 tokens of every document) of every available model on:
  1. "Web (held-out)" : ALL documents of data/eval/heldout_fineweb.parquet, built by build_heldout_set.py from a
     FineWeb-2 shard that was NOT consumed by the training corpus, cleaned exactly like the corpus
     (prepare_scale_corpus.clean_text_content) and with every document whose corpus deduplication key (first 64
     cleaned characters) or MD5 occurs in the corpus or in the reflection sidecar removed. This is the ONLY set on
     which an alignment-tax conclusion may be drawn.
  2. "Web held-out, semnalat de clasificator" / "Web held-out, nesemnalat" : the same documents split by the safety
     classifier (models/safety_clf_ro, the one that defines the Filtered baseline's loss-masked set and the SPP
     annotation candidates). The flagged slice is where the Filtered baseline is expected to pay (no loss on such
     documents during training) and SPP is expected not to (the documents stay in the loss); the unflagged slice is
     the control. Both slices use every held-out document on their side of the threshold; they are computed from the
     same per-document losses as (1). Classifier scores are cached in data/eval/heldout_clf_scores.parquet.
  3. "Wikipedia (în antrenare)" / "Presă (în antrenare)" : the last --num-samples documents of wiki_clean /
     news_recent, kept for transparency only. They were used in the first version of the thesis as "held-out", but
     92-99 % of these passages are inside the training corpus, so their perplexity measures memorisation.

Outputs: evals/thesis_alignment_tax_report.json, evals/thesis_alignment_tax_table.tex and
evals/thesis_alignment_tax_perdoc_losses.json (the mean loss of every held-out document for every model, in the same
order, for the paired tests of eval_alignment_paired.py).
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config import TOKENIZER_DIR, CLEAN_DATA_DIR, EVALS_DIR, EVAL_DATA_DIR, HELDOUT_FILE, ALL_MODELS, base_ref_key
    from train_safety_classifier import CLF_DIR, predict_probs
    from eval_utils import tex_escape
except ImportError:
    from src.config import TOKENIZER_DIR, CLEAN_DATA_DIR, EVALS_DIR, EVAL_DATA_DIR, HELDOUT_FILE, ALL_MODELS, base_ref_key
    from src.train_safety_classifier import CLF_DIR, predict_probs
    from src.eval_utils import tex_escape

EVALS_DIR.mkdir(parents=True, exist_ok=True)

HELDOUT_KEY = "Web (held-out)"
FLAGGED_KEY = "Web held-out, semnalat de clasificator"
UNFLAGGED_KEY = "Web held-out, nesemnalat"
INTRAIN_KEYS = ("Wikipedia (în antrenare)", "Presă (în antrenare)")
SLICE_NAMES = {HELDOUT_KEY: "heldout", FLAGGED_KEY: "flagged", UNFLAGGED_KEY: "unflagged"}   # keys of the per-doc file
CLF_SCORES_FILE = EVAL_DATA_DIR / "heldout_clf_scores.parquet"
PERDOC_FILE = EVALS_DIR / "thesis_alignment_tax_perdoc_losses.json"
DEFAULT_CLF_THRESHOLD = 0.1   # the threshold of the sidecar expansion; the Filtered mask set is built from the same classifier
MAX_LEN = 512


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
    """First `max_length` tokens of every text; `kept[j]` is the index (in `texts`) of sample j."""
    def __init__(self, texts: List[str], tokenizer, max_length: int = MAX_LEN):
        self.samples, self.kept = [], []
        for i, t in enumerate(texts):
            if len(t.strip()) < 50:
                continue
            enc = tokenizer.encode(t.strip(), add_special_tokens=False)
            if len(enc) > 16:
                self.samples.append(enc[:max_length])
                self.kept.append(i)

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
def score_documents(model, dataloader, device) -> Tuple[List[float], List[int]]:
    """Per-document summed cross-entropy (nats) and number of predicted tokens, in dataset order. Token t is
    predicted from tokens < t (the first token of a document is never a target), exactly like the HF loss."""
    model.eval()
    sums: List[float] = []
    counts: List[int] = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
        tgt = labels[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100,
                              reduction="none").view(tgt.shape)
        valid = tgt != -100
        sums.extend(nll.masked_fill(~valid, 0.0).double().sum(dim=1).tolist())
        counts.extend(valid.sum(dim=1).tolist())
    return sums, counts


def aggregate(sums: List[float], counts: List[int], idx: Optional[List[int]] = None) -> Dict[str, Any]:
    """Token-weighted loss / perplexity over the documents in `idx` (all if None)."""
    if idx is not None:
        sums = [sums[i] for i in idx]
        counts = [counts[i] for i in idx]
    ntok = int(sum(counts))
    if ntok == 0:
        return {"loss": 0.0, "ppl": float("nan"), "tokens": 0, "docs": 0}
    avg = math.fsum(sums) / ntok
    return {"loss": round(avg, 4), "ppl": round(math.exp(min(avg, 20.0)), 2), "tokens": ntok, "docs": len(counts)}


def load_passages(num_samples: int, clf_threshold: Optional[float] = DEFAULT_CLF_THRESHOLD,
                  device: str = "cpu") -> Tuple[Dict[str, List[str]], Optional[List[float]]]:
    """Returns ({domain: texts}, classifier score of every held-out document or None)."""
    domains: Dict[str, List[str]] = {}
    if not HELDOUT_FILE.exists():
        raise FileNotFoundError(
            f"{HELDOUT_FILE} missing. Run `py src/build_heldout_set.py` first; the wiki/news tails are NOT held out.")
    heldout_all = pd.read_parquet(HELDOUT_FILE)["text"].dropna().tolist()
    domains[HELDOUT_KEY] = heldout_all
    scores = heldout_classifier_scores(heldout_all, device) if clf_threshold is not None else None
    wiki = CLEAN_DATA_DIR / "wiki_clean.parquet"
    news = CLEAN_DATA_DIR / "news_recent.parquet"
    if wiki.exists():
        domains[INTRAIN_KEYS[0]] = pd.read_parquet(wiki, columns=["text"])["text"].dropna().tail(num_samples).tolist()
    if news.exists():
        domains[INTRAIN_KEYS[1]] = pd.read_parquet(news, columns=["text"])["text"].dropna().tail(num_samples).tolist()
    return domains, scores


def run_alignment_tax_evaluation(num_samples: int = 1000, batch_size: int = 8,
                                 clf_threshold: Optional[float] = DEFAULT_CLF_THRESHOLD):
    print("\n" + "=" * 72)
    print(" ALIGNMENT TAX EVALUATION (held-out web set + classifier slices + in-training reference sets)")
    print("=" * 72)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    passages, scores = load_passages(num_samples, clf_threshold, device)
    datasets = {dom: ValidationTextDataset(texts, tokenizer, max_length=MAX_LEN) for dom, texts in passages.items()}
    loaders = {dom: DataLoader(ds, batch_size=batch_size, shuffle=False,
                               collate_fn=lambda b, p=pad_id: collate_fn_pad(b, pad_token_id=p))
               for dom, ds in datasets.items()}
    held_ds = datasets[HELDOUT_KEY]
    held_texts = passages[HELDOUT_KEY]
    slices: Dict[str, Optional[List[int]]] = {HELDOUT_KEY: None}         # positions inside the held-out dataset
    if scores is not None:
        slices[FLAGGED_KEY] = [j for j, i in enumerate(held_ds.kept) if scores[i] >= clf_threshold]
        slices[UNFLAGGED_KEY] = [j for j, i in enumerate(held_ds.kept) if scores[i] < clf_threshold]
        print(f"  [clasificator] prag {clf_threshold:.2f}: {len(slices[FLAGGED_KEY]):,} din {len(held_ds):,} documente held-out "
              f"semnalate ({100.0 * len(slices[FLAGGED_KEY]) / max(1, len(held_ds)):.1f}%)", flush=True)
    for dom, ds in datasets.items():
        print(f"  {dom}: {len(ds)} passages")
    for key in (FLAGGED_KEY, UNFLAGGED_KEY):
        if key in slices:
            print(f"  {key}: {len(slices[key])} passages (subset of {HELDOUT_KEY})")

    results: List[Dict[str, Any]] = []
    ref_rows: Dict[str, Dict[str, Any]] = {}      # every model can be a reference (Base for T0, re-warm for MT)
    perdoc: Dict[str, Any] = {}
    for label, path, desc in ALL_MODELS:
        if not (path / "model.safetensors").exists():
            print(f"[skip] {label}: {path} not found")
            continue
        print(f"\n[Evaluating] {label} ({desc})")
        dtype = torch.float32   # PPL deltas are reported to 0.01: bf16 weight rounding alone would move them by ~0.3
        model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype).to(device)
        row: Dict[str, Any] = {"model": label, "description": desc, "domains": {}}
        held_sums, held_counts = None, None
        for dom, dl in loaders.items():
            sums, counts = score_documents(model, dl, device)
            if dom == HELDOUT_KEY:
                held_sums, held_counts = sums, counts
                for key, idx in slices.items():
                    row["domains"][key] = aggregate(sums, counts, idx)
            else:
                row["domains"][dom] = aggregate(sums, counts)
        for dom in list(slices) + [d for d in loaders if d != HELDOUT_KEY]:
            r = row["domains"][dom]
            print(f"  -> {dom:40s} loss={r['loss']:.4f}  ppl={r['ppl']:.2f}  ({r['tokens']:,} tokens, {r['docs']} docs)")
        perdoc[label] = {SLICE_NAMES[k]: [round(held_sums[j] / held_counts[j], 6) for j in (idx if idx is not None else range(len(held_counts)))]
                         for k, idx in slices.items()}
        row["heldout_ppl"] = row["domains"][HELDOUT_KEY]["ppl"]
        row["heldout_loss"] = row["domains"][HELDOUT_KEY]["loss"]
        if FLAGGED_KEY in row["domains"]:
            row["flagged_ppl"] = row["domains"][FLAGGED_KEY]["ppl"]
            row["unflagged_ppl"] = row["domains"][UNFLAGGED_KEY]["ppl"]
        ref = row if label == base_ref_key(label) else ref_rows.get(base_ref_key(label))   # the reference row gets 0.00
        fields = [("delta_ppl_heldout_vs_base", "heldout_ppl")]
        if FLAGGED_KEY in row["domains"]:
            fields += [("delta_ppl_flagged_vs_base", "flagged_ppl"), ("delta_ppl_unflagged_vs_base", "unflagged_ppl")]
        for out_key, ppl_key in fields:
            row[out_key] = round(row[ppl_key] - ref[ppl_key], 2) if ref is not None else None
        ref_rows[label] = row
        results.append(row)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    slice_sizes = {k: (len(held_ds) if idx is None else len(idx)) for k, idx in slices.items()}
    slice_sizes.update({k: len(ds) for k, ds in datasets.items() if k != HELDOUT_KEY})
    report_path = EVALS_DIR / "thesis_alignment_tax_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"note": "Alignment tax is judged on 'Web (held-out)' only; the wiki/news tails are inside the training corpus. "
                           "The flagged / unflagged slices split ALL held-out documents with the safety classifier that defines the "
                           "Filtered baseline's loss-masked set (threshold %s). Perplexity = exp(token-weighted mean cross-entropy) "
                           "on the first %d tokens of every document." % (clf_threshold, MAX_LEN),
                   "heldout_file": HELDOUT_FILE.relative_to(_ROOT_DIR).as_posix(), "heldout_docs_in_file": len(held_texts),
                   "clf_threshold": clf_threshold, "max_len": MAX_LEN, "passages_per_domain": slice_sizes,
                   "results": results}, f, indent=2, ensure_ascii=False)

    order_md5 = hashlib.md5("".join(_doc_hash(held_texts[i]) for i in held_ds.kept).encode()).hexdigest()
    meta = {"note": "Mean cross-entropy (nats/token, first %d tokens) of every held-out document, same document order for "
                    "every model (paired). 'flagged'/'unflagged' are the subsets of 'heldout' given by flagged_index / "
                    "unflagged_index (positions in 'heldout'). Produced by eval_alignment_tax.py; analysed by "
                    "eval_alignment_paired.py." % MAX_LEN,
            "heldout_file": HELDOUT_FILE.relative_to(_ROOT_DIR).as_posix(), "heldout_order_md5": order_md5,
            "clf_threshold": clf_threshold, "n_tokens": held_counts if results else [],
            "flagged_index": slices.get(FLAGGED_KEY), "unflagged_index": slices.get(UNFLAGGED_KEY)}
    with open(PERDOC_FILE, "w", encoding="utf-8") as f:
        json.dump({"_meta": meta, **perdoc}, f, ensure_ascii=False)

    def _delta(r: Dict[str, Any], field: str) -> str:
        if r["model"] == base_ref_key(r["model"]):
            return "0.00 (ref)"
        if r.get(field) is None:
            return "n/a"          # reference model not evaluated
        return f"{r[field]:+.2f}"

    tex_path = EVALS_DIR / "thesis_alignment_tax_table.tex"
    has_flagged = FLAGGED_KEY in slices
    doms = [HELDOUT_KEY] + ([FLAGGED_KEY, UNFLAGGED_KEY] if has_flagged else []) + [k for k in INTRAIN_KEYS if k in loaders]
    short = {HELDOUT_KEY: "held-out", FLAGGED_KEY: "held-out semnalat", UNFLAGGED_KEY: "held-out nesemnalat",
             INTRAIN_KEYS[0]: "Wikipedia (antr.)", INTRAIN_KEYS[1]: "Presă (antr.)"}
    delta_cols = ["delta_ppl_heldout_vs_base"] + (["delta_ppl_flagged_vs_base"] if has_flagged else [])
    delta_heads = ["$\\Delta$PPL held-out"] + (["$\\Delta$PPL semnalat"] if has_flagged else [])
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Taxa de aliniere: perplexitate pe set held-out verificat (FineWeb-2 shard neconsumat, normalizat ca si corpusul, "
                "dedup. fata de corpus si sidecar cu cheia corpusului)\n")
        f.write("% 'semnalat' / 'nesemnalat' = toate documentele held-out de o parte si de alta a pragului clasificatorului de siguranta "
                "(acelasi care defineste setul mascat al Filtered).\n")
        f.write("% Coloanele 'in antrenare' sunt pastrate doar pentru transparenta (set folosit gresit ca held-out in v1).\n")
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{l l " + "c " * len(doms) + "c " * len(delta_cols) + "}\n\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Variantă} & " + " & ".join("\\textbf{PPL " + short[d] + "}" for d in doms)
                + " & " + " & ".join("\\textbf{" + h + "}" for h in delta_heads) + " \\\\\n\\midrule\n")
        for r in results:
            cells = [f"{r['domains'][d]['ppl']:.2f}" for d in doms]
            deltas = [_delta(r, c) for c in delta_cols]
            f.write(f"{tex_escape(r['model'])} & {tex_escape(r['description'])} & " + " & ".join(cells) + " & " + " & ".join(deltas) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        flagged_note = (f" Feliile `semnalat' ({slice_sizes.get(FLAGGED_KEY, 0)} documente, scor clasificator $\\geq$ {clf_threshold}) și `nesemnalat' "
                        f"({slice_sizes.get(UNFLAGGED_KEY, 0)} documente) împart toate documentele held-out cu clasificatorul de siguranță care definește setul "
                        "mascat al liniei de bază Filtered; pe felia semnalată Filtered nu a primit loss la antrenare, SPP da. Testele pereche pe documente "
                        "sunt în tabelul generat de eval\\_alignment\\_paired.py." if has_flagged else "")
        f.write(f"\\caption{{Perplexitate (primii {MAX_LEN} tokeni) pe {slice_sizes[HELDOUT_KEY]} pagini web held-out, normalizate exact ca și corpusul de antrenare și "
                "verificate ca absente din corpus și din sidecar-ul de reflecții (build\\_heldout\\_set.py), alături de seturile Wikipedia/presă care sunt "
                "conținute în corpusul de antrenare și pe care diferențele reflectă memorare. Taxa de aliniere ($\\Delta$PPL față de referința fiecărui "
                "model) se raportează exclusiv pe setul held-out." + flagged_note + "}\n")
        f.write("\\label{tab:thesis_alignment_tax}\n\\end{table}\n")

    print("\n[SUCCESS] JSON:", report_path, "\n          per-document losses:", PERDOC_FILE, "\n          LaTeX:", tex_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=1000, help="documents taken from the end of the in-training wiki/news sets "
                                                                 "(the held-out set is always used in full)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--clf-threshold", type=float, default=DEFAULT_CLF_THRESHOLD,
                    help="safety-classifier threshold for the flagged held-out slice (default 0.1 = the sidecar expansion threshold)")
    ap.add_argument("--no-flagged", action="store_true", help="skip the classifier-flagged / unflagged held-out slices")
    args = ap.parse_args()
    run_alignment_tax_evaluation(num_samples=args.num_samples, batch_size=args.batch_size,
                                 clf_threshold=None if args.no_flagged else args.clf_threshold)

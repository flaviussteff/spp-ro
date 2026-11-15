"""
Safety classifier (the paper's "SafeLM" step), Romanian.

1. Fine-tunes dumitrescustefan/bert-base-romanian-cased-v1 on Qwen-labelled documents
   (data/sidecar/severity_labels.parquet from `generate_reflections_v2.py --label-only`):
   binary target = severity >= 3 (harmful / sensitive) vs <= 2.
2. Picks the decision threshold on a held-out split so that recall >= --recall (the paper
   favours coverage over precision: 61% of its flagged documents were benign).
3. Scores the WHOLE training corpus (first --max-len tokens of every document) and writes
   data/clean/spp_candidates_v2.parquet = all flagged documents + an equal number of random
   low-score benign documents. `generate_reflections_v2.py --candidates <file>` then annotates
   exactly these documents (classifier defines the sets, as in the paper).

GPU: ~10 min training + ~1-2 h scoring 1.5M documents at 384 tokens on an RTX 3060.
Usage:  py src/train_safety_classifier.py            (train + score)
        py src/train_safety_classifier.py --score-only   (reuse models/safety_clf_ro)
"""
import re
import sys
import json
import math
import time
import random
import hashlib
import argparse
import contextlib
from pathlib import Path
from typing import List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from config import CLEAN_DATA_DIR, SIDECAR_DIR, MODELS_DIR, HELDOUT_FILE
except ImportError:
    from src.config import CLEAN_DATA_DIR, SIDECAR_DIR, MODELS_DIR, HELDOUT_FILE

BASE_CLF = "dumitrescustefan/bert-base-romanian-cased-v1"
DEV_CACHE = {"y": None, "p": None}
LABELS_FILE = SIDECAR_DIR / "severity_labels.parquet"
CLF_DIR = MODELS_DIR / "safety_clf_ro"
CANDIDATES_FILE = CLEAN_DATA_DIR / "spp_candidates_v2.parquet"
SCORES_SUMMARY = CLEAN_DATA_DIR / "spp_candidates_v2_summary.json"
CORPUS_FILE = CLEAN_DATA_DIR / "corpus_scale_stream.parquet"


def doc_key(text: str) -> str:
    return hashlib.md5(text.strip()[:200].encode("utf-8")).hexdigest()


def amp(device: str):
    """bf16 autocast on CUDA; no-op on CPU (torch.autocast('cuda') fails when CUDA is hidden)."""
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()


ADULT_SPAM_RE = re.compile(r"(?i)\b(porn\w*|xxx|escort\w*|escorte\w*|matrimonial\w*|fut\w*|curv[aăe]\w*|pizd\w*|pul[aă]|muie|nimfoman\w*|xvideos|brazzers|beeg|publi24|orgi[ei]|ejacul\w*|sex (gratis|oral|anal|la tel\w*)|filme (xxx|porno))\b")


def looks_like_spam(text: str) -> bool:
    """Keyword-list / adult-escort spam pages (most severity-5 documents in the calibration sample).
    Sexual content is outside this thesis' constitution, so such pages are excluded from the classifier
    training set and from the annotation candidates; the count is reported in the scoring summary.
    Tuned on the 3,000 calibration labels: catches ~75% of severity-5 pages and <5% of severity-3 news."""
    w = text.split()
    if len(w) < 60:
        return True
    diversity = len(set(x.lower() for x in w)) / len(w)
    alpha = sum(c.isalpha() or c.isspace() for c in text) / max(1, len(text))
    punct_per_100w = sum(text.count(c) for c in ".!?") / len(w) * 100
    adult_per_100w = len(ADULT_SPAM_RE.findall(text)) / len(w) * 100
    if alpha < 0.80 or diversity < 0.40:
        return True
    if adult_per_100w >= 0.8 and punct_per_100w < 6.0:   # tuned on calibration labels: 20/29 sev-5 spam pages, 8/201 sev-3 news
        return True
    if punct_per_100w < 2.0 and diversity < 0.60:
        return True
    return False


def head_words(text: str, n: int = 350) -> str:
    w = text.split()
    return " ".join(w[:n])


class TextDS(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tok, max_len: int):
        self.enc = tok([head_words(t) for t in texts], truncation=True, max_length=max_len, padding=False)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i], "label": self.labels[i]}


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        att[i, :n] = 1
    return ids, att, torch.tensor([b["label"] for b in batch], dtype=torch.long)


@torch.no_grad()
def predict_probs(model, tok, texts: List[str], device, max_len: int, bs: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(texts), bs):
        enc = tok([head_words(t) for t in texts[i:i + bs]], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
        with amp(device):
            logits = model(**enc).logits.float()
        out.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train(args, device):
    df = pd.read_parquet(LABELS_FILE)
    df = df[df["severity"] >= 0].reset_index(drop=True)
    n0 = len(df)
    df = df[~df["text"].apply(looks_like_spam)].reset_index(drop=True)
    print(f"[labels] spam/keyword-list pages removed from the training set: {n0 - len(df)} of {n0}")
    y = (df["severity"] >= 3).astype(int).tolist()
    print(f"[labels] {len(df):,} documents; harmful (sev>=3): {sum(y):,} ({100*sum(y)/len(y):.1f}%)")
    idx = list(range(len(df)))
    random.Random(0).shuffle(idx)
    n_dev = min(len(idx) // 2, max(20, int(0.15 * len(idx))))
    dev_idx, tr_idx = idx[:n_dev], idx[n_dev:]
    tok = AutoTokenizer.from_pretrained(BASE_CLF)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_CLF, num_labels=2).to(device)
    ds = TextDS(df["text"].iloc[tr_idx].tolist(), [y[i] for i in tr_idx], tok, args.max_len)
    pos = sum(y[i] for i in tr_idx)
    w = torch.tensor([1.0, max(1.0, (len(tr_idx) - pos) / max(1, pos))], device=device)   # class weight for the minority class
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * total))) * max(0.0, (total - s) / total))
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    model.train()
    step = 0
    for ep in range(args.epochs):
        t0 = time.time()
        tot = 0.0
        for ids, att, lab in dl:
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            with amp(device):
                logits = model(input_ids=ids, attention_mask=att).logits.float()
            loss = loss_fn(logits, lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item(); step += 1
        print(f"[train] epoch {ep+1}/{args.epochs} loss {tot/len(dl):.4f} ({time.time()-t0:.0f}s)")
    model.eval()
    p = predict_probs(model, tok, df["text"].iloc[dev_idx].tolist(), device, args.max_len)
    yd = np.array([y[i] for i in dev_idx])
    DEV_CACHE["y"], DEV_CACHE["p"] = yd, p
    # threshold with recall >= args.recall
    best = None
    for thr in np.linspace(0.05, 0.95, 91):
        pred = p >= thr
        tp = int(((pred == 1) & (yd == 1)).sum()); fp = int(((pred == 1) & (yd == 0)).sum()); fn = int(((pred == 0) & (yd == 1)).sum())
        rec = tp / max(1, tp + fn); prec = tp / max(1, tp + fp)
        if rec >= args.recall:
            best = (thr, prec, rec)
    if best is None:
        # recall target unreachable on this dev split: fall back to 0.5 and report the REAL metrics there
        pred = p >= 0.5
        tp = int(((pred == 1) & (yd == 1)).sum()); fp = int(((pred == 1) & (yd == 0)).sum()); fn = int(((pred == 0) & (yd == 1)).sum())
        best = (0.5, tp / max(1, tp + fp), tp / max(1, tp + fn))
        print(f"[dev] WARNING: no threshold in [0.05, 0.95] reaches recall >= {args.recall:.2f}; using 0.50")
    thr, prec, rec = best
    auc = roc_auc(yd, p)
    print(f"[dev] n={len(yd)} harmful={int(yd.sum())} | AUC {auc:.3f} | threshold {thr:.2f} -> precision {prec:.2f}, recall {rec:.2f}")
    CLF_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(CLF_DIR)); tok.save_pretrained(str(CLF_DIR))
    with open(CLF_DIR / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": float(thr), "dev_precision": prec, "dev_recall": rec, "dev_auc": auc, "n_train": len(tr_idx), "n_dev": len(dev_idx),
                   "base_model": BASE_CLF, "max_len": args.max_len, "label_rule": "severity>=3"}, f, indent=2)
    return model, tok, float(thr)


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def score_corpus(model, tok, thr: float, args, device):
    pf = pq.ParquetFile(str(CORPUS_FILE))
    heldout = set(doc_key(t) for t in pq.read_table(str(HELDOUT_FILE), columns=["text"])["text"].to_pylist()) if HELDOUT_FILE.exists() else set()
    if getattr(args, "exclude_parquet", None):
        # expansion run (v3): documents already in the sidecar are neither flagged again nor used as benign fill
        heldout |= set(pq.read_table(args.exclude_parquet, columns=["doc_key"])["doc_key"].to_pylist())
        skip_file = SIDECAR_DIR / "reflections_v2_skip_keys.json"   # adult/escort pages removed in v2: never candidates again
        if skip_file.exists():
            heldout |= set(json.load(open(skip_file, encoding="utf-8")))
        print(f"[score] excluding {len(heldout):,} documents (held-out + already annotated in {Path(args.exclude_parquet).name})")
    rng = random.Random(0)
    flagged, benign_pool = [], []
    n_scanned = 0
    n_spam = 0
    t0 = time.time()
    hist = np.zeros(10, dtype=np.int64)
    n_rg = pf.num_row_groups if args.max_rowgroups <= 0 else min(pf.num_row_groups, args.max_rowgroups)
    for rg in range(n_rg):
        tb = pf.read_row_group(rg, columns=["text", "source"])
        texts, srcs = tb["text"].to_pylist(), tb["source"].to_pylist()
        keep = [(t, s) for t, s in zip(texts, srcs) if t and not looks_like_spam(t)]
        n_spam += len(texts) - len(keep)
        if not keep:
            continue
        probs = predict_probs(model, tok, [t for t, _ in keep], device, args.max_len, bs=args.score_bs)
        hist += np.histogram(probs, bins=10, range=(0, 1))[0]
        for (t, s), pr in zip(keep, probs):
            k = doc_key(t)
            if k in heldout:
                continue
            if pr >= thr:
                flagged.append((k, s, t, float(pr)))
            elif pr < args.benign_max and rng.random() < 0.25:
                benign_pool.append((k, s, t, float(pr)))
        n_scanned += len(texts)
        if (rg + 1) % 10 == 0 or rg == n_rg - 1:
            el = time.time() - t0
            eta = el / (rg + 1) * (n_rg - rg - 1)
            print(f"[score] row-group {rg+1}/{n_rg} | scanned {n_scanned:,} | flagged {len(flagged):,} ({100*len(flagged)/max(1,n_scanned):.2f}%) | {el/60:.0f} min, ETA {eta/60:.0f} min", flush=True)
    # Budget cap: keep only the --max-flagged highest-scoring documents (reports the implied recall on dev).
    cap_thr = thr
    if args.max_flagged > 0 and len(flagged) > args.max_flagged:
        flagged.sort(key=lambda x: -x[3])
        cap_thr = flagged[args.max_flagged - 1][3]
        print(f"[cap] {len(flagged):,} documents above threshold {thr:.2f}; keeping the top {args.max_flagged:,} (score >= {cap_thr:.3f})")
        flagged = flagged[: args.max_flagged]
        if DEV_CACHE["p"] is not None:
            yd, pd_ = DEV_CACHE["y"], DEV_CACHE["p"]
            rec_cap = float(((pd_ >= cap_thr) & (yd == 1)).sum() / max(1, (yd == 1).sum()))
            prec_cap = float(((pd_ >= cap_thr) & (yd == 1)).sum() / max(1, (pd_ >= cap_thr).sum()))
            print(f"[cap] implied on dev split: recall {rec_cap:.2f}, precision {prec_cap:.2f}")
    rng.shuffle(benign_pool)
    benign = benign_pool[: len(flagged)]
    rows = [{"doc_key": k, "source": s, "text": t, "clf_score": pr, "clf_flag": 1} for k, s, t, pr in flagged] + \
           [{"doc_key": k, "source": s, "text": t, "clf_score": pr, "clf_flag": 0} for k, s, t, pr in benign]
    rng.shuffle(rows)
    pq.write_table(pa.Table.from_pylist(rows), str(CANDIDATES_FILE), compression="zstd")
    summary = {"scanned": n_scanned, "spam_or_short_skipped": n_spam, "flagged": len(flagged), "flagged_fraction": len(flagged) / max(1, n_scanned), "benign_sampled": len(benign),
               "threshold": thr, "effective_threshold_after_cap": cap_thr, "max_flagged": args.max_flagged, "score_histogram_10bins": hist.tolist(), "candidates_file": str(CANDIDATES_FILE),
               "api_calls_needed_batch3": math.ceil(len(rows) / 3)}
    with open(SCORES_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {len(rows):,} candidate documents -> {CANDIDATES_FILE}")
    print(f"Next: py src/generate_reflections_v2.py --candidates {CANDIDATES_FILE}  (~{summary['api_calls_needed_batch3']:,} API calls with batch 3)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--recall", type=float, default=0.9, help="target recall on the dev split when picking the threshold")
    ap.add_argument("--benign-max", type=float, default=0.10, help="max classifier score for a document to enter the benign pool")
    ap.add_argument("--max-flagged", type=int, default=80000, help="budget cap: keep at most this many top-scoring flagged documents (0 = no cap). 80k = 5.3%% of the corpus")
    ap.add_argument("--score-bs", type=int, default=96)
    ap.add_argument("--max-rowgroups", type=int, default=0, help="score only the first N row groups (smoke test); 0 = whole corpus")
    ap.add_argument("--labels", type=str, default=None, help="labels parquet (default data/sidecar/severity_labels.parquet)")
    ap.add_argument("--out-candidates", type=str, default=None, help="output parquet (default data/clean/spp_candidates_v2.parquet)")
    ap.add_argument("--clf-dir", type=str, default=None, help="where to save/load the classifier (default models/safety_clf_ro)")
    ap.add_argument("--device", type=str, default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--threshold", type=float, default=None, help="override the saved decision threshold when scoring (v3 expansion: 0.1 -> higher recall, as the paper favours coverage)")
    ap.add_argument("--exclude-parquet", type=str, default=None, help="parquet with a doc_key column (the current sidecar): those documents are skipped, so the candidates file holds only NEW documents")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")
    global LABELS_FILE, CANDIDATES_FILE, SCORES_SUMMARY, CLF_DIR
    if args.labels:
        LABELS_FILE = Path(args.labels)
    if args.out_candidates:
        CANDIDATES_FILE = Path(args.out_candidates)
        SCORES_SUMMARY = CANDIDATES_FILE.with_name(CANDIDATES_FILE.stem + "_summary.json")
    if args.clf_dir:
        CLF_DIR = Path(args.clf_dir)
    if args.score_only:
        tok = AutoTokenizer.from_pretrained(str(CLF_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(CLF_DIR)).to(device).eval()
        thr = json.load(open(CLF_DIR / "threshold.json", encoding="utf-8"))["threshold"]
        if args.threshold is not None:
            print(f"[score] threshold override: {thr:.2f} -> {args.threshold:.2f}")
            thr = args.threshold
    else:
        if not LABELS_FILE.exists():
            raise SystemExit(f"{LABELS_FILE} missing: run `py src/generate_reflections_v2.py --label-only` first")
        model, tok, thr = train(args, device)
    if not args.train_only:
        score_corpus(model, tok, thr, args, device)


if __name__ == "__main__":
    main()

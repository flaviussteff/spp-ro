"""
Trim the expanded sidecar to exactly the paper's annotated fraction (10% of the corpus documents) and report
the reflection-token fraction (paper: 0.55% of tokens).

Default (balanced, like the paper's "all flagged + equal benign"): keep the N/2 highest-scoring classifier-flagged
documents and N/2 random benign ones (seed 0). `--mode benign-only` keeps every flagged document and cuts only benign.
The full file is kept as reflections_full.parquet; the pipeline reads reflections.parquet.

Usage (CPU, ~1 min; run BEFORE the pipeline reaches the SPP training):
  py src\trim_sidecar.py                 # 150,000 documents = 10% of the 1.5M-document stream
  py src\trim_sidecar.py --dry-run       # only the numbers
  py src\trim_sidecar.py --mode benign-only
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_SRC = Path(__file__).resolve().parent
for _p in [str(_SRC), str(_SRC.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config import SIDECAR_DIR, CLEAN_DATA_DIR, TOKENIZER_DIR

SIDECAR = SIDECAR_DIR / "reflections.parquet"
FULL = SIDECAR_DIR / "reflections_full.parquet"
CORPUS = CLEAN_DATA_DIR / "corpus_scale_stream.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150_000, help="documents to keep (default 10%% of the 1.5M stream)")
    ap.add_argument("--mode", choices=["balanced", "benign-only"], default="balanced")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-only", action="store_true", help="keep only documents whose reflection fits the training window under the head-cap rule (prefix + <assistant> + reflection <= --max-length), so every kept document carries its reflection at T0")
    ap.add_argument("--max-length", type=int, default=1023, help="window budget used by the training encoder (1024 minus the EOS)")
    args = ap.parse_args()

    src = FULL if FULL.exists() else SIDECAR
    df = pd.read_parquet(src)
    print(f"sidecar sursă: {src.name}: {len(df):,} documente | clf_flag=1: {int((df.clf_flag == 1).sum()):,} | benigne: {int((df.clf_flag == 0).sum()):,} | sev>=3: {int((df.severity >= 3).sum()):,}")
    if df.doc_key.duplicated().any():
        raise SystemExit("chei duplicate în sidecar; nu continui")
    n_corpus = pq.ParquetFile(str(CORPUS)).metadata.num_rows
    rng = np.random.RandomState(args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    if args.fit_only:
        # head-cap rule of spp_collator.encode_item: len(encode(prefix.rstrip())) + len(encode(reflection)[:128]) + 1 <= max_length
        pre_len, refl_len = [], []
        texts, poss, refls = df.text.tolist(), df.reflection_char_position.astype(int).tolist(), df.reflection_text.tolist()
        B = 4000
        for i in range(0, len(df), B):
            pre_len += [len(x) for x in tok([t[:p].rstrip() for t, p in zip(texts[i:i + B], poss[i:i + B])], add_special_tokens=False)["input_ids"]]
            refl_len += [min(128, len(x)) for x in tok(refls[i:i + B], add_special_tokens=False)["input_ids"]]
        fits = np.array([a + b + 1 <= args.max_length for a, b in zip(pre_len, refl_len)])
        print(f"reflecția încape în fereastră (head-cap, {args.max_length} tokeni): {int(fits.sum()):,} din {len(df):,} ({100 * fits.mean():.1f}%) | semnalate {int(fits[df.clf_flag.values == 1].sum()):,}/{int((df.clf_flag == 1).sum()):,}, benigne {int(fits[df.clf_flag.values == 0].sum()):,}/{int((df.clf_flag == 0).sum()):,}")
        df = df[fits].reset_index(drop=True)
    flagged = df[df.clf_flag == 1].sort_values("clf_score", ascending=False)
    benign = df[df.clf_flag == 0]
    if args.mode == "balanced":
        n_half = args.target // 2
        keep_f = flagged.head(n_half)
        keep_b = benign.iloc[rng.permutation(len(benign))[:args.target - len(keep_f)]]
    else:
        keep_f = flagged
        keep_b = benign.iloc[rng.permutation(len(benign))[:max(0, args.target - len(keep_f))]]
    out = pd.concat([keep_f, keep_b]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    thr_eff = float(keep_f.clf_score.min()) if len(keep_f) else float("nan")
    print(f"păstrate: {len(out):,} = {100 * len(out) / n_corpus:.2f}% din {n_corpus:,} documente | semnalate {len(keep_f):,} (prag efectiv clasificator {thr_eff:.3f}) + benigne {len(keep_b):,} | sev>=3: {int((out.severity >= 3).sum()):,}")

    # reflection-token fraction: reflection tokens / corpus tokens (corpus tokens estimated on a 20k-document sample)
    refl_tok = sum(len(tok.encode(t, add_special_tokens=False)[:128]) + 1 for t in out.reflection_text)   # +1 for <assistant>
    pf = pq.ParquetFile(str(CORPUS))
    sample = pf.read_row_group(0, columns=["text"])["text"].to_pylist()[:20000]
    avg_doc = np.mean([min(len(tok.encode(t, add_special_tokens=False)), 1023) + 1 for t in sample])   # truncated at the window, +EOS
    corpus_tok = avg_doc * n_corpus
    print(f"tokeni reflecție (incl. <assistant>): {refl_tok / 1e6:.2f}M | corpus ≈ {corpus_tok / 1e9:.2f}B tokeni ({avg_doc:.0f}/document) | fracția reflecțiilor ≈ {100 * refl_tok / corpus_tok:.2f}% (articol: 0,55%)")
    full_refl = sum(len(tok.encode(t, add_special_tokens=False)[:128]) + 1 for t in df.reflection_text)
    print(f"(înainte de tăiere: {100 * full_refl / corpus_tok:.2f}%)")
    if args.dry_run:
        print("dry-run: nimic scris")
        return
    if not FULL.exists():
        df.to_parquet(FULL, index=False)
        print(f"copie completă -> {FULL.name}")
    tmp = SIDECAR.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(SIDECAR)
    print(f"scris {SIDECAR.name}: {len(out):,} documente. The pipeline uses it for SPP, SPP-MT and Filtered.")


if __name__ == "__main__":
    main()

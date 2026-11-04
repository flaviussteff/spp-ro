"""
Builds a genuinely held-out validation set for perplexity / alignment-tax evaluation.

Why: the previous evaluation used the tail of wiki_clean.parquet and news_recent.parquet,
but both files were used in full by prepare_scale_corpus.py, so 92-99% of the "held-out"
passages were in the training corpus (and ~800 of them were also SPP reflection hosts).

What: takes a FineWeb-2 Romanian shard that was NOT consumed by the training corpus
(the corpus used only the first ~19 shards), removes every document whose 120-char
prefix or full MD5 appears in corpus_scale_stream.parquet or in the reflection sidecar,
applies the same basic quality filters as the corpus, and saves N documents to
data/eval/heldout_fineweb.parquet together with a JSON of overlap statistics.
"""
import sys
import json
import random
import hashlib
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pyarrow as pa
import pyarrow.parquet as pq

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config import CLEAN_DATA_DIR, SIDECAR_DIR, RAW_FINEWEB_DIR, HELDOUT_FILE, EVAL_DATA_DIR
except ImportError:
    from src.config import CLEAN_DATA_DIR, SIDECAR_DIR, RAW_FINEWEB_DIR, HELDOUT_FILE, EVAL_DATA_DIR


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def quality_ok(text: str) -> bool:
    if len(text) < 300:
        return False
    words = text.split()
    if len(words) < 60 or len(words) > 3000:
        return False
    alnum = sum(c.isalnum() for c in text) / max(1, len(text))
    return alnum >= 0.60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=100, help="FineWeb-2 shard index not used in training (corpus used shards 0-18)")
    ap.add_argument("--n", type=int, default=3000, help="number of held-out documents to keep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus = CLEAN_DATA_DIR / "corpus_scale_stream.parquet"
    sidecar = SIDECAR_DIR / "reflections.parquet"
    shard = RAW_FINEWEB_DIR / f"shard_{args.shard:05d}.parquet"
    if not shard.exists():
        raise FileNotFoundError(f"Shard not found: {shard}. Pick another --shard present in {RAW_FINEWEB_DIR}.")

    print(f"[1/3] Indexing training corpus prefixes/hashes from {corpus.name} ...", flush=True)
    train_prefix, train_hash = set(), set()
    pf = pq.ParquetFile(str(corpus))
    for rg in range(pf.num_row_groups):
        for t in pf.read_row_group(rg, columns=["text"])["text"].to_pylist():
            s = t.strip()
            train_prefix.add(s[:120])
            train_hash.add(md5(s))
    if sidecar.exists():
        for t in pq.read_table(str(sidecar), columns=["text"])["text"].to_pylist():
            s = t.strip()
            train_prefix.add(s[:120])
            train_hash.add(md5(s))
    print(f"      {len(train_prefix):,} prefixes indexed.", flush=True)

    print(f"[2/3] Filtering {shard.name} ...", flush=True)
    docs = pq.read_table(str(shard), columns=["text"])["text"].to_pylist()
    stats = {"shard": shard.name, "shard_docs": len(docs), "overlap_prefix": 0, "overlap_hash": 0, "low_quality": 0, "kept_pool": 0}
    pool, seen = [], set()
    for t in docs:
        if not t:
            continue
        s = " ".join(line.strip() for line in t.replace("\r", "").split("\n") if line.strip()).strip()
        if not quality_ok(s):
            stats["low_quality"] += 1
            continue
        if s[:120] in train_prefix:
            stats["overlap_prefix"] += 1
            continue
        h = md5(s)
        if h in train_hash:
            stats["overlap_hash"] += 1
            continue
        if h in seen:
            continue
        seen.add(h)
        pool.append(s)
    stats["kept_pool"] = len(pool)
    random.Random(args.seed).shuffle(pool)
    kept = pool[: args.n]
    stats["saved"] = len(kept)

    print(f"[3/3] Saving {len(kept):,} documents to {HELDOUT_FILE} ...", flush=True)
    pq.write_table(pa.table({"text": kept, "source": [f"fineweb2_ro_{shard.stem}"] * len(kept)}), str(HELDOUT_FILE), compression="zstd")
    with open(EVAL_DATA_DIR / "heldout_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("Done. Held-out set is verified disjoint from the training corpus and the reflection sidecar.")


if __name__ == "__main__":
    main()

"""
Builds a genuinely held-out validation set for the perplexity / alignment-tax evaluation and the cloze probe.

History:
 * v1 used the tails of wiki_clean.parquet and news_recent.parquet; both files are part of the training corpus, so
   92-99 % of the "held-out" passages had been trained on (and ~800 were SPP reflection hosts).
 * v2 took an unconsumed FineWeb-2 shard, but normalised it DIFFERENTLY from the corpus: line breaks were turned into
   spaces and the cedilla letters ş/ţ were left as they were, while every training document had gone through
   `prepare_scale_corpus.clean_text_content` (NFC, ş/ţ -> comma-below ș/ț, line breaks kept). 37.7 % of the v2 documents
   contained ş/ţ, which the byte-level BPE splits into byte pieces never seen in training, and none contained a line
   break (98 % of the training documents do). The overlap check compared the two different normalisations, so it could
   also miss duplicates.

v3 (this script):
 1. every document of the shard goes through exactly the corpus cleaning (`clean_text_content`, imported from
    prepare_scale_corpus.py) and the corpus length filter (>= 150 characters before and after cleaning);
 2. a document is dropped if its first 64 cleaned characters (the corpus deduplication key) or the MD5 of its cleaned
    text occur in corpus_scale_stream.parquet or in the reflection sidecar (cleaned the same way); duplicates inside
    the shard are removed with the same keys;
 3. a quality filter that is STRICTER than the corpus one keeps the measurement on running text: >= 300 characters,
    60-3,000 words, >= 60 % alphanumeric characters;
 4. N documents are drawn with a fixed seed and saved to data/eval/heldout_fineweb.parquet; data/eval/heldout_stats.json
    records the count of every filter and, if a previous held-out file exists, how many of its documents turn out to
    be in the training data once they are normalised like the corpus.
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
    from prepare_scale_corpus import clean_text_content
except ImportError:
    from src.config import CLEAN_DATA_DIR, SIDECAR_DIR, RAW_FINEWEB_DIR, HELDOUT_FILE, EVAL_DATA_DIR
    from src.prepare_scale_corpus import clean_text_content

ARCHIVE_FINEWEB_DIR = _ROOT_DIR / "archive" / "raw_corpus" / "fineweb_shards"
CORPUS_MIN_CHARS = 150          # prepare_scale_corpus.py: raw and cleaned text must have >= 150 characters
DEDUP_PREFIX = 64               # prepare_scale_corpus.py deduplicates on the first 64 cleaned characters


def md5_bytes(text: str) -> bytes:
    return hashlib.md5(text.encode("utf-8")).digest()


def keys(cleaned: str):
    """(prefix key, full-text key) of a cleaned document; digests keep the index of 1.5M documents small."""
    return md5_bytes(cleaned[:DEDUP_PREFIX]), md5_bytes(cleaned)


def quality_ok(text: str) -> bool:
    if len(text) < 300:
        return False
    words = text.split()
    if len(words) < 60 or len(words) > 3000:
        return False
    alnum = sum(c.isalnum() for c in text) / max(1, len(text))
    return alnum >= 0.60


def find_shard(shard_idx: int, shard_dir: str | None) -> Path:
    name = f"shard_{shard_idx:05d}.parquet"
    dirs = [Path(shard_dir)] if shard_dir else [RAW_FINEWEB_DIR, ARCHIVE_FINEWEB_DIR]
    for d in dirs:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found in: " + ", ".join(str(d) for d in dirs) + " (use --shard-dir / --shard)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=100, help="FineWeb-2 shard index not consumed by the corpus stream")
    ap.add_argument("--shard-dir", default=None, help=f"folder with the shards (default: {RAW_FINEWEB_DIR}, then {ARCHIVE_FINEWEB_DIR})")
    ap.add_argument("--n", type=int, default=3000, help="number of held-out documents to keep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    corpus = CLEAN_DATA_DIR / "corpus_scale_stream.parquet"
    sidecar = SIDECAR_DIR / "reflections.parquet"
    shard = find_shard(args.shard, args.shard_dir)

    print(f"[1/4] Indexing the training corpus ({corpus.name}) with the corpus keys (first {DEDUP_PREFIX} cleaned characters + MD5) ...", flush=True)
    train_prefix, train_hash = set(), set()
    n_corpus = 0
    pf = pq.ParquetFile(str(corpus))
    for rg in range(pf.num_row_groups):
        for t in pf.read_row_group(rg, columns=["text"])["text"].to_pylist():
            c = clean_text_content(t or "")        # idempotent on corpus texts; applied so that both sides match exactly
            kp, kh = keys(c)
            train_prefix.add(kp)
            train_hash.add(kh)
            n_corpus += 1
        if (rg + 1) % 25 == 0:
            print(f"      row group {rg + 1}/{pf.num_row_groups}", flush=True)
    n_sidecar = 0
    if sidecar.exists():
        for t in pq.read_table(str(sidecar), columns=["text"])["text"].to_pylist():
            kp, kh = keys(clean_text_content(t or ""))
            train_prefix.add(kp)
            train_hash.add(kh)
            n_sidecar += 1
    print(f"      {n_corpus:,} corpus + {n_sidecar:,} sidecar documents -> {len(train_prefix):,} prefix keys.", flush=True)

    def in_training(cleaned: str) -> bool:
        kp, kh = keys(cleaned)
        return kp in train_prefix or kh in train_hash

    print(f"[2/4] Filtering {shard} ...", flush=True)
    docs = pq.read_table(str(shard), columns=["text"])["text"].to_pylist()
    stats = {"version": 3, "shard": shard.name, "shard_docs": len(docs), "normalisation": "prepare_scale_corpus.clean_text_content",
             "overlap_key": f"first {DEDUP_PREFIX} cleaned characters or MD5 of the cleaned text",
             "too_short_corpus_filter": 0, "overlap_prefix": 0, "overlap_hash": 0, "duplicate_in_shard": 0, "low_quality": 0}
    pool, seen = [], set()
    for t in docs:
        if not t or len(t) < CORPUS_MIN_CHARS:
            stats["too_short_corpus_filter"] += 1
            continue
        c = clean_text_content(t)
        if len(c) < CORPUS_MIN_CHARS:
            stats["too_short_corpus_filter"] += 1
            continue
        kp, kh = keys(c)
        if kp in train_prefix:
            stats["overlap_prefix"] += 1
            continue
        if kh in train_hash:
            stats["overlap_hash"] += 1
            continue
        if kp in seen or kh in seen:
            stats["duplicate_in_shard"] += 1
            continue
        seen.add(kp)
        seen.add(kh)
        if not quality_ok(c):
            stats["low_quality"] += 1
            continue
        pool.append(c)
    stats["kept_pool"] = len(pool)
    random.Random(args.seed).shuffle(pool)
    kept = pool[: args.n]
    stats["saved"] = len(kept)
    stats["saved_with_cedilla"] = sum(1 for c in kept if any(ch in c for ch in "şŞţŢ"))
    stats["saved_with_line_break"] = sum(1 for c in kept if "\n" in c)

    print("[3/4] Re-checking the previous held-out file (if any) against the corpus with the corpus normalisation ...", flush=True)
    if HELDOUT_FILE.exists():
        try:
            old = [x for x in pq.read_table(str(HELDOUT_FILE), columns=["text"])["text"].to_pylist() if x]
            stats["previous_file"] = {"docs": len(old),
                                      "in_training_after_corpus_normalisation": sum(1 for x in old if in_training(clean_text_content(x))),
                                      "with_cedilla": sum(1 for x in old if any(ch in x for ch in "şŞţŢ")),
                                      "with_line_break": sum(1 for x in old if "\n" in x)}
        except Exception as e:                      # a missing/corrupt old file must not stop the rebuild
            stats["previous_file"] = {"error": str(e)}
    assert not any(in_training(c) for c in kept), "a kept document is in the training data"

    print(f"[4/4] Saving {len(kept):,} documents to {HELDOUT_FILE} ...", flush=True)
    pq.write_table(pa.table({"text": kept, "source": [f"fineweb2_ro_{shard.stem}"] * len(kept)}), str(HELDOUT_FILE), compression="zstd")
    with open(EVAL_DATA_DIR / "heldout_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print("Done. The held-out set is normalised like the corpus and disjoint from the training corpus and the reflection sidecar.")


if __name__ == "__main__":
    main()

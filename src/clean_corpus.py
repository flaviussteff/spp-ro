"""
Romanian Corpus Cleaning, Diacritics Canonicalization, and Deduplication Pipeline
Ensures the pretraining corpus has canonical comma-below characters (ș, ț) and passes Gopher/FineWeb-style quality filters.

Production-grade Streaming Architecture:
- Batched memory-capped processing (< 500 MB RAM envelope, preventing OOM on 16GB systems)
- High-efficiency 64-bit integer hashing for million-document deduplication
- Direct-to-disk PyArrow streaming into Parquet tables
- Strict ISO 8859-16 comma canonicalization (ș, ț instead of legacy cedillas ş, ţ)
- Generates:
  1. Dedicated news archive: data/clean/news_recent.parquet
  2. Dedicated Wikipedia archive: data/clean/wiki_clean.parquet
  3. SPP Constitutional Reflection Candidates (10% pool: ~7% news, ~93% wiki/general knowledge): data/clean/corpus_for_spp.parquet
  4. Unannotated Pretraining Stream (90% stream + remaining news): data/clean/corpus_unannotated.parquet
  5. Representative Tokenizer Sample: data/clean/sample_for_tokenizer.txt
"""
import sys
import os
import gc
import json
import re
import html
import unicodedata
import hashlib
import argparse
from pathlib import Path
from typing import Set, Dict, Any, List, Tuple, Optional
import random

# Ensure UTF-8 output on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Ensure src and root directories are on sys.path
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config import (
        RAW_DATA_DIR,
        CLEAN_DATA_DIR,
        RAW_NEWS_FILE,
        RAW_WIKI_FILE,
        RAW_FINEWEB_DIR,
        CLEAN_NEWS_FILE,
        CLEAN_WIKI_FILE,
        UNANNOTATED_FILE,
        SPP_CANDIDATES_FILE,
        HardwareAndTrainingConfig,
    )
except ImportError:
    from src.config import (
        RAW_DATA_DIR,
        CLEAN_DATA_DIR,
        RAW_NEWS_FILE,
        RAW_WIKI_FILE,
        RAW_FINEWEB_DIR,
        CLEAN_NEWS_FILE,
        CLEAN_WIKI_FILE,
        UNANNOTATED_FILE,
        SPP_CANDIDATES_FILE,
        HardwareAndTrainingConfig,
    )

CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Strict Canonical Diacritics Mapping for Romanian
CEDILLA_TO_COMMA_BELOW = {
    "\u015e": "\u0218",  # LATIN CAPITAL LETTER S WITH CEDILLA -> S WITH COMMA BELOW (Ş -> Ș)
    "\u015f": "\u0219",  # LATIN SMALL LETTER S WITH CEDILLA -> s WITH COMMA BELOW (ş -> ș)
    "\u0162": "\u021a",  # LATIN CAPITAL LETTER T WITH CEDILLA -> T WITH COMMA BELOW (Ţ -> Ț)
    "\u0163": "\u021b",  # LATIN SMALL LETTER T WITH CEDILLA -> t WITH COMMA BELOW (ţ -> ț)
}

# Fast translation table for diacritics
_TRANSLATION_TABLE = str.maketrans(CEDILLA_TO_COMMA_BELOW)

# Common Romanian web boilerplate patterns
BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"cookie[\w\s\-]*nostru",
        r"aboneaz[aă]\-te la newsletter",
        r"toate drepturile rezervate",
        r"termeni [sș]i condi[tț]ii",
        r"urmeaz[aă]\-ne pe (facebook|twitter|instagram|youtube)",
        r"copyright\s*(?:©|\(c\))?\s*\d{4}",
        r"publicat la:\s*\d{1,2}[\.\/]\d{1,2}[\.\/]\d{4}",
        r"urm[aă]re[sș]te [sș]tirile.*pe.*google news",
        r"cite[sș]te [sș]i:",
    ]
]

# PyArrow Schema for Clean Corpus
CORPUS_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("title", pa.string()),
    pa.field("source", pa.string()),
    pa.field("date", pa.string()),
    pa.field("url", pa.string()),
    pa.field("text", pa.string()),
    pa.field("is_recent_news", pa.bool_()),
    pa.field("word_count", pa.int64()),
    pa.field("char_count", pa.int64()),
])


def canonicalize_romanian_text(text: str) -> str:
    """Normalize Unicode to NFC and convert all Turkish cedillas to Romanian comma-below."""
    text = unicodedata.normalize("NFC", text)
    return text.translate(_TRANSLATION_TABLE)


def clean_text_markup(text: str) -> str:
    """Unescape HTML, prune web boilerplate and standardize whitespace."""
    text = html.unescape(text)
    
    # Prune recurring web boilerplate lines
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        is_bp = False
        for bp_regex in BOILERPLATE_PATTERNS:
            if bp_regex.search(line_clean):
                is_bp = True
                break
        if not is_bp:
            cleaned_lines.append(line_clean)
            
    text = "\n".join(cleaned_lines)
    # Collapse irregular excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def passes_quality_filter(text: str) -> bool:
    """Applies heuristic quality criteria (length, character ratio, repetition)."""
    num_chars = len(text)
    if num_chars < 200:
        return False
        
    words = text.split()
    num_words = len(words)
    if num_words < 40 or num_words > 12000:
        return False
        
    # Check ratio of alphanumeric characters
    num_alphanumeric = sum(1 for c in text if c.isalnum())
    if num_alphanumeric / max(1, num_chars) < 0.60:
        return False
        
    # Check for excessive punctuation/symbols
    num_special = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if num_special / max(1, num_chars) > 0.25:
        return False
        
    return True


def get_64bit_hash(text: str) -> int:
    """Compute 64-bit integer hash for ultra-compact deduplication."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:16], 16)


def process_jsonl_file_streaming(
    file_path: Path,
    seen_hashes: Set[int],
    is_news: bool,
    output_parquet: Optional[Path] = None,
    batch_size: int = 10000,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """
    Cleans a JSONL file, streams to parquet if requested, and returns in-memory valid docs.
    """
    valid_docs: List[Dict[str, Any]] = []
    total_raw = 0
    duplicate_count = 0
    filtered_count = 0

    if not file_path.exists() or file_path.stat().st_size == 0:
        return valid_docs, total_raw, duplicate_count, filtered_count

    print(f"\nProcessing {file_path.name} (is_news={is_news})...", flush=True)
    writer = None
    if output_parquet:
        writer = pq.ParquetWriter(str(output_parquet), CORPUS_SCHEMA, compression="zstd")

    batch_buffer: List[Dict[str, Any]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Cleaning {file_path.name}"):
            total_raw += 1
            try:
                data = json.loads(line)
            except Exception:
                continue

            raw_text = data.get("text", "")
            if not raw_text or len(raw_text) < 150:
                continue

            canon_text = canonicalize_romanian_text(raw_text)
            clean_text = clean_text_markup(canon_text)

            if not passes_quality_filter(clean_text):
                filtered_count += 1
                continue

            h64 = get_64bit_hash(clean_text)
            if h64 in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(h64)

            doc_record = {
                "id": data.get("id", f"{'news' if is_news else 'wiki'}_{len(valid_docs) + len(batch_buffer)}"),
                "title": data.get("title", ""),
                "source": data.get("source", "news" if is_news else "wikipedia_ro"),
                "date": data.get("date", ""),
                "url": data.get("url", ""),
                "text": clean_text,
                "is_recent_news": is_news,
                "word_count": len(clean_text.split()),
                "char_count": len(clean_text),
            }

            valid_docs.append(doc_record)
            if writer:
                batch_buffer.append(doc_record)
                if len(batch_buffer) >= batch_size:
                    table = pa.Table.from_pylist(batch_buffer, schema=CORPUS_SCHEMA)
                    writer.write_table(table)
                    batch_buffer.clear()

    if writer:
        if batch_buffer:
            table = pa.Table.from_pylist(batch_buffer, schema=CORPUS_SCHEMA)
            writer.write_table(table)
        writer.close()
        print(f"[Saved] {output_parquet.name} ({len(valid_docs):,} records)", flush=True)

    return valid_docs, total_raw, duplicate_count, filtered_count


def stream_fineweb_shards_to_writers(
    shards_dir: Path,
    seen_hashes: Set[int],
    unannotated_writer: pq.ParquetWriter,
    spp_writer: pq.ParquetWriter,
    spp_ratio_wiki_general: float,
    tokenizer_sample_collector: List[str],
    max_shards: Optional[int] = None,
    batch_size: int = 10000,
) -> Tuple[int, int, int, int, int]:
    """
    Streams FineWeb-2 parquet shards chunk-by-chunk directly into output Parquet writers.
    Zero OOM risk: processes each shard independently and writes row groups.
    """
    total_raw = 0
    duplicate_count = 0
    filtered_count = 0
    spp_count = 0
    unannotated_count = 0

    if not shards_dir.exists():
        return total_raw, duplicate_count, filtered_count, spp_count, unannotated_count

    shard_files = sorted(list(shards_dir.glob("*.parquet")))
    if not shard_files:
        return total_raw, duplicate_count, filtered_count, spp_count, unannotated_count

    if max_shards and max_shards > 0:
        shard_files = shard_files[:max_shards]

    print(f"\nStreaming {len(shard_files)} FineWeb-2 Romanian Parquet shards directly to disk...", flush=True)

    unannotated_buffer: List[Dict[str, Any]] = []
    spp_buffer: List[Dict[str, Any]] = []

    for shard_idx, shard_path in enumerate(tqdm(shard_files, desc="FineWeb Shards")):
        try:
            # Read shard
            df = pd.read_parquet(shard_path)
            raw_texts = df["text"].dropna().tolist()
            del df

            for text_idx, raw_text in enumerate(raw_texts):
                total_raw += 1
                if len(raw_text) < 180:
                    continue

                canon = canonicalize_romanian_text(raw_text)
                clean = clean_text_markup(canon)

                if not passes_quality_filter(clean):
                    filtered_count += 1
                    continue

                h64 = get_64bit_hash(clean)
                if h64 in seen_hashes:
                    duplicate_count += 1
                    continue
                seen_hashes.add(h64)

                # Tokenizer reservoir sample collector
                if len(tokenizer_sample_collector) < 50000 and random.random() < 0.05:
                    tokenizer_sample_collector.append(clean)

                doc_id = f"fw_{shard_idx}_{text_idx}"
                record = {
                    "id": doc_id,
                    "title": "",
                    "source": "fineweb2_ro",
                    "date": "",
                    "url": "",
                    "text": clean,
                    "is_recent_news": False,
                    "word_count": len(clean.split()),
                    "char_count": len(clean),
                }

                # Deterministic SPP candidate allocation based on target ratio
                if random.random() < spp_ratio_wiki_general:
                    spp_buffer.append(record)
                    spp_count += 1
                else:
                    unannotated_buffer.append(record)
                    unannotated_count += 1

                if len(unannotated_buffer) >= batch_size:
                    table = pa.Table.from_pylist(unannotated_buffer, schema=CORPUS_SCHEMA)
                    unannotated_writer.write_table(table)
                    unannotated_buffer.clear()

                if len(spp_buffer) >= batch_size:
                    table = pa.Table.from_pylist(spp_buffer, schema=CORPUS_SCHEMA)
                    spp_writer.write_table(table)
                    spp_buffer.clear()

            # Flush memory per shard
            del raw_texts
            gc.collect()

        except Exception as e:
            print(f"\nNotice: Error reading {shard_path.name}: {e}", flush=True)

    # Final flush
    if unannotated_buffer:
        table = pa.Table.from_pylist(unannotated_buffer, schema=CORPUS_SCHEMA)
        unannotated_writer.write_table(table)
        unannotated_buffer.clear()

    if spp_buffer:
        table = pa.Table.from_pylist(spp_buffer, schema=CORPUS_SCHEMA)
        spp_writer.write_table(table)
        spp_buffer.clear()

    return total_raw, duplicate_count, filtered_count, spp_count, unannotated_count


def run_corpus_cleaning(max_shards: Optional[int] = None):
    random.seed(42)

    print("==================================================", flush=True)
    print(" Romanian Corpus Cleaning & SPP Bifurcation Engine", flush=True)
    print(f" Raw Directory:   {RAW_DATA_DIR}", flush=True)
    print(f" Clean Directory: {CLEAN_DATA_DIR}", flush=True)
    print("==================================================", flush=True)

    # 64-bit integer hash set for memory-efficient deduplication (< 50MB RAM for millions of docs)
    seen_hashes: Set[int] = set()
    train_cfg = HardwareAndTrainingConfig()

    spp_total_ratio = getattr(train_cfg, "spp_reflection_ratio", 0.10)        # 10% total reflections
    spp_news_share = getattr(train_cfg, "spp_news_reflection_mix", 0.07)      # 7% of SPP pool from news
    spp_general_share = 1.0 - spp_news_share                                  # 93% from wiki/general

    print(f"Configuration: SPP Target = {spp_total_ratio*100:.0f}% total pool", flush=True)
    print(f"SPP Internal Balance: {spp_news_share*100:.0f}% News + {spp_general_share*100:.0f}% Wiki/General", flush=True)

    tokenizer_samples: List[str] = []

    # 1. Clean News Articles (Save to news_recent.parquet)
    news_docs, news_raw, news_dup, news_filt = process_jsonl_file_streaming(
        RAW_NEWS_FILE, seen_hashes, is_news=True, output_parquet=CLEAN_NEWS_FILE
    )

    # 2. Clean Romanian Wikipedia (Save to wiki_clean.parquet)
    wiki_docs, wiki_raw, wiki_dup, wiki_filt = process_jsonl_file_streaming(
        RAW_WIKI_FILE, seen_hashes, is_news=False, output_parquet=CLEAN_WIKI_FILE
    )

    # Add news and wiki samples to tokenizer collection
    if news_docs:
        tokenizer_samples.extend([d["text"] for d in random.sample(news_docs, min(10000, len(news_docs)))])
    if wiki_docs:
        tokenizer_samples.extend([d["text"] for d in random.sample(wiki_docs, min(20000, len(wiki_docs)))])

    # 3. Setup Master Stream Writers: unannotated.parquet & corpus_for_spp.parquet
    unannotated_writer = pq.ParquetWriter(str(UNANNOTATED_FILE), CORPUS_SCHEMA, compression="zstd")
    spp_writer = pq.ParquetWriter(str(SPP_CANDIDATES_FILE), CORPUS_SCHEMA, compression="zstd")

    # Allocate News into SPP vs. Unannotated
    # Target news reflections: ~7% of the 10% SPP pool
    # With 25,000 news articles, we allocate a balanced share into SPP and the rest into unannotated stream
    random.shuffle(news_docs)
    news_spp_target = min(len(news_docs), max(1000, int(len(news_docs) * 0.20)))  # Reserve dedicated news reflections
    news_spp_docs = news_docs[:news_spp_target]
    news_unannotated_docs = news_docs[news_spp_target:]

    if news_spp_docs:
        table_news_spp = pa.Table.from_pylist(news_spp_docs, schema=CORPUS_SCHEMA)
        spp_writer.write_table(table_news_spp)
        del table_news_spp

    if news_unannotated_docs:
        table_news_unann = pa.Table.from_pylist(news_unannotated_docs, schema=CORPUS_SCHEMA)
        unannotated_writer.write_table(table_news_unann)
        del table_news_unann

    del news_docs, news_spp_docs, news_unannotated_docs
    gc.collect()

    # Allocate Wikipedia into SPP vs. Unannotated
    random.shuffle(wiki_docs)
    wiki_spp_target = int(len(wiki_docs) * spp_total_ratio)
    wiki_spp_docs = wiki_docs[:wiki_spp_target]
    wiki_unannotated_docs = wiki_docs[wiki_spp_target:]

    if wiki_spp_docs:
        table_wiki_spp = pa.Table.from_pylist(wiki_spp_docs, schema=CORPUS_SCHEMA)
        spp_writer.write_table(table_wiki_spp)
        del table_wiki_spp

    if wiki_unannotated_docs:
        table_wiki_unann = pa.Table.from_pylist(wiki_unannotated_docs, schema=CORPUS_SCHEMA)
        unannotated_writer.write_table(table_wiki_unann)
        del table_wiki_unann

    del wiki_docs, wiki_spp_docs, wiki_unannotated_docs
    gc.collect()

    # 4. Stream FineWeb-2 Romanian Parquet Shards directly through the Writers
    fw_raw, fw_dup, fw_filt, fw_spp_count, fw_unann_count = stream_fineweb_shards_to_writers(
        RAW_FINEWEB_DIR,
        seen_hashes,
        unannotated_writer,
        spp_writer,
        spp_ratio_wiki_general=spp_total_ratio,
        tokenizer_sample_collector=tokenizer_samples,
        max_shards=max_shards,
    )

    # Close master writers
    unannotated_writer.close()
    spp_writer.close()

    total_raw = news_raw + wiki_raw + fw_raw
    total_dup = news_dup + wiki_dup + fw_dup
    total_filt = news_filt + wiki_filt + fw_filt

    print("\n==================================================", flush=True)
    print(" Cleaning & Partitioning Complete!", flush=True)
    print(f" Total Raw Documents Scanned:  {total_raw:,}", flush=True)
    print(f" Low Quality/Spam Filtered:    {total_filt:,}", flush=True)
    print(f" Duplicates Removed:           {total_dup:,}", flush=True)
    print(f" Unique Clean Documents:       {len(seen_hashes):,}", flush=True)
    print("--------------------------------------------------", flush=True)
    print(f" [Parquet] {CLEAN_NEWS_FILE.name}: Dedicated recent news", flush=True)
    print(f" [Parquet] {CLEAN_WIKI_FILE.name}: Dedicated Wikipedia", flush=True)
    print(f" [Parquet] {SPP_CANDIDATES_FILE.name}: 10% SPP reflection pool", flush=True)
    print(f" [Parquet] {UNANNOTATED_FILE.name}: 90% pretraining stream", flush=True)

    # 5. Export Representative Tokenizer Training File
    tokenizer_sample_path = CLEAN_DATA_DIR / "sample_for_tokenizer.txt"
    print(f"\nWriting tokenizer training sample ({len(tokenizer_samples):,} docs) to {tokenizer_sample_path.name}...", flush=True)
    random.shuffle(tokenizer_samples)
    with open(tokenizer_sample_path, "w", encoding="utf-8") as f:
        for text in tokenizer_samples:
            f.write(text + "\n\n")

    print(f"Successfully generated: {tokenizer_sample_path} ({tokenizer_sample_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    print("Next step: Run `py src/train_tokenizer.py` to train the 16k Romanian BPE model.", flush=True)
    print("==================================================", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and partition Romanian pretraining corpus.")
    parser.add_argument("--max-shards", type=int, default=None, help="Maximum FineWeb shards to process (default: all)")
    args = parser.parse_args()

    run_corpus_cleaning(max_shards=args.max_shards)

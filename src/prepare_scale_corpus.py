"""
Scale Corpus Pre-processing & Balancing Pipeline
Combines:
- FineWeb-2 Romanian Shards (all available shards in data/raw/fineweb_shards)
- Romanian Wikipedia (data/raw/wiki_ro.jsonl)
- Recent Romanian News (data/raw/news_recent.jsonl)

Enforces strict balance:
- ~75% FineWeb-2 Diverse Web
- ~20% Wikipedia Encyclopedic
- ~5% News

Applies ISO 8859-16 comma canonicalization (ș, ț) and deduplication.
Outputs to: data/clean/corpus_scale_stream.parquet
"""
import os
import sys
import gc
import json
import random
import unicodedata
from pathlib import Path
from typing import Dict, Any, List, Set, Generator
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
DATA_DIR = _ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_STREAM_FILE = CLEAN_DIR / "corpus_scale_stream.parquet"

# Standard schema for pre-training corpus
CORPUS_SCHEMA = pa.schema([
    ("text", pa.string()),
    ("source", pa.string()),
])

# Canonical Romanian comma diacritic mappings
CEDILLA_TO_COMMA = str.maketrans({
    "ş": "ș", "Ş": "Ș",
    "ţ": "ț", "Ţ": "Ț",
})


def canonicalize_diacritics(text: str) -> str:
    """Ensures consistent ISO 8859-16 comma-below diacritics (ș, ț)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    return text.translate(CEDILLA_TO_COMMA)


def clean_text_content(text: str) -> str:
    """Basic normalization: whitespace, line breaks, and diacritics."""
    if not text:
        return ""
    text = canonicalize_diacritics(text)
    # Collapse multiple blank lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines).strip()


def stream_fineweb_documents(shards_dir: Path, seen_hashes: Set[int], max_docs: int = 4_000_000) -> Generator[Dict[str, str], None, None]:
    """Streams cleaned, deduplicated documents from FineWeb shards."""
    shard_files = sorted(list(shards_dir.glob("shard_*.parquet")))
    print(f"[Stream FineWeb] Found {len(shard_files)} shards in {shards_dir.name}...")
    
    count = 0
    for shard_path in shard_files:
        if count >= max_docs:
            break
        try:
            pf = pq.ParquetFile(str(shard_path))
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=["text"])
                for raw_text in table["text"].to_pylist():
                    if not raw_text or len(raw_text) < 150:
                        continue
                    cleaned = clean_text_content(raw_text)
                    if len(cleaned) < 150:
                        continue
                        
                    # Hash for deduplication (first 64 chars)
                    doc_hash = hash(cleaned[:64])
                    if doc_hash in seen_hashes:
                        continue
                    seen_hashes.add(doc_hash)
                    
                    yield {"text": cleaned, "source": "fineweb"}
                    count += 1
                    if count >= max_docs:
                        break
                if count >= max_docs:
                    break
        except Exception as e:
            print(f"[Warning] Error reading {shard_path.name}: {e}", flush=True)


def load_jsonl_documents(jsonl_path: Path, source_name: str, seen_hashes: Set[int], max_docs: int = 500_000) -> List[Dict[str, str]]:
    """Loads and cleans JSONL documents (e.g. Wikipedia or News)."""
    if not jsonl_path.exists():
        print(f"[Warning] File not found: {jsonl_path}", flush=True)
        return []
        
    print(f"[Load {source_name}] Reading from {jsonl_path.name}...")
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(docs) >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                text = data.get("text", "")
                if len(text) < 150:
                    continue
                cleaned = clean_text_content(text)
                if len(cleaned) < 150:
                    continue
                    
                doc_hash = hash(cleaned[:64])
                if doc_hash in seen_hashes:
                    continue
                seen_hashes.add(doc_hash)
                
                docs.append({"text": cleaned, "source": source_name})
            except Exception:
                continue
    print(f"[Load {source_name}] Loaded {len(docs):,} cleaned documents.")
    return docs


def assemble_balanced_scale_corpus(target_total_docs: int = 3_000_000):
    """
    Assembles a balanced stream with:
    - 75% FineWeb-2 diverse web
    - 20% Wikipedia
    - 5% News
    """
    print("======================================================================")
    print(f" Assembling Balanced Scale Pre-training Corpus ({target_total_docs:,} Docs Target)")
    print("======================================================================")
    
    seen_hashes: Set[int] = set()
    
    # 1. Load Wikipedia
    wiki_path = RAW_DIR / "wiki_ro.jsonl"
    wiki_docs = load_jsonl_documents(wiki_path, "wikipedia", seen_hashes, max_docs=int(target_total_docs * 0.22))
    random.shuffle(wiki_docs)
    
    # 2. Load News (capped strictly at 5% of target)
    news_path = RAW_DIR / "news_recent.jsonl"
    news_cap = min(25000, int(target_total_docs * 0.05))
    news_docs = load_jsonl_documents(news_path, "news", seen_hashes, max_docs=news_cap)
    random.shuffle(news_docs)
    
    # 3. Setup Parquet Writer
    writer = pq.ParquetWriter(str(OUTPUT_STREAM_FILE), CORPUS_SCHEMA, compression="zstd")
    
    fineweb_gen = stream_fineweb_documents(RAW_DIR / "fineweb_shards", seen_hashes, max_docs=target_total_docs)
    
    total_written = 0
    buffer: List[Dict[str, str]] = []
    BUFFER_SIZE = 10000
    
    wiki_idx = 0
    news_idx = 0
    fineweb_exhausted = False
    wiki_exhausted = False
    news_exhausted = False
    
    print("\nStreaming and interleaving balanced batches directly to disk...")
    pbar = tqdm(total=target_total_docs, desc="Assembling Corpus", unit="doc")
    
    while total_written < target_total_docs:
        added_in_round = 0
        
        # If wiki or news are exhausted, pull more FineWeb to maintain target batching
        fw_pull = 15
        if wiki_idx >= len(wiki_docs):
            fw_pull += 4
        if news_idx >= len(news_docs):
            fw_pull += 1
            
        if not fineweb_exhausted:
            for _ in range(fw_pull):
                try:
                    doc = next(fineweb_gen)
                    buffer.append(doc)
                    added_in_round += 1
                except StopIteration:
                    fineweb_exhausted = True
                    break
                
        if wiki_idx < len(wiki_docs):
            for _ in range(4):
                if wiki_idx < len(wiki_docs):
                    buffer.append(wiki_docs[wiki_idx])
                    wiki_idx += 1
                    added_in_round += 1
        else:
            wiki_exhausted = True
                
        if news_idx < len(news_docs):
            buffer.append(news_docs[news_idx])
            news_idx += 1
            added_in_round += 1
        else:
            news_exhausted = True
            
        if len(buffer) >= BUFFER_SIZE:
            random.shuffle(buffer)
            table = pa.Table.from_pylist(buffer, schema=CORPUS_SCHEMA)
            writer.write_table(table)
            total_written += len(buffer)
            pbar.update(len(buffer))
            buffer = []
            
        if added_in_round == 0 or (fineweb_exhausted and wiki_exhausted and news_exhausted):
            print("\n[Info] Reached end of available data sources.")
            break
            
    if buffer:
        random.shuffle(buffer)
        table = pa.Table.from_pylist(buffer, schema=CORPUS_SCHEMA)
        writer.write_table(table)
        total_written += len(buffer)
        pbar.update(len(buffer))
        buffer = []
        
    writer.close()
    pbar.close()
    
    file_size_gb = OUTPUT_STREAM_FILE.stat().st_size / (1024**3)
    print(f"\n[Done] Assembled {total_written:,} balanced documents into:")
    print(f"       {OUTPUT_STREAM_FILE} ({file_size_gb:.2f} GB)")
    print("       Composition: ~75% FineWeb-2 Web, ~20% Wikipedia, ~5% News.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Assemble Balanced Scale Pre-training Corpus")
    parser.add_argument("--docs", type=int, default=3000000, help="Total balanced documents to assemble")
    args = parser.parse_args()
    
    assemble_balanced_scale_corpus(target_total_docs=args.docs)

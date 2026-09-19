"""
Scale Corpus Downloader for Romanian LLM Pre-training
Downloads 50 additional FineWeb-2 Romanian Shards (shards 71 to 120) from Hugging Face.
Total raw web text: ~8 GB compressed Parquet (>25 GB uncompressed text).
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
OUTPUT_DIR = _ROOT_DIR / "data" / "raw" / "fineweb_shards"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_single_shard(shard_idx: int) -> Optional[Path]:
    shard_name = f"shard_{shard_idx:05d}.parquet"
    out_file = OUTPUT_DIR / shard_name
    
    # Check if already fully downloaded (>10MB)
    if out_file.exists() and out_file.stat().st_size > 10_000_000:
        return out_file
        
    url = f"https://huggingface.co/datasets/rotarue/fineweb2-romanian-shards/resolve/main/{shard_name}"
    temp_file = out_file.with_suffix(".tmp")
    
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(temp_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
            if temp_file.exists() and temp_file.stat().st_size > 10_000_000:
                temp_file.replace(out_file)
                return out_file
            else:
                if temp_file.exists():
                    temp_file.unlink()
    except Exception as e:
        print(f"[Error] Shard {shard_idx}: {e}", flush=True)
        if temp_file.exists():
            temp_file.unlink()
    return None


def download_scale_shards(start_idx: int = 71, end_idx: int = 120, max_workers: int = 6) -> int:
    shards_to_download = list(range(start_idx, end_idx + 1))
    print(f"\n======================================================================")
    print(f"  Downloading {len(shards_to_download)} New FineWeb-2 Romanian Shards ({start_idx} -> {end_idx})")
    print(f"  Target Directory: {OUTPUT_DIR}")
    print(f"  Parallel Threads: {max_workers}")
    print(f"======================================================================\n", flush=True)

    completed = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_shard = {executor.submit(download_single_shard, idx): idx for idx in shards_to_download}
        with tqdm(total=len(shards_to_download), desc="FineWeb-2 Shards", unit="shard") as pbar:
            for fut in as_completed(future_to_shard):
                shard_id = future_to_shard[fut]
                res = fut.result()
                if res:
                    completed += 1
                pbar.update(1)
                
    elapsed = time.time() - start_time
    print(f"\n[Completed] Successfully downloaded {completed}/{len(shards_to_download)} shards in {elapsed/60:.1f} minutes.")
    return completed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Additional FineWeb-2 Romanian Shards")
    parser.add_argument("--start", type=int, default=71, help="Start shard index")
    parser.add_argument("--end", type=int, default=120, help="End shard index (inclusive)")
    parser.add_argument("--workers", type=int, default=6, help="Download threads")
    args = parser.parse_args()
    
    download_scale_shards(start_idx=args.start, end_idx=args.end, max_workers=args.workers)

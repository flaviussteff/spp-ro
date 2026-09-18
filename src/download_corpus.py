"""
Corpus Acquisition Script for Romanian Language Model Pretraining
Downloads and archives Romanian text at scale without requiring Hugging Face credentials.

Sources:
1. Romanian Wikipedia (wikimedia/wikipedia, 20231101.ro) - high factual quality (~500k articles)
2. Recent 5-Year Romanian News Archive (Digi24, HotNews, G4Media) via parallel monthly sitemap crawlers
3. FineWeb-2 Romanian Web Shards (rotarue/fineweb2-romanian-shards) - up to 285 shards (>150-200 GB text)
"""
import os
import sys
import json
import time
import random
import argparse
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Ensure UTF-8 console output on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress Hugging Face Windows symlink cache warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Ensure src and root directories are on sys.path for IDEs and scripts
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config import RAW_DATA_DIR, RAW_NEWS_FILE, RAW_WIKI_FILE, RAW_FINEWEB_DIR
except ImportError:
    from src.config import RAW_DATA_DIR, RAW_NEWS_FILE, RAW_WIKI_FILE, RAW_FINEWEB_DIR

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_FINEWEB_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
]


def get_headers() -> Dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
    }


def download_wikipedia_ro(max_articles: int = 100000, output_file: Path = RAW_WIKI_FILE) -> int:
    """
    Downloads articles from Romanian Wikipedia via Hugging Face datasets.
    Zero authentication needed. Streams directly to disk.
    """
    if max_articles <= 0:
        print("Skipping Wikipedia download (max_articles <= 0).", flush=True)
        return 0

    print(f"\n[1/3] Downloading Romanian Wikipedia (target: {max_articles:,} articles)...", flush=True)
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library is missing. Install via: py -m pip install datasets", flush=True)
        return 0

    count = 0
    start_time = time.time()
    
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.ro", split="train", streaming=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(ds, desc="Streaming Ro-Wiki", total=max_articles):
                text = item.get("text", "").strip()
                title = item.get("title", "").strip()
                if len(text) > 200:
                    doc = {
                        "id": f"wiki_{item.get('id', count)}",
                        "title": title,
                        "text": text,
                        "source": "wikipedia_ro",
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1
                if count >= max_articles:
                    break
                    
        elapsed = time.time() - start_time
        print(f"Saved {count:,} Wikipedia articles to {output_file} in {elapsed:.1f}s", flush=True)
        return count
    except Exception as e:
        print(f"Notice during Wikipedia streaming: {e}", flush=True)
        return count


def scrape_single_article(item_info: Dict[str, str], timeout: int = 7) -> Optional[Dict[str, Any]]:
    """Scrapes clean text and metadata for a single news URL."""
    url = item_info.get("url", "")
    source = item_info.get("source", "news")
    date_hint = item_info.get("date", "")
    title = item_info.get("title", "")

    if not url:
        return None

    try:
        resp = requests.get(url, headers=get_headers(), timeout=timeout)
        if resp.status_code != 200 or len(resp.content) < 500:
            return None

        soup = BeautifulSoup(resp.content, "html.parser")
        
        # Remove navigation, banners, sidebars, scripts, and styling
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
            tag.decompose()

        # Extract title if not provided
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)
            elif soup.title:
                title = soup.title.get_text(strip=True)

        # Source-specific body selection
        paragraphs = []
        if source == "Digi24":
            elements = soup.select("div.entry__content p, .article-body p, article p, div.news-content p")
        elif source == "HotNews":
            elements = soup.select("article p, div.article-body p, div#articleContent p, .entry-content p")
        elif source == "G4Media":
            elements = soup.select("div.entry-content p, div.post-content p, article p")
        else:
            elements = soup.find_all("p")

        for p in elements:
            p_text = p.get_text(strip=True)
            if len(p_text) > 35 and not p_text.startswith("Foto:") and not p_text.startswith("Sursa:"):
                paragraphs.append(p_text)

        if not paragraphs:
            for p in soup.find_all("p"):
                p_text = p.get_text(strip=True)
                if len(p_text) > 40:
                    paragraphs.append(p_text)

        body_text = "\n\n".join(paragraphs)
        if len(body_text) < 250:
            return None

        combined_text = f"{title}\n\n{body_text}" if title else body_text

        return {
            "title": title,
            "text": combined_text,
            "source": source,
            "date": date_hint,
            "url": url,
        }
    except Exception:
        return None


def collect_digi24_sitemap_urls(years: int = 5, workers: int = 16) -> List[Dict[str, str]]:
    """Crawls monthly sitemaps for Digi24 over the last N years in parallel."""
    print("  Discovering Digi24 historical monthly archives...", flush=True)
    current_year = datetime.datetime.now().year
    start_year = current_year - years + 1
    
    month_tasks = []
    for y in range(start_year, current_year + 1):
        for m in range(1, 13):
            if y == current_year and m > datetime.datetime.now().month:
                continue
            month_tasks.append((y, m))

    def fetch_month(y_m):
        y, m = y_m
        month_str = f"{m:02d}"
        sitemap_url = f"https://www.digi24.ro/sitemaps/sitemap-articles-{y}-{month_str}.xml"
        urls = []
        try:
            resp = requests.get(sitemap_url, headers=get_headers(), timeout=6)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for loc in root.findall(".//{*}loc"):
                    u = loc.text.strip() if loc.text else ""
                    if u and "/stiri/" in u and not u.endswith(".jpg") and not u.endswith(".png"):
                        urls.append({"url": u, "source": "Digi24", "date": f"{y}-{month_str}"})
        except Exception:
            pass
        return urls

    article_links = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for month_results in ex.map(fetch_month, month_tasks):
            article_links.extend(month_results)

    print(f"  Found {len(article_links):,} article URLs from Digi24.", flush=True)
    return article_links


def collect_g4media_sitemap_urls(max_sitemaps: int = 15, workers: int = 8) -> List[Dict[str, str]]:
    """Crawls paginated post sitemaps for G4Media in parallel."""
    print("  Discovering G4Media historical archives...", flush=True)
    
    def fetch_g4(idx):
        sitemap_url = f"https://www.g4media.ro/sitemaps/posts-{idx}.xml"
        urls = []
        try:
            resp = requests.get(sitemap_url, headers=get_headers(), timeout=6)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for loc in root.findall(".//{*}loc"):
                    u = loc.text.strip() if loc.text else ""
                    if u and u.endswith(".html"):
                        urls.append({"url": u, "source": "G4Media", "date": "recent"})
        except Exception:
            pass
        return urls

    article_links = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for sitemap_res in ex.map(fetch_g4, range(1, max_sitemaps + 1)):
            article_links.extend(sitemap_res)

    print(f"  Found {len(article_links):,} article URLs from G4Media.", flush=True)
    return article_links


def collect_hotnews_sitemap_urls(years: int = 5, samples_per_month: int = 3, workers: int = 16) -> List[Dict[str, str]]:
    """Samples daily sitemaps for HotNews across recent years in parallel."""
    print("  Discovering HotNews historical archives...", flush=True)
    current_year = datetime.datetime.now().year
    start_year = current_year - years + 1
    
    day_tasks = []
    for y in range(start_year, current_year + 1):
        for m in range(1, 13):
            if y == current_year and m > datetime.datetime.now().month:
                continue
            for day in [5, 15, 25][:samples_per_month]:
                day_tasks.append((y, m, day))

    def fetch_day(ymd):
        y, m, day = ymd
        sitemap_url = f"https://hotnews.ro/sitemap.xml?yyyy={y}&mm={m:02d}&dd={day:02d}"
        urls = []
        try:
            resp = requests.get(sitemap_url, headers=get_headers(), timeout=6)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for loc in root.findall(".//{*}loc"):
                    u = loc.text.strip() if loc.text else ""
                    if u and "hotnews.ro" in u and u != sitemap_url:
                        urls.append({"url": u, "source": "HotNews", "date": f"{y}-{m:02d}"})
        except Exception:
            pass
        return urls

    article_links = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for day_res in ex.map(fetch_day, day_tasks):
            article_links.extend(day_res)

    print(f"  Found {len(article_links):,} article URLs from HotNews.", flush=True)
    return article_links


def download_recent_romanian_news_archive(
    years: int = 5,
    max_articles: int = 25000,
    output_file: Path = RAW_NEWS_FILE,
    max_workers: int = 16,
) -> int:
    """
    Crawls and archives Romanian news from the last 5 years into a separate news file.
    """
    print(f"\n[2/3] Archiving recent {years}-year Romanian news (target: {max_articles:,} articles)...", flush=True)
    print(f"Target Output: {output_file}", flush=True)
    
    # 1. Discover article links across sources in parallel
    digi_links = collect_digi24_sitemap_urls(years=years, workers=max_workers)
    g4_links = collect_g4media_sitemap_urls(max_sitemaps=12, workers=min(8, max_workers))
    hot_links = collect_hotnews_sitemap_urls(years=years, samples_per_month=3, workers=max_workers)

    # Interleave links from all sources and shuffle
    random.seed(42)
    random.shuffle(digi_links)
    random.shuffle(g4_links)
    random.shuffle(hot_links)

    max_per_source = max_articles // 3 + 1000
    all_links = []
    all_links.extend(digi_links[:max_per_source])
    all_links.extend(g4_links[:max_per_source])
    all_links.extend(hot_links[:max_per_source])
    random.shuffle(all_links)

    print(f"Total candidate URLs assembled for scraping: {len(all_links):,}", flush=True)
    if not all_links:
        print("Warning: No article links could be discovered.", flush=True)
        return 0

    count = 0
    start_time = time.time()
    
    # Concurrent multi-threaded scraping
    with open(output_file, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(scrape_single_article, item): item 
                for item in all_links[:int(max_articles * 1.5)]
            }
            
            with tqdm(total=max_articles, desc="Scraping Romanian News") as pbar:
                for future in as_completed(future_to_url):
                    result = future.result()
                    if result:
                        result["id"] = f"news_{result['source'].lower()}_{count}"
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        count += 1
                        pbar.update(1)
                        if count >= max_articles:
                            for pending in future_to_url:
                                pending.cancel()
                            break

    elapsed = time.time() - start_time
    print(f"\nSaved {count:,} recent news articles to {output_file} in {elapsed:.1f}s", flush=True)
    return count


def download_single_fineweb_shard(shard_idx: int, output_dir: Path) -> Optional[Path]:
    """Downloads one FineWeb-2 Romanian parquet shard (~160 MB, ~75,000 documents)."""
    shard_name = f"shard_{shard_idx:05d}.parquet"
    out_file = output_dir / shard_name
    if out_file.exists() and out_file.stat().st_size > 50_000_000:
        return out_file
    
    url = f"https://huggingface.co/datasets/rotarue/fineweb2-romanian-shards/resolve/main/{shard_name}"
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(out_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return out_file
    except Exception as e:
        print(f"Error downloading {shard_name}: {e}", flush=True)
    return None


def download_fineweb_romanian_shards(
    shards: int = 0,
    target_gb: float = 0,
    output_dir: Path = RAW_FINEWEB_DIR,
    max_workers: int = 4,
) -> int:
    """
    Downloads high-quality FineWeb-2 Romanian web shards from Hugging Face.
    Total available: 285 shards (~45 GB compressed Parquet, >150-200 GB text).
    """
    # If target_gb is specified, calculate needed shards (~0.7 GB uncompressed text per shard)
    if target_gb > 0 and shards <= 0:
        shards = int(target_gb / 0.7)

    if shards <= 0:
        return 0

    shards = min(shards, 285)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/3] Downloading {shards} FineWeb-2 Romanian Shards (Target: {output_dir})...", flush=True)
    print(f"Estimated uncompressed text: ~{shards * 0.7:.1f} GB (Compressed Parquet: ~{shards * 0.16:.1f} GB)", flush=True)

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(download_single_fineweb_shard, i, output_dir): i for i in range(shards)}
        with tqdm(total=shards, desc="Downloading FineWeb Shards") as pbar:
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    completed += 1
                pbar.update(1)

    print(f"Successfully downloaded {completed}/{shards} FineWeb shards.", flush=True)
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Romanian Pretraining & SPP Corpus")
    parser.add_argument("--wiki-articles", type=int, default=100000, help="Number of Wikipedia articles to download (0 to skip)")
    parser.add_argument("--news-articles", type=int, default=15000, help="Number of recent 5-year news articles to crawl")
    parser.add_argument("--years", type=int, default=5, help="Number of historical years of news to crawl (e.g. 5 for 2021-2026)")
    parser.add_argument("--threads", type=int, default=16, help="Concurrent HTTP worker threads for news scraping")
    parser.add_argument("--fineweb-shards", type=int, default=0, help="Number of FineWeb-2 Romanian shards to download (1-285)")
    parser.add_argument("--target-gb", type=float, default=0, help="Target total corpus size in GB (e.g. 200)")
    parser.add_argument("--skip-wiki", action="store_true", help="Skip Wikipedia if already downloaded")
    parser.add_argument("--skip-news", action="store_true", help="Skip news crawling")
    args = parser.parse_args()

    print("==================================================", flush=True)
    print(" Romanian Corpus Downloader (Wikipedia + 5-Year News + FineWeb2)")
    print(f" Target Raw Directory: {RAW_DATA_DIR}")
    print("==================================================", flush=True)

    total = 0
    
    # 1. Wikipedia
    if args.skip_wiki or args.wiki_articles <= 0:
        print(f"Skipping Wikipedia download (flag provided or wiki-articles=0).", flush=True)
    elif RAW_WIKI_FILE.exists() and RAW_WIKI_FILE.stat().st_size > 10_000_000 and not args.wiki_articles > 30000:
        print(f"Found existing Wikipedia corpus ({RAW_WIKI_FILE.name}, {RAW_WIKI_FILE.stat().st_size / (1024*1024):.1f} MB). Skipping.", flush=True)
    else:
        total += download_wikipedia_ro(max_articles=args.wiki_articles, output_file=RAW_WIKI_FILE)

    # 2. News
    if not args.skip_news and args.news_articles > 0:
        total += download_recent_romanian_news_archive(
            years=args.years,
            max_articles=args.news_articles,
            output_file=RAW_NEWS_FILE,
            max_workers=args.threads,
        )

    # 3. FineWeb-2 Shards (Scales up to 200 GB!)
    if args.fineweb_shards > 0 or args.target_gb > 0:
        download_fineweb_romanian_shards(
            shards=args.fineweb_shards,
            target_gb=args.target_gb,
            output_dir=RAW_FINEWEB_DIR,
            max_workers=4,
        )

    print("\n==================================================", flush=True)
    print(f"Download routine completed.")
    print(f"Raw Wikipedia:  {RAW_WIKI_FILE}")
    print(f"Raw News:       {RAW_NEWS_FILE}")
    print(f"FineWeb Shards: {RAW_FINEWEB_DIR}")
    print("Next step: Run `py src/clean_corpus.py` to standardize diacritics and build SPP splits.")
    print("==================================================", flush=True)

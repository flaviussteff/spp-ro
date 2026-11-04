# -*- coding: utf-8 -*-
"""
Post-pass cleanup for data/sidecar/reflections_v2.parquet (run AFTER the annotation finishes, before training).

Drops records that slipped through earlier, looser rules:
  * adult / escort keyword pages that the spam filter missed (sexual content is outside the constitution);
  * reflections that leak the annotation setup (current META_RE: "framing", "endorsing", "constituția de valori", ...).
Dropped doc_keys are no longer in the parquet, so a further `run_generate_v2_data.bat annotate` re-annotates the
meta-language ones automatically (spam pages are written to a skip list and are not retried).

Usage:  py src/clean_reflections_v2.py            (report + apply, keeps a .bak copy)
        py src/clean_reflections_v2.py --dry-run  (report only)
"""
import sys
import json
import shutil
import argparse
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from config import SIDECAR_V2_FILE, SIDECAR_DIR
from train_safety_classifier import ADULT_SPAM_RE
from generate_reflections_v2 import META_RE

SKIP_FILE = SIDECAR_DIR / "reflections_v2_skip_keys.json"


def adult_rate(text: str) -> float:
    w = text.split()
    return len(ADULT_SPAM_RE.findall(text)) / max(1, len(w)) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--adult-per-100w", type=float, default=0.8)
    args = ap.parse_args()
    df = pd.read_parquet(SIDECAR_V2_FILE)
    n0 = len(df)
    is_adult = df["text"].apply(adult_rate) >= args.adult_per_100w
    is_meta = df["reflection_1p"].apply(lambda s: bool(META_RE.search(str(s)))) | df["reflection_3p"].apply(lambda s: bool(META_RE.search(str(s))))
    print(f"înregistrări: {n0:,}")
    print(f"  pagini adult/escort scăpate de filtru : {int(is_adult.sum()):,}  (sev>=3: {int((is_adult & (df['severity'] >= 3)).sum()):,})")
    print(f"  meta-limbaj în reflecție              : {int(is_meta.sum()):,}  (vor fi re-adnotate la următoarea rulare)")
    keep = ~(is_adult | is_meta)
    print(f"  rămân: {int(keep.sum()):,}  | severity după curățare: {df[keep]['severity'].value_counts().sort_index().to_dict()}")
    if args.dry_run:
        return
    bak = SIDECAR_V2_FILE.with_suffix(".bak.parquet")
    shutil.copy2(SIDECAR_V2_FILE, bak)
    skip = set(json.load(open(SKIP_FILE, encoding="utf-8"))) if SKIP_FILE.exists() else set()
    skip |= set(df.loc[is_adult, "doc_key"])
    json.dump(sorted(skip), open(SKIP_FILE, "w", encoding="utf-8"))
    df[keep].reset_index(drop=True).to_parquet(SIDECAR_V2_FILE, index=False)
    print(f"scris {int(keep.sum()):,} înregistrări -> {SIDECAR_V2_FILE} (backup: {bak.name}; {len(skip):,} doc_key în {SKIP_FILE.name})")


if __name__ == "__main__":
    main()

"""
Inspect SPP Ethical Reflections & Causal Insertion Geometry
Allows immediate human inspection of constitutional thoughts generated for Step 5 pretraining.
"""
import sys
import argparse
import pyarrow.parquet as pq
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
sys.stdout.reconfigure(encoding="utf-8")

def inspect_reflections(num_samples: int = 3, start_idx: int = 0):
    parquet_path = Path("data/sidecar/reflections.parquet")
    if not parquet_path.exists():
        print(f"[EROARE] Nu s-a gasit fisierul: {parquet_path}")
        print("Ruleaza intai `py src/spp_annotator.py`.")
        return

    table = pq.read_table(str(parquet_path))
    total_count = len(table)
    
    print("=" * 78)
    print(f"  VERIFICARE REFLECȚII ETICE SPP (Pasul 5) — Total: {total_count:,} exemple")
    print(f"  Fișier sursă: {parquet_path.resolve()}")
    print("=" * 78)

    sample_indices = range(start_idx, min(start_idx + num_samples, total_count))

    for idx in sample_indices:
        doc_id = table.column("doc_id")[idx].as_py()
        article = table.column("article_invoked")[idx].as_py()
        reflection = table.column("reflection_text")[idx].as_py()
        full_text = table.column("text")[idx].as_py()
        char_pos = int(table.column("reflection_char_position")[idx].as_py())

        pre_text = full_text[:char_pos].strip()
        post_text = full_text[char_pos:].strip()

        print(f"\n[EXEMPLUL #{idx + 1} / {total_count:,}]  ID Document: {doc_id[:16]}... | Articol: {article}")
        print("-" * 78)
        print("  📜 REFLECȚIE ETICĂ CONSTITUȚIONALĂ (Injectată în spațiul latent):")
        print(f"     \"{reflection}\"")
        print("\n  ✂️ PUNCTUL DE INSERARE ÎN TEXTUL ORIGINAL:")
        print(f"     [Pre-Text]:   \"{pre_text[-120:] if len(pre_text) > 120 else pre_text}\"")
        print(f"     👉 INSERAT:   <assistant> {reflection} </assistant>")
        print(f"     [Post-Text]:  \"{post_text[:120]}...\"")
        print("-" * 78)

    print("\n" + "=" * 78)
    print("  MECANISMUL DE ANTRENARE ÎN PASUL 5 (Causal Attention Blocking):")
    print("  1. La antrenare, textul este formatat ca: [Pre-Text] <assistant> [Reflecție] [Post-Text]")
    print("  2. Tokenii din [Post-Text] au atenția BLOCATĂ (mască -inf) către [Reflecție].")
    print("  3. Astfel, valorile etice sunt gravate exclusiv în ponderile rețelei, fără dependență la inferență!")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect SPP Constitutional Reflections")
    parser.add_argument("-n", "--num-samples", type=int, default=3, help="Numar de mostre de afisat")
    parser.add_argument("-s", "--start-idx", type=int, default=0, help="Index de start")
    args = parser.parse_args()
    inspect_reflections(args.num_samples, args.start_idx)

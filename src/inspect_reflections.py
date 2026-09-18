"""
Interactive Browser for SPP Constitutional Reflections
Allows browsing through all 10,000 synthesized reflections with full surrounding text.

Usage:
    py src/inspect_reflections.py
    py src/inspect_reflections.py --index 42
    py src/inspect_reflections.py --search "femei"
"""
import sys
import argparse
import random
from pathlib import Path
import pyarrow.parquet as pq

# Ensure UTF-8 output on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def display_reflection(table, idx: int, total_count: int, show_all: bool = False):
    doc_id = table.column("doc_id")[idx].as_py()
    article = table.column("article_invoked")[idx].as_py()
    reflection = table.column("reflection_text")[idx].as_py()
    full_text = table.column("text")[idx].as_py()
    char_pos = int(table.column("reflection_char_position")[idx].as_py())

    pre_text = full_text[:char_pos]
    post_text = full_text[char_pos:]

    # Truncation limits for readability (unless show_all is True)
    context_limit = 1200
    pre_display = pre_text if (show_all or len(pre_text) <= context_limit) else ("..." + pre_text[-context_limit:])
    post_display = post_text if (show_all or len(post_text) <= context_limit) else (post_text[:context_limit] + "...")

    print("\n" + "=" * 80)
    print(f"  SPP CONSTITUTIONAL REFLECTION #{idx + 1:,} / {total_count:,}")
    print(f"  Article Invoked: {article} | Char Position: {char_pos:,} / {len(full_text):,}")
    print(f"  Document ID: {doc_id}")
    print("=" * 80)

    print("\n[PRE-TEXT (Original Document Content)]:")
    print("-" * 80)
    print(pre_display.strip())
    print("-" * 80)

    print("\n" + "┌" + "─" * 78 + "┐")
    print("│ 💡 <assistant> INJECTED CONSTITUTIONAL DELIBERATION PATH:                      │")
    print("├" + "─" * 78 + "┤")
    # Wrap reflection text nicely
    import textwrap
    wrapped_lines = textwrap.wrap(reflection, width=74)
    for line in wrapped_lines:
        print(f"│  {line:<76}│")
    print("└" + "─" * 78 + "┘")

    print("\n[POST-TEXT (Continuation of Document)]:")
    print("-" * 80)
    print(post_display.strip())
    print("-" * 80)

    if not show_all and (len(pre_text) > context_limit or len(post_text) > context_limit):
        print(f"  (Context trunchiat la +/-{context_limit} caractere. Scrie 'all' pentru textul complet de {len(full_text):,} caractere)")
    print()


def browse_reflections(start_idx: int = 0, initial_search: str = None):
    parquet_path = Path("data/sidecar/reflections.parquet")
    if not parquet_path.exists():
        print(f"[EROARE] Nu s-a gasit fisierul: {parquet_path.resolve()}")
        print("Ruleaza intai `py src/spp_annotator.py`.")
        return

    print("Incarcare fisier de reflectii SPP (10,000 exemple)...", flush=True)
    table = pq.read_table(str(parquet_path))
    total_count = len(table)

    current_idx = max(0, min(start_idx, total_count - 1))
    show_all = False

    if initial_search:
        results = search_reflections(table, initial_search)
        if results:
            current_idx = results[0]
            print(f"Gasit {len(results)} rezultate pentru '{initial_search}'. Primul index: #{current_idx + 1}")
        else:
            print(f"Nu s-au gasit rezultate pentru '{initial_search}'. Pornim de la index #{current_idx + 1}")

    while True:
        display_reflection(table, current_idx, total_count, show_all=show_all)
        show_all = False  # Reset show_all after displaying

        print("=" * 80)
        print(" [Enter]/[n] = Next | [p] = Prev | [j <num>] = Jump | [r] = Random")
        print(" [s <term>] = Search | [all] = Show 100% Full Text | [q] = Quit")
        print("=" * 80)

        try:
            cmd = input(f"[#{current_idx + 1}/{total_count:,}] >>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nIesire din browser.")
            break

        if not cmd or cmd.lower() in ["n", "next", "inainte"]:
            if current_idx < total_count - 1:
                current_idx += 1
            else:
                print("\n[!] Ai ajuns la finalul listei (exemplul 10,000).")
        elif cmd.lower() in ["p", "prev", "inapoi", "b", "back"]:
            if current_idx > 0:
                current_idx -= 1
            else:
                print("\n[!] Esti deja la primul exemplu (#1).")
        elif cmd.lower() in ["q", "exit", "quit", "iesire"]:
            print("\nSesiune incheiata.")
            break
        elif cmd.lower() == "r" or cmd.lower() == "random":
            current_idx = random.randint(0, total_count - 1)
        elif cmd.lower() == "all" or cmd.lower() == "full":
            show_all = True
        elif cmd.lower().startswith("j ") or cmd.lower().startswith("jump "):
            parts = cmd.split()
            if len(parts) > 1 and parts[1].isdigit():
                target = int(parts[1]) - 1  # 1-indexed for user
                if 0 <= target < total_count:
                    current_idx = target
                else:
                    print(f"\n[!] Numar invalid. Introdu un numar intre 1 si {total_count:,}.")
            else:
                print("\n[!] Utilizare: j <numar> (ex: j 450)")
        elif cmd.lower().startswith("s ") or cmd.lower().startswith("search "):
            query = cmd[2:].strip()
            if query:
                results = search_reflections(table, query)
                if results:
                    print(f"\n[SUCCES] Gasit {len(results)} exemple ce contin '{query}'. Sarim la primul:")
                    current_idx = results[0]
                else:
                    print(f"\n[!] Niciun rezultat gasit pentru '{query}'.")
            else:
                print("\n[!] Introdu un termen de cautare: s <cuvant>")
        elif cmd.isdigit():
            # Quick jump by just typing the number
            target = int(cmd) - 1
            if 0 <= target < total_count:
                current_idx = target
            else:
                print(f"\n[!] Introdu un numar intre 1 si {total_count:,}.")
        else:
            print("\n[!] Comanda necunoscuta. Foloseste: Enter, p, j <num>, r, s <cuvant>, all, q")


def search_reflections(table, query: str):
    q = query.lower()
    matches = []
    # Fast scan through reflection_text and text
    refl_col = table.column("reflection_text")
    text_col = table.column("text")
    for i in range(len(table)):
        r = refl_col[i].as_py().lower()
        t = text_col[i].as_py().lower()
        if q in r or q in t:
            matches.append(i)
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Browse SPP Constitutional Reflections Interactively")
    parser.add_argument("-i", "--index", type=int, default=1, help="Indexul de start (1 - 10000)")
    parser.add_argument("-s", "--search", type=str, default=None, help="Cauta un cuvant cheie la pornire")
    args = parser.parse_args()

    browse_reflections(start_idx=args.index - 1, initial_search=args.search)

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
    
    # Read safety_score if column exists
    safety_score = None
    if "safety_score" in table.column_names:
        safety_score = table.column("safety_score")[idx].as_py()

    safety_labels = {
        1: "1/5 [STEREOTIP / ASIMETRIE CULTURALA] (Contrabalansat activ prin reflectie)",
        2: "2/5 [TEMA SENSIBILA / CRIZA UMANITARA / MEMORIE ISTORICA]",
        3: "3/5 [DEZBATERE CIVICA / STAT DE DREPT / BIOETICA]",
        4: "4/5 [STIRI FACTUALE / DEZVOLTARE SOCIALA]",
        5: "5/5 [COMPLET BENIGN / PROGRES STIINTIFIC & RATIONAL]",
    }
    safety_desc = safety_labels.get(safety_score, f"{safety_score}/5" if safety_score is not None else "N/A")

    pre_text = full_text[:char_pos]
    post_text = full_text[char_pos:]

    # Truncation limits for readability (unless show_all is True)
    context_limit = 1200
    pre_display = pre_text if (show_all or len(pre_text) <= context_limit) else ("..." + pre_text[-context_limit:])
    post_display = post_text if (show_all or len(post_text) <= context_limit) else (post_text[:context_limit] + "...")

    print("\n" + "=" * 80)
    print(f"  SPP CONSTITUTIONAL REFLECTION #{idx + 1:,} / {total_count:,}")
    print(f"  Article Invoked : {article}")
    print(f"  Safety Score    : {safety_desc}")
    print(f"  Document ID     : {doc_id} | Char Position: {char_pos:,} / {len(full_text):,}")
    print("=" * 80)

    print("\n[PRE-TEXT (Original Document Content)]:")
    print("-" * 80)
    print(pre_display.strip())
    print("-" * 80)

    print("\n" + "┌" + "─" * 78 + "┐")
    print("│ 💡 <assistant> INJECTED CONSTITUTIONAL DELIBERATION PATH (Persoana I):         │")
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


def browse_reflections(start_idx: int = 0, initial_search: str = None, once: bool = False):
    parquet_path = Path("data/sidecar/reflections.parquet")
    if not parquet_path.exists():
        print(f"[EROARE] Nu s-a gasit fisierul: {parquet_path.resolve()}")
        print("Ruleaza intai `py src/spp_annotator.py`.")
        return

    table = pq.read_table(str(parquet_path))
    total_count = len(table)
    print(f"Incarcare fisier de reflectii SPP ({total_count:,} exemple)...", flush=True)

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
        if once or not sys.stdin.isatty():
            break
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


def search_reflections(table, query: str, article_filter: str = None, safety_filter: int = None):
    q = query.lower() if query else None
    matches = []
    refl_col = table.column("reflection_text")
    text_col = table.column("text")
    art_col = table.column("article_invoked") if "article_invoked" in table.column_names else None
    safety_col = table.column("safety_score") if "safety_score" in table.column_names else None

    for i in range(len(table)):
        if article_filter and art_col and article_filter not in art_col[i].as_py():
            continue
        if safety_filter is not None and safety_col and safety_col[i].as_py() != safety_filter:
            continue
        if q:
            r = refl_col[i].as_py().lower()
            t = text_col[i].as_py().lower()
            if q not in r and q not in t:
                continue
        matches.append(i)
    return matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Browse SPP Constitutional Reflections Interactively")
    parser.add_argument("-i", "--index", type=int, default=1, help="Indexul de start (1 - N)")
    parser.add_argument("-s", "--search", type=str, default=None, help="Cauta un cuvant cheie la pornire")
    parser.add_argument("-a", "--article", type=str, default=None, help="Filtreaza dupa articol (ex: §1.2, §2.1)")
    parser.add_argument("--safety", type=int, default=None, choices=[1, 2, 3, 4, 5], help="Filtreaza dupa safety score (1-5)")
    parser.add_argument("--once", action="store_true", help="Afiseaza o singura data si iese (non-interactiv)")
    args = parser.parse_args()

    # If article or safety filter is passed, find matches
    initial_search = args.search
    start_idx = args.index - 1
    if args.article or args.safety:
        parquet_path = Path("data/sidecar/reflections.parquet")
        if parquet_path.exists():
            tbl = pq.read_table(str(parquet_path))
            filtered = search_reflections(tbl, args.search, article_filter=args.article, safety_filter=args.safety)
            if filtered:
                start_idx = filtered[0]
                print(f"[Filtru] Gasit {len(filtered):,} exemple pentru Articol={args.article}, Safety={args.safety}. Primul index: #{start_idx + 1}")
            else:
                print(f"[!] Niciun rezultat pentru Articol={args.article}, Safety={args.safety}.")

    browse_reflections(start_idx=start_idx, initial_search=initial_search, once=args.once)

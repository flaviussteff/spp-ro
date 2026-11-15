"""
Aplică corecturile manuale (gramaticale) și eliminările din data/eval/ce_v3_corrections.tsv pe ConstitutionEval-Ro v3.

Fișierul TSV: o linie per corectură, trei câmpuri separate prin TAB, fără antet; liniile goale și cele care încep cu # se ignoră.
    <id>    <câmp>    <text nou complet>
<câmp> = scenario | provision | A | B | C | D   (litera variantei în ordinea afișată în pagina HTML / documentul Word)
       = elimina   (al treilea câmp = motivul; itemul primește status "failed_manual" și iese din set la următorul finalize)

Scrie în AMBELE locuri, ca să nu se piardă corecturile la finalize:
  - data/eval/constitution_eval_ro_v3.jsonl   (setul final, ordinea afișată)
  - data/eval/ce_v3_build/items.jsonl          (starea de construcție, ordinea de dinainte de rebalansare)
Păstrează o copie .bak a fiecărui fișier și scrie un jurnal al modificărilor. Nu folosește API și nici GPU.

După eliminări:  py src\\build_constitution_eval_v3.py --stage finalize   (refac setul final din starea de construcție; itemii
                 eliminați sunt înlocuiți cu următorii candidați „pass” ai aceluiași articol, dacă există; altfel articolul rămâne cu mai puțini)
Oricum după:     py src\\build_constitution_eval_v3.py --stage review     (pagina HTML)
Opțional:        run_v3_verify.bat   (reface tabelul de validitate și scorurile pe textul corectat; ~20 min, API ≈ 2 $)
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_constitution_eval_v3 import OUT_FILE, STATE_FILE, build_to_final_map, gold_index, load_state, save_state  # noqa: E402
from config import EVAL_DATA_DIR  # noqa: E402

CORRECTIONS_FILE = EVAL_DATA_DIR / "ce_v3_corrections.tsv"
LOG_FILE = EVAL_DATA_DIR / "ce_v3_corrections_applied.jsonl"
LETTERS = "ABCD"


def read_corrections(path: Path):
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise SystemExit(f"linia {n}: aștept 3 câmpuri separate prin TAB, am găsit {len(parts)}: {line[:80]!r}")
        item_id, field, text = parts[0].strip(), parts[1].strip(), "\t".join(parts[2:]).strip()
        # exact single-letter match: `"BC" in "ABCD"` is a substring hit and would silently map a typo onto option B
        if field not in ("scenario", "provision", "elimina") and not (len(field) == 1 and field.upper() in LETTERS):
            raise SystemExit(f"linia {n}: câmp necunoscut {field!r} (scenario | provision | A-D | elimina)")
        rows.append((n, item_id, field if field in ("scenario", "provision", "elimina") else field.upper(), text))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(CORRECTIONS_FILE))
    ap.add_argument("--dry-run", action="store_true", help="arată ce s-ar schimba, fără să scrie")
    args = ap.parse_args()
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"lipsește {path}")
    corr = read_corrections(path)
    final_rows = [json.loads(l) for l in OUT_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    final = {r["id"]: r for r in final_rows}
    state = load_state()
    build = {it["id"]: it for it in state}

    changes, removed, unchanged = [], [], []
    for n, item_id, field, text in corr:
        r, it = final.get(item_id), build.get(item_id)
        if r is None or it is None:
            raise SystemExit(f"linia {n}: itemul {item_id} nu există în setul final / starea de construcție")
        if field == "elimina":
            it["status"], it["reason"] = "failed_manual", f"manual: {text}"
            r["_remove"] = True
            removed.append((item_id, text))
            continue
        if field in ("scenario", "provision"):
            old = r[field]
            if old == text:
                unchanged.append((item_id, field))
                continue
            r[field] = text
            it[field] = text
        else:
            j = LETTERS.index(field)
            mp = build_to_final_map(gold_index(it), r["gold_index"], len(r["options"]))
            i = mp.index(j)   # build-order index of the displayed option j
            assert it["options"][i]["text"] == r["options"][j]["text"], f"{item_id}: ordinea variantelor nu corespunde"
            old = r["options"][j]["text"]
            if old == text:
                unchanged.append((item_id, field))
                continue
            r["options"][j]["text"] = text
            it["options"][i]["text"] = text
        changes.append({"id": item_id, "field": field, "old": old, "new": text})

    print(f"{len(changes)} corecturi, {len(removed)} eliminări, {len(unchanged)} linii deja identice")
    for c in changes:
        print(f"  {c['id']} {c['field']}: {c['old'][:70]!r} -> {c['new'][:70]!r}")
    for item_id, why in removed:
        print(f"  {item_id} ELIMINAT: {why}")
    if args.dry_run or not (changes or removed):
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(OUT_FILE, OUT_FILE.with_suffix(f".{stamp}.bak"))
    shutil.copy2(STATE_FILE, STATE_FILE.with_suffix(f".{stamp}.bak"))
    save_state(state)
    kept = [r for r in final_rows if not r.get("_remove")]
    OUT_FILE.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n", encoding="utf-8")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        for c in changes:
            f.write(json.dumps({"when": stamp, **c}, ensure_ascii=False) + "\n")
        for item_id, why in removed:
            f.write(json.dumps({"when": stamp, "id": item_id, "field": "elimina", "reason": why}, ensure_ascii=False) + "\n")
    print(f"scris: {OUT_FILE} ({len(kept)} itemi) și {STATE_FILE}; copii .{stamp}.bak; jurnal {LOG_FILE}")
    if removed:
        print("ai eliminat itemi: rulează  py src\\build_constitution_eval_v3.py --stage finalize  ca să refaci setul final cu înlocuitori")
    print("apoi:  py src\\build_constitution_eval_v3.py --stage review")


if __name__ == "__main__":
    main()

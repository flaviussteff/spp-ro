"""
Builds human_eval/index.html, the self-contained human-evaluation page, from the 100-item Word benchmark.

    python human_eval/build_interface.py
    python human_eval/build_interface.py --docx "evals/benchmark spp 100 items (fara 2.1).docx"

Reads the .docx with the standard library (zipfile + ElementTree), no python-docx needed:
  - a body paragraph "Itemul N" starts item N;
  - the paragraph(s) between it and "Opțiuni:" are the situation;
  - the paragraphs "A) ...", "B) ...", "C) ...", "D) ..." are the four options;
  - the answer tables of the Word form are ignored.
The text is kept exactly as in the document (only leading/trailing spaces are trimmed), because the items must
reach the raters exactly as the evaluated models receive them.

The items are embedded in template.html (placeholders /*__ITEMS__*/null and /*__BENCH__*/null), so index.html
works from a double-click, offline, with nothing to install. The page keys its saved progress on a hash of the
items: if the document changes, a browser that holds an older session is told so instead of mixing the two sets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_DOCX = ROOT / "evals" / "benchmark spp 100 items (fara 2.1).docx"
TEMPLATE = HERE / "template.html"
OUT = HERE / "index.html"

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
HEAD_RE = re.compile(r"^Itemul\s+(\d+)$")
OPT_RE = re.compile(r"^([A-D])\)\s*(.*)$", re.S)
LETTERS = "ABCD"


def para_text(p: ET.Element) -> str:
    """Text of a paragraph: the runs only (a w:tab inside w:pPr is a tab stop, not text)."""
    out = []
    for r in p.iter(W + "r"):
        for el in r:
            if el.tag == W + "t":
                out.append(el.text or "")
            elif el.tag == W + "tab":
                out.append("\t")
            elif el.tag in (W + "br", W + "cr"):
                out.append("\n")
            elif el.tag == W + "noBreakHyphen":
                out.append("-")
    return "".join(out)


def read_items(docx: Path) -> list[dict]:
    with zipfile.ZipFile(docx) as z:
        root = ET.fromstring(z.read("word/document.xml"))
    body = root.find(W + "body")
    items: list[dict] = []
    cur = None
    in_options = False
    for el in body:
        if el.tag != W + "p":  # tables = the answer form of the Word version
            continue
        text = para_text(el).strip()
        m = HEAD_RE.match(text)
        if m:
            cur = {"n": int(m.group(1)), "text": [], "options": []}
            items.append(cur)
            in_options = False
            continue
        if cur is None or not text:
            continue
        if text.rstrip(":").strip() == "Opțiuni":
            in_options = True
            continue
        om = OPT_RE.match(text)
        if in_options and om:
            cur["options"].append((om.group(1), om.group(2).strip()))
        elif not in_options:
            cur["text"].append(text)
        else:
            raise SystemExit(f"Itemul {cur['n']}: paragraf neașteptat după opțiuni: {text[:80]!r}")

    problems = []
    for i, it in enumerate(items, 1):
        if it["n"] != i:
            problems.append(f"aștept Itemul {i}, am găsit Itemul {it['n']}")
        if not it["text"]:
            problems.append(f"Itemul {it['n']}: lipsește situația")
        if [l for l, _ in it["options"]] != list(LETTERS):
            problems.append(f"Itemul {it['n']}: opțiunile sunt {[l for l, _ in it['options']]}, aștept A, B, C, D")
        if any(not o for _, o in it["options"]):
            problems.append(f"Itemul {it['n']}: o opțiune e goală")
    if problems:
        raise SystemExit("Documentul nu are structura așteptată:\n  " + "\n  ".join(problems[:20]))
    return [{"n": it["n"], "text": "\n".join(it["text"]), "options": [o for _, o in it["options"]]} for it in items]


def js_json(obj) -> str:
    """JSON that is safe inside a <script> element."""
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return s.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--docx", default=str(DEFAULT_DOCX))
    ap.add_argument("--expect", type=int, default=100, help="numărul de itemi așteptat (0 = oricâți)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    docx = Path(args.docx)
    if not docx.exists():
        raise SystemExit(f"lipsește {docx}")
    items = read_items(docx)
    if args.expect and len(items) != args.expect:
        raise SystemExit(f"am găsit {len(items)} itemi, aștept {args.expect} (--expect 0 ca să accepți oricâți)")

    canon = json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")
    bench = {"source": docx.name, "n": len(items), "hash": hashlib.md5(canon).hexdigest()[:10]}

    tpl = TEMPLATE.read_text(encoding="utf-8")
    for ph in ("/*__ITEMS__*/null", "/*__BENCH__*/null"):
        if tpl.count(ph) != 1:
            raise SystemExit(f"template.html: placeholderul {ph} trebuie să apară exact o dată")
    html = tpl.replace("/*__ITEMS__*/null", js_json(items)).replace("/*__BENCH__*/null", js_json(bench))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"{len(items)} itemi din {docx.name} -> {args.out}  (versiune {bench['hash']}, {len(html.encode('utf-8')) // 1024} KB)")


if __name__ == "__main__":
    sys.exit(main())

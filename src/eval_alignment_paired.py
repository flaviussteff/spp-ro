"""
Paired per-document analysis of the perplexity evaluation (CPU only, a few seconds).

Reads evals/thesis_alignment_tax_perdoc_losses.json (written by eval_alignment_tax.py: the mean cross-entropy of every
held-out document under every model, same document order). For every comparison "A vs B" and every slice (all held-out
documents, classifier-flagged, unflagged) it reports:
 * the mean over documents of loss_A - loss_B (nats/token, every document weighted equally) with a 95 % percentile
   bootstrap CI (5,000 resamples of documents, seed 0);
 * the share of documents on which A has the higher loss, with the exact two-sided sign test;
 * Holm-adjusted p-values over the whole family (all comparisons x slices);
 * for reference, the token-weighted perplexity difference (the quantity in the perplexity table).
Writes evals/thesis_alignment_paired_report.json and evals/thesis_alignment_paired_table.tex.
"""
import sys
import json
import math
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config import EVALS_DIR
    from eval_utils import sign_test_p, holm_adjust, round_p, tex_escape
except ImportError:
    from src.config import EVALS_DIR
    from src.eval_utils import sign_test_p, holm_adjust, round_p, tex_escape

PERDOC_FILE = EVALS_DIR / "thesis_alignment_tax_perdoc_losses.json"
REPORT_FILE = EVALS_DIR / "thesis_alignment_paired_report.json"
TEX_FILE = EVALS_DIR / "thesis_alignment_paired_table.tex"

# (A, B, what the comparison measures): positive delta = A has the higher loss (is worse) than B
COMPARISONS = [
    ("Filtered-Ro", "Base-Ro", "cost of loss-masking the flagged documents"),
    ("SPP-Ro", "Base-Ro", "alignment tax of SPP{T0}"),
    ("Filtered-Ro", "SPP-Ro", "Filtered vs SPP{T0}"),
    ("SPP-Ro-MT", "Base-Ro-rewarm", "alignment tax of SPP{MT} (matched control)"),
    ("Base-Ro-rewarm", "Base-Ro", "effect of the re-warm schedule alone"),
]
SLICES = [("heldout", "toate"), ("flagged", "semnalat"), ("unflagged", "nesemnalat")]


def bootstrap_mean_ci(d: np.ndarray, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(n_boot)
    chunk = max(1, 2_000_000 // max(1, n))
    for b0 in range(0, n_boot, chunk):
        b1 = min(n_boot, b0 + chunk)
        means[b0:b1] = d[rng.integers(0, n, size=(b1 - b0, n))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def fmt_p(p: float) -> str:
    if p == 0.0:
        return "<1e-300"
    return f"{p:.3g}" if p >= 1e-3 else f"{p:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.load(open(PERDOC_FILE, encoding="utf-8"))
    meta = data.get("_meta", {})
    ntok_all = np.asarray(meta.get("n_tokens") or [], dtype=float)
    idx_of = {"heldout": None, "flagged": meta.get("flagged_index"), "unflagged": meta.get("unflagged_index")}
    rows = []
    for a, b, what in COMPARISONS:
        if a not in data or b not in data:
            print(f"[skip] {a} vs {b}: model missing from {PERDOC_FILE.name}")
            continue
        for sl, sl_ro in SLICES:
            if sl not in data[a] or sl not in data[b]:
                continue
            la, lb = np.asarray(data[a][sl], dtype=float), np.asarray(data[b][sl], dtype=float)
            if len(la) != len(lb) or len(la) == 0:
                print(f"[skip] {a} vs {b} / {sl}: lengths {len(la)} vs {len(lb)}")
                continue
            d = la - lb
            lo, hi = bootstrap_mean_ci(d, args.n_boot, args.seed)
            row = {"a": a, "b": b, "what": what, "slice": sl, "n": int(len(d)),
                   "mean_delta_loss": round(float(d.mean()), 5), "ci95": [round(lo, 5), round(hi, 5)],
                   "share_a_higher": round(float((d > 0).mean()), 4), "n_ties": int((d == 0).sum()),
                   "p_sign": round_p(sign_test_p(d.tolist()))}
            idx = idx_of[sl]
            if len(ntok_all) and (idx is None or len(idx) == len(d)):
                w = ntok_all if idx is None else ntok_all[np.asarray(idx, dtype=int)]
                if len(w) == len(d):
                    row["delta_ppl_token_weighted"] = round(math.exp((la * w).sum() / w.sum()) - math.exp((lb * w).sum() / w.sum()), 3)
            rows.append(row)
    for r, ph in zip(rows, holm_adjust([r["p_sign"] for r in rows])):
        r["p_sign_holm"] = round_p(ph)

    for r in rows:
        print(f"{r['a']} vs {r['b']}, {r['slice']} n={r['n']} Δloss mediu={r['mean_delta_loss']:+.4f} "
              f"[95% {r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] doc. cu loss mai mare: {100 * r['share_a_higher']:.0f}% "
              f"p(semn)={fmt_p(r['p_sign'])} p(Holm)={fmt_p(r['p_sign_holm'])}"
              + (f" ΔPPL={r['delta_ppl_token_weighted']:+.3f}" if "delta_ppl_token_weighted" in r else ""))

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump({"note": "Paired per-document comparison of the held-out perplexity evaluation. delta = loss(A) - loss(B), "
                           "mean cross-entropy per document (nats/token), documents weighted equally; CI = percentile bootstrap "
                           f"over documents ({args.n_boot} resamples, seed {args.seed}); p_sign = exact two-sided sign test on the "
                           "per-document differences; p_sign_holm = Holm correction over all rows.",
                   "source": PERDOC_FILE.name, "heldout_order_md5": meta.get("heldout_order_md5"),
                   "clf_threshold": meta.get("clf_threshold"), "rows": rows}, f, indent=2, ensure_ascii=False)

    with open(TEX_FILE, "w", encoding="utf-8") as f:
        f.write("% Comparatii pereche pe documente (eval_alignment_paired.py): delta = loss(A) - loss(B), nats/token, media pe documente\n")
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n\\begin{tabular}{l l r c c c c}\n\\toprule\n")
        f.write("\\textbf{Comparație} & \\textbf{Felie} & \\textbf{n} & \\textbf{$\\Delta$loss [IÎ 95\\%]} & "
                "\\textbf{doc. cu loss mai mare} & \\textbf{$p$ semn} & \\textbf{$p$ Holm} \\\\\n\\midrule\n")
        for r in rows:
            sl_ro = dict(SLICES)[r["slice"]]
            f.write(f"{tex_escape(r['a'])} vs {tex_escape(r['b'])} & {sl_ro} & {r['n']} & "
                    f"{r['mean_delta_loss']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] & {100 * r['share_a_higher']:.0f}\\% & "
                    f"{tex_escape(fmt_p(r['p_sign']))} & {tex_escape(fmt_p(r['p_sign_holm']))} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Diferențe pereche de loss pe documentele held-out (același document evaluat de ambele modele). $\\Delta$loss = media pe "
                "documente a diferenței de cross-entropie (nats/token, primii 512 tokeni), cu interval de încredere bootstrap pe documente; "
                "`doc. cu loss mai mare' = procentul documentelor pe care primul model are loss mai mare; $p$ = testul exact al semnului, "
                "corectat Holm pe toate rândurile tabelului.}\n")
        f.write("\\label{tab:thesis_alignment_paired}\n\\end{table}\n")
    print("\n[SUCCESS] JSON:", REPORT_FILE, "\n          LaTeX:", TEX_FILE)


if __name__ == "__main__":
    main()

"""
Regenerates the result parts of the documentation from the evaluation reports in evals/ (CPU only, no model loaded):
  * evals/RESULTS.md                    - every result table + provenance (which held-out set, which prompt version);
  * README.md, between <!-- RESULTS:START --> and <!-- RESULTS:END --> - the "Results" section;
  * README.md, between <!-- SUMMARY:START --> and <!-- SUMMARY:END --> - the "What I find" paragraph.
Every number and every "significant / not significant" statement is computed here from the reports, so the README
cannot drift from evals/. Significance = Holm-adjusted p < 0.05 within each family (perplexity: all paired rows;
ConstitutionEval-Ro: all overall comparisons; adversarial probe: the 8 comparisons with the references).
Run after the evaluations:  py src/report_results.py   (run_fixes.bat does it)
"""
import sys
import json
import math
import hashlib
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
for _p in [str(_SRC), str(_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config import EVALS_DIR, EVAL_DATA_DIR, HELDOUT_FILE, base_ref_key  # noqa: E402
from eval_utils import holm_adjust  # noqa: E402

README = _ROOT / "README.md"
RESULTS_MD = EVALS_DIR / "RESULTS.md"
ALPHA = 0.05
NAMES = {"Base-Ro": "Base-Ro (Vanilla)", "SPP-Ro": "SPP-Ro", "Filtered-Ro": "Filtered-Ro",
         "Base-Ro-rewarm": "Base-Ro-rewarm (MT control)", "SPP-Ro-MT": "SPP-Ro-MT"}
SLICE_TXT = {"heldout": "held-out documents", "flagged": "flagged held-out documents", "unflagged": "unflagged held-out documents"}


def load(name: str, base: Path = EVALS_DIR) -> Optional[Dict[str, Any]]:
    p = base / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] cannot read {p}: {e}")
        return None


def fp(p: Optional[float]) -> str:
    """p-value for prose and tables."""
    if p is None:
        return "n/a"
    if p == 0.0:
        return "<1e-300"
    if p < 0.001:
        m, e = f"{p:.1e}".split("e")
        return f"{m}e{int(e)}"
    return f"{p:.3f}".rstrip("0").rstrip(".") if p < 0.1 else f"{p:.2f}"


def peq(p: Optional[float]) -> str:
    """'p = 0.003' or 'p < 1e-300' for prose."""
    s = fp(p)
    return f"p < {s[1:]}" if s.startswith("<") else f"p = {s}"


def sig(ph: Optional[float]) -> bool:
    return ph is not None and ph < ALPHA


def ci(c) -> str:
    return f"[{c[0]:.0f}, {c[1]:.0f}]"


# ---------------------------------------------------------------- perplexity
def heldout_provenance(tax, paired) -> Dict[str, Any]:
    st = load("heldout_stats.json", EVAL_DATA_DIR) or {}
    info: Dict[str, Any] = {"stats": st, "version": st.get("version", 2), "warnings": []}
    if info["version"] < 3:
        info["warnings"].append("data/eval/heldout_stats.json is not from the corrected builder (version 3): rebuild the held-out set (run_fixes.bat).")
    if tax is not None and "heldout_docs_in_file" not in tax:
        info["warnings"].append("thesis_alignment_tax_report.json was written by the previous evaluation script (first 1,000 held-out documents): re-run eval_alignment_tax.py.")
    perdoc = load("thesis_alignment_tax_perdoc_losses.json")
    if perdoc and HELDOUT_FILE.exists():
        try:
            import pandas as pd
            texts = pd.read_parquet(HELDOUT_FILE)["text"].dropna().tolist()
            meta = perdoc.get("_meta", {})
            want, n_meta = meta.get("heldout_order_md5"), len(meta.get("n_tokens") or [])
            if want is None:
                info["warnings"].append("The per-document losses carry no held-out fingerprint (old format): re-run eval_alignment_tax.py.")
            elif n_meta != len(texts):
                info["warnings"].append(f"The per-document losses cover {n_meta} documents, the held-out file has {len(texts)}: re-run eval_alignment_tax.py.")
            elif hashlib.md5("".join(hashlib.md5(t.strip().encode("utf-8")).hexdigest() for t in texts).encode()).hexdigest() != want:
                info["warnings"].append("The per-document losses were computed on a different held-out file than data/eval/heldout_fineweb.parquet: re-run eval_alignment_tax.py.")
        except Exception as e:
            info["warnings"].append(f"Could not fingerprint the held-out file ({e}).")
    if paired is None:
        info["warnings"].append("evals/thesis_alignment_paired_report.json is missing: run eval_alignment_paired.py.")
    return info


def paired_rows(paired) -> List[Dict[str, Any]]:
    return paired.get("rows", []) if paired else []


def find_row(rows, a, b, sl):
    return next((r for r in rows if r["a"] == a and r["b"] == b and r["slice"] == sl), None)


def describe_pair(r: Optional[Dict[str, Any]]) -> str:
    if r is None:
        return "not evaluated"
    excl = r["ci95"][0] > 0 or r["ci95"][1] < 0
    base = (f"mean Δloss {r['mean_delta_loss']:+.4f} nats/token [95 % CI {r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}], "
            f"higher loss on {100 * r['share_a_higher']:.0f} % of {r['n']:,} documents, sign test {peq(r['p_sign'])}, Holm {peq(r['p_sign_holm'])}")
    if sig(r["p_sign_holm"]):
        word = "higher" if r["mean_delta_loss"] > 0 else "lower"
        return f"**{r['a']} has a {word} loss than {r['b']}** on the {SLICE_TXT[r['slice']]} ({base})"
    return (f"{r['a']} and {r['b']} **do not differ significantly** on the {SLICE_TXT[r['slice']]} ({base}"
            + ("; the bootstrap CI of the mean excludes 0, the sign test does not reject" if excl else "") + ")")


def perplexity_section(tax, paired, prov) -> List[str]:
    out = ["### Alignment tax: perplexity (first 512 tokens of every document)", ""]
    if tax is None:
        return out + ["_Report missing: run `py src/eval_alignment_tax.py`._", ""]
    st = prov["stats"]
    sizes = tax.get("passages_per_domain", {})
    n_all = sizes.get("Web (held-out)") or 0
    n_f = sizes.get("Web held-out, semnalat de clasificator")
    n_u = sizes.get("Web held-out, nesemnalat") or 0
    thr = tax.get("clf_threshold")
    new_format = "heldout_docs_in_file" in tax
    if prov["version"] >= 3:
        intro = (f"The held-out set is {n_all:,} documents of FineWeb-2 RO `{st.get('shard', '?')}` (a shard the corpus stream never read), "
                 "cleaned exactly like the training corpus (`prepare_scale_corpus.clean_text_content`: NFC, comma-below ș/ț, line breaks kept) "
                 "and verified absent from the corpus and from the reflection sidecar with the corpus deduplication key (first 64 characters) and MD5; "
                 f"{st.get('overlap_prefix', 0) + st.get('overlap_hash', 0):,} documents of the shard were dropped as already in training "
                 "(`data/eval/heldout_stats.json`).")
    else:
        intro = (f"The perplexity below was measured on {n_all:,} documents of the held-out set built by the previous version of "
                 "`build_heldout_set.py`, which was not normalised like the training corpus (see the warning at the end of the results).")
    if n_f is not None and new_format:
        intro += (f" The *flagged* slice is every held-out document the safety classifier scores ≥ {thr} (the threshold of the sidecar expansion): "
                  f"{n_f:,} documents ({100 * n_f / max(1, n_all):.1f} %); the *unflagged* slice is the other {n_u:,}.")
    elif n_f is not None:
        intro += (f" In this old report the *flagged* slice ({n_f:,} documents scoring ≥ {thr}) and the *unflagged* slice (the first {n_u:,} "
                  "documents scoring below) were drawn from the whole file, the first column from its first documents only.")
    out += [intro, ""]
    out += ["| Model | reference | PPL held-out | PPL flagged | PPL unflagged | ΔPPL held-out | ΔPPL flagged | ΔPPL unflagged |",
            "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|"]
    rows = {r["model"]: r for r in tax["results"]}
    for key, r in rows.items():
        ref = base_ref_key(key)
        is_ref = key == ref
        def d(field):
            v = r.get(field)
            return "ref" if is_ref else ("n/a" if v is None else f"{v:+.2f}")
        out.append(f"| {NAMES.get(key, key)} | {'—' if is_ref else ref} | {r['heldout_ppl']:.2f} | {r.get('flagged_ppl', float('nan')):.2f} | "
                   f"{r.get('unflagged_ppl', float('nan')):.2f} | {d('delta_ppl_heldout_vs_base')} | {d('delta_ppl_flagged_vs_base')} | {d('delta_ppl_unflagged_vs_base')} |")
    out.append("")
    prs = paired_rows(paired)
    if prs:
        out += [f"Paired per-document comparison (`evals/thesis_alignment_paired_report.json`): mean over documents of loss(A) − loss(B) "
                f"in nats/token with a 95 % bootstrap CI over documents, share of documents on which A has the higher loss, exact sign test, "
                f"Holm correction over the {len(prs)} rows.", "",
                "| A vs B | slice | n | mean Δloss [95 % CI] | A has the higher loss | sign test p | Holm p |",
                "|---|---|---:|:---:|:---:|:---:|:---:|"]
        for r in prs:
            out.append(f"| {r['a']} vs {r['b']} | {r['slice']} | {r['n']:,} | {r['mean_delta_loss']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | "
                       f"{100 * r['share_a_higher']:.0f} % | {fp(r['p_sign'])} | {'**' + fp(r['p_sign_holm']) + '**' if sig(r['p_sign_holm']) else fp(r['p_sign_holm'])} |")
        out.append("")
        out += ["Reading (generated from the table above):", ""]
        for a, b, sl in [("Filtered-Ro", "Base-Ro", "flagged"), ("Filtered-Ro", "Base-Ro", "unflagged"), ("SPP-Ro", "Base-Ro", "flagged"),
                         ("SPP-Ro", "Base-Ro", "heldout"), ("Filtered-Ro", "SPP-Ro", "flagged"), ("SPP-Ro-MT", "Base-Ro-rewarm", "heldout")]:
            r = find_row(prs, a, b, sl)
            if r is not None:
                out.append(f"* {describe_pair(r)}.")
        out.append("")
    return out


# ---------------------------------------------------------------- cloze
def capabilities_section(cap) -> List[str]:
    out = ["### Capabilities: held-out 4-way cloze", ""]
    if cap is None:
        return out + ["_Report missing: run `py src/eval_capabilities.py --n 1000`._", ""]
    out += [f"{cap['n_items']:,} last-word cloze items from held-out sentences (chance 25 %); p = exact McNemar test against the model's reference on the same items.", "",
            "| Model | accuracy [95 % CI] | reference | items gained / lost | McNemar p |", "|---|:---:|---|:---:|:---:|"]
    for key, r in cap["results"].items():
        has = "p_vs_base" in r
        is_ref = key == base_ref_key(key)
        out.append(f"| {NAMES.get(key, key)} | {r['accuracy']:.1f} % {ci(r['ci95'])} | {'—' if is_ref else r.get('reference', base_ref_key(key))} | "
                   f"{(str(r['items_gained']) + ' / ' + str(r['items_lost'])) if has else ''} | "
                   f"{'ref' if is_ref else (fp(r['p_vs_base']) if has else 'n/a (old report)')} |")
    out.append("")
    return out


def cloze_summary(cap) -> str:
    if cap is None:
        return "the held-out cloze test was not run"
    res = cap["results"]
    ps = {k: r["p_vs_base"] for k, r in res.items() if "p_vs_base" in r}
    accs = [r["accuracy"] for r in res.values()]
    low = [k for k, p in ps.items() if p < ALPHA]
    s = f"held-out cloze accuracy {min(accs):.1f}–{max(accs):.1f} % for all models"
    if low:
        s += "; differences with p < 0.05 against the reference (uncorrected): " + ", ".join(f"{k} {res[k]['accuracy']:.1f} % (p = {fp(ps[k])})" for k in low)
    else:
        s += f", no model differs from its reference (smallest McNemar p = {fp(min(ps.values()))})" if ps else ""
    return s


# ---------------------------------------------------------------- ConstitutionEval-Ro
PROTOS = [("continuation", 25), ("letter", 25), ("pmi", 25), ("pairwise", 50)]


def ce_cell(o: Dict[str, Any]) -> str:
    c = f"{o['accuracy']:.1f} {ci(o['ci95'])}"
    if "p_vs_base" in o:
        ph = o.get("p_holm")
        c += f" (p={fp(o['p_vs_base'])}" + (f"; H={'**' + fp(ph) + '**' if sig(ph) else fp(ph)}" if ph is not None else "") + ")"
    return c


def ce_levels(ce) -> List[str]:
    """Where each protocol's model accuracies sit relative to chance (CI below / containing / above)."""
    res = {k: r for k, r in ce["results"].items() if not k.startswith("Euristic")}
    lines = []
    for proto, chance in PROTOS:
        below = [k for k, r in res.items() if r[proto]["overall"]["ci95"][1] < chance]
        above = [k for k, r in res.items() if r[proto]["overall"]["ci95"][0] > chance]
        accs = [r[proto]["overall"]["accuracy"] for r in res.values()]
        if len(below) == len(res):
            where = f"every model below chance (95 % CI entirely under {chance} %)"
        elif len(above) == len(res):
            where = f"every model above chance (95 % CI entirely over {chance} %)"
        else:
            parts = []
            if below:
                parts.append(f"{len(below)} below")
            if above:
                parts.append(f"{len(above)} above")
            parts.append(f"{len(res) - len(below) - len(above)} not distinguishable from chance")
            where = ", ".join(parts) + f" (chance {chance} %)"
        lines.append(f"{proto}: {min(accs):.1f}–{max(accs):.1f} %, {where}")
    return lines


def ce_holm(ce) -> Dict[str, Any]:
    h = ce.get("holm") or {}
    rows = h.get("rows", [])
    return {"m": h.get("m", len(rows)), "sig": [r for r in rows if sig(r.get("p_holm"))],
            "min": min(rows, key=lambda r: r["p_holm"]) if rows else None}


def ce_section(ce, valid) -> List[str]:
    out = ["### ConstitutionEval-Ro", ""]
    if ce is None:
        return out + ["_Report missing: run `py src/eval_constitution.py`._", ""]
    h = ce_holm(ce)
    if str(ce.get("delib_prompt", "")).startswith("v2"):
        delib = ("The *deliberative* rows place the `<assistant>` marker directly after the scenario, with no whitespace in between, "
                 "as in training.")
    else:
        delib = ("The *deliberative* rows of this report were scored with the old prompt (a line break before `<assistant>`, "
                 "which never occurs in training); see the warning at the end of the results.")
    out += [f"{ce['n_items']} items, accuracy in % with the 95 % bootstrap CI (chance 25 %; pairwise: 1,800 gold-vs-distractor pairs, chance 50 %). "
            "p = exact McNemar test against the model's reference on the same items (Base-Ro; Base-Ro-rewarm for the MT models; pairwise: exact "
            f"sign-flip test on the per-item number of pairs won); H = Holm-adjusted p over the {h['m']} overall comparisons. " + delib, "",
            "| Model | continuation | letter (4 rotations) | PMI | pairwise |", "|---|:---:|:---:|:---:|:---:|"]
    for key, r in ce["results"].items():
        label = key.replace(" (delib)", ", deliberative") if not key.startswith("Euristic") else (
            "*Heuristic: shortest option*" if "scurt" in key else "*Heuristic: longest option*")
        out.append(f"| {label} | " + " | ".join(ce_cell(r[p]["overall"]) for p, _ in PROTOS) + " |")
    out += ["", "Levels (generated): " + "; ".join(ce_levels(ce)) + ".", ""]
    if h["sig"]:
        out.append(f"After Holm correction {len(h['sig'])} of {h['m']} comparisons are significant: "
                   + ", ".join(f"{r['model']} vs {r['reference']} on {r['protocol']} (p = {fp(r['p'])}, Holm p = {fp(r['p_holm'])})" for r in h["sig"]) + ".")
    elif h["min"] is not None:
        r = h["min"]
        out.append(f"After Holm correction none of the {h['m']} comparisons is significant (smallest: {r['model']} vs {r['reference']} on "
                   f"{r['protocol']}, p = {fp(r['p'])}, Holm p = {fp(r['p_holm'])}).")
    if valid:
        m = valid.get("models", {})
        q, g, pr = m.get("qwen27b_blind", {}), m.get("gptoss120b_open", {}), m.get("probe_blind", {})
        if q and g:
            out.append(f"Benchmark validity (`evals/constitution_eval_validity.json`): the items are solved blind by {q.get('model')} at "
                       f"{q['accuracy']['overall']} % and open-book by {g.get('model')} at {g['accuracy']['overall']} %"
                       + (f"; the local 7B probe ({pr.get('model')}) solves {pr['accuracy']['overall']} %" if pr else "") + ".")
    out.append("")
    return out


# ---------------------------------------------------------------- adversarial probe
def adv_section(adv):
    """Returns (lines, number of Holm-significant comparisons, number of comparisons)."""
    out = ["### Adversarial-prefix probe", ""]
    if adv is None:
        return out + ["_Report missing: run `py src/eval_jailbreaks.py`._", ""]
    res = adv["results"]
    n = next(iter(res.values()))["n"]
    out += [f"{n} items. A model *prefers the stereotype* on an item when the stereotyped continuation has the higher length-normalised "
            "log-likelihood; the *margin shift* is how far the adversarial prefix moves that preference towards the stereotype (nats/token). "
            "The last column compares each model with its reference: adversarial-prefix rate (exact McNemar) / margin shift (exact sign test).", "",
            "| Model | stereotype preferred, neutral prefix | stereotype preferred, adversarial prefix | neutral → adversarial (McNemar p) | margin shift (nats/token) | vs reference: adversarial rate / margin shift |",
            "|---|:---:|:---:|:---:|:---:|:---:|"]
    fam = []
    for key, r in res.items():
        v = r.get("vs_reference")
        if v:
            fam += [(key, "adversarial rate", v["p_adv_rate"]), (key, "margin shift", v["p_margin_shift_sign"])]
    adj = dict(zip([(k, w) for k, w, _ in fam], holm_adjust([p for _, _, p in fam]))) if fam else {}
    for key, r in res.items():
        v = r.get("vs_reference")
        vs = "ref" if not v else (f"{v['adv_rate_diff']:+.1f} pp (p={fp(v['p_adv_rate'])}; H={fp(adj[(key, 'adversarial rate')])}) / "
                                  f"{v['margin_shift_diff']:+.3f} (p={fp(v['p_margin_shift_sign'])}; H={fp(adj[(key, 'margin shift')])})")
        out.append(f"| {NAMES.get(key, key)} | {r['neutral_rate']:.1f} % {ci(r['neutral_rate_ci95'])} | {r['adv_rate']:.1f} % {ci(r['adv_rate_ci95'])} | "
                   f"{fp(r['p_neutral_vs_adv_mcnemar'])} | {r['mean_margin_shift']:+.3f} [{r['margin_shift_ci95'][0]:+.3f}, {r['margin_shift_ci95'][1]:+.3f}] | {vs} |")
    out.append("")
    if fam:
        best = min(fam, key=lambda x: x[2])
        n_sig = sum(1 for k, w, _ in fam if adj[(k, w)] < ALPHA)
        out.append(f"Holm over the {len(fam)} comparisons with the references: {n_sig} significant; smallest p = {fp(best[2])} "
                   f"({best[0]}, {best[1]}), Holm p = {fp(adj[(best[0], best[1])])}.")
        out.append("")
    return out, (n_sig if fam else 0), len(fam)


# ---------------------------------------------------------------- summary
def summary_text(tax, paired, cap, ce, adv_counts) -> str:
    prs = paired_rows(paired)
    parts = ["**What I find** (this paragraph is generated by `src/report_results.py` from the reports in `evals/`; significance = Holm-adjusted p < 0.05)."]
    r = find_row(prs, "SPP-Ro", "Base-Ro", "heldout")
    if r is not None:
        parts.append(f"*Cost of SPP.* {describe_pair(r)}; {cloze_summary(cap)}.")
    rf, ru, rs = find_row(prs, "Filtered-Ro", "Base-Ro", "flagged"), find_row(prs, "Filtered-Ro", "Base-Ro", "unflagged"), find_row(prs, "SPP-Ro", "Base-Ro", "flagged")
    if rf is not None:
        parts.append(f"*Filtered baseline (the flagged documents are in its context but not in its loss).* {describe_pair(rf)}"
                     + (f"; {describe_pair(ru)}" if ru is not None else "") + ".")
    if rs is not None:
        parts.append(f"*SPP on the same flagged documents (kept in the loss, 10 % of the corpus annotated).* {describe_pair(rs)}.")
    rm = find_row(prs, "SPP-Ro-MT", "Base-Ro-rewarm", "heldout")
    if rm is not None:
        parts.append(f"*Midtraining.* {describe_pair(rm)}.")
    if ce is not None:
        h = ce_holm(ce)
        lv = "; ".join(ce_levels(ce))
        if h["sig"]:
            parts.append(f"*ConstitutionEval-Ro.* {len(h['sig'])} of the {h['m']} model-vs-reference comparisons survive Holm correction ("
                         + ", ".join(f"{x['model']} vs {x['reference']} on {x['protocol']}, Holm p = {fp(x['p_holm'])}" for x in h["sig"])
                         + f"). Absolute levels: {lv}.")
        else:
            parts.append(f"*ConstitutionEval-Ro.* None of the {h['m']} model-vs-reference comparisons survives Holm correction. Absolute levels: "
                         f"{lv}; the Vanilla model sits at the same levels, so at 88M parameters the benchmark does not separate the training recipes.")
    if adv_counts is not None:
        n_sig, m = adv_counts
        how = (f"none of the {m} comparisons with the references is significant after Holm correction" if n_sig == 0 else
               f"{n_sig} of the {m} comparisons with the references {'is' if n_sig == 1 else 'are'} significant after Holm correction")
        parts.append(f"*Adversarial-prefix probe.* The adversarial prefix moves every model's margin towards the stereotype; {how}.")
    return " ".join(p.rstrip() for p in parts)


# ---------------------------------------------------------------- write
def replace_block(text: str, tag: str, body: str) -> Optional[str]:
    start, end = f"<!-- {tag}:START -->", f"<!-- {tag}:END -->"
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0 or j < i:
        return None
    return text[:i + len(start)] + "\n" + body.rstrip() + "\n" + text[j:]


def main():
    tax = load("thesis_alignment_tax_report.json")
    paired = load("thesis_alignment_paired_report.json")
    cap = load("thesis_capabilities_report.json")
    ce = load("thesis_constitution_eval_report.json")
    adv = load("thesis_adversarial_prefix_report.json")
    valid = load("constitution_eval_validity.json")
    prov = heldout_provenance(tax, paired)
    if ce is not None and not str(ce.get("delib_prompt", "")).startswith("v2"):
        prov["warnings"].append("thesis_constitution_eval_report.json was scored with the old deliberative prompt: re-run eval_constitution.py.")
    if ce is not None and "holm" not in ce:
        prov["warnings"].append("thesis_constitution_eval_report.json has no Holm correction (old scorer): run `py src/eval_constitution.py --stats-only`.")

    results: List[str] = []
    results += perplexity_section(tax, paired, prov)
    results += capabilities_section(cap)
    results += ce_section(ce, valid)
    adv_out = adv_section(adv) if adv is not None else (["### Adversarial-prefix probe", "", "_Report missing._", ""], None, None)
    adv_lines, n_sig, m = adv_out if isinstance(adv_out, tuple) else (adv_out, None, None)
    results += adv_lines
    if prov["warnings"]:
        results += ["> **Warning (generated):** " + " ".join(prov["warnings"]), ""]
    summary = summary_text(tax, paired, cap, ce, (n_sig, m) if m else None)

    st = prov["stats"]
    prev = st.get("previous_file") or {}
    header = ["# SPP-Ro: results", "",
              f"Generated by `src/report_results.py` on {datetime.datetime.now():%Y-%m-%d %H:%M} from the reports in `evals/`. "
              "Do not edit by hand: re-run the script (or `run_fixes.bat`).", ""]
    if st and prov["version"] >= 3 and isinstance(st.get("shard_docs"), int):
        header += [f"Held-out set: version {st['version']}, `{st.get('shard', '?')}`, {st.get('saved', 0):,} documents "
                   f"(shard {st['shard_docs']:,} documents; dropped: {st.get('too_short_corpus_filter', 0):,} by the corpus length filter, "
                   f"{st.get('overlap_prefix', 0):,} + {st.get('overlap_hash', 0):,} already in training (prefix / MD5), "
                   f"{st.get('duplicate_in_shard', 0):,} duplicates in the shard, {st.get('low_quality', 0):,} by the quality filter).", ""]
    elif st:
        header += [f"Held-out set: built by the previous builder (version {prov['version']}; not normalised like the corpus): `{json.dumps(st)}`.", ""]
    if prev and "docs" in prev:
        fi = lambda v: f"{v:,}" if isinstance(v, int) else str(v)
        header += [f"Previous held-out file (replaced): {fi(prev['docs'])} documents, of which {fi(prev.get('in_training_after_corpus_normalisation', '?'))} "
                   f"are in the training data once normalised like the corpus; {fi(prev.get('with_cedilla', '?'))} contained cedilla ş/ţ, "
                   f"{fi(prev.get('with_line_break', '?'))} contained a line break.", ""]
    md = header + ["## Summary", "", summary, ""] + [("#" + l if l.startswith("### ") else l) for l in results]
    RESULTS_MD.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    print(f"[OK] {RESULTS_MD}")

    if README.exists():
        text = README.read_text(encoding="utf-8")
        new = replace_block(text, "SUMMARY", summary)
        new = replace_block(new, "RESULTS", "\n".join(results)) if new is not None else None
        if new is None:
            print("[warn] README.md has no <!-- SUMMARY/RESULTS:START/END --> markers: README not updated.")
        elif new != text:
            README.write_text(new, encoding="utf-8")
            print(f"[OK] {README} (summary + results sections regenerated)")
        else:
            print("[OK] README.md already up to date")
    for w in prov["warnings"]:
        print("[warn]", w)


if __name__ == "__main__":
    main()

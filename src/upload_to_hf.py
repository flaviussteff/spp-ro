"""
Publish the SPP-Ro models on the Hugging Face Hub.

Uploads every model folder listed in HF_RELEASES (the five reported models plus the safety classifier) to its repo,
with a model card generated from the actual evaluation reports in evals/ (perplexity, cloze, ConstitutionEval-Ro),
so the numbers on the Hub are the ones in the repository. Training checkpoints and training_args.bin are not uploaded.

Usage:
    py src/upload_to_hf.py                  (all models, public repos)
    py src/upload_to_hf.py --private        (private repos)
    py src/upload_to_hf.py --only spp-ro spp-ro-base
    py src/upload_to_hf.py --dry-run        (print the cards, upload nothing)
Login once with `hf auth login` (or set HF_TOKEN); the token needs write permission.
"""
import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config import (EVALS_DIR, CLEAN_DATA_DIR, base_ref_key, BASE_MODEL_DIR, SPP_MODEL_DIR, FILTERED_MODEL_DIR,  # noqa: E402
                    BASE_REWARM_MODEL_DIR, SPP_MT_MODEL_DIR, SAFETY_CLF_DIR)
from report_results import describe_pair, find_row, ce_holm, ce_levels, fp  # noqa: E402  (same wording as the README)

HF_USER = "flaviussteff"
GITHUB_URL = "https://github.com/flaviussteff/spp-ro"
PAPER = "Minder et al., *Synthetic Persona Pretraining*, EPFL-dlab, 2026 (arXiv:2608.13482)"

# Hub repo name -> (model folder, key in the evals/ reports, one-line role)
HF_RELEASES = {
    "spp-ro":             (SPP_MODEL_DIR, "SPP-Ro", "Synthetic Persona Pretraining from token zero (SPP{T0}): 10% of the training documents carry an inline constitutional reflection."),
    "spp-ro-base":        (BASE_MODEL_DIR, "Base-Ro", "Vanilla control: the same documents, the same stream, no reflections."),
    "spp-ro-filtered":    (FILTERED_MODEL_DIR, "Filtered-Ro", "Filtered baseline: the same documents; the 75,000 classifier-flagged ones stay in context but are removed from the loss."),
    "spp-ro-mt":          (SPP_MT_MODEL_DIR, "SPP-Ro-MT", "Midtraining variant (SPP{MT}): the Vanilla model at step 45,000 + 5,000 steps on the annotated stream (67.6% of the stream's documents carry a reflection; for those documents the loss is on the reflection only)."),
    "spp-ro-base-rewarm": (BASE_REWARM_MODEL_DIR, "Base-Ro-rewarm", "Control for spp-ro-mt: the Vanilla model at step 45,000 + 5,000 steps with the same fresh LR schedule, no reflections."),
    "spp-ro-safety-classifier": (SAFETY_CLF_DIR, None, "Romanian BERT safety classifier (severity >= 3 vs <= 2) that selects the documents to annotate and the Filtered mask set."),
}

ARCH_TABLE = """| Parameter | Value |
|---|---|
| Parameters | 88,099,584 (~88.1M), tied embeddings |
| hidden / intermediate | 768 / 2048 (SwiGLU) |
| Layers | 12, GQA 12 query / 4 key-value heads |
| Context | 1,024 tokens, RoPE theta 10,000 |
| Vocabulary | 16,384 byte-level BPE (Romanian), `<assistant>` special token |"""


def _load(name: str) -> Optional[dict]:
    p = EVALS_DIR / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def eval_numbers(key: str) -> Dict[str, str]:
    """Headline numbers for one model key, read from the current evals/ reports (missing -> omitted)."""
    out: Dict[str, str] = {}
    tax = _load("thesis_alignment_tax_report.json")
    if tax:
        sizes = tax.get("passages_per_domain", {})
        out["n_heldout"] = f"{sizes.get('Web (held-out)', 0):,}"
        out["n_flagged"] = f"{sizes.get('Web held-out, semnalat de clasificator', 0):,}"
        out["n_unflagged"] = f"{sizes.get('Web held-out, nesemnalat', 0):,}"
        rows = {r["model"]: r for r in tax["results"]}
        r, ref = rows.get(key), rows.get(base_ref_key(key))
        if r:
            out["ppl"] = f"{r['heldout_ppl']:.2f}"
            if "flagged_ppl" in r:
                out["ppl_flagged"], out["ppl_unflagged"] = f"{r['flagged_ppl']:.2f}", f"{r['unflagged_ppl']:.2f}"
            if ref and ref is not r:
                out["dppl"] = f"{r['heldout_ppl'] - ref['heldout_ppl']:+.2f}"
                if "flagged_ppl" in r and "flagged_ppl" in ref:
                    out["dppl_flagged"] = f"{r['flagged_ppl'] - ref['flagged_ppl']:+.2f}"
                    out["dppl_unflagged"] = f"{r['unflagged_ppl'] - ref['unflagged_ppl']:+.2f}"
    cap = _load("thesis_capabilities_report.json")
    if cap:
        out["n_cloze"] = f"{cap.get('n_items', 0):,}"
        r = cap["results"].get(key) if isinstance(cap["results"], dict) else None
        if r:
            out["cloze"] = f"{r['accuracy']:.1f}"
            if "p_vs_base" in r:
                out["cloze_p"] = fp(r["p_vs_base"])
    ce = _load("thesis_constitution_eval_report.json")
    if ce and key in ce["results"]:
        res = ce["results"][key]
        for proto in ("continuation", "letter", "pmi", "pairwise"):
            if proto in res and isinstance(res[proto], dict) and "overall" in res[proto]:
                o = res[proto]["overall"]
                out[f"ce_{proto}"] = f"{o['accuracy']:.1f}"
                if "p_vs_base" in o:
                    out[f"ce_{proto}_p"] = f"p = {fp(o['p_vs_base'])}" + (f", Holm p = {fp(o['p_holm'])}" if "p_holm" in o else "")
        delib = ce["results"].get(f"{key} (delib)")
        if delib and "letter" in delib:
            out["ce_letter_delib"] = f"{delib['letter']['overall']['accuracy']:.1f}"
    adv = _load("thesis_adversarial_prefix_report.json")
    if adv and isinstance(adv.get("results"), dict) and key in adv["results"]:
        r = adv["results"][key]
        out["adv_n"] = str(adv.get("n_items", r.get("n", "")))
        out["adv_neutral"], out["adv_adv"] = f"{r['neutral_rate']:.1f}", f"{r['adv_rate']:.1f}"
        out["adv_shift"] = f"{r['mean_margin_shift']:+.3f}"
    return out


def flag_rates() -> str:
    """How much of the corpus / held-out set the classifier flags, from the scoring summary and the perplexity report."""
    parts = []
    s = CLEAN_DATA_DIR / "spp_candidates_summary.json"
    if s.exists():
        d = json.load(open(s, encoding="utf-8"))
        h, n = d.get("score_histogram_10bins"), d.get("scanned")
        if h and n:
            parts.append(f"On the 1.5M-document corpus it flags {100 * sum(h[5:]) / n:.1f} % of the documents at threshold 0.5 "
                         f"and {100 * sum(h[1:]) / n:.1f} % at 0.1 ({d.get('spam_or_short_skipped', 0):,} spam/short documents not scored)")
    tax = _load("thesis_alignment_tax_report.json")
    if tax and tax.get("clf_threshold") is not None:
        sz = tax.get("passages_per_domain", {})
        nf, na = sz.get("Web held-out, semnalat de clasificator"), sz.get("Web (held-out)")
        if nf is not None and na:
            parts.append(f"on the held-out web set of the evaluation it flags {nf:,} of {na:,} documents ({100 * nf / na:.1f} %) at threshold {tax['clf_threshold']}")
    return ("; ".join(parts) + ".") if parts else ""


def filtered_reading() -> str:
    paired = _load("thesis_alignment_paired_report.json")
    r = find_row(paired.get("rows", []), "Filtered-Ro", "Base-Ro", "flagged") if paired else None
    return f"Measured: {describe_pair(r)}." if r else ""


def reading(key: str) -> str:
    """Generated reading of this model's results against its reference (same wording as the README)."""
    out = []
    ref = base_ref_key(key)
    paired = _load("thesis_alignment_paired_report.json")
    if paired and key != ref:
        rows = paired.get("rows", [])
        for sl in ("heldout", "flagged", "unflagged"):
            r = find_row(rows, key, ref, sl)
            if r:
                out.append(f"* Perplexity, paired per document: {describe_pair(r)}.")
    ce = _load("thesis_constitution_eval_report.json")
    if ce and "holm" in ce:
        h = ce_holm(ce)
        mine = [x for x in h["sig"] if x["model"] in (key, f"{key} (delib)")]
        if key == ref:
            s = f"reference model ({len(h['sig'])} of the {h['m']} comparisons of the family survive Holm correction)"
        elif mine:
            s = (f"significant after Holm correction over the {h['m']} comparisons of the family: "
                 + ", ".join(f"{x['model']} vs {x['reference']} on {x['protocol']} (Holm p = {fp(x['p_holm'])})" for x in mine))
        else:
            s = f"no difference from the reference survives Holm correction over the {h['m']} comparisons of the family"
        out.append(f"* ConstitutionEval-Ro: {s}. Levels of the whole family: " + "; ".join(ce_levels(ce)) + ".")
    return ("\n" + "\n".join(out) + "\n\n") if out else ""


def model_card(repo: str, key: Optional[str], role: str) -> str:
    repo_id = f"{HF_USER}/{repo}"
    if key is None:   # classifier
        thr = {}
        tp = SAFETY_CLF_DIR / "threshold.json"
        if tp.exists():
            thr = json.load(open(tp, encoding="utf-8"))
        return f"""---
language:
- ro
license: mit
pipeline_tag: text-classification
base_model: dumitrescustefan/bert-base-romanian-cased-v1
tags:
- romanian
- safety
- synthetic-persona-pretraining
---

# {repo}

{role}

Part of **SPP-Ro** ({GITHUB_URL}), a compute-matched replication in Romanian of {PAPER}.

Fine-tuned from `dumitrescustefan/bert-base-romanian-cased-v1` on {thr.get('n_train', 0):,} real web documents whose severity (0-5) was labelled by an LLM; the target is severity >= 3. Dev split ({thr.get('n_dev', 0):,} documents): AUC {thr.get('dev_auc', float('nan')):.3f}, threshold {thr.get('threshold', 'n/a')} -> precision {thr.get('dev_precision', 'n/a')}, recall {thr.get('dev_recall', 'n/a')} (severity 4-5: {thr.get('dev_recall_sev45', 'n/a')}, severity 3: {thr.get('dev_recall_sev3', 'n/a')}). Input: the first {thr.get('max_len', 384)} tokens of the document. {flag_rates()}

Used to (1) select the documents that receive a reflection in the SPP sidecar, (2) define the loss-masked set of the Filtered baseline, and (3) define the "flagged" held-out slice of the alignment-tax evaluation (where the Filtered baseline, which had such documents in context but never received loss on them, is expected to lose knowledge). {filtered_reading()}

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
tok = AutoTokenizer.from_pretrained("{repo_id}")
clf = AutoModelForSequenceClassification.from_pretrained("{repo_id}").eval()
p = torch.softmax(clf(**tok("text românesc ...", truncation=True, max_length=384, return_tensors="pt")).logits, -1)[0, 1]
```
"""
    n = eval_numbers(key)
    ref = base_ref_key(key)
    ref_repo = next((r for r, (_, k, _) in HF_RELEASES.items() if k == ref), None)
    is_ref = key == ref
    res_rows = []
    if "ppl" in n:
        res_rows.append(f"| Perplexity, held-out web ({n['n_heldout']} docs) | {n['ppl']} | {'ref' if is_ref else n.get('dppl', '')} |")
    if "ppl_flagged" in n:
        res_rows.append(f"| Perplexity, held-out docs flagged by the safety classifier ({n['n_flagged']} docs) | {n['ppl_flagged']} | {'ref' if is_ref else n.get('dppl_flagged', '')} |")
        res_rows.append(f"| Perplexity, held-out docs not flagged ({n['n_unflagged']} docs) | {n['ppl_unflagged']} | {'ref' if is_ref else n.get('dppl_unflagged', '')} |")
    if "cloze" in n:
        res_rows.append(f"| Held-out 4-way cloze accuracy ({n['n_cloze']} items) | {n['cloze']} % | {'ref' if is_ref else ('McNemar p = ' + n['cloze_p'] if 'cloze_p' in n else '')} |")
    for proto, label in (("letter", "letter, 4-rotation debiased (chance 25 %)"), ("pmi", "PMI (chance 25 %)"),
                         ("continuation", "continuation (chance 25 %)"), ("pairwise", "gold vs each distractor (chance 50 %)")):
        if f"ce_{proto}" in n:
            res_rows.append(f"| ConstitutionEval-Ro accuracy, {label}, 600 items | {n[f'ce_{proto}']} % | {'ref' if is_ref else n.get(f'ce_{proto}_p', '')} |")
    if "ce_letter_delib" in n:
        res_rows.append(f"| ConstitutionEval-Ro, letter, deliberative (`<assistant>` after the scenario) | {n['ce_letter_delib']} % | |")
    if "adv_adv" in n:
        res_rows.append(f"| Adversarial-prefix probe ({n['adv_n']} items): stereotyped continuation preferred, neutral prefix | {n['adv_neutral']} % | |")
        res_rows.append(f"| Adversarial-prefix probe: stereotyped continuation preferred, adversarial prefix | {n['adv_adv']} % | |")
        res_rows.append(f"| Adversarial-prefix probe: margin shift towards the stereotype (nats/token) | {n['adv_shift']} | |")
    results = ("| Metric | Value | vs. reference |\n|---|---|---|\n" + "\n".join(res_rows)) if res_rows else "_(evaluation reports not found at upload time)_"
    ref_note = "" if is_ref else f"\nReference model: [`{HF_USER}/{ref_repo}`](https://huggingface.co/{HF_USER}/{ref_repo})." if ref_repo else ""
    mt = "-MT" in key or "rewarm" in key
    steps = ("5,000 steps from the Vanilla checkpoint at step 45,000, fresh schedule (peak LR 5e-5, 250 warm-up steps, cosine to 0)"
             if mt else "50,000 steps from scratch (peak LR 1.5e-4, 500 warm-up steps, cosine to 0, weight decay 0.1)")
    spp_note = ""
    if key.startswith("SPP"):
        spp_note = """
**SPP document mode.** `c1 ... ck <assistant> r1 ... rn ck+1 ...`: the reflection (written by an LLM for this very document, grounded in a six-article Romanian value constitution) is inserted at a sentence boundary; the continuation **cannot attend to the reflection** (asymmetric attention mask) and its RoPE positions alias back to the insertion point, so the document itself is modelled exactly as in the Vanilla model. The `<assistant>` token carries no loss. At inference, `<assistant>` is the marker after which the model was trained to write a reflection (there is no post-training).
"""
    return f"""---
language:
- ro
license: mit
pipeline_tag: text-generation
tags:
- romanian
- llama
- causal-lm
- from-scratch
- synthetic-persona-pretraining
---

# {repo}

{role}

Part of **SPP-Ro** ({GITHUB_URL}): a compute-matched, from-scratch replication in Romanian of {PAPER}, comparing Vanilla, Filtered and SPP pretraining on **exactly the same document stream**. Bachelor's thesis, Faculty of Mathematics and Computer Science, University of Bucharest (2026).

## Training

* **Data.** 1.5M Romanian documents (FineWeb-2 RO, Wikipedia RO, press), read in a fixed-seed order that is identical for every model of the family (≈4 passes over the corpus in 50,000 steps; documents longer than 1,023 tokens are truncated). 150,000 documents (10 %) have a reflection in the sidecar: 75,000 documents flagged by a safety classifier (score ≥ 0.12, chosen among those whose insertion point falls inside the 1,024-token window) plus 75,000 benign ones; reflections average 36 tokens (~0.68 % of the trained tokens).
* **Schedule.** {steps}; 64 packed windows of 1,024 positions per step (EOS-terminated documents packed first-fit, block-diagonal attention, per-document positions, 2.4 % padding); {"≈0.33" if mt else "≈3.28"} billion token positions in this run{"" if mt else " (~66 h)"}; bf16 autocast with fp32 master weights; single NVIDIA RTX 3060 12 GB.
{spp_note}
## Architecture

{ARCH_TABLE}

## Results

{results}
{ref_note}
{reading(key)}Details, per-item predictions and LaTeX tables: `evals/` and `evals/RESULTS.md` in the GitHub repository.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}", dtype=torch.float32).eval()
prompt = "{'<assistant>' if key.startswith('SPP') else ''}În România, egalitatea de șanse între femei și bărbați"
out = model.generate(**tok(prompt, return_tensors="pt"), max_new_tokens=60, do_sample=True, temperature=0.7)
print(tok.decode(out[0], skip_special_tokens=False))
```

## Limitations

Small (88M), monolingual Romanian, pretraining only (no instruction tuning or post-training), single seed. Trained on web text: it reproduces the biases of its data. Not intended for production use.

## Citation

```bibtex
@mastersthesis{{habeanu2026sppro,
  title  = {{Pre-antrenarea Modelelor Generative Românești cu Personalitate Sintetică: Aliniere Constituțională de la Token Zero}},
  author = {{Hăbeanu, Flavius-Ștefan}},
  school = {{Faculty of Mathematics and Computer Science, University of Bucharest}},
  year   = {{2026}},
  url    = {{{GITHUB_URL}}}
}}
```
"""


def main():
    global HF_USER
    ap = argparse.ArgumentParser(description="Publish the SPP-Ro models on the Hugging Face Hub")
    ap.add_argument("--only", nargs="*", default=None, help="repo names to upload (default: all): " + " ".join(HF_RELEASES))
    ap.add_argument("--private", action="store_true", help="create the repos as private")
    ap.add_argument("--dry-run", action="store_true", help="print the generated model cards, upload nothing")
    ap.add_argument("--user", default=HF_USER, help=f"Hugging Face namespace (default {HF_USER})")
    args = ap.parse_args()
    HF_USER = args.user

    todo = {k: v for k, v in HF_RELEASES.items() if not args.only or k in args.only}
    unknown = set(args.only or []) - set(HF_RELEASES)
    if unknown:
        raise SystemExit(f"repo-uri necunoscute: {sorted(unknown)}; disponibile: {list(HF_RELEASES)}")

    if not args.dry_run:
        from huggingface_hub import HfApi, get_token
        token = os.environ.get("HF_TOKEN") or get_token()
        if not token:
            raise SystemExit("Nu esti autentificat pe Hugging Face: ruleaza `hf auth login` (token cu permisiune Write) sau seteaza HF_TOKEN.")
        api = HfApi(token=token)
        who = api.whoami().get("name")
        print(f"Autentificat ca: {who}")

    for repo, (folder, key, role) in todo.items():
        repo_id = f"{HF_USER}/{repo}"
        print("\n" + "=" * 72 + f"\n  {repo_id}  <-  {folder}\n" + "=" * 72)
        if not (folder / "model.safetensors").exists():
            print(f"  [skip] lipseste {folder / 'model.safetensors'}")
            continue
        card = model_card(repo, key, role)
        if args.dry_run:
            print(card)
            continue
        api.create_repo(repo_id=repo_id, private=args.private, exist_ok=True)
        print("  [1/2] upload folder (fara checkpoints/, training_args.bin, README.md local) ...")
        api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model",
                          ignore_patterns=["checkpoints/*", "checkpoints", "checkpoints_keep/*", "checkpoints_keep", "training_args.bin", "README.md", "*.bin"],
                          commit_message=f"Upload {repo}")
        print("  [2/2] upload model card ...")
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(card)
            tmp = f.name
        api.upload_file(path_or_fileobj=tmp, path_in_repo="README.md", repo_id=repo_id, repo_type="model", commit_message=f"Model card for {repo}")
        os.unlink(tmp)
        print(f"  [OK] https://huggingface.co/{repo_id}")

    if not args.dry_run:
        print("\nGata. Colectia: https://huggingface.co/" + HF_USER)


if __name__ == "__main__":
    main()

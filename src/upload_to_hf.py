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
from config import (EVALS_DIR, base_ref_key, BASE_MODEL_DIR, SPP_MODEL_DIR, FILTERED_MODEL_DIR,  # noqa: E402
                    BASE_REWARM_MODEL_DIR, SPP_MT_MODEL_DIR, SAFETY_CLF_DIR)

HF_USER = "flaviussteff"
GITHUB_URL = "https://github.com/flaviussteff/spp-ro"
PAPER = "Minder et al., *Synthetic Persona Pretraining*, EPFL-dlab, 2026 (arXiv:2608.13482)"

# Hub repo name -> (model folder, key in the evals/ reports, one-line role)
HF_RELEASES = {
    "spp-ro":             (SPP_MODEL_DIR, "SPP-Ro", "Synthetic Persona Pretraining from token zero (SPP{T0}): 10% of the training documents carry an inline constitutional reflection."),
    "spp-ro-base":        (BASE_MODEL_DIR, "Base-Ro", "Vanilla control: the same documents, the same stream, no reflections."),
    "spp-ro-filtered":    (FILTERED_MODEL_DIR, "Filtered-Ro", "Filtered baseline: the same documents; the 75,000 classifier-flagged ones stay in context but are removed from the loss."),
    "spp-ro-mt":          (SPP_MT_MODEL_DIR, "SPP-Ro-MT", "Midtraining variant (SPP{MT}): the Vanilla model at step 45,000 + 5,000 steps on the annotated stream (67.6% annotated sequences, loss on the reflection only)."),
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
    cap = _load("thesis_capabilities_report.json")
    if cap:
        rows = cap["results"]
        rows = rows if isinstance(rows, list) else [dict(v, model=k) for k, v in rows.items()]
        for r in rows:
            if r.get("model") == key:
                out["cloze"] = f"{r['accuracy']:.1f}"
    ce = _load("thesis_constitution_eval_report.json")
    if ce and key in ce["results"]:
        res = ce["results"][key]
        for proto in ("continuation", "letter", "pmi", "pairwise"):
            if proto in res and isinstance(res[proto], dict) and "overall" in res[proto]:
                out[f"ce_{proto}"] = f"{res[proto]['overall']['accuracy']:.1f}"
        delib = ce["results"].get(f"{key} (delib)")
        if delib and "letter" in delib:
            out["ce_letter_delib"] = f"{delib['letter']['overall']['accuracy']:.1f}"
    return out


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

Fine-tuned from `dumitrescustefan/bert-base-romanian-cased-v1` on {thr.get('n_train', 'n/a')} real web documents whose severity (0-5) was labelled by an LLM; the target is severity >= 3. Dev split ({thr.get('n_dev', 'n/a')} documents): AUC {thr.get('dev_auc', float('nan')):.3f}, threshold {thr.get('threshold', 'n/a')} -> precision {thr.get('dev_precision', 'n/a')}, recall {thr.get('dev_recall', 'n/a')}. Input: the first {thr.get('max_len', 384)} tokens of the document. It flags about 3% of the 1.5M-document corpus at threshold 0.5 and 6.6% of the held-out web set at threshold 0.1.

Used to (1) select the documents that receive a reflection in the SPP sidecar, (2) define the loss-masked set of the Filtered baseline, and (3) define the "flagged" held-out slice of the alignment-tax evaluation, on which the Filtered model loses knowledge and SPP does not.

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
        res_rows.append(f"| Perplexity, held-out web (1,000 docs) | {n['ppl']} | {'ref' if is_ref else n.get('dppl', '')} |")
    if "ppl_flagged" in n:
        res_rows.append(f"| Perplexity, held-out docs flagged by the safety classifier (199 docs) | {n['ppl_flagged']} | {'ref' if is_ref else n.get('dppl_flagged', '')} |")
        res_rows.append(f"| Perplexity, held-out docs not flagged (1,000 docs) | {n['ppl_unflagged']} | |")
    if "cloze" in n:
        res_rows.append(f"| Held-out 4-way cloze accuracy (1,000 items) | {n['cloze']} % | |")
    for proto, label in (("letter", "letter, 4-rotation debiased (chance 25 %)"), ("pmi", "PMI (chance 25 %)"),
                         ("continuation", "continuation (chance 25 %)"), ("pairwise", "gold vs each distractor (chance 50 %)")):
        if f"ce_{proto}" in n:
            res_rows.append(f"| ConstitutionEval-Ro accuracy, {label}, 600 items | {n[f'ce_{proto}']} % | |")
    if "ce_letter_delib" in n:
        res_rows.append(f"| ConstitutionEval-Ro, letter, deliberative (`<assistant>` prefix) | {n['ce_letter_delib']} % | |")
    results = ("| Metric | Value | vs. reference |\n|---|---|---|\n" + "\n".join(res_rows)) if res_rows else "_(evaluation reports not found at upload time)_"
    ref_note = "" if is_ref else f"\nReference model: [`{HF_USER}/{ref_repo}`](https://huggingface.co/{HF_USER}/{ref_repo})." if ref_repo else ""
    mt = "-MT" in key or "rewarm" in key
    steps = ("5,000 steps from the Vanilla checkpoint at step 45,000, fresh schedule (peak LR 5e-5, 250 warm-up steps, cosine to 0)"
             if mt else "50,000 steps from scratch (peak LR 1.5e-4, 500 warm-up steps, cosine to 0, weight decay 0.1)")
    spp_note = ""
    if key.startswith("SPP"):
        spp_note = """
**SPP document mode.** `c1 ... ck <assistant> r1 ... rn ck+1 ...`: the reflection (written by an LLM for this very document, grounded in a six-article Romanian value constitution) is inserted at a sentence boundary; the continuation **cannot attend to the reflection** (asymmetric attention mask) and its RoPE positions alias back to the insertion point, so the document itself is modelled exactly as in the Vanilla model. The `<assistant>` token carries no loss. Prefixing a prompt with `<assistant>` activates the reflective persona at inference.
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

* **Data.** 1.5M Romanian documents (FineWeb-2 RO, Wikipedia RO, press), read in a fixed-seed order that is identical for every model of the family. 150,000 documents (10 %) have a reflection in the sidecar: the 75,000 best-scored documents of a safety classifier plus 75,000 benign ones; reflections average 36 tokens (~0.67 % of the trained tokens).
* **Schedule.** {steps}; 64 packed windows of 1,024 positions per step (EOS-terminated documents packed first-fit, block-diagonal attention, per-document positions, 2.4 % padding); {"≈0.33" if mt else "≈3.28"} billion tokens in this run{"" if mt else " (~66 h)"}; bf16 autocast with fp32 master weights; single NVIDIA RTX 3060 12 GB.
{spp_note}
## Architecture

{ARCH_TABLE}

## Results

{results}
{ref_note}
All five models of the family are at chance level on ConstitutionEval-Ro (an 88M-parameter model does not solve the benchmark, so it cannot show an alignment effect at this scale); the measurable result is the alignment tax: SPP costs nothing in perplexity, while the Filtered baseline loses knowledge exactly on the classifier-flagged documents (+1.13 perplexity, worse on 85 % of them, paired sign test p ≈ 1e-24). Details, per-item predictions and LaTeX tables: `evals/` in the GitHub repository.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}", dtype=torch.float32).eval()
prompt = "{'<assistant> ' if key.startswith('SPP') else ''}În România, egalitatea de șanse între femei și bărbați"
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

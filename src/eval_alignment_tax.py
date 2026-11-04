"""
Alignment Tax Evaluation Harness (corrected)

Measures cross-entropy / perplexity of every available model on:
  1. "Web (held-out)" : data/eval/heldout_fineweb.parquet, built by build_heldout_set.py from a
     FineWeb-2 shard that was NOT consumed by the training corpus, with every document whose
     prefix or hash appears in the corpus or in the reflection sidecar removed. This is the
     ONLY set on which an alignment-tax conclusion may be drawn.
  2. "Wikipedia (în antrenare)" / "Presă (în antrenare)" : the tails of wiki_clean / news_recent,
     kept for transparency. They were used in the first version of the thesis as "held-out",
     but 92-99% of these passages are inside the training corpus and ~800 are reflection hosts,
     so their perplexity measures memorisation, not generalisation.

Delta PPL is computed on the held-out web set only.
"""
import sys
import math
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config import (BASE_MODEL_DIR, SPP_MODEL_DIR, SPP_20PCT_MODEL_DIR, SPP_MT_MODEL_DIR,
                        TOKENIZER_DIR, CLEAN_DATA_DIR, EVALS_DIR, HELDOUT_FILE, ALL_MODELS)
except ImportError:
    from src.config import (BASE_MODEL_DIR, SPP_MODEL_DIR, SPP_20PCT_MODEL_DIR, SPP_MT_MODEL_DIR,
                            TOKENIZER_DIR, CLEAN_DATA_DIR, EVALS_DIR, HELDOUT_FILE, ALL_MODELS)

EVALS_DIR.mkdir(parents=True, exist_ok=True)

HELDOUT_KEY = "Web (held-out)"
INTRAIN_KEYS = ("Wikipedia (în antrenare)", "Presă (în antrenare)")


class ValidationTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.samples = []
        for t in texts:
            if len(t.strip()) < 50:
                continue
            enc = tokenizer.encode(t.strip(), add_special_tokens=False)
            if len(enc) > 16:
                self.samples.append(enc[:max_length])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def collate_fn_pad(batch, pad_token_id=0):
    max_len = max(len(b["input_ids"]) for b in batch)
    inputs, labels, masks = [], [], []
    for b in batch:
        inp = b["input_ids"].tolist()
        pad = max_len - len(inp)
        inputs.append(inp + [pad_token_id] * pad)
        labels.append(inp + [-100] * pad)
        masks.append([1] * len(inp) + [0] * pad)
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


@torch.no_grad()
def compute_dataset_perplexity(model, dataloader, device) -> Tuple[float, float, int]:
    """Token-weighted cross-entropy and perplexity. Returns (loss, ppl, n_tokens)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # HF averages over all valid label positions (shifted); recover the token count accordingly
        valid = (labels[:, 1:] != -100).sum().item()
        if valid > 0:
            total_loss += out.loss.item() * valid
            total_tokens += valid
    if total_tokens == 0:
        return 0.0, float("nan"), 0
    avg = total_loss / total_tokens
    return round(avg, 4), round(math.exp(min(avg, 20.0)), 2), total_tokens


def load_passages(num_samples: int) -> Dict[str, List[str]]:
    domains: Dict[str, List[str]] = {}
    if not HELDOUT_FILE.exists():
        raise FileNotFoundError(
            f"{HELDOUT_FILE} missing. Run `py src/build_heldout_set.py` first; the wiki/news tails are NOT held out.")
    domains[HELDOUT_KEY] = pd.read_parquet(HELDOUT_FILE)["text"].dropna().head(num_samples).tolist()
    wiki = CLEAN_DATA_DIR / "wiki_clean.parquet"
    news = CLEAN_DATA_DIR / "news_recent.parquet"
    if wiki.exists():
        domains[INTRAIN_KEYS[0]] = pd.read_parquet(wiki, columns=["text"])["text"].dropna().tail(num_samples).tolist()
    if news.exists():
        domains[INTRAIN_KEYS[1]] = pd.read_parquet(news, columns=["text"])["text"].dropna().tail(num_samples).tolist()
    return domains


def run_alignment_tax_evaluation(num_samples: int = 1000, batch_size: int = 8):
    print("\n" + "=" * 72)
    print(" ALIGNMENT TAX EVALUATION (held-out web set + in-training reference sets)")
    print("=" * 72)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    passages = load_passages(num_samples)
    loaders = {}
    for dom, texts in passages.items():
        ds = ValidationTextDataset(texts, tokenizer, max_length=512)
        loaders[dom] = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b, p=pad_id: collate_fn_pad(b, pad_token_id=p))
        print(f"  {dom}: {len(ds)} passages")

    models_to_test = list(ALL_MODELS)
    results: List[Dict[str, Any]] = []
    base_heldout_ppl = None
    for label, path, desc in models_to_test:
        if not (path / "model.safetensors").exists():
            print(f"[skip] {label}: {path} not found")
            continue
        print(f"\n[Evaluating] {label} ({desc})")
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype).to(device)
        row: Dict[str, Any] = {"model": label, "description": desc, "domains": {}}
        for dom, dl in loaders.items():
            loss, ppl, ntok = compute_dataset_perplexity(model, dl, device)
            row["domains"][dom] = {"loss": loss, "ppl": ppl, "tokens": ntok}
            print(f"  -> {dom:28s} loss={loss:.4f}  ppl={ppl:.2f}  ({ntok:,} tokens)")
        row["heldout_ppl"] = row["domains"][HELDOUT_KEY]["ppl"]
        row["heldout_loss"] = row["domains"][HELDOUT_KEY]["loss"]
        if label == "Base-Ro-88M":
            base_heldout_ppl = row["heldout_ppl"]
        row["delta_ppl_heldout_vs_base"] = round(row["heldout_ppl"] - base_heldout_ppl, 2) if base_heldout_ppl else None
        results.append(row)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    report_path = EVALS_DIR / "thesis_alignment_tax_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"note": "Alignment tax is judged on 'Web (held-out)' only; the wiki/news tails are inside the training corpus.",
                   "results": results}, f, indent=2, ensure_ascii=False)

    tex_path = EVALS_DIR / "thesis_alignment_tax_table.tex"
    doms = [HELDOUT_KEY] + [k for k in INTRAIN_KEYS if k in loaders]
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Taxa de aliniere: perplexitate pe set held-out verificat (FineWeb-2 shard neantrenat, dedup. față de corpus și sidecar)\n")
        f.write("% Coloanele 'în antrenare' sunt păstrate doar pentru transparență (set folosit greșit ca held-out în v1).\n")
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{l l " + "c " * len(doms) + "c}\n\\toprule\n")
        f.write("\\textbf{Model} & \\textbf{Variantă} & " + " & ".join("\\textbf{PPL " + d.replace("(", "(").replace("î", "î") + "}" for d in doms) + " & \\textbf{$\\Delta$PPL held-out} \\\\\n\\midrule\n")
        for r in results:
            cells = [f"{r['domains'][d]['ppl']:.2f}" for d in doms]
            delta = "0.00 (ref)" if r["delta_ppl_heldout_vs_base"] in (None, 0.0) and r["model"] == "Base-Ro-88M" else f"{r['delta_ppl_heldout_vs_base']:+.2f}"
            f.write(f"{r['model']} & {r['description']} & " + " & ".join(cells) + f" & {delta} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Perplexitate (context 512 tokeni) pe un set held-out de pagini web verificate ca absente din corpusul de antrenare și din sidecar-ul de reflecții "
                "(build\\_heldout\\_set.py), alături de seturile Wikipedia/presă care sunt conținute în corpusul de antrenare și pe care diferențele reflectă memorare. "
                "Taxa de aliniere ($\\Delta$PPL) se raportează exclusiv pe setul held-out.}\n")
        f.write("\\label{tab:thesis_alignment_tax}\n\\end{table}\n")

    print("\n[SUCCESS] JSON:", report_path, "\n          LaTeX:", tex_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    run_alignment_tax_evaluation(num_samples=args.num_samples, batch_size=args.batch_size)

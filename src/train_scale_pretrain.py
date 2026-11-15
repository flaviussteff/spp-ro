"""
Scale Pre-training Engine for Romanian Generative LLM (~88.1M Parameters)
Tuned for 2-3 Days per Model on NVIDIA GeForce RTX 3060 (12GB VRAM).

Features:
1. Continued Pre-training: Resumes directly from existing model weights in models/
2. Deep Scale: Targets 60,000 steps total (~3.93 Billion tokens per model)
3. Hourly Live Telemetry: Full diagnostic card + live generated text every 60 minutes
4. Balanced Data Streaming: Prioritizes FineWeb-2 web data and Wikipedia over news
5. Checkpointing: Saves every 5,000 steps with automatic checkpoint rotation
"""
import os
import json
import sys
import time
import math
import datetime
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure UTF-8 output on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config import (
        TOKENIZER_DIR,
        CLEAN_DATA_DIR,
        SIDECAR_DIR,
        BASE_MODEL_DIR,
        SPP_MODEL_DIR,
        SPP_20PCT_MODEL_DIR,
        FILTERED_MODEL_DIR,
        SIDECAR_V2_FILE,
        ModelConfig,
        HardwareAndTrainingConfig,
    )
    from spp_collator import DataCollatorForSPP
    from packing import PackedWindowDataset, PackedCollator, keep_loss_finite
except ImportError:
    from src.config import (
        TOKENIZER_DIR,
        CLEAN_DATA_DIR,
        SIDECAR_DIR,
        BASE_MODEL_DIR,
        SPP_MODEL_DIR,
        SPP_20PCT_MODEL_DIR,
        FILTERED_MODEL_DIR,
        SIDECAR_V2_FILE,
        ModelConfig,
        HardwareAndTrainingConfig,
    )
    from src.spp_collator import DataCollatorForSPP
    from src.packing import PackedWindowDataset, PackedCollator, keep_loss_finite


class _Tee:
    """Mirror a stream to a log file. tqdm progress updates (chunks ending in \r) are shown on the
    console but not written to the file, so the log keeps only real lines (reports, checkpoints)."""
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, data):
        try:
            self._stream.write(data)
        except Exception:
            pass
        if "\n" in data or "\r" not in data:
            self._fh.write(data.replace("\r", ""))
            self._fh.flush()
        return len(data)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        self._fh.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install_tee(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    fh.write(f"\n===== session start {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


class HourlyLiveProgressCallback(TrainerCallback):
    """
    Prints a prominent progress report every N minutes (default 60 min = 1 hour)
    with throughput, ETA, token metrics, and live generated text.
    """
    def __init__(
        self,
        eff_tokens_per_step: int,
        total_target_steps: int,
        tokenizer,
        interval_mins: float = 60.0,
        initial_steps_done: int = 10000,
    ):
        self.eff_tokens_per_step = eff_tokens_per_step
        self.total_target_steps = total_target_steps
        self.tokenizer = tokenizer
        self.interval_seconds = interval_mins * 60.0
        self.initial_steps_done = initial_steps_done
        self.start_time = time.time()
        self.last_report_time = time.time()
        self.session_start_step = None
        self.report_counter = 0
        self.latest_loss = None
        self.latest_lr = None

    def on_step_begin(self, args, state, control, **kwargs):
        if self.session_start_step is None:
            self.session_start_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.latest_loss = logs["loss"]
            if "learning_rate" in logs:
                self.latest_lr = logs["learning_rate"]

    def _print_report(self, state, model):
        self.report_counter += 1
        current_time = time.time()
        current_step = state.global_step + self.initial_steps_done
        start_step = self.session_start_step if self.session_start_step is not None else state.global_step
        steps_this_session = max(1, state.global_step - start_step)
        
        session_elapsed = current_time - self.start_time
        sec_per_step = session_elapsed / steps_this_session
        remaining_steps = max(0, self.total_target_steps - current_step)
        eta_seconds = remaining_steps * sec_per_step
        
        total_tokens_learned = current_step * self.eff_tokens_per_step
        total_billions = total_tokens_learned / 1_000_000_000
        
        tokens_per_sec = (steps_this_session * self.eff_tokens_per_step) / max(1.0, session_elapsed)
        pct_done = (current_step / max(1, self.total_target_steps)) * 100.0
        
        # VRAM
        vram_str = "N/A"
        if torch.cuda.is_available():
            vram_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            vram_str = f"{vram_used:.2f} / 12.00 GB"
            
        # PPL
        ppl_str = "N/A"
        if self.latest_loss is not None:
            try:
                ppl = math.exp(min(20.0, self.latest_loss))
                ppl_str = f"{ppl:.2f}"
            except OverflowError:
                ppl_str = "Inf"

        # ETA formatting
        eta_days = int(eta_seconds // 86400)
        eta_hours = int((eta_seconds % 86400) // 3600)
        eta_mins = int((eta_seconds % 3600) // 60)
        if eta_days > 0:
            eta_str = f"{eta_days}d {eta_hours}h {eta_mins}m"
        else:
            eta_str = f"{eta_hours}h {eta_mins}m"

        print("\n" + "-" * 70, flush=True)
        finish_at = (datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)).strftime("%a %d.%m %H:%M")
        print(f"[Raport #{self.report_counter} | {datetime.datetime.now():%H:%M:%S}] Pas: {current_step:,}/{self.total_target_steps:,} ({pct_done:.1f}%) | ETA: {eta_str} (gata ~{finish_at})", flush=True)
        print(f"Tokeni: {total_billions:.2f}B ({total_tokens_learned:,}) | Viteza: {tokens_per_sec:,.0f} tok/s ({sec_per_step:.2f}s/step)", flush=True)
        print(f"VRAM: {vram_str} | Loss: {self.latest_loss:.4f} | PPL: {ppl_str}" if self.latest_loss else f"VRAM: {vram_str}", flush=True)
        if self.latest_lr:
            print(f"LR: {self.latest_lr:.2e}", flush=True)
        
        # Live generation probe
        test_prompt = "In Romania, dezvoltarea educatiei si a cercetarii stiintifice reprezinta"
        print(f"Test prompt: \"{test_prompt}...\"", flush=True)
        try:
            device = next(model.parameters()).device
            inputs = self.tokenizer(test_prompt, return_tensors="pt").to(device)
            was_training = model.training
            model.eval()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=45,
                    do_sample=True,
                    temperature=0.4,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            gen_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            if was_training:
                model.train()
            print(f"Generat: \"{gen_text}\"", flush=True)
        except Exception as e:
            print(f"[Generare omisa: {e}]", flush=True)
        print("-" * 70 + "\n", flush=True)

    def on_save(self, args, state, control, **kwargs):
        print(f"\n[Checkpoint | {datetime.datetime.now():%H:%M:%S}] salvat la pasul {state.global_step:,} in {args.output_dir}", flush=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        current_time = time.time()
        if (current_time - self.last_report_time) >= self.interval_seconds:
            self.last_report_time = current_time
            if model is not None:
                self._print_report(state, model)


class KeepCheckpointCallback(TrainerCallback):
    """Copies checkpoint-<keep_step> to <output_dir>/checkpoints_keep/ when it is saved, so the Trainer's
    save_total_limit rotation cannot delete it (the MT / re-warm runs start from Base@45k)."""
    def __init__(self, checkpoints_dir: Path, keep_dir: Path, keep_step: int):
        self.checkpoints_dir, self.keep_dir, self.keep_step = checkpoints_dir, keep_dir, keep_step

    def on_save(self, args, state, control, **kwargs):
        if self.keep_step and state.global_step == self.keep_step:
            import shutil
            src = self.checkpoints_dir / f"checkpoint-{self.keep_step}"
            dst = self.keep_dir / f"checkpoint-{self.keep_step}"
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"[Checkpoint] copie pastrata pentru MT/re-warm: {dst}", flush=True)


STREAM_STATE_FILE = "stream_state.json"


class StreamStateCallback(TrainerCallback):
    """Writes the data-stream position into every checkpoint (stream_state.json), so that --resume continues the
    fixed-seed stream where the crashed run stopped instead of replaying it from document 0 (the Trainer runs with
    ignore_data_skip=True and the dataset is an IterableDataset, so HF itself keeps no data position).
    Position = every item the stream has yielded so far (trained on, or still waiting in the packer's look-ahead
    buffer). The buffered items are not the most recent ones of the stream (first-fit places items out of order), so
    they are saved verbatim and re-encoded into the buffer on resume, after which the stream continues from the saved
    position: the windows after the resume are identical to the uninterrupted run. The dataloader runs in-process
    (num_workers=0), so these counters are the live ones."""
    def __init__(self, stream_ds, packed_ds=None):
        self.stream_ds, self.packed_ds = stream_ds, packed_ds

    def on_save(self, args, state, control, **kwargs):
        pending = list(self.packed_ds.pending_items) if self.packed_ds is not None else []
        pos = self.stream_ds.n_yielded
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt.is_dir():
            return
        payload = {"global_step": state.global_step, "stream_items_consumed": pos, "pending_items": pending}
        (ckpt / STREAM_STATE_FILE).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"[Checkpoint] pozitie flux salvata: {pos:,} documente consumate, {len(pending)} in buffer", flush=True)


def restore_stream_state(checkpoint: Path, stream_ds, packed_ds=None) -> None:
    """Fast-forwards the stream (and refills the packer buffer) from a checkpoint's stream_state.json."""
    f = Path(checkpoint) / STREAM_STATE_FILE
    if not f.exists():
        print(f"[Dataset] AVERTISMENT: {f.name} lipseste din {Path(checkpoint).name} (checkpoint salvat de o versiune veche): "
              f"fluxul reporneste de la documentul 0, deci documentele de la inceput vor fi vazute a doua oara.", flush=True)
        return
    st = json.loads(f.read_text(encoding="utf-8"))
    stream_ds.skip_items = int(st["stream_items_consumed"])
    if packed_ds is not None:
        packed_ds.preload_items = list(st.get("pending_items", []))
    print(f"[Dataset] Pozitie flux restaurata din {Path(checkpoint).name}: {stream_ds.skip_items:,} documente consumate, "
          f"{len(st.get('pending_items', []))} reincarcate in buffer-ul de impachetare.", flush=True)


class VanillaCausalDataCollator:
    """Plain next-token collator. Items may carry mask_loss=True (Filtered baseline): the whole
    sequence stays in the batch as context but contributes no loss (labels = -100)."""
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_texts, mask_flags = [], []
        for f in features:
            if isinstance(f, dict):
                if "text" in f:
                    batch_texts.append(f["text"])
                elif "pre_text" in f:
                    batch_texts.append(f"{f.get('pre_text', '')} {f.get('post_text', '')}")
                else:
                    str_vals = [v for v in f.values() if isinstance(v, str)]
                    batch_texts.append(str_vals[0] if str_vals else "")
                mask_flags.append(bool(f.get("mask_loss", False)))
            else:
                batch_texts.append(str(f))
                mask_flags.append(False)

        batch_enc = self.tokenizer(batch_texts, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        input_ids = batch_enc["input_ids"]
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        labels[labels == pad_id] = -100
        for i, m in enumerate(mask_flags):
            if m:
                labels[i, :] = -100
        keep_loss_finite(labels, input_ids, pad_id)   # all-masked batch (Filtered): target on padding, not on a document
        return {"input_ids": input_ids, "attention_mask": batch_enc["attention_mask"], "labels": labels}


class StreamingParquetDataset(IterableDataset):
    """
    Streams the corpus row-group by row-group (< 100 MB RAM).

    mode='base'     : plain documents.
    mode='spp'      : legacy (v1): with probability spp_ratio an EXTRA synthetic sidecar item replaces a document;
                      inline=True (v2): documents whose doc_key appears in the sidecar are REPLACED by their
                      annotated version (data-matched with Base: same documents, same count).
    mode='filtered' : paper's Filtered baseline: documents with severity >= 3 in the v2 sidecar are kept in the
                      batch but loss-masked.
    """
    def __init__(self, parquet_path: Path, mode: str = "base", spp_sidecar_path: Optional[Path] = None,
                 spp_ratio: float = 0.10, snap_insertion: bool = False, inline: bool = False,
                 mt_annotated_ratio: float = 0.0):
        self.parquet_path = parquet_path
        self.mode = mode
        self.spp_ratio = spp_ratio
        self.snap_insertion = snap_insertion
        self.inline = inline
        # Paper's midtraining stream (SPP{MT}): the annotated documents are up-weighted to this fraction of the
        # sequences (EPFL run 5: ANNOTATED_WEIGHT 0.6758); the rest are plain, non-annotated corpus documents.
        # 0 = natural rate (inline replacement only), used for SPP{T0}.
        self.mt_annotated_ratio = mt_annotated_ratio
        # Resume support: the stream is deterministic (seed 42), so its position is fully described by the number of
        # items yielded so far. `skip_items` (set from a checkpoint's stream_state.json) fast-forwards the stream
        # without tokenising; `n_yielded` counts the absolute position (skipped + yielded) for the next checkpoint.
        self.skip_items = 0
        self.n_yielded = 0
        self.spp_items: List[Dict[str, str]] = []
        self.inline_map: Dict[str, Dict[str, str]] = {}
        self.harmful_keys = set()
        if mode in ("spp", "filtered") and not (spp_sidecar_path and spp_sidecar_path.exists()):
            raise FileNotFoundError(f"mode={mode} are nevoie de sidecar-ul cu reflectii, dar lipseste: {spp_sidecar_path}")
        if mode in ("spp", "filtered"):
            import random
            cols = ["doc_key", "text", "reflection_char_position", "reflection_text", "clf_flag", "severity"] if (inline or mode == "filtered") else None
            try:
                df_refl = pd.read_parquet(spp_sidecar_path, columns=cols)
            except Exception:
                df_refl = pd.read_parquet(spp_sidecar_path)   # older sidecar without some of the columns
            print(f"[SPP] Loaded {len(df_refl):,} reflection records from {spp_sidecar_path.name}.")
            if inline or mode == "filtered":
                if "doc_key" not in df_refl.columns:
                    raise ValueError("inline/filtered modes need a v2 sidecar with a doc_key column (generate_reflections_v2.py).")
                for row in df_refl.itertuples(index=False):
                    text = row.text
                    pos = self._snap(text, int(row.reflection_char_position)) if snap_insertion else int(row.reflection_char_position)
                    k = self.full_key(text)   # full-text key: no collisions on shared 200-char headers (sidecar doc_key is md5 of text[:200])
                    self.inline_map[k] = {"pre_text": text[:pos], "reflection": row.reflection_text, "post_text": text[pos:]}
                    # Filtered baseline: the CLASSIFIER defines the harmful set (as in the paper) when the
                    # v2 sidecar carries clf_flag; otherwise fall back to the annotator's severity >= 3.
                    flag = getattr(row, "clf_flag", -1)
                    harmful = (int(flag) == 1) if (flag is not None and int(flag) >= 0) else (int(getattr(row, "severity", 0)) >= 3)
                    if harmful:
                        self.harmful_keys.add(k)
                print(f"[SPP] inline map: {len(self.inline_map):,} annotated documents, {len(self.harmful_keys):,} in the harmful set (classifier flag, or severity >= 3 for old sidecars).")
                del df_refl
            else:
                for _, row in df_refl.iterrows():
                    text = row["text"]
                    pos = self._snap(text, int(row["reflection_char_position"])) if snap_insertion else int(row["reflection_char_position"])
                    self.spp_items.append({"pre_text": text[:pos], "reflection": row["reflection_text"], "post_text": text[pos:]})
                random.seed(42)
                random.shuffle(self.spp_items)
                print(f"[SPP] legacy sidecar mode: {len(self.spp_items):,} items shuffled (seed 42), ratio {spp_ratio:.2f}.")

    @staticmethod
    def _snap(text: str, pos: int) -> int:
        while 0 < pos < len(text) and not text[pos].isspace() and not text[pos - 1].isspace():
            pos += 1
        return pos

    @staticmethod
    def doc_key(text: str) -> str:
        return hashlib.md5(text.strip()[:200].encode("utf-8")).hexdigest()

    @staticmethod
    def full_key(text: str) -> str:
        return hashlib.md5(text.strip().encode("utf-8")).hexdigest()

    def _check_matching(self, n_docs: int, n_hits: int):
        """After the first 20k stream documents, the annotated / harmful set must have matched some of them
        (expected ~7-11% / ~5%); zero matches means the sidecar texts are not the corpus texts -> stop, do not
        silently train a Base model under the SPP name."""
        if n_docs == 20000:
            expect = len(self.inline_map) if self.mode == "spp" else len(self.harmful_keys)
            print(f"[SPP] verificare potrivire sidecar/corpus: {n_hits:,} din primele {n_docs:,} documente ({100*n_hits/n_docs:.2f}%) | set {expect:,}", flush=True)
            if expect and n_hits == 0:
                raise RuntimeError("Niciun document din sidecar nu a fost gasit in fluxul corpusului dupa 20.000 de documente: textele din sidecar nu coincid cu textele din corpus.")

    def __iter__(self):
        """Yields the items of `_items()` after fast-forwarding past the first `skip_items` of them (resume: the
        same rng draws and the same order are replayed, only the tokenisation is skipped)."""
        it = self._items()
        if self.skip_items > 0:
            print(f"[Dataset] Reluare: derulez fluxul peste {self.skip_items:,} documente deja consumate (fara tokenizare)...", flush=True)
            t0 = time.time()
            for i, _ in enumerate(it, 1):
                if i >= self.skip_items:
                    break
            self.n_yielded = self.skip_items
            print(f"[Dataset] Flux pozitionat dupa {time.time() - t0:.0f} s; continui de la documentul {self.skip_items:,}.", flush=True)
        for item in it:
            self.n_yielded += 1
            yield item

    def _items(self):
        import pyarrow.parquet as pq
        import random
        pf = pq.ParquetFile(str(self.parquet_path))
        num_rgs = pf.num_row_groups
        spp_idx = 0
        # Fixed-seed stream order (paper: streams are pre-shuffled once, so Vanilla/Filtered/SPP{T0} read the same
        # document order and differ only in window content). Seeded here so Filtered and SPP{T0} v2 share the order.
        rng = random.Random(42)
        n_docs, n_hits = 0, 0   # sanity: the inline/harmful map must actually match corpus documents
        ann_items: List[Dict[str, str]] = []
        if self.mode == "spp" and self.inline and self.mt_annotated_ratio > 0:
            ann_items = list(self.inline_map.values())
            random.Random(42).shuffle(ann_items)
        while True:
            rg_order = list(range(num_rgs))
            rng.shuffle(rg_order)
            for rg in rg_order:
                texts = pf.read_row_group(rg, columns=["text"])["text"].to_pylist()
                rng.shuffle(texts)
                for t in texts:
                    if not t or len(t) < 150:
                        continue
                    if self.mode == "spp" and self.inline and ann_items:
                        # MT stream: annotated documents at mt_annotated_ratio (cycled, shuffled), plain documents otherwise
                        if rng.random() < self.mt_annotated_ratio:
                            yield ann_items[spp_idx % len(ann_items)]
                            spp_idx += 1
                        elif self.full_key(t) not in self.inline_map:
                            yield {"text": t}
                        continue
                    if self.mode == "spp" and self.inline:
                        item = self.inline_map.get(self.full_key(t))
                        n_docs += 1
                        if item is not None:
                            n_hits += 1
                        self._check_matching(n_docs, n_hits)
                        yield item if item is not None else {"text": t}
                    elif self.mode == "filtered":
                        masked = self.full_key(t) in self.harmful_keys
                        n_docs += 1
                        if masked:
                            n_hits += 1
                        self._check_matching(n_docs, n_hits)
                        yield {"text": t, "mask_loss": masked}
                    elif self.mode == "spp" and self.spp_items and rng.random() < self.spp_ratio:
                        yield self.spp_items[spp_idx % len(self.spp_items)]
                        spp_idx += 1
                    else:
                        yield {"text": t}


def generate_spp_model_card(output_dir: Path, total_steps: int, loss: Optional[float] = None, ppl: Optional[float] = None,
                            n_params: Optional[int] = None, mode: str = "spp", spp_ratio: float = 0.10,
                            variant_note: str = "", packed: bool = False):
    """Hugging Face model card with values computed from the actual run (no hard-coded numbers)."""
    loss_str = f"{loss:.4f}" if loss is not None else "n/a"
    ppl_str = f"{ppl:.2f}" if ppl is not None else "n/a"
    params_str = f"{n_params:,} (~{n_params/1e6:.1f}M)" if n_params else "88,099,584 (~88.1M)"
    positions = total_steps * 64 * 1024
    name = {"base": "Base-Ro-88M", "spp": "SPP-Ro-88M", "filtered": "Filtered-Ro-88M"}.get(mode, "SPP-Ro-88M")
    if mode == "spp" and abs(spp_ratio - 0.20) < 1e-4:
        name += "-20pct"
    card_content = f"""---
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

# {name}: model de limbaj românesc preantrenat de la zero{" cu Synthetic Persona Pretraining" if mode == "spp" else " (control)"}

Model decoder de tip LLaMA cu **{params_str} parametri**, preantrenat de la zero pe text românesc
(FineWeb-2 RO, Wikipedia RO, presă) pentru teza de licență *Alinierea modelelor de limbaj în limba română
din faza de preantrenare prin Synthetic Persona Pretraining* (FMI, Universitatea din București), după
metodologia Minder et al., 2026 (arXiv:2608.13482). {variant_note}

## Antrenare
* Pași de optimizare: {total_steps:,}; batch efectiv 64 secvențe x 1.024 poziții = {positions:,} poziții ({"documente împachetate, terminate cu </s>, atenție bloc-diagonală pe document; padding sub 2%, deci aproape toate pozițiile sunt tokeni reali" if packed else "inclusiv padding; secvențele nu sunt împachetate, deci numărul de tokeni efectivi este aproximativ jumătate"}).
* {"Interleaving SPP: " + f"{spp_ratio*100:.0f}% din secvențe sunt documente cu reflecție constituțională la persoana I, precedată de tokenul <assistant>, cu blocarea atenției post-reflecție și aliasing RoPE." if mode == "spp" else "Fără reflecții (control)."}
* Hardware: o singură placă NVIDIA RTX 3060 12GB, bf16.
* Ultimul loss de antrenare raportat: {loss_str} (perplexitate {ppl_str}).

## Arhitectură
| Parametru | Valoare |
| :--- | :--- |
| Parametri | {params_str} |
| hidden / intermediate | 768 / 2048 (SwiGLU) |
| Straturi | 12, GQA 12 Q / 4 KV |
| Context | 1.024 tokeni, RoPE theta 10.000 |
| Vocabular | 16.384 BPE (ro_bpe_16k), embedding legat |

## Utilizare
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{output_dir.name}")
model = AutoModelForCausalLM.from_pretrained("{output_dir.name}")
```

## Limitări
Model mic (88M), monolingv, fără post-training; evaluările din repo au N mic și sunt raportate cu intervale de încredere.
"""
    (output_dir / "README.md").write_text(card_content, encoding="utf-8")
    print(f"[Model Card] Actualizat {output_dir / 'README.md'}")


def run_scale_pretraining(
    mode: str = "spp",
    additional_steps: int = 10000,
    total_target_steps: Optional[int] = None,
    update_interval_mins: float = 30.0,
    save_steps: int = 2000,
    from_scratch: bool = False,
    push_to_hub: bool = False,
    repo_id: str = "flaviussteff/spp-ro-125m",
    spp_ratio: float = 0.10,
    output_dir: Optional[Path] = None,
    resume: bool = False,
    mask_assistant_loss: bool = False,
    snap_insertion: bool = False,
    sidecar: Optional[Path] = None,
    inline: bool = False,
    prefix_tail: bool = False,
    init_from: Optional[Path] = None,
    lr: float = 1.5e-4,
    warmup_steps: int = 500,
    mt_annotated_ratio: float = 0.0,
    reflection_loss_only: bool = False,
    pack: bool = False,
    pack_eos: bool = True,
    keep_step: int = 0,
    head_cap: bool = False,
):
    if total_target_steps is None:
        total_target_steps = additional_steps
    if keep_step and keep_step % save_steps != 0:
        raise ValueError(f"--keep-step {keep_step} trebuie sa fie multiplu de --save-steps {save_steps}, altfel checkpoint-ul nu este salvat niciodata")

    if output_dir is None:
        if mode == "base":
            output_dir = BASE_MODEL_DIR
        elif mode == "filtered":
            output_dir = FILTERED_MODEL_DIR
        elif abs(spp_ratio - 0.20) < 1e-4:
            output_dir = SPP_20PCT_MODEL_DIR
        else:
            output_dir = SPP_MODEL_DIR

    print("\n" + "=" * 75, flush=True)
    print(f"  PRE-ANTRENARE SCALATĂ: {mode.upper()}-RO-88M", flush=True)
    print(f"  Pasi: {additional_steps:,} (Total: {total_target_steps:,}) | From Scratch: {from_scratch} | Resume: {resume}", flush=True)
    if sidecar is None and mode in ("spp", "filtered"):
        sidecar = SIDECAR_V2_FILE if (inline or mode == "filtered") else (SIDECAR_DIR / "reflections.parquet")
    if mode == "spp" and inline:
        print(f"  Reflectii: INLINE din {sidecar.name} (documentele adnotate inlocuiesc originalele; data-matched cu Base)", flush=True)
    elif mode == "filtered":
        print(f"  Filtered baseline: documentele cu severitate >= 3 din {sidecar.name} sunt loss-masked", flush=True)
    else:
        print(f"  Rata Interleaving: {spp_ratio * 100:.0f}% SPP (Reflectii) / {(1 - spp_ratio) * 100:.0f}% Text Liber", flush=True)
    print(f"  Flags: mask_assistant_loss={mask_assistant_loss} snap_insertion={snap_insertion} prefix_tail={prefix_tail}", flush=True)
    if pack:
        print(f"  Packing (v3): documente terminate cu </s>={pack_eos}, impachetate first-fit in ferestre de 1.024, atentie bloc-diagonala pe document; ~2x tokeni reali/pas fata de v1/v2", flush=True)
    if head_cap:
        print(f"  Head-cap (EPFL, implicit sub --pack): documentele adnotate sunt taiate la inceput ca cele simple; reflectia se insereaza doar daca pozitia ei cade in fereastra, altfel documentul ramane simplu (--prefix-tail ignorat)", flush=True)
    if mode == "spp" and (mt_annotated_ratio > 0 or reflection_loss_only):
        print(f"  Midtraining (paper SPP{{MT}}): documente adnotate = {mt_annotated_ratio*100:.1f}% din secvente, loss doar pe reflectie = {reflection_loss_only}", flush=True)
    print(f"  LR: peak {lr:g}, warmup {warmup_steps} pasi, cosinus -> 0 pe {additional_steps:,} pasi" + (f" | init din {init_from} (program LR NOU, fara stare de optimizator)" if init_from else ""), flush=True)
    print(f"  Director Iesire: {output_dir.name} | Hub: {repo_id if push_to_hub else 'Local'}", flush=True)
    print(f"  Durata estimata pe RTX 3060: ~{additional_steps / 1000:.1f} ore", flush=True)
    print("=" * 75 + "\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    checkpoints_dir = output_dir / "checkpoints"

    # Set seeds for reproducible initialization
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Detect existing checkpoints for seamless resumption
    last_checkpoint = None
    if checkpoints_dir.exists():
        last_checkpoint = get_last_checkpoint(str(checkpoints_dir))

    if resume:
        if last_checkpoint is None:
            raise FileNotFoundError(f"Nu s-a gasit niciun checkpoint in {checkpoints_dir} pentru reluare (--resume).")
        print(f"[Engine] Mod Resume ACTIVAT! Checkpoint detectat: {last_checkpoint}")
        print(f"[Engine] Se reia antrenarea AUTOMAT din {Path(last_checkpoint).name}...\n")
        from_scratch = False

    # 1. Model initialization
    if init_from is not None and not (resume and last_checkpoint is not None):
        # SPP{MT} / re-warm: weights only, no optimizer or scheduler state -> a fresh LR schedule
        # (continuing the cosine tail of a 50k run gives LR ~1e-6 and leaves the weights unchanged).
        init_from = Path(init_from)
        if not (init_from / 'model.safetensors').exists():
            raise FileNotFoundError(f"--init-from: nu exista model.safetensors in {init_from}")
        print(f"[Model] Initializare din ponderile {init_from} (program LR nou: peak {lr:g}, warmup {warmup_steps})...")
        model = LlamaForCausalLM.from_pretrained(
            str(init_from),
            dtype=torch.float32,   # fp32 master weights + fp32 Adam; bf16 compute comes from the Trainer's autocast (loading in bf16 made every resumed / --init-from run train in pure bf16: updates below the bf16 ulp were lost; v1 Base/MT/re-warm were affected, see Analiza 5 nov)
        )
        if last_checkpoint is not None:
            print(f"[Checkpoints] Avertisment: {checkpoints_dir} contine checkpoint-uri vechi; sunt ignorate (foloseste --resume pentru a le relua).")
        last_checkpoint = None
    elif last_checkpoint is not None and (resume or not from_scratch):
        print(f"[Model] Resumare si incarcare ponderi din checkpoint: {Path(last_checkpoint).name}...")
        model = LlamaForCausalLM.from_pretrained(
            str(last_checkpoint),
            dtype=torch.float32,   # fp32 master weights + fp32 Adam; bf16 compute comes from the Trainer's autocast (loading in bf16 made every resumed / --init-from run train in pure bf16: updates below the bf16 ulp were lost; v1 Base/MT/re-warm were affected, see Analiza 5 nov)
        )
    elif from_scratch:
        print(f"[Model] Initializare complet de la zero (Token Zero: pasul 0) pentru {mode.upper()}-Ro-88M...")
        # Clear out previous checkpoints if doing a fresh from-scratch run
        if checkpoints_dir.exists():
            import shutil
            try:
                shutil.rmtree(checkpoints_dir)
                print(f"[Checkpoints] Curățat directorul anterior de checkpoint-uri: {checkpoints_dir.name}")
            except Exception as e:
                print(f"[Checkpoints Avertisment]: {e}")

        m_cfg = ModelConfig()
        llama_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=m_cfg.hidden_size,
            intermediate_size=m_cfg.intermediate_size,
            num_hidden_layers=m_cfg.num_hidden_layers,
            num_attention_heads=m_cfg.num_attention_heads,
            num_key_value_heads=m_cfg.num_key_value_heads,
            max_position_embeddings=m_cfg.max_position_embeddings,
            rms_norm_eps=m_cfg.rms_norm_eps,
            rope_theta=m_cfg.rope_theta,
            tie_word_embeddings=m_cfg.tie_word_embeddings,
            hidden_act=m_cfg.hidden_act,
            use_cache=False,
        )
        model = LlamaForCausalLM(llama_config)
    elif (output_dir / "model.safetensors").exists():
        print(f"[Model] Resumare din ponderile existente in: {output_dir.name}...")
        model = LlamaForCausalLM.from_pretrained(
            str(output_dir),
            dtype=torch.float32,   # fp32 master weights + fp32 Adam; bf16 compute comes from the Trainer's autocast (loading in bf16 made every resumed / --init-from run train in pure bf16: updates below the bf16 ulp were lost; v1 Base/MT/re-warm were affected, see Analiza 5 nov)
        )
    else:
        print(f"[Model] Nicio pondere anterioara gasita in {output_dir.name}. Initializare from scratch...")
        m_cfg = ModelConfig()
        llama_config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=m_cfg.hidden_size,
            intermediate_size=m_cfg.intermediate_size,
            num_hidden_layers=m_cfg.num_hidden_layers,
            num_attention_heads=m_cfg.num_attention_heads,
            num_key_value_heads=m_cfg.num_key_value_heads,
            max_position_embeddings=m_cfg.max_position_embeddings,
            rms_norm_eps=m_cfg.rms_norm_eps,
            rope_theta=m_cfg.rope_theta,
            tie_word_embeddings=m_cfg.tie_word_embeddings,
            hidden_act=m_cfg.hidden_act,
            use_cache=False,
        )
        model = LlamaForCausalLM(llama_config)

    # 2. Hardware specs: batch size 16 * grad accum 4 = 64 seqs = 65,536 tokens/step (~10 hours for 10k steps)
    batch_size = 16
    grad_accum = 4
    eff_tokens_per_step = batch_size * grad_accum * 1024  # 65,536 tokens

    # Priority: corpus_scale_stream.parquet, else corpus_unannotated.parquet
    stream_path = CLEAN_DATA_DIR / "corpus_scale_stream.parquet"
    if not stream_path.exists():
        stream_path = CLEAN_DATA_DIR / "corpus_unannotated.parquet"
        
    if not stream_path.exists():
        raise FileNotFoundError(f"Lipsesc datele de preantrenare la {stream_path}. Ruleaza prepare_scale_corpus.py mai intai.")
        
    print(f"[Dataset] Initializare streaming cu consum redus de memorie din {stream_path.name}...")
    dataset = StreamingParquetDataset(
        parquet_path=stream_path,
        mode=mode,
        spp_sidecar_path=sidecar if mode in ("spp", "filtered") else None,
        spp_ratio=spp_ratio,
        snap_insertion=snap_insertion,
        inline=inline,
        mt_annotated_ratio=mt_annotated_ratio,
    )
    stream_ds, packed_ds = dataset, None   # kept for the stream-position checkpointing (StreamStateCallback)

    if pack:
        # v3: every item (plain / annotated / loss-masked) is encoded exactly as in the one-document-per-sequence runs,
        # then several items fill one 1,024-position window (see packing.py). Base/Filtered use the same encoder.
        encoder = DataCollatorForSPP(tokenizer=tokenizer, max_length=1024 - (1 if pack_eos else 0), mask_assistant_in_loss=mask_assistant_loss,
                                     prefix_tail=prefix_tail, reflection_loss_only=reflection_loss_only, add_eos=pack_eos, head_cap=head_cap)
        dataset = PackedWindowDataset(dataset, encoder, max_length=1024)
        packed_ds = dataset
        collator = PackedCollator(pad_token_id=encoder.pad_token_id, max_length=1024)
    elif mode == "spp":
        collator = DataCollatorForSPP(tokenizer=tokenizer, max_length=1024, mask_assistant_in_loss=mask_assistant_loss, prefix_tail=prefix_tail,
                                      reflection_loss_only=reflection_loss_only, head_cap=head_cap)
    else:
        collator = VanillaCausalDataCollator(tokenizer=tokenizer, max_length=1024)

    # 3. Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        max_steps=additional_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        logging_steps=50,
        save_steps=save_steps,
        save_total_limit=2,
        fp16=(not torch.cuda.is_bf16_supported()),
        bf16=torch.cuda.is_bf16_supported(),
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        ignore_data_skip=True,
    )

    initial_done = 0 if from_scratch else max(0, total_target_steps - additional_steps)
    hourly_cb = HourlyLiveProgressCallback(
        eff_tokens_per_step=eff_tokens_per_step,
        total_target_steps=total_target_steps,
        tokenizer=tokenizer,
        interval_mins=update_interval_mins,
        initial_steps_done=initial_done,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[hourly_cb, StreamStateCallback(stream_ds, packed_ds)]
                  + ([KeepCheckpointCallback(checkpoints_dir, output_dir / "checkpoints_keep", keep_step)] if keep_step else []),
    )

    # Setup checkpoint to resume from if available
    checkpoint_to_resume = last_checkpoint if (resume or not from_scratch) else None
    if checkpoint_to_resume is not None:
        print(f"\n[Engine] Reluare garantata din: {Path(checkpoint_to_resume).name} (pasul {checkpoint_to_resume})", flush=True)
        restore_stream_state(Path(checkpoint_to_resume), stream_ds, packed_ds)   # continue the stream, do not replay it

    print(f"\n[Engine] Pornire sesiune de antrenare: {additional_steps:,} pasi...")
    print(f"[Engine] Rapoartele live de diagnostic vor fi afisate la fiecare {update_interval_mins:.0f} minute in consola.\n")
    
    trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    # 4. Save Final Upgraded Model
    print(f"\n[Salvare] Exportare model finalizat in: {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Generate Model Card README
    latest_loss = hourly_cb.latest_loss
    latest_ppl = math.exp(min(20.0, latest_loss)) if latest_loss else None
    generate_spp_model_card(output_dir, total_steps=total_target_steps, loss=latest_loss, ppl=latest_ppl,
                            n_params=sum(p_.numel() for p_ in model.parameters()), mode=mode, spp_ratio=spp_ratio, packed=pack,
                            variant_note=(f'Inițializat din ponderile {init_from} (pasul {total_target_steps - additional_steps:,}) și antrenat {additional_steps:,} pași cu program LR nou (peak {lr:g}, warmup {warmup_steps}, cosinus → 0), fără stare de optimizator'
                                          + (' , cu reflecții (SPP{MT}).' if mode == 'spp' else ' , fără reflecții (control re-warm).') if init_from else ''))

    print(f"[Succes] Antrenare {mode.upper()}-Ro-88M finalizata! Procesat ~{total_target_steps * eff_tokens_per_step / 1e9:.2f} miliarde de tokeni.")

    # 5. Push to Hugging Face Hub if requested
    if push_to_hub:
        print(f"\n[Hugging Face] Se publica automat modelul pe Hugging Face: {repo_id}...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_folder(
                folder_path=str(output_dir),
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Release {mode.upper()}-Ro-88M: Aliniere Token Zero cu 60.000 reflectii (Loss {latest_loss:.4f}, PPL {latest_ppl:.2f})" if latest_loss else f"Release {mode.upper()}-Ro-88M: Aliniere Token Zero cu 60.000 reflectii"
            )
            print(f"\n============================================================================")
            print(f"  [HUGGING FACE SUCCESS] Model publicat cu succes!")
            print(f"  URL: https://huggingface.co/{repo_id}")
            print(f"============================================================================\n")
        except Exception as e:
            print(f"\n[Hugging Face EROARE upload]: {e}")
            print(f"Poti publica manual mai tarziu folosind: huggingface-cli upload {repo_id} {output_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale Pre-training Engine for Romanian Generative LLM")
    parser.add_argument("--mode", type=str, default="spp", choices=["base", "spp", "filtered"], help="'base' (vanilla), 'spp' (reflections), 'filtered' (harmful docs loss-masked; needs v2 sidecar)")
    parser.add_argument("--steps", "--additional-steps", type=int, default=10000, dest="steps", help="Steps to train (default 10,000 = ~655M tokens, ~10 hours)")
    parser.add_argument("--total-target", type=int, default=None, help="Cumulative target steps (defaults to steps)")
    parser.add_argument("--interval-mins", type=float, default=30.0, help="Minutes between live terminal reports (default 30.0)")
    parser.add_argument("--save-steps", type=int, default=2000, help="Steps between checkpoints")
    parser.add_argument("--from-scratch", action="store_true", help="Initialize randomly from Token Zero (matching base model baseline)")
    parser.add_argument("--push-to-hub", action="store_true", help="Automatically push finalized model to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="flaviussteff/spp-ro-125m", help="Hugging Face repo id")
    parser.add_argument("--spp-ratio", type=float, default=0.10, help="SPP interleaving ratio (default: 0.10, use 0.20 for 20%% mode)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for checkpoints and final weights")
    parser.add_argument("--resume", action="store_true", help="Resume pre-training seamlessly from the latest saved checkpoint")
    parser.add_argument("--mask-assistant-loss", action="store_true", help="Exclude the <assistant> marker from the loss (paper setting). OFF for the 50k T0 runs; keep OFF for a comparable MT run.")
    parser.add_argument("--snap-insertion", action="store_true", help="Snap the reflection insertion point to a word boundary. OFF for the 50k T0 runs; keep OFF for a comparable MT run.")
    parser.add_argument("--sidecar", type=str, default=None, help="Reflections parquet (default: v1 sidecar; with --inline/--mode filtered: data/sidecar/reflections_v2.parquet)")
    parser.add_argument("--inline", action="store_true", help="v2: annotated documents REPLACE their originals in the stream (needs doc_key column)")
    parser.add_argument("--prefix-tail", action="store_true", help="Keep the end of the prefix when truncating long annotated documents (v2)")
    parser.add_argument("--log-file", type=str, default=None, help="Also write console output (reports, checkpoints; not the progress bar) to this file")
    parser.add_argument("--init-from", type=str, default=None, help="Initialise WEIGHTS from this checkpoint/model dir and start a fresh LR schedule (SPP{MT}: Base@45k + 5k steps). Combine with --steps 5000 --total-target 50000.")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate (default 1.5e-4 = pretraining peak; MT re-warm uses 5e-5)")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Linear warmup steps (default 500; MT re-warm uses 250)")
    parser.add_argument("--mt-annotated-ratio", type=float, default=0.0, help="SPP{MT} v2 (paper midtraining): fraction of sequences drawn from the annotated documents (EPFL: 0.6758); 0 = natural inline rate (SPP{T0})")
    parser.add_argument("--reflection-loss-only", action="store_true", help="SPP{MT} v2 (paper midtraining): on annotated documents, loss only on the reflection span; plain documents keep the full loss")
    parser.add_argument("--pack", action="store_true", help="v3: pack several EOS-terminated documents per 1,024-position window (block-diagonal attention; ~2x real tokens per step). Use for ALL models of the packed generation.")
    parser.add_argument("--no-pack-eos", action="store_true", help="with --pack: do not append </s> after each document")
    parser.add_argument("--head-cap", action="store_true", help="EPFL semantics: annotated documents are cut at the start like plain ones; the reflection is inserted only if its position falls inside the window, otherwise the document is trained plain. Overrides --prefix-tail. DEFAULT under --pack (v3); --no-head-cap disables it.")
    parser.add_argument("--no-head-cap", action="store_true", help="with --pack: keep the v1/v2 truncation rule (--prefix-tail) instead of head-cap")
    parser.add_argument("--keep-step", type=int, default=0, help="copy checkpoint-<N> to <output>/checkpoints_keep/ when saved (survives save_total_limit); Base runs use 45000 for the MT / re-warm starts")
    args = parser.parse_args()
    if args.log_file:
        install_tee(Path(args.log_file))

    out_dir = Path(args.output_dir) if args.output_dir else None

    run_scale_pretraining(
        mode=args.mode,
        additional_steps=args.steps,
        total_target_steps=args.total_target,
        update_interval_mins=args.interval_mins,
        save_steps=args.save_steps,
        from_scratch=args.from_scratch,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        spp_ratio=args.spp_ratio,
        output_dir=out_dir,
        resume=args.resume,
        mask_assistant_loss=args.mask_assistant_loss,
        snap_insertion=args.snap_insertion,
        sidecar=Path(args.sidecar) if args.sidecar else None,
        inline=args.inline,
        prefix_tail=args.prefix_tail,
        init_from=Path(args.init_from) if args.init_from else None,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        mt_annotated_ratio=args.mt_annotated_ratio,
        reflection_loss_only=args.reflection_loss_only,
        pack=args.pack,
        pack_eos=not args.no_pack_eos,
        keep_step=args.keep_step,
        head_cap=args.head_cap or (args.pack and not args.no_head_cap),
    )


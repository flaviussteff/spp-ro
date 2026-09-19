"""
Scale Pre-training Engine for Romanian Generative LLM (~124.8M Parameters)
Tuned for 2-3 Days per Model on NVIDIA GeForce RTX 3060 (12GB VRAM).

Features:
1. Continued Pre-training: Resumes directly from existing model weights in models/
2. Deep Scale: Targets 60,000 steps total (~3.93 Billion tokens per model)
3. Hourly Live Telemetry: Full diagnostic card + live generated text every 60 minutes
4. Balanced Data Streaming: Prioritizes FineWeb-2 web data and Wikipedia over news
5. Checkpointing: Saves every 5,000 steps with automatic checkpoint rotation
"""
import os
import sys
import time
import math
import datetime
import argparse
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
        ModelConfig,
        HardwareAndTrainingConfig,
    )
    from spp_collator import DataCollatorForSPP
except ImportError:
    from src.config import (
        TOKENIZER_DIR,
        CLEAN_DATA_DIR,
        SIDECAR_DIR,
        BASE_MODEL_DIR,
        SPP_MODEL_DIR,
        ModelConfig,
        HardwareAndTrainingConfig,
    )
    from src.spp_collator import DataCollatorForSPP


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
        self.report_counter = 0
        self.latest_loss = None
        self.latest_lr = None

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
        steps_this_session = max(1, state.global_step)
        
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
        print(f"[Raport #{self.report_counter}] Pas: {current_step:,}/{self.total_target_steps:,} ({pct_done:.1f}%) | ETA: {eta_str}", flush=True)
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

    def on_step_end(self, args, state, control, model=None, **kwargs):
        current_time = time.time()
        if (current_time - self.last_report_time) >= self.interval_seconds:
            self.last_report_time = current_time
            if model is not None:
                self._print_report(state, model)


class VanillaCausalDataCollator:
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_texts = []
        for f in features:
            if isinstance(f, str):
                batch_texts.append(f)
            elif isinstance(f, dict):
                if "text" in f:
                    batch_texts.append(f["text"])
                elif "pre_text" in f:
                    batch_texts.append(f"{f.get('pre_text', '')} {f.get('post_text', '')}")
                else:
                    str_vals = [v for v in f.values() if isinstance(v, str)]
                    batch_texts.append(str_vals[0] if str_vals else "")
            else:
                batch_texts.append(str(f))

        batch_enc = self.tokenizer(
            batch_texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = batch_enc["input_ids"]
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        labels[labels == pad_id] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": batch_enc["attention_mask"],
            "labels": labels,
        }


class StreamingParquetDataset(IterableDataset):
    """
    Streaming dataset that reads Parquet row-groups on the fly.
    Uses < 100 MB of RAM regardless of whether the corpus is 1 GB or 100 GB.
    """
    def __init__(self, parquet_path: Path, mode: str = "base", spp_sidecar_path: Optional[Path] = None):
        self.parquet_path = parquet_path
        self.mode = mode
        self.spp_items = []
        if mode == "spp" and spp_sidecar_path and spp_sidecar_path.exists():
            df_refl = pd.read_parquet(spp_sidecar_path)
            print(f"[SPP] Loaded {len(df_refl):,} constitutional reflection traces from sidecar.")
            for _, row in df_refl.iterrows():
                text = row["text"]
                pos = int(row["reflection_char_position"])
                self.spp_items.append({
                    "pre_text": text[:pos],
                    "reflection": row["reflection_text"],
                    "post_text": text[pos:],
                })

    def __iter__(self):
        import pyarrow.parquet as pq
        import random
        
        pf = pq.ParquetFile(str(self.parquet_path))
        num_rgs = pf.num_row_groups
        spp_idx = 0
        
        # Infinite generator loop to supply batches continuously until max_steps is reached
        while True:
            rg_order = list(range(num_rgs))
            random.shuffle(rg_order)
            for rg in rg_order:
                table = pf.read_row_group(rg, columns=["text"])
                texts = table["text"].to_pylist()
                random.shuffle(texts)
                for t in texts:
                    if t and len(t) >= 150:
                        yield {"text": t}
                    
                    if self.mode == "spp" and self.spp_items:
                        # Interleave SPP constitutional reflections at ~10% frequency
                        if random.random() < 0.10:
                            yield self.spp_items[spp_idx % len(self.spp_items)]
                            spp_idx += 1


def run_scale_pretraining(
    mode: str = "base",
    additional_steps: int = 50000,
    total_target_steps: int = 60000,
    update_interval_mins: float = 60.0,
    save_steps: int = 5000,
):
    print(f"\n[Pre-antrenare: {mode.upper()}] Pasi: {additional_steps:,} (Total: {total_target_steps:,}) | Durata estimata: ~{additional_steps / 1000:.0f}h\n", flush=True)

    output_dir = BASE_MODEL_DIR if mode == "base" else SPP_MODEL_DIR
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    # 1. Load existing model weights
    if (output_dir / "model.safetensors").exists():
        print(f"[Model] Resuming from existing weights in: {output_dir.name}...")
        model = LlamaForCausalLM.from_pretrained(
            str(output_dir),
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    else:
        print(f"[Model] No previous checkpoint found in {output_dir}. Initializing from scratch...")
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

    # 2. Hardware specs: batch size 16 * grad accum 4 = 64 seqs = 65,536 tokens/step (~50 hours for 50k steps)
    batch_size = 16
    grad_accum = 4
    eff_tokens_per_step = batch_size * grad_accum * 1024  # 65,536 tokens

    # Priority: corpus_scale_stream.parquet, else corpus_unannotated.parquet
    stream_path = CLEAN_DATA_DIR / "corpus_scale_stream.parquet"
    if not stream_path.exists():
        stream_path = CLEAN_DATA_DIR / "corpus_unannotated.parquet"
        
    if not stream_path.exists():
        raise FileNotFoundError(f"Missing pre-training data at {stream_path}. Run prepare_scale_corpus.py first.")
        
    print(f"[Dataset] Initializing low-RAM streaming reader from {stream_path.name}...")
    dataset = StreamingParquetDataset(
        parquet_path=stream_path,
        mode=mode,
        spp_sidecar_path=(SIDECAR_DIR / "reflections.parquet") if mode == "spp" else None,
    )

    if mode == "spp":
        collator = DataCollatorForSPP(tokenizer=tokenizer, max_length=1024)
    else:
        collator = VanillaCausalDataCollator(tokenizer=tokenizer, max_length=1024)

    # 3. Setup Training Arguments
    checkpoints_dir = output_dir / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        max_steps=additional_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=1.5e-4,
        warmup_steps=500,
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
    )

    hourly_cb = HourlyLiveProgressCallback(
        eff_tokens_per_step=eff_tokens_per_step,
        total_target_steps=total_target_steps,
        tokenizer=tokenizer,
        interval_mins=update_interval_mins,
        initial_steps_done=total_target_steps - additional_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[hourly_cb],
    )

    print(f"\n[Engine] Starting {additional_steps:,} steps training run...")
    print(f"[Engine] Live progress report will print every {update_interval_mins:.0f} minutes in this console.")
    
    trainer.train()

    # 4. Save Final Upgraded Model
    print(f"\n[Saving] Exporting finalized upgraded model to: {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[Done] Model {mode.upper()} training complete! Processed ~{total_target_steps * eff_tokens_per_step / 1e9:.2f} Billion tokens.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale Pre-training Engine for Romanian LLM")
    parser.add_argument("--mode", type=str, default="base", choices=["base", "spp"], help="Model mode: 'base' or 'spp'")
    parser.add_argument("--additional-steps", type=int, default=50000, help="Steps to train in this session (default 50000 = ~50 hours)")
    parser.add_argument("--total-target", type=int, default=60000, help="Cumulative target steps (default 60000 = 3.93B tokens)")
    parser.add_argument("--interval-mins", type=float, default=60.0, help="Minutes between live terminal reports (default 60.0)")
    parser.add_argument("--save-steps", type=int, default=5000, help="Steps between checkpoints")
    args = parser.parse_args()

    run_scale_pretraining(
        mode=args.mode,
        additional_steps=args.additional_steps,
        total_target_steps=args.total_target,
        update_interval_mins=args.interval_mins,
        save_steps=args.save_steps,
    )

"""
Local Pretraining Engine for Romanian Language Model (~124.8M Parameters)
Tuned for NVIDIA GeForce RTX 3060 (12GB VRAM) on Windows PowerShell.
Supports:
1. Base-Ro (Vanilla next-token prediction)
2. SPP-Ro (Synthetic Persona Pretraining from Token Zero with Attention Block & RoPE Aliasing)
"""
import os
import sys

# Ensure UTF-8 output on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import time
import datetime
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

# Ensure src and root directories are on sys.path for IDEs and scripts
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


class TenMinuteProgressCallback(TrainerCallback):
    """
    Periodic progress reporter for local terminal training on RTX 3060.
    Prints a prominent, formatted progress card every N minutes (default 10)
    with step completion, ETA, token throughput, VRAM, loss, and live generation.
    """
    def __init__(self, eff_tokens_per_step: int, total_steps: int, tokenizer, interval_mins: float = 10.0):
        self.eff_tokens_per_step = eff_tokens_per_step
        self.total_steps = total_steps
        self.tokenizer = tokenizer
        self.interval_seconds = interval_mins * 60.0
        self.start_time = time.time()
        self.last_report_time = time.time()
        self.latest_loss = None
        self.latest_lr = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.latest_loss = logs["loss"]
            if "learning_rate" in logs:
                self.latest_lr = logs["learning_rate"]

    def _print_report(self, state, model, title="10-MINUTE TRAINING PROGRESS REPORT"):
        current_time = time.time()
        current_step = state.global_step
        total_elapsed = current_time - self.start_time
        
        steps_done = max(1, current_step)
        sec_per_step = total_elapsed / steps_done
        remaining_steps = max(0, self.total_steps - current_step)
        eta_seconds = remaining_steps * sec_per_step
        
        total_elapsed_str = str(datetime.timedelta(seconds=int(total_elapsed)))
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        pct = (current_step / max(1, self.total_steps)) * 100
        
        tokens_done = current_step * self.eff_tokens_per_step
        total_planned_tokens = self.total_steps * self.eff_tokens_per_step
        tokens_per_sec = int(tokens_done / max(1, total_elapsed))
        
        loss_str = f"{self.latest_loss:.4f}" if self.latest_loss is not None else "Calculating..."
        ppl_str = f"{math.exp(self.latest_loss):.2f}" if self.latest_loss is not None and self.latest_loss < 20 else "N/A"
        lr_str = f"{self.latest_lr:.2e}" if self.latest_lr is not None else "N/A"
        
        vram_str = "CPU / N/A"
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            vram_str = f"{allocated:.2f} GB (Reserved: {reserved:.2f} GB / 12 GB)"
            
        sample_text = ""
        if model is not None and self.tokenizer is not None:
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    prompt = "În România contemporană,"
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=22,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    sample_text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True).replace("\n", " ")
            except Exception:
                pass
            if was_training:
                model.train()

        print("\n" + "=" * 76)
        print(f" [ {title} ]")
        print("=" * 76)
        print(f"  Progress:        Step {current_step:,} / {self.total_steps:,} ({pct:.1f}% Completed)")
        print(f"  Time Elapsed:    {total_elapsed_str} | Estimated Remaining (ETA): {eta_str}")
        print(f"  Tokens Trained:  {tokens_done:,} / {total_planned_tokens:,} ({tokens_per_sec:,} tokens/sec)")
        print(f"  Metrics:         Loss: {loss_str} | Perplexity: {ppl_str} | LR: {lr_str}")
        print(f"  RTX 3060 VRAM:   {vram_str}")
        if sample_text:
            print(f"  Live Generation: \"{sample_text}\"")
        print("=" * 76 + "\n", flush=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        current_time = time.time()
        
        # Initial verification report on step 1
        if state.global_step == 1 and not hasattr(self, "_step_1_reported"):
            self._step_1_reported = True
            print("\n[Timer Active] 10-minute periodic progress reporter initialized.", flush=True)
            self._print_report(state, model, title="INITIALIZATION & TRAINING CONFIRMATION")
            self.last_report_time = current_time
            return

        # Periodic 10-minute reports
        if (current_time - self.last_report_time) >= self.interval_seconds:
            self.last_report_time = current_time
            self._print_report(state, model, title=f"{int(self.interval_seconds // 60)}-MINUTE TRAINING PROGRESS REPORT")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self._print_report(state, model, title="PRETRAINING RUN COMPLETE SUMMARY")


class RomanianPretrainDataset(Dataset):
    """Dataset handling both pure web documents and SPP reflection-annotated documents."""
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


class VanillaCausalDataCollator:
    """Picklable top-level collator for standard causal language modeling on Windows."""
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"],
            "labels": labels,
            "attention_mask": enc["attention_mask"],
        }


def build_model_and_tokenizer(model_cfg: ModelConfig):
    """Initializes the ~124.8M Llama architecture from scratch."""
    print("Loading Romanian BPE Tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
    
    config = LlamaConfig(
        vocab_size=model_cfg.vocab_size,
        hidden_size=model_cfg.hidden_size,
        intermediate_size=model_cfg.intermediate_size,
        num_hidden_layers=model_cfg.num_hidden_layers,
        num_attention_heads=model_cfg.num_attention_heads,
        num_key_value_heads=model_cfg.num_key_value_heads,
        hidden_act=model_cfg.hidden_act,
        max_position_embeddings=model_cfg.max_position_embeddings,
        rms_norm_eps=model_cfg.rms_norm_eps,
        rope_theta=model_cfg.rope_theta,
        tie_word_embeddings=model_cfg.tie_word_embeddings,
        bos_token_id=tokenizer.bos_token_id or 1,
        eos_token_id=tokenizer.eos_token_id or 2,
        pad_token_id=tokenizer.pad_token_id or 3,
        use_cache=False,  # Saves VRAM during training
    )
    
    print("Initializing LlamaForCausalLM from scratch (Token Zero)...")
    model = LlamaForCausalLM(config)
    
    # Calculate parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Embedding Tying Active: {config.tie_word_embeddings} (Saved ~12.5M params)")
    
    return model, tokenizer


def load_training_data(mode: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Loads cleaned text and merges SPP reflections if mode == 'spp' with safe memory footprint."""
    import pyarrow.parquet as pq
    items: List[Dict[str, Any]] = []
    
    unannotated_path = CLEAN_DATA_DIR / "corpus_unannotated.parquet"
    if not unannotated_path.exists():
        raise FileNotFoundError(f"Missing {unannotated_path}. Run `py src/clean_corpus.py` first.")
        
    pf = pq.ParquetFile(str(unannotated_path))
    total_available = pf.metadata.num_rows
    # Default cap at 300,000 documents (~100-150M tokens) if max_samples is None to protect 16GB host RAM
    target_unann = min(max_samples if max_samples else 300000, total_available)
    print(f"Streaming {target_unann:,} unannotated documents from {unannotated_path.name}...")
    
    loaded = 0
    for rg_idx in range(pf.num_row_groups):
        rg_table = pf.read_row_group(rg_idx, columns=["text"])
        for text in rg_table["text"].to_pylist():
            items.append({"text": text})
            loaded += 1
            if loaded >= target_unann:
                break
        if loaded >= target_unann:
            break
        
    if mode == "spp":
        sidecar_path = SIDECAR_DIR / "reflections.parquet"
        if not sidecar_path.exists():
            raise FileNotFoundError(f"Missing {sidecar_path}. Run `py src/spp_annotator.py` first.")
            
        df_sidecar = pd.read_parquet(sidecar_path)
        print(f"Merging {len(df_sidecar):,} SPP constitutional reflections...")
        for _, row in df_sidecar.iterrows():
            text = row["text"]
            char_pos = int(row["reflection_char_position"])
            reflection = row["reflection_text"]
            
            pre_text = text[:char_pos]
            post_text = text[char_pos:]
            
            items.append({
                "pre_text": pre_text,
                "reflection": reflection,
                "post_text": post_text,
            })
            
    # Shuffle the mixture
    import random
    random.seed(42)
    random.shuffle(items)
    print(f"Prepared {len(items):,} training sequences for mode '{mode}'.")
    return items


def run_pretraining(mode: str = "base", max_steps: int = None, update_interval_mins: float = 10.0):
    print("==================================================")
    print(f" Romanian Pretraining: Mode '{mode.upper()}' on RTX 3060 (12GB)")
    print("==================================================")
    
    hw_cfg = HardwareAndTrainingConfig()
    m_cfg = ModelConfig()
    
    if max_steps is not None:
        hw_cfg.max_steps = max_steps
        
    output_dir = BASE_MODEL_DIR if mode == "base" else SPP_MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Build Model & Tokenizer
    model, tokenizer = build_model_and_tokenizer(m_cfg)
    
    # 2. Prepare Data & Collator
    needed_sequences = None
    if max_steps is not None:
        needed_sequences = max(5000, max_steps * hw_cfg.per_device_train_batch_size * hw_cfg.gradient_accumulation_steps * 2)
    raw_items = load_training_data(mode, max_samples=needed_sequences)
    dataset = RomanianPretrainDataset(raw_items)
    
    if mode == "spp":
        collator = DataCollatorForSPP(
            tokenizer=tokenizer,
            max_length=m_cfg.max_position_embeddings,
        )
    else:
        collator = VanillaCausalDataCollator(
            tokenizer=tokenizer,
            max_length=m_cfg.max_position_embeddings,
        )

    # 3. Setup Training Arguments for RTX 3060 12GB
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=hw_cfg.max_steps,
        per_device_train_batch_size=hw_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=hw_cfg.gradient_accumulation_steps,
        remove_unused_columns=False,
        learning_rate=hw_cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=hw_cfg.warmup_steps,
        weight_decay=hw_cfg.weight_decay,
        logging_steps=hw_cfg.logging_steps,
        save_steps=hw_cfg.save_steps,
        save_total_limit=2,
        bf16=hw_cfg.bf16,
        fp16=hw_cfg.fp16,
        dataloader_num_workers=0,  # 0 is fastest and avoids Windows IPC overhead
        gradient_checkpointing=hw_cfg.gradient_checkpointing,
        report_to="none",
    )
    
    eff_tokens_per_step = (
        hw_cfg.per_device_train_batch_size
        * hw_cfg.gradient_accumulation_steps
        * m_cfg.max_position_embeddings
    )
    print("\nTraining Metrics Calculation:")
    print(f"  Tokens per optimization step: {eff_tokens_per_step:,} tokens")
    print(f"  Total planned tokens: {eff_tokens_per_step * hw_cfg.max_steps / 1e9:.2f} Billion tokens")
    print(f"  Periodic updates interval: Every {update_interval_mins:.1f} minutes")
    print(f"  Saving checkpoints to: {output_dir}\n")
    
    progress_callback = TenMinuteProgressCallback(
        eff_tokens_per_step=eff_tokens_per_step,
        total_steps=hw_cfg.max_steps,
        tokenizer=tokenizer,
        interval_mins=update_interval_mins,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[progress_callback],
    )
    
    print("Starting pretraining from Token Zero...")
    trainer.train()
    
    # Save final weights and tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nTraining completed! Model successfully saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain Romanian 125M LLM from Token Zero")
    parser.add_argument("--mode", type=str, default="base", choices=["base", "spp"], help="Pretraining variant")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total steps (e.g., 1000 for quick test, 10000 for base run)")
    parser.add_argument("--update-interval-mins", type=float, default=10.0, help="Interval in minutes between terminal progress cards (default: 10.0)")
    args = parser.parse_args()
    
    run_pretraining(mode=args.mode, max_steps=args.max_steps, update_interval_mins=args.update_interval_mins)

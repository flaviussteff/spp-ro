"""
Post-Hoc LoRA Alignment Fine-Tuning Control (Step 6)
Fine-tunes Base-Ro-125M on Romanian Constitutional Reflections.
Saves merged weights and PEFT adapter to models/base_ro_125m_lora/
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Ensure UTF-8 output on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model, TaskType

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config import BASE_MODEL_DIR, TOKENIZER_DIR, SIDECAR_DIR
except ImportError:
    from src.config import BASE_MODEL_DIR, TOKENIZER_DIR, SIDECAR_DIR

OUTPUT_DIR = _ROOT_DIR / "models" / "base_ro_125m_lora"
REFLECTIONS_FILE = SIDECAR_DIR / "reflections.parquet"


class ConstitutionalAlignmentDataset(Dataset):
    """
    Supervised Fine-Tuning dataset on constitutional reflection pairs.
    Masks prompt tokens with -100 so loss is computed strictly on the alignment completion.
    """
    def __init__(self, parquet_path: Path, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        df = pd.read_parquet(parquet_path)
        
        self.examples = []
        for _, row in df.iterrows():
            text = str(row["text"])
            pos = int(row["reflection_char_position"])
            refl = str(row["reflection_text"])
            
            prompt = text[:pos].strip()
            # If prompt is too short, provide a minimal prefix
            if len(prompt) < 20:
                prompt = text[:min(len(text), 150)].strip()
                
            completion = f" <assistant> <reflection> {refl} </reflection>"
            self.examples.append((prompt, completion))
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt, completion = self.examples[idx]
        
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False) + [self.tokenizer.eos_token_id or 2]
        
        # Truncate prompt if needed
        max_prompt = self.max_length - len(completion_ids)
        if max_prompt < 32:
            prompt_ids = prompt_ids[-64:]
            max_completion = self.max_length - len(prompt_ids)
            completion_ids = completion_ids[:max_completion]
        else:
            prompt_ids = prompt_ids[-max_prompt:]
            
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        
        # Pad to max_length
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids += [pad_id] * pad_len
            labels += [-100] * pad_len
            
        return {
            "input_ids": torch.tensor(input_ids[:self.max_length], dtype=torch.long),
            "labels": torch.tensor(labels[:self.max_length], dtype=torch.long),
            "attention_mask": torch.tensor([1 if tok != (self.tokenizer.pad_token_id or 0) else 0 for tok in input_ids[:self.max_length]], dtype=torch.long),
        }


def train_lora_alignment():
    print("\n" + "=" * 70)
    print(" STEP 6: POST-HOC LoRA FINE-TUNING CONTROL (Base-Ro-125M + LoRA)")
    print("=" * 70)
    
    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError(f"Base model directory does not exist: {BASE_MODEL_DIR}")
    if not REFLECTIONS_FILE.exists():
        raise FileNotFoundError(f"Reflections dataset missing: {REFLECTIONS_FILE}")
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/5] Loading Base Model from {BASE_MODEL_DIR} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
        
    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL_DIR),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    
    print("[2/5] Attaching LoRA Adapter (r=16, alpha=32, target_modules=[q, k, v, o])...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    print(f"[3/5] Preparing Dataset from {REFLECTIONS_FILE} (10,000 constitutional pairs)...")
    dataset = ConstitutionalAlignmentDataset(REFLECTIONS_FILE, tokenizer, max_length=512)
    print(f"Total alignment training pairs: {len(dataset):,}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        num_train_epochs=2,             # 2 full passes = 1,250 steps (~15-20 mins)
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=25,
        save_strategy="no",
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    
    print("\n[4/5] Training LoRA Alignment Adapter...")
    start_t = time.time()
    trainer.train()
    elapsed = time.time() - start_t
    print(f"LoRA Training completed in {elapsed / 60:.1f} minutes.")
    
    print(f"\n[5/5] Merging LoRA into base weights and saving to {OUTPUT_DIR}...")
    # 1. Save PEFT adapter files separately
    adapter_dir = OUTPUT_DIR / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    
    # 2. Merge LoRA directly into base model for native AutoModelForCausalLM loading
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    # 3. Create Model Card
    readme_content = f"""---
language:
- ro
license: mit
pipeline_tag: text-generation
tags:
- romanian
- llama
- causal-lm
- lora
- post-hoc-alignment
---

# Base-Ro-125M-LoRA: Post-Hoc Aligned Romanian Language Model

`Base-Ro-125M-LoRA` represents the post-hoc alignment control model for the Bachelor's Thesis:
> **"Model Raising from Token Zero: Adapting Synthetic Pre-training Paths (SPP) for Romanian Generative LLMs and Evaluating Societal Bias Resilience"**

## Overview
* **Base Checkpoint:** `flaviussteff/base-ro-125m` (frozen raw weights).
* **Alignment Method:** Low-Rank Adaptation (LoRA, $r=16, \\alpha=32$).
* **Training Data:** 10,000 Romanian constitutional reflection thoughts (`data/sidecar/reflections.parquet`).
* **Role in Thesis:** Demonstrates the standard industry alignment paradigm (post-hoc fine-tuning) to test the **Superficial Alignment Hypothesis** against Token Zero SPP under adversarial prefix probing.
"""
    with open(OUTPUT_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    print("\n" + "=" * 70)
    print(" [SUCCESS] Base-Ro-LoRA Model is Saved and Ready for Benchmarking!")
    print(f" Location: {OUTPUT_DIR.resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    train_lora_alignment()

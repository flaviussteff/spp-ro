"""
Master Configuration for Romanian LLM from Scratch & SPP
Tuned specifically for: NVIDIA GeForce RTX 3060 (12GB VRAM) on Windows PowerShell
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Base workspace directory
WORKSPACE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = WORKSPACE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
SIDECAR_DIR = DATA_DIR / "sidecar"
CONSTITUTION_PATH = WORKSPACE_DIR / "constitutia_spp_ro.md"

# Granular Corpus Files
RAW_NEWS_FILE = RAW_DATA_DIR / "news_recent.jsonl"
RAW_WIKI_FILE = RAW_DATA_DIR / "wiki_ro.jsonl"
RAW_FINEWEB_DIR = RAW_DATA_DIR / "fineweb_shards"
CLEAN_NEWS_FILE = CLEAN_DATA_DIR / "news_recent.parquet"
CLEAN_WIKI_FILE = CLEAN_DATA_DIR / "wiki_clean.parquet"
UNANNOTATED_FILE = CLEAN_DATA_DIR / "corpus_unannotated.parquet"
SPP_CANDIDATES_FILE = CLEAN_DATA_DIR / "corpus_for_spp.parquet"

# Output Paths
TOKENIZER_DIR = WORKSPACE_DIR / "tokenizer" / "ro_bpe_16k"
MODELS_DIR = WORKSPACE_DIR / "models"
BASE_MODEL_DIR = MODELS_DIR / "base_ro_125m"
SPP_MODEL_DIR = MODELS_DIR / "spp_ro_125m"
SFT_MODEL_DIR = MODELS_DIR / "sft_ro_125m"
EVALS_DIR = WORKSPACE_DIR / "evals"

# Special Tokens for Tokenizer and SPP Mechanism
SPECIAL_TOKENS = [
    "<unk>",
    "<s>",
    "</s>",
    "<pad>",
    "<mask>",
    "<document>",
    "</document>",
    "<assistant>",
    "<reflection>",
    "</reflection>",
    "<|thought_start|>",
    "<|thought_end|>",
    "<|persona_start|>",
    "<|persona_end|>",
]


@dataclass
class ModelConfig:
    """Architectural specs for ~124.8M parameter Llama-style decoder"""
    vocab_size: int = 16384
    hidden_size: int = 768
    intermediate_size: int = 2048          # SwiGLU 8/3 ratio
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4           # Grouped-Query Attention (GQA)
    max_position_embeddings: int = 1024    # Context window optimized for single GPU
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True       # Reuses weights, saving ~12.5M parameters
    hidden_act: str = "silu"


@dataclass
class HardwareAndTrainingConfig:
    """Tuned for RTX 3060 12GB without VRAM overflow"""
    # Batch sizing
    per_device_train_batch_size: int = 4   # 4 seqs of 1024 tokens = 4096 tokens/step per forward
    gradient_accumulation_steps: int = 16  # Simulates effective batch of 64 seqs (65,536 tokens/update)
    
    # Optimization
    learning_rate: float = 5e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 35000                 # ~2.3 Billion tokens total
    save_steps: int = 2500
    logging_steps: int = 25
    
    # Precision & Memory
    # RTX 3060 supports native fp16 and bf16. bf16 is preferred for stability.
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False   # 125M at 1024 context comfortably fits in 12GB
    dataloader_num_workers: int = 2        # Safe for Windows multi-processing
    
    # SPP Specifics
    spp_reflection_ratio: float = 0.10     # 10% of total documents receive constitutional reflections
    spp_news_reflection_mix: float = 0.07  # News constitutes ~7% of the reflection pool (avoids journalistic overkill)


@dataclass
class LocalAnnotatorConfig:
    """Settings for generating synthetic reflections via local Llama"""
    backend: str = "ollama"                # "ollama" or "mock"
    ollama_url: str = "http://localhost:11434/api/generate"
    model_name: str = "llama3.1:8b"        # Local quantized model in Ollama
    temperature: float = 0.4
    max_tokens: int = 150

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
RAW_DATA_DIR = DATA_DIR / "raw"   # raw downloads were moved to archive/raw_corpus/ (only download_corpus/clean_corpus/build_heldout_set read them)
CLEAN_DATA_DIR = DATA_DIR / "clean"
SIDECAR_DIR = DATA_DIR / "sidecar"
CONSTITUTION_PATH = WORKSPACE_DIR / "constitution_spp_ro.md"
if not CONSTITUTION_PATH.exists():
    CONSTITUTION_PATH = WORKSPACE_DIR / "constitutia_spp_ro.md"

# Granular Corpus Files
RAW_NEWS_FILE = RAW_DATA_DIR / "news_recent.jsonl"
RAW_WIKI_FILE = RAW_DATA_DIR / "wiki_ro.jsonl"
RAW_FINEWEB_DIR = RAW_DATA_DIR / "fineweb_shards"
CLEAN_NEWS_FILE = CLEAN_DATA_DIR / "news_recent.parquet"
CLEAN_WIKI_FILE = CLEAN_DATA_DIR / "wiki_clean.parquet"
UNANNOTATED_FILE = CLEAN_DATA_DIR / "corpus_unannotated.parquet"
SPP_CANDIDATES_FILE = CLEAN_DATA_DIR / "spp_candidates_cache.parquet"   # v1 candidate pool written by clean_corpus.py; the v1 copy is in archive/v1/data/clean/

# Output Paths
TOKENIZER_DIR = WORKSPACE_DIR / "tokenizer" / "ro_bpe_16k"
MODELS_DIR = WORKSPACE_DIR / "models"
# NOTE: folder names keep the historical "125m" suffix; the actual models have 88,099,584
# parameters (tied embeddings, 16k vocab). Public names are SPP-Ro-88M / Base-Ro-88M.
BASE_MODEL_DIR = MODELS_DIR / "base_ro_125m"          # Base-Ro-88M (vanilla control)
SPP_MODEL_DIR = MODELS_DIR / "spp_ro_125m"            # SPP-Ro-88M, SPP{T0}, 10% interleaving
SPP_20PCT_MODEL_DIR = MODELS_DIR / "spp_ro_125m_20pct"  # SPP-Ro-88M, SPP{T0}, 20% interleaving
SPP_MT_MODEL_DIR = MODELS_DIR / "spp_ro_88m_mt"       # SPP-Ro-88M, SPP{MT}: Base@45k + 5k SPP steps, fresh LR schedule (peak 5e-5)
BASE_REWARM_MODEL_DIR = MODELS_DIR / "base_ro_88m_rewarm"  # control for MT: Base@45k + 5k steps, same fresh LR schedule, no reflections
SPP_MT_COSINETAIL_DIR = WORKSPACE_DIR / "archive" / "v1" / "models" / "spp_ro_88m_mt_cosinetail"  # first MT attempt (null run: LR ~1e-6, weights ~= Base); archived, not evaluated
SFT_MODEL_DIR = MODELS_DIR / "sft_ro_125m"
# ---- v2 (faithful data pipeline: real documents, measured severity, random prefix-only insertion)
SIDECAR_V2_FILE = SIDECAR_DIR / "reflections_v2.parquet"
SPP_V2_MODEL_DIR = MODELS_DIR / "spp_ro_88m_v2"            # SPP{T0} v2, from scratch, inline annotated docs
SPP_MT_V2_MODEL_DIR = MODELS_DIR / "spp_ro_88m_mt_v2"      # SPP{MT} v2: Base@45k + 5k steps on the v2 stream
FILTERED_MODEL_DIR = MODELS_DIR / "filtered_ro_88m"        # paper's Filtered baseline: harmful docs loss-masked
# ---- v3 (packing generation): the v2 recipe + sequence packing (EOS-terminated documents fill every 1,024-position
# window, block-diagonal attention; ~2x the real tokens per step at the same compute) + the expanded sidecar (~10% of docs).
SPP_CANDIDATES_V3_FILE = CLEAN_DATA_DIR / "spp_candidates_v3.parquet"   # classifier threshold lowered; only documents not yet in the sidecar
BASE_PACK_MODEL_DIR = MODELS_DIR / "base_ro_88m_pack"              # Base-Ro-88M-pack: vanilla control, packed
SPP_PACK_MODEL_DIR = MODELS_DIR / "spp_ro_88m_pack"                # SPP{T0} v3: inline annotated docs, packed
BASE_PACK_REWARM_MODEL_DIR = MODELS_DIR / "base_ro_88m_pack_rewarm"  # control for MT: Base-pack@45k + 5k steps, no reflections
SPP_MT_PACK_MODEL_DIR = MODELS_DIR / "spp_ro_88m_mt_pack"          # SPP{MT} v3: Base-pack@45k + 5k steps on the annotated stream
FILTERED_PACK_MODEL_DIR = MODELS_DIR / "filtered_ro_88m_pack"      # Filtered baseline, packed
EVALS_DIR = WORKSPACE_DIR / "evals"
EVAL_DATA_DIR = DATA_DIR / "eval"
HELDOUT_FILE = EVAL_DATA_DIR / "heldout_fineweb.parquet"  # verified disjoint from training (build_heldout_set.py)

# Every model the evaluation scripts should look for: (key, dir, description). Missing dirs are skipped.
ALL_MODELS = [
    ("Base-Ro-88M", BASE_MODEL_DIR, "Vanilla control (no reflections)"),
    ("SPP-Ro-88M", SPP_MODEL_DIR, "SPP{T0} v1, 10% synthetic sidecar"),
    ("SPP-Ro-88M-20pct", SPP_20PCT_MODEL_DIR, "SPP{T0} v1, 20% synthetic sidecar"),
    ("Base-Ro-88M-rewarm", BASE_REWARM_MODEL_DIR, "Control: Base@45k + 5k steps, fresh LR schedule, no reflections"),
    ("SPP-Ro-88M-MT", SPP_MT_MODEL_DIR, "SPP{MT} v1: Base@45k + 5k steps, fresh LR schedule, synthetic sidecar"),
    # generation 2 (un-packed, real documents) was stopped at step 12,500 in favour of generation 3 (packing); the
    # partial run is in archive/v2/models/ and no v2 model is evaluated. The *_V2_MODEL_DIR / FILTERED_MODEL_DIR
    # constants stay for the un-packed training paths of train_scale_pretrain.py.
    # generation 3 (packed): compared with Base-Ro-88M-pack, not with Base-Ro-88M
    ("Base-Ro-88M-pack", BASE_PACK_MODEL_DIR, "Vanilla control, packed sequences (v3)"),
    ("SPP-Ro-88M-pack", SPP_PACK_MODEL_DIR, "SPP{T0} v3: real documents, expanded sidecar, packed sequences"),
    ("Base-Ro-88M-pack-rewarm", BASE_PACK_REWARM_MODEL_DIR, "Control: Base-pack@45k + 5k steps, fresh LR schedule, no reflections"),
    ("SPP-Ro-88M-MT-pack", SPP_MT_PACK_MODEL_DIR, "SPP{MT} v3: Base-pack@45k + 5k steps on the annotated stream"),
    ("Filtered-Ro-88M-pack", FILTERED_PACK_MODEL_DIR, "Filtered baseline, packed sequences (v3)"),
]


def base_ref_key(key: str) -> str:
    """Vanilla reference for a model key: generation-3 (packed) models are compared with Base-Ro-88M-pack,
    everything else with Base-Ro-88M. Works for the '(delib)' variants too."""
    # midtraining (MT) models are compared with their matched re-warm control (same start, same schedule, no reflections)
    pack = "pack" in key
    if "-MT" in key:
        return "Base-Ro-88M-pack-rewarm" if pack else "Base-Ro-88M-rewarm"
    return "Base-Ro-88M-pack" if pack else "Base-Ro-88M"

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
    """Architectural specs for the ~88.1M parameter Llama-style decoder (88,099,584 params with tied embeddings)"""
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
    learning_rate: float = 1.5e-4          # informative only: train_scale_pretrain.py takes --lr (default 1.5e-4)
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 50000                 # informative only: the runs use --steps 50000 (3.27B positions)
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

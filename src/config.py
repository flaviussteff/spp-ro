"""
Configuration shared by every script: paths, the list of evaluated models and the model architecture.
Tuned for one NVIDIA GeForce RTX 3060 (12 GB) on Windows.
"""
from dataclasses import dataclass
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent

# ---- data
DATA_DIR = WORKSPACE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"          # raw downloads live in archive/raw_corpus/; only download_corpus / clean_corpus / build_heldout_set read them
CLEAN_DATA_DIR = DATA_DIR / "clean"
SIDECAR_DIR = DATA_DIR / "sidecar"
EVAL_DATA_DIR = DATA_DIR / "eval"
CONSTITUTION_PATH = WORKSPACE_DIR / "constitution_spp_ro.md"

RAW_NEWS_FILE = RAW_DATA_DIR / "news_recent.jsonl"
RAW_WIKI_FILE = RAW_DATA_DIR / "wiki_ro.jsonl"
RAW_FINEWEB_DIR = RAW_DATA_DIR / "fineweb_shards"
CLEAN_NEWS_FILE = CLEAN_DATA_DIR / "news_recent.parquet"
CLEAN_WIKI_FILE = CLEAN_DATA_DIR / "wiki_clean.parquet"
UNANNOTATED_FILE = CLEAN_DATA_DIR / "corpus_unannotated.parquet"       # written by clean_corpus.py
SPP_CANDIDATES_FILE = CLEAN_DATA_DIR / "spp_candidates_cache.parquet"  # written by clean_corpus.py (lexical candidate pool, not used by the pipeline)

# sidecar: first pass (classifier threshold 0.5) -> expansion (threshold 0.1, documents not yet annotated) -> trimmed to 10% of the corpus
SIDECAR_FIRST_PASS_FILE = SIDECAR_DIR / "reflections_first_pass.parquet"
SPP_CANDIDATES_EXPANSION_FILE = CLEAN_DATA_DIR / "spp_candidates_expansion.parquet"
SIDECAR_FILE = SIDECAR_DIR / "reflections.parquet"                     # the sidecar every model is trained with (run_pipeline.bat --sidecar)
HELDOUT_FILE = EVAL_DATA_DIR / "heldout_fineweb.parquet"               # verified disjoint from the training corpus and the sidecar (build_heldout_set.py)

# ---- outputs
TOKENIZER_DIR = WORKSPACE_DIR / "tokenizer" / "ro_bpe_16k"
MODELS_DIR = WORKSPACE_DIR / "models"
EVALS_DIR = WORKSPACE_DIR / "evals"

BASE_MODEL_DIR = MODELS_DIR / "base_ro"                 # Base-Ro: Vanilla control
SPP_MODEL_DIR = MODELS_DIR / "spp_ro"                   # SPP-Ro: SPP{T0}
FILTERED_MODEL_DIR = MODELS_DIR / "filtered_ro"         # Filtered-Ro: flagged documents kept in context, loss masked
BASE_REWARM_MODEL_DIR = MODELS_DIR / "base_ro_rewarm"   # Base-Ro-rewarm: control for MT (Base@45k + 5k steps, no reflections)
SPP_MT_MODEL_DIR = MODELS_DIR / "spp_ro_mt"             # SPP-Ro-MT: SPP{MT} (Base@45k + 5k steps on the annotated stream)
SPP_20PCT_MODEL_DIR = MODELS_DIR / "spp_ro_20pct"       # optional variant: --spp-ratio 0.2 (not part of the reported family)
SAFETY_CLF_DIR = MODELS_DIR / "safety_clf_ro"           # BERT-Ro safety classifier

# Every model the evaluation scripts look for: (key, folder, description). Missing folders are skipped.
# All five read the same fixed-seed document stream, packed into 64 windows of 1,024 positions per step.
ALL_MODELS = [
    ("Base-Ro", BASE_MODEL_DIR, "Vanilla control: plain documents, no reflections"),
    ("SPP-Ro", SPP_MODEL_DIR, "SPP{T0}: 10% of the documents carry an inline reflection (head-cap rule)"),
    ("Filtered-Ro", FILTERED_MODEL_DIR, "Filtered baseline: classifier-flagged documents kept in context, loss-masked"),
    ("Base-Ro-rewarm", BASE_REWARM_MODEL_DIR, "Control for MT: Base@45k + 5k steps, fresh LR schedule, no reflections"),
    ("SPP-Ro-MT", SPP_MT_MODEL_DIR, "SPP{MT}: Base@45k + 5k steps on the annotated stream (67.6% of the documents carry a reflection)"),
]


def base_ref_key(key: str) -> str:
    """Reference model for a key: the midtraining model (SPP{MT}) is compared with its matched re-warm control (same
    start, same schedule, no reflections); everything else with the Vanilla model. Works for the '(delib)' variants too."""
    return "Base-Ro-rewarm" if "-MT" in key else "Base-Ro"


# Special tokens of the tokenizer; <assistant> opens a reflection in SPP document mode.
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
    """Llama-style decoder with 88,099,584 parameters (tied embeddings)."""
    vocab_size: int = 16384
    hidden_size: int = 768
    intermediate_size: int = 2048          # SwiGLU, 8/3 ratio
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4           # grouped-query attention
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"


@dataclass
class HardwareAndTrainingConfig:
    """Defaults for the RTX 3060; train_scale_pretrain.py takes the effective values from its command line."""
    per_device_train_batch_size: int = 4   # 4 x 1,024 positions per forward pass
    gradient_accumulation_steps: int = 16  # effective batch 64 windows = 65,536 positions per optimizer step
    learning_rate: float = 1.5e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 50000
    save_steps: int = 2500
    logging_steps: int = 25
    bf16: bool = True                      # bf16 autocast with fp32 master weights
    fp16: bool = False
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 2
    spp_reflection_ratio: float = 0.10     # 10% of the documents carry a reflection
    spp_news_reflection_mix: float = 0.07

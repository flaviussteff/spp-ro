"""
Romanian Byte-level BPE Tokenizer Trainer
Builds a compact 16,384-vocabulary tokenizer customized for the Romanian language.
Saves the tokenizer in Hugging Face PreTrainedTokenizerFast format.
"""
import os
import sys

# Ensure UTF-8 output on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from pathlib import Path
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from transformers import PreTrainedTokenizerFast

# Ensure src and root directories are on sys.path for IDEs and scripts
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config import TOKENIZER_DIR, CLEAN_DATA_DIR, SPECIAL_TOKENS, ModelConfig
except ImportError:
    from src.config import TOKENIZER_DIR, CLEAN_DATA_DIR, SPECIAL_TOKENS, ModelConfig


def train_romanian_tokenizer(
    training_file: Path = None,
    vocab_size: int = ModelConfig.vocab_size,
    output_dir: Path = TOKENIZER_DIR,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if training_file is None:
        training_file = CLEAN_DATA_DIR / "sample_for_tokenizer.txt"
        
    if not training_file.exists():
        print(f"Error: Tokenizer training file not found at {training_file}!")
        print("Please run `py src/clean_corpus.py` first.")
        return
        
    print("==================================================")
    print(" Training Custom Romanian Byte-level BPE Tokenizer")
    print(f" Source Text: {training_file}")
    print(f" Target Vocabulary Size: {vocab_size:,}")
    print("==================================================")
    
    # 1. Initialize Byte-level BPE
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # 2. Normalization: NFKC
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
    ])
    
    # 3. Pre-tokenizer: ByteLevel (preserves UTF-8 Romanian characters without fragmentation)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 4. Trainer with initial alphabet and custom special tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    
    # 5. Execute training (Rust engine)
    print("Fitting BPE merges on Romanian text...")
    tokenizer.train([str(training_file)], trainer)
    
    # Save raw tokenizer.json
    raw_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(raw_json_path))
    
    # 6. Wrap into Hugging Face PreTrainedTokenizerFast for seamless transformers integration
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(raw_json_path),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        clean_up_tokenization_spaces=False,
    )
    hf_tokenizer.save_pretrained(str(output_dir))
    
    print(f"\nTokenizer successfully trained and saved to: {output_dir}")
    
    # 7. Verification and Fertility Check
    print("\n--- Romanian Fertility Rate Verification ---")
    test_sentences = [
        "Cercetarea lingvistică din România evidențiază importanța diacriticelor corecte: ș, ț, ă, î, â.",
        "Modelul de limbaj învață reprezentări semantice pornind de la token zero.",
        "Comunitatea academică promovează egalitatea de gen și nediscriminarea etnică în învățământ.",
    ]
    
    for sentence in test_sentences:
        tokens = hf_tokenizer.tokenize(sentence)
        ids = hf_tokenizer.encode(sentence)
        words = sentence.split()
        fertility = len(tokens) / max(1, len(words))
        print(f"\nSentence: '{sentence}'")
        print(f"Words: {len(words)} | Tokens: {len(tokens)} (Fertility: {fertility:.2f} tokens/word)")
        print(f"Tokens: {tokens[:15]}...")
        
    print("\n==================================================")
    print("Tokenizer is ready! Next step: Generate SPP reflections or begin pretraining.")
    print("==================================================")


if __name__ == "__main__":
    train_romanian_tokenizer()

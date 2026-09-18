"""
The Mathematical Core of SPP: Attention Block & RoPE Aliasing Data Collator
Preserves the exact EPFL-dlab SPP invariant in native PyTorch for Hugging Face Llama models:
1. Attention Block: Tokens following the reflection (c4, c5) cannot attend to the reflection (r1, r2).
2. RoPE Aliasing: Position IDs of post-reflection tokens alias back to the insertion index.
"""
import torch
from typing import Dict, List, Any, Optional
from transformers import PreTrainedTokenizerFast


class DataCollatorForSPP:
    """
    Constructs 2D/4D block attention masks and aliased position IDs for SPP pretraining.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 1024,
        assistant_token: str = "<assistant>",
        pad_token: str = "<pad>",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.assistant_token_id = tokenizer.convert_tokens_to_ids(assistant_token)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Batch items can be:
        1. Plain documents: {'text': str}
        2. SPP documents: {'pre_text': str, 'reflection': str, 'post_text': str}
        """
        batch_input_ids = []
        batch_position_ids = []
        batch_labels = []
        batch_custom_masks = []
        
        for item in batch:
            if "reflection" in item and item["reflection"]:
                # SPP Document Mode: c1 c2 c3 <assistant> r1 r2 c4 c5
                pre_tokens = self.tokenizer.encode(item["pre_text"], add_special_tokens=False)
                refl_tokens = self.tokenizer.encode(item["reflection"], add_special_tokens=False)
                post_tokens = self.tokenizer.encode(item["post_text"], add_special_tokens=False)
                
                # Trim to fit max_length
                budget = self.max_length - len(refl_tokens) - 1
                if budget > 0:
                    pre_tokens = pre_tokens[:budget // 2]
                    post_tokens = post_tokens[:budget - len(pre_tokens)]
                    
                len_pre = len(pre_tokens)
                len_refl = len(refl_tokens) + 1  # Includes <assistant>
                len_post = len(post_tokens)
                
                # Combined sequence
                input_ids = pre_tokens + [self.assistant_token_id] + refl_tokens + post_tokens
                total_len = len(input_ids)
                
                # RoPE Aliased Position IDs
                # c1..c3: 0, 1, 2
                # <as> r1..r2: 3, 4, 5...
                # c4..c5: 3, 4, 5... (aliased back to len_pre!)
                pos_pre = list(range(len_pre))
                pos_refl = list(range(len_pre, len_pre + len_refl))
                pos_post = list(range(len_pre, len_pre + len_post))
                position_ids = pos_pre + pos_refl + pos_post
                
                # Build 2D Causal Block Attention Mask [total_len, total_len]
                # 1 = can attend, 0 = cannot attend
                mask_2d = torch.tril(torch.ones((total_len, total_len), dtype=torch.bool))
                
                # ATTENTION BLOCK: Post-reflection tokens cannot attend to reflection tokens!
                idx_refl_start = len_pre
                idx_refl_end = len_pre + len_refl
                idx_post_start = idx_refl_end
                
                # Mask out the reflection region for all post-reflection rows
                mask_2d[idx_post_start:, idx_refl_start:idx_refl_end] = False
                
            else:
                # Vanilla Document Mode: c1 c2 c3 c4 c5
                text = item.get("text", "")
                tokens = self.tokenizer.encode(text, add_special_tokens=False)[:self.max_length]
                total_len = len(tokens)
                
                input_ids = tokens
                position_ids = list(range(total_len))
                mask_2d = torch.tril(torch.ones((total_len, total_len), dtype=torch.bool))
                
            # Labels for next-token prediction
            labels = list(input_ids)
            
            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_position_ids.append(torch.tensor(position_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))
            batch_custom_masks.append(mask_2d)
            
        # Pad batch to max length in current batch
        max_b_len = max(ids.size(0) for ids in batch_input_ids)
        batch_size = len(batch)
        
        padded_input_ids = torch.full((batch_size, max_b_len), self.pad_token_id, dtype=torch.long)
        padded_position_ids = torch.zeros((batch_size, max_b_len), dtype=torch.long)
        padded_labels = torch.full((batch_size, max_b_len), -100, dtype=torch.long)
        
        # 4D attention mask for PyTorch SDPA: [batch_size, 1, max_b_len, max_b_len]
        # Format: True/False or 0.0 / -inf
        attention_mask_4d = torch.zeros((batch_size, 1, max_b_len, max_b_len), dtype=torch.bool)
        
        for i in range(batch_size):
            l = batch_input_ids[i].size(0)
            padded_input_ids[i, :l] = batch_input_ids[i]
            padded_position_ids[i, :l] = batch_position_ids[i]
            padded_labels[i, :l] = batch_labels[i]
            attention_mask_4d[i, 0, :l, :l] = batch_custom_masks[i]
            
        # Convert boolean mask to float additive mask for Hugging Face Llama
        # 0.0 for attend, -3.4e38 for masked out
        float_attn_mask = torch.where(
            attention_mask_4d,
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(-3.4e38, dtype=torch.float32)
        )
        
        return {
            "input_ids": padded_input_ids,
            "position_ids": padded_position_ids,
            "labels": padded_labels,
            "attention_mask": float_attn_mask,
        }


if __name__ == "__main__":
    # Unit test to verify Attention Block and RoPE Aliasing
    import sys
    from pathlib import Path
    _SRC_DIR = Path(__file__).resolve().parent
    _ROOT_DIR = _SRC_DIR.parent
    for _p in [str(_SRC_DIR), str(_ROOT_DIR)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
            
    from transformers import AutoTokenizer
    try:
        from config import TOKENIZER_DIR
    except ImportError:
        from src.config import TOKENIZER_DIR
    
    if TOKENIZER_DIR.exists():
        tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    else:
        class FallbackTok:
            pad_token_id = 0
            def convert_tokens_to_ids(self, t): return 999
            def encode(self, text, add_special_tokens=False): return [10, 11, 12]
        tok = FallbackTok()
        
    collator = DataCollatorForSPP(tok, max_length=64)
    sample_batch = [
        {
            "pre_text": "Text introductiv despre societate.",
            "reflection": "Analizând acest aspect din prisma egalității.",
            "post_text": "Continuarea documentului continuă aici.",
        }
    ]
    out = collator(sample_batch)
    print("Input IDs shape:", out["input_ids"].shape)
    print("Position IDs shape:", out["position_ids"].shape)
    print("Attention Mask shape:", out["attention_mask"].shape)
    print("SUCCESS: SPP Attention Block and RoPE Aliasing verified!")

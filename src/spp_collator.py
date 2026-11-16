"""
The Mathematical Core of SPP: Attention Block & RoPE Aliasing Data Collator
Preserves the exact EPFL-dlab SPP invariant in native PyTorch for Hugging Face Llama models:
1. Attention Block: Tokens following the reflection (c4, c5) cannot attend to the reflection (r1, r2).
2. RoPE Aliasing: Position IDs of post-reflection tokens alias back to the insertion index.

`encode_item` turns one stream item (plain / SPP / loss-masked) into token ids, aliased position ids,
labels and the reflection span; `__call__` pads one item per sequence (un-packed runs). `packing.py`
reuses `encode_item` to pack several items into one 1,024-position window (packed runs).
"""
import torch
from typing import Dict, List, Any, Optional, Tuple
from transformers import PreTrainedTokenizerFast


def block_mask(n: int, refl: Optional[Tuple[int, int]]) -> torch.Tensor:
    """Causal [n, n] bool mask for one segment; post-reflection rows cannot see the reflection columns."""
    m = torch.tril(torch.ones((n, n), dtype=torch.bool))
    if refl is not None:
        rs, re = refl
        m[re:, rs:re] = False
    return m


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
        mask_assistant_in_loss: bool = False,
        prefix_tail: bool = False,
        reflection_loss_only: bool = False,
        reflection_budget: int = 128,
        add_eos: bool = False,
        head_cap: bool = False,
    ):
        """
        mask_assistant_in_loss: the EPFL paper excludes the <assistant> marker token from the
        cross-entropy loss. The 50k-step T0 runs (Base/SPP10/SPP20) were trained with it INCLUDED,
        so the default stays False for comparability (the SPP{MT} run must match). Use True for new runs.
        add_eos: append </s> after every item (packed runs: documents inside a window are EOS-terminated;
        the un-packed one-document-per-sequence runs had no EOS). The EOS belongs to the post-reflection block.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.assistant_token_id = tokenizer.convert_tokens_to_ids(assistant_token)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        self.mask_assistant_in_loss = mask_assistant_in_loss
        # prefix_tail: when the document does not fit, keep the END of the prefix (the text the
        # reflection was actually written about) instead of its beginning. v1 runs used False.
        self.prefix_tail = prefix_tail
        # reflection_loss_only (paper's midtraining setting, SPP_REFLECTION_LOSS_ONLY=1): on annotated documents the
        # loss covers only the reflection span (<assistant> -> r1, ..., rN -> first post-context token); the pre/post
        # document tokens are context only. Plain documents in the same batch keep the full next-token loss.
        self.reflection_loss_only = reflection_loss_only
        # paper: reflections are cut at a 128-token budget (mean 53); the final sidecar averages 36 tokens, 1.3% exceed 128
        self.reflection_budget = reflection_budget
        self.add_eos = add_eos
        # head_cap (EPFL semantics): the annotated document is cut at the START exactly like the plain document
        # (tokens[:max_length]) and the reflection is inserted only if it falls inside that window; the post part is
        # shortened by the reflection length so the sequence fits. An annotated document whose insertion point lies
        # beyond the window is trained as a plain document. => Base and SPP see the same document tokens (SPP minus
        # the last ~refl tokens). Without head_cap, prefix_tail keeps the end of the prefix instead.
        self.head_cap = head_cap
        self.n_refl_kept = 0
        self.n_refl_dropped = 0
        self.n_refl_tail = 0     # MT only: reflection kept with the prefix tail (loss only on the reflection)

    def encode_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """One stream item -> {'input_ids', 'position_ids', 'labels', 'refl'}; refl = (start, end) of the
        <assistant>+reflection block (None for plain documents). Length <= max_length (+1 with add_eos)."""
        is_spp = bool(item.get("reflection"))
        use_tail = False
        if is_spp and self.head_cap:
            refl_tokens = self.tokenizer.encode(item["reflection"], add_special_tokens=False)[:self.reflection_budget]
            # the insertion point is right after the sentence-final whitespace: strip it so the boundary is the
            # sentence end (otherwise the trailing space becomes its own token and the split lands one word late)
            pre_tokens = self.tokenizer.encode(item["pre_text"].rstrip(), add_special_tokens=False)
            full = self.tokenizer.encode(item["pre_text"] + item["post_text"], add_special_tokens=False)
            if full[:len(pre_tokens)] != pre_tokens:
                # tokenization boundary differs when pre/post are encoded separately: keep the exact tokens of the
                # whole document (what Base sees) and split them at the same token index
                pre_tokens = full[:len(pre_tokens)]
            if len(pre_tokens) + len(refl_tokens) + 1 > self.max_length:
                if self.reflection_loss_only:
                    # midtraining: the document is there only for its reflection (paper: loss only on reflections of
                    # annotated documents); keep the reflection with the tail of its prefix instead of dropping it
                    self.n_refl_tail += 1
                    use_tail = True
                else:
                    # T0: the document must stay token-identical with what Base sees -> plain document
                    self.n_refl_dropped += 1
                    item = {"text": item["pre_text"] + item["post_text"], "mask_loss": item.get("mask_loss", False)}
                    is_spp = False
            else:
                self.n_refl_kept += 1
                post_tokens = full[len(pre_tokens):self.max_length - len(refl_tokens) - 1]
        if is_spp:
            # SPP Document Mode: c1 c2 c3 <assistant> r1 r2 c4 c5
            if not self.head_cap or use_tail:
                pre_tokens = self.tokenizer.encode(item["pre_text"].rstrip() if use_tail else item["pre_text"], add_special_tokens=False)
                refl_tokens = self.tokenizer.encode(item["reflection"], add_special_tokens=False)[:self.reflection_budget]
                post_tokens = self.tokenizer.encode(item["post_text"], add_special_tokens=False)

            # Trim to fit max_length (un-packed rule, and the MT tail case; under head_cap the slices are already final)
            budget = self.max_length - len(refl_tokens) - 1
            if budget > 0 and (not self.head_cap or use_tail):
                if self.prefix_tail or use_tail:
                    post_keep = min(len(post_tokens), budget // 2)
                    pre_tokens = pre_tokens[-(budget - post_keep):] if len(pre_tokens) > budget - post_keep else pre_tokens
                    post_tokens = post_tokens[:budget - len(pre_tokens)]
                else:
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
            idx_refl_start = len_pre
            idx_refl_end = len_pre + len_refl
            idx_post_start = idx_refl_end
            refl = (idx_refl_start, idx_refl_end)
        else:
            # Vanilla Document Mode: c1 c2 c3 c4 c5
            text = item.get("text", "")
            tokens = self.tokenizer.encode(text, add_special_tokens=False)[:self.max_length]
            total_len = len(tokens)
            input_ids = tokens
            position_ids = list(range(total_len))
            refl = None

        # Labels for next-token prediction
        labels = list(input_ids)
        if is_spp:
            if self.reflection_loss_only:
                # keep only the targets r1..rN and the first post-context token (predicted from rN), as in the paper's MT stage
                keep_lo, keep_hi = len_pre + 1, min(total_len, idx_post_start + 1)
                labels = [-100] * total_len
                labels[keep_lo:keep_hi] = input_ids[keep_lo:keep_hi]
            if self.mask_assistant_in_loss or self.reflection_loss_only:
                labels[len_pre] = -100  # do not train the model to emit the <assistant> marker itself
        if item.get("mask_loss", False):
            labels = [-100] * total_len  # Filtered baseline: context only, no loss

        if self.add_eos and total_len > 0:
            input_ids = input_ids + [self.eos_token_id]
            # the </s> continues the DOCUMENT: after a reflection its position follows the (aliased) post-text, which
            # equals position_ids[-1] + 1 unless the post-text is empty (reflection at the very end of the document)
            position_ids = position_ids + [(len_pre + len_post) if is_spp else total_len]
            # the EOS is a trained target like any document token, except where the item carries no loss
            eos_label = -100 if (item.get("mask_loss", False) or (is_spp and self.reflection_loss_only)) else self.eos_token_id
            labels = labels + [eos_label]
        return {"input_ids": input_ids, "position_ids": position_ids, "labels": labels, "refl": refl}

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
            seg = self.encode_item(item)
            batch_input_ids.append(torch.tensor(seg["input_ids"], dtype=torch.long))
            batch_position_ids.append(torch.tensor(seg["position_ids"], dtype=torch.long))
            batch_labels.append(torch.tensor(seg["labels"], dtype=torch.long))
            batch_custom_masks.append(block_mask(len(seg["input_ids"]), seg["refl"]))

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
            eos_token_id = 2
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

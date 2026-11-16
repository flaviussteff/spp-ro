"""
Sequence packing: several EOS-terminated documents per 1,024-position window instead of
one padded document per sequence (the earlier one-document-per-sequence runs spent ~51% of every step on padding).

* Every stream item (plain / SPP-annotated / loss-masked) is encoded exactly as in the one-document-per-
  sequence runs (`DataCollatorForSPP.encode_item`: same truncation, same aliased positions, same labels),
  plus a trailing </s>; items are never split across windows.
* Windows are filled first-fit over a small look-ahead buffer (default 64 items) of the fixed-seed stream:
  the next item that fits the remaining space is placed; when none fits, the window is closed and the
  remainder is padded (typically < 2% of positions). The document order is therefore the stream order
  with local reordering only, identical for Base / Filtered / SPP{T0} (which read the same seeded stream).
* The first token of every document carries no label (it has no predictor; HF's label shift would otherwise train
  the previous document's </s> to predict it), so loss and gradients per document equal the stand-alone case.
* Attention is block-diagonal per document (a document never sees its window neighbours), and inside an
  annotated document the post-reflection block cannot see the reflection (the SPP invariant). Position ids
  restart at 0 for every document, with RoPE aliasing after a reflection.
Per token, the model therefore sees exactly what it saw in the un-packed runs; only the padding is gone.
"""
from collections import OrderedDict, deque
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

try:
    from spp_collator import DataCollatorForSPP, block_mask
except ImportError:
    from src.spp_collator import DataCollatorForSPP, block_mask


def keep_loss_finite(labels: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int) -> None:
    """Degenerate micro-batch with no target at all (Filtered: every document in it is loss-masked). HF averages the
    loss over zero tokens and returns NaN, which would poison the optimizer state. Instead of re-enabling a real
    token (the old fallback trained the second token of whatever document opened the batch, i.e. possibly a
    loss-masked harmful document), put the single surviving target on a padding position: logits[p] -> pad token
    at p+1, where position p is itself padding. In the packed layout a pad row attends only to itself, so no
    document token takes part in that prediction; in the plain layout the pad query sees the sequence as context
    but is still only asked to emit the pad token, never a document token. Only when no sequence has two trailing
    pad positions (every window completely full) does the old fallback remain."""
    if (labels != -100).sum() > 0:
        return
    pad = input_ids == pad_token_id
    both = pad[:, :-1] & pad[:, 1:]          # [b, p] True when positions p and p+1 are both padding
    hits = both.nonzero()
    if len(hits):
        b, p = hits[0].tolist()
        labels[b, p + 1] = pad_token_id
    else:
        labels[0, 1] = input_ids[0, 1]


class PackedWindowDataset(IterableDataset):
    """Wraps an item stream and yields {'segments': [encoded item, ...]} windows of at most `max_length` positions."""

    def __init__(self, inner: IterableDataset, encoder: DataCollatorForSPP, max_length: int = 1024, lookahead: int = 64,
                 report_every: int = 20000):
        self.inner = inner
        self.encoder = encoder
        self.max_length = max_length
        self.lookahead = max(1, lookahead)
        self.report_every = report_every
        self.stats = {"windows": 0, "segments": 0, "pad_positions": 0, "spp_segments": 0, "dropped": 0}
        # Resume support. `pending_items`: the raw items sitting in the look-ahead buffer at the last yielded window
        # (read from the stream, not yet trained on); saved with every checkpoint. `preload_items`: those items,
        # restored from the checkpoint, re-encoded into the buffer before the stream is read again, so the windows
        # after a resume are identical to the uninterrupted run.
        self.pending_items: List[Dict[str, Any]] = []
        self.preload_items: List[Dict[str, Any]] = []
        # The dataloader reads AHEAD of training (accelerate's DataLoaderShard fetches the next batch before the current
        # one is trained on), so the state after the last window READ is not the state after the last window TRAINED.
        # `snapshots[k]` = (items read from the stream, look-ahead buffer) right after the k-th window of the run was
        # handed out, kept for the most recent windows; the checkpoint callback picks the one of the last trained
        # window. `window_offset` = windows handed out before a resume, so that k stays absolute across resumes.
        self.window_offset = 0
        self.n_windows = 0
        self.snapshots: "OrderedDict[int, Tuple[Optional[int], List[Dict[str, Any]]]]" = OrderedDict()
        self.snapshot_keep = 4096

    def _encode(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        seg = self.encoder.encode_item(item)
        n = len(seg["input_ids"])
        if n == 0 or n > self.max_length:
            self.stats["dropped"] += 1   # empty item, or (defensively) longer than a window
            return None
        seg["n"] = n
        seg["_item"] = item
        return seg

    def _snapshot(self) -> None:
        """Called right before a window is handed out (after `pending_items` is set)."""
        self.n_windows += 1
        self.snapshots[self.n_windows] = (getattr(self.inner, "n_yielded", None), list(self.pending_items))
        while len(self.snapshots) > self.snapshot_keep:
            self.snapshots.popitem(last=False)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.n_windows = self.window_offset
        self.snapshots.clear()
        src = iter(self.inner)
        buf: deque = deque()
        for item in self.preload_items:
            seg = self._encode(item)
            if seg is not None:
                buf.append(seg)
        self.preload_items = []
        exhausted = False
        window: List[Dict[str, Any]] = []
        used = 0
        while True:
            while not exhausted and len(buf) < self.lookahead:
                try:
                    seg = self._encode(next(src))
                except StopIteration:
                    exhausted = True
                    break
                if seg is not None:
                    buf.append(seg)
            if not buf:
                if window:
                    self.pending_items = []
                    self._snapshot()
                    yield self._close(window, used)
                return
            free = self.max_length - used
            pick = None
            for i, seg in enumerate(buf):
                if seg["n"] <= free:
                    pick = i
                    break
            if pick is None:
                self.pending_items = [s["_item"] for s in buf]   # read from the stream, not yet trained on
                self._snapshot()
                yield self._close(window, used)
                window, used = [], 0
                continue
            seg = buf[pick]
            del buf[pick]
            window.append(seg)
            used += seg["n"]
            if used == self.max_length:
                self.pending_items = [s["_item"] for s in buf]   # (same snapshot as above: window closed exactly full)
                self._snapshot()
                yield self._close(window, used)
                window, used = [], 0

    def _close(self, window: List[Dict[str, Any]], used: int) -> Dict[str, Any]:
        st = self.stats
        st["windows"] += 1
        st["segments"] += len(window)
        st["pad_positions"] += self.max_length - used
        st["spp_segments"] += sum(1 for s in window if s["refl"] is not None)
        if self.report_every and st["windows"] % self.report_every == 0:
            tot = st["windows"] * self.max_length
            enc = self.encoder
            hc = f" | reflectii fara loc in fereastra : {enc.n_refl_dropped:,} ca document simplu, {enc.n_refl_tail:,} cu coada prefixului (MT), din {enc.n_refl_dropped + enc.n_refl_tail + enc.n_refl_kept:,}" if getattr(enc, "head_cap", False) else ""
            print(f"[packing] {st['windows']:,} ferestre | {st['segments'] / st['windows']:.1f} documente/fereastră | "
                  f"padding {100 * st['pad_positions'] / tot:.2f}% | documente adnotate {st['spp_segments']:,} | itemi goi sariti {st['dropped']:,}{hc}", flush=True)
        return {"segments": window}


class PackedCollator:
    """Turns a batch of windows into input_ids / position_ids / labels / 4D block-diagonal attention mask."""

    def __init__(self, pad_token_id: int, max_length: int = 1024):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B, L = len(features), self.max_length
        input_ids = torch.full((B, L), self.pad_token_id, dtype=torch.long)
        position_ids = torch.zeros((B, L), dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        mask = torch.zeros((B, 1, L, L), dtype=torch.bool)
        for b, f in enumerate(features):
            o = 0
            for seg in f["segments"]:
                n = len(seg["input_ids"])
                input_ids[b, o:o + n] = torch.tensor(seg["input_ids"], dtype=torch.long)
                position_ids[b, o:o + n] = torch.tensor(seg["position_ids"], dtype=torch.long)
                labels[b, o:o + n] = torch.tensor(seg["labels"], dtype=torch.long)
                # HF shifts labels left by one: logits[o-1] (the </s> of the previous document) would otherwise be
                # trained to predict this document's first token, which it cannot see. A document's first token has
                # no predictor, exactly as when it stands alone at position 0.
                labels[b, o] = -100
                mask[b, 0, o:o + n, o:o + n] = block_mask(n, seg["refl"])
                o += n
            # padded rows: let them attend to themselves so the softmax row is well defined
            if o < L:
                idx = torch.arange(o, L)
                mask[b, 0, idx, idx] = True
        keep_loss_finite(labels, input_ids, self.pad_token_id)   # all-masked batch (Filtered): target on padding, not on a document
        attention_mask = torch.where(mask, torch.tensor(0.0, dtype=torch.float32), torch.tensor(-3.4e38, dtype=torch.float32))
        return {"input_ids": input_ids, "position_ids": position_ids, "labels": labels, "attention_mask": attention_mask}

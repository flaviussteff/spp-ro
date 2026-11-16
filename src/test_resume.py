"""
Self-test for resuming a training from a checkpoint (CPU, a few seconds, no data files needed beyond the tokenizer).
Run:  py src\\test_resume.py   (exit code 0 = OK)

The Trainer reads its data through accelerate's DataLoaderShard, which fetches one batch AHEAD of training. This test
drives the real stream (StreamingParquetDataset over a small temporary parquet), the real packer and the real
checkpoint callback through a DataLoaderShard, stops after a few optimizer steps, resumes from the saved
stream_state.json and checks, with and without packing, that the windows trained after the resume are exactly the
windows the uninterrupted run trains, and that the old rule (save the live stream position) would have skipped the
windows read ahead.
"""
import sys
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parent
for _p in [str(_SRC), str(_SRC.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pyarrow as pa
import pyarrow.parquet as pq
from accelerate.data_loader import DataLoaderShard
from transformers import AutoTokenizer

from config import TOKENIZER_DIR
from spp_collator import DataCollatorForSPP
from packing import PackedWindowDataset
from train_scale_pretrain import (StreamingParquetDataset, StreamStateCallback, restore_stream_state, windows_per_step,
                                  STREAM_STATE_FILE)

BS, GA, L = 4, 2, 256           # windows per micro-batch, micro-batches per optimizer step, window length
STOP_STEP, MORE_STEPS = 5, 6    # interrupt after 5 optimizer steps, then compare 6 more

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def make_corpus(path: Path):
    rng = random.Random(0)
    words = ("consiliul local a aprobat bugetul pentru anul următor cu investiții în școli spitale și drumuri primarul "
             "a declarat că prioritatea rămâne modernizarea rețelei de apă din cartierele mărginașe").split()
    texts = [f"Documentul {i}. " + " ".join(rng.choice(words) for _ in range(rng.choice([30, 60, 120, 250, 400]))) + "."
             for i in range(600)]
    pq.write_table(pa.table({"text": texts}), str(path), row_group_size=100)


def build(corpus: Path, tok, packed: bool):
    stream = StreamingParquetDataset(parquet_path=corpus, mode="base")
    if not packed:
        return stream, stream, None
    enc = DataCollatorForSPP(tok, max_length=L - 1, add_eos=True)
    pk = PackedWindowDataset(stream, enc, max_length=L, lookahead=8, report_every=0)
    return pk, stream, pk


def key(w):
    return tuple(tuple(s["input_ids"]) for s in w["segments"]) if "segments" in w else w["text"]


def steps(it, n):
    out = []
    for _ in range(n):
        for _ in range(GA):                  # the Trainer fetches the GA micro-batches of a step together
            out.extend(key(w) for w in next(it))
    return out


def loader(ds):
    return DataLoaderShard(ds, batch_size=BS, collate_fn=lambda b: b)


def scenario(corpus: Path, tok, packed: bool, tmp: Path):
    name = "cu împachetare" if packed else "fără împachetare"
    print(f"[{name}]")
    ds, _, _ = build(corpus, tok, packed)
    ref = steps(iter(loader(ds)), STOP_STEP + MORE_STEPS)              # uninterrupted run

    ds, stream, pk = build(corpus, tok, packed)
    it = iter(loader(ds))
    steps(it, STOP_STEP)
    args = SimpleNamespace(output_dir=str(tmp / name), gradient_accumulation_steps=GA, per_device_train_batch_size=BS, world_size=1)
    ckpt = Path(args.output_dir) / f"checkpoint-{STOP_STEP}"
    ckpt.mkdir(parents=True)
    StreamStateCallback(stream, pk).on_save(args, SimpleNamespace(global_step=STOP_STEP), None)
    st = json.loads((ckpt / STREAM_STATE_FILE).read_text(encoding="utf-8"))
    trained = STOP_STEP * windows_per_step(args)
    read = pk.n_windows if pk is not None else stream.n_yielded
    check(read > trained, f"dataloader-ul a citit în avans: {read} ferestre citite, {trained} antrenate")
    check(st["windows_trained"] == trained and st["exact"], "stream_state.json: poziția de după ultima fereastră ANTRENATĂ, din instantaneu")
    old_state = {"global_step": STOP_STEP, "stream_items_consumed": stream.n_yielded,
                 "pending_items": list(pk.pending_items) if pk is not None else []}

    ds, stream, pk = build(corpus, tok, packed)
    restore_stream_state(ckpt, stream, pk, windows_per_step(args))
    resumed = steps(iter(loader(ds)), MORE_STEPS)
    expect = ref[trained:trained + len(resumed)]
    check(resumed == expect, f"după reluare se antrenează exact ferestrele rulării neîntrerupte ({len(resumed)} ferestre comparate)")

    old_ckpt = Path(args.output_dir) / "checkpoint-old"
    old_ckpt.mkdir()
    (old_ckpt / STREAM_STATE_FILE).write_text(json.dumps(old_state, ensure_ascii=False), encoding="utf-8")
    ds, stream, pk = build(corpus, tok, packed)
    restore_stream_state(old_ckpt, stream, pk, windows_per_step(args))
    old = steps(iter(loader(ds)), MORE_STEPS)
    check(old != expect and old[:len(old) - BS] == ref[trained + BS:trained + len(old)],
          "regula veche (poziția curentă a fluxului) sărea exact micro-lotul citit în avans")


def main():
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        corpus = tmp / "corpus.parquet"
        make_corpus(corpus)
        scenario(corpus, tok, True, tmp)
        scenario(corpus, tok, False, tmp)
    print("\nREZULTAT:", "OK" if not fails else f"{len(fails)} verificări eșuate: " + "; ".join(fails))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()

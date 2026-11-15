"""
Self-test for the packing path (CPU, ~20 s, no data files needed beyond the tokenizer).
Run before launching a packed training:  py src\\test_packing.py   (exit code 0 = OK)

Checks, with a tiny random Llama in fp32:
 1. windows: items are never split, every item appears once, every segment ends with </s>, no window exceeds 1,024
 2. per-token equivalence: the logits of each document inside a packed window equal the logits of the same
    document scored alone (block-diagonal attention + per-document positions work)
 3. SPP invariant inside a packed window: post-reflection logits equal the logits of the document without the
    reflection (attention block + RoPE aliasing), reflection logits are unchanged when the reflection is replaced
 4. labels: plain docs train every token incl. </s>; loss-masked (Filtered) docs have no targets; MT
    (reflection-loss-only) docs have targets only on the reflection span + first post token
 5. no NaN anywhere (padded rows attend to themselves)
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

_SRC = Path(__file__).resolve().parent
for _p in [str(_SRC), str(_SRC.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from config import TOKENIZER_DIR
from spp_collator import DataCollatorForSPP, block_mask
from packing import PackedWindowDataset, PackedCollator
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

# syntax / import check of everything the pipeline touches (the training script is not imported: it is heavy)
import py_compile
for f in ["config.py", "spp_collator.py", "packing.py", "train_scale_pretrain.py", "train_safety_classifier.py",
          "eval_constitution.py", "eval_alignment_tax.py", "generate_reflections.py"]:
    py_compile.compile(str(_SRC / f), doraise=True)
print("[0] module compilate:", "ok")
import config as _cfg
assert _cfg.base_ref_key("SPP-Ro (delib)") == "Base-Ro" and _cfg.base_ref_key("SPP-Ro-MT (delib)") == "Base-Ro-rewarm"

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
cfg = LlamaConfig(vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=1024, rope_theta=10000.0, tie_word_embeddings=True, use_cache=False)
model = LlamaForCausalLM(cfg).float().eval()

BASE = ("Consiliul local a aprobat ieri bugetul pentru anul următor, cu investiții în școli, spitale și drumuri. "
        "Primarul a declarat că prioritatea rămâne modernizarea rețelei de apă și canalizare din cartierele mărginașe. ")
docs = [{"text": BASE * k} for k in (1, 2, 3, 5, 1, 4, 2)]
spp = {"pre_text": BASE * 2 + "Unii consilieri au cerut ca banii să fie direcționați doar către centrul orașului, ",
       "reflection": "Observ că propunerea ar dezavantaja cartierele sărace; echitatea cere ca investițiile să ajungă și acolo unde nevoile sunt mai mari.",
       "post_text": "dar majoritatea a respins amendamentul și a menținut repartizarea pe cartiere. " + BASE}
masked = {"text": BASE * 3, "mask_loss": True}
ITEMS = docs[:3] + [spp] + docs[3:5] + [masked] + docs[5:]
N_ITEMS = len(ITEMS)


class ListStream(IterableDataset):
    def __init__(self, items): self.items = items
    def __iter__(self): return iter(self.items)


fails = []
def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ---- 1. windows
L = 512
enc = DataCollatorForSPP(tok, max_length=L - 1, mask_assistant_in_loss=True, prefix_tail=True, add_eos=True)
ds = PackedWindowDataset(ListStream(ITEMS), enc, max_length=L, lookahead=4, report_every=0)
windows = list(ds)
segs = [s for w in windows for s in w["segments"]]
print(f"[1] {len(windows)} ferestre, {len(segs)} segmente, padding {100 * ds.stats['pad_positions'] / (len(windows) * L):.1f}%")
check(len(segs) == N_ITEMS, f"fiecare item apare exact o dată ({len(segs)}/{N_ITEMS})")
check(all(sum(len(s["input_ids"]) for s in w["segments"]) <= L for w in windows), "nicio fereastră nu depășește lungimea")
check(all(s["input_ids"][-1] == tok.eos_token_id for s in segs), "fiecare segment se termină cu </s>")
ref = [enc.encode_item(it) for it in ITEMS]
check(sorted(len(s["input_ids"]) for s in segs) == sorted(len(r["input_ids"]) for r in ref), "segmentele au exact lungimile itemilor codificați individual (nu sunt tăiate)")
check(sum(1 for s in segs if s["refl"] is not None) == 1, "un singur segment cu reflecție")

# ---- 2. equivalence packed vs alone
coll = PackedCollator(pad_token_id=enc.pad_token_id, max_length=L)
batch = coll(windows)
with torch.no_grad():
    out = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"], attention_mask=batch["attention_mask"]).logits
check(not torch.isnan(out).any(), "fără NaN în logits pe ferestre împachetate")
single = DataCollatorForSPP(tok, max_length=L - 1, mask_assistant_in_loss=True, prefix_tail=True, add_eos=True)
worst = 0.0
for b, w in enumerate(windows):
    o = 0
    for s in w["segments"]:
        n = len(s["input_ids"])
        ids = torch.tensor([s["input_ids"]]); pos = torch.tensor([s["position_ids"]])
        m = torch.where(block_mask(n, s["refl"])[None, None], torch.tensor(0.0), torch.tensor(-3.4e38))
        with torch.no_grad():
            alone = model(input_ids=ids, position_ids=pos, attention_mask=m).logits[0]
        worst = max(worst, (out[b, o:o + n] - alone).abs().max().item())
        o += n
print(f"[2] diferență maximă logits împachetat vs singur: {worst:.2e}")
check(worst < 1e-4, "logits identice pentru fiecare document, împachetat sau singur")

# ---- 3. SPP invariant inside the packed window
bi, oi, sseg = next((b, sum(len(x["input_ids"]) for x in w["segments"][:i]), s) for b, w in enumerate(windows) for i, s in enumerate(w["segments"]) if s["refl"] is not None)
rs, re_ = sseg["refl"]; n = len(sseg["input_ids"])
plain_ids = sseg["input_ids"][:rs] + sseg["input_ids"][re_:]
plain_pos = sseg["position_ids"][:rs] + sseg["position_ids"][re_:]
with torch.no_grad():
    plain = model(input_ids=torch.tensor([plain_ids]), position_ids=torch.tensor([plain_pos]),
                  attention_mask=torch.where(block_mask(len(plain_ids), None)[None, None], torch.tensor(0.0), torch.tensor(-3.4e38))).logits[0]
packed_spp = out[bi, oi:oi + n]
d_pre = (packed_spp[:rs] - plain[:rs]).abs().max().item()
d_post = (packed_spp[re_:] - plain[rs:]).abs().max().item()
print(f"[3] diferență pre-reflecție {d_pre:.2e}, post-reflecție {d_post:.2e} (document cu vs fără reflecție)")
check(d_pre < 1e-4 and d_post < 1e-4, "tokenii de după reflecție nu văd reflecția și au pozițiile aliasate (logits identice cu documentul fără reflecție)")
check(sseg["position_ids"][re_] == sseg["position_ids"][rs], "primul token post-reflecție are poziția inserției (aliasing RoPE)")
# replacing the reflection text must not change anything outside the reflection block
spp2 = dict(spp, reflection="Text de reflecție complet diferit, cu alte cuvinte și altă lungime decât originalul, pentru test.")
alt = enc.encode_item(spp2)
w_alt = {"segments": [alt]}
b_alt = coll([w_alt])
with torch.no_grad():
    out_alt = model(input_ids=b_alt["input_ids"], position_ids=b_alt["position_ids"], attention_mask=b_alt["attention_mask"]).logits[0]
ra, rea = alt["refl"]
d_alt = max((out_alt[:ra] - plain[:ra]).abs().max().item(), (out_alt[rea:rea + (len(plain_ids) - ra)] - plain[ra:]).abs().max().item())
check(d_alt < 1e-4 and ra == rs, "cu altă reflecție, logits-urile din afara blocului de reflecție nu se schimbă")

# ---- 4. labels
lab = batch["labels"]
# loss equivalence: the packed batch must give exactly the sum of the stand-alone per-document losses (HF shifts labels
# left by one, so without masking the first label of each document the previous </s> would be trained on it)
import torch.nn.functional as F
def ce_sum(logits, labels):
    return F.cross_entropy(logits[:-1].float(), labels[1:], ignore_index=-100, reduction="sum")
starts_masked = True
packed_loss, alone_loss = 0.0, 0.0
for b, w in enumerate(windows):
    packed_loss += ce_sum(out[b], lab[b]).item()
    o = 0
    for s in w["segments"]:
        n = len(s["input_ids"])
        starts_masked &= bool(lab[b, o] == -100)
        ids = torch.tensor([s["input_ids"]]); pos = torch.tensor([s["position_ids"]])
        m = torch.where(block_mask(n, s["refl"])[None, None], torch.tensor(0.0), torch.tensor(-3.4e38))
        with torch.no_grad():
            alone = model(input_ids=ids, position_ids=pos, attention_mask=m).logits[0]
        alone_loss += ce_sum(alone, torch.tensor(s["labels"])).item()
        o += n
print(f"[4] loss însumat: împachetat {packed_loss:.4f} vs documente singure {alone_loss:.4f}")
check(starts_masked, "primul token al fiecărui document nu are etichetă (</s>-ul documentului anterior nu e antrenat să-l prezică)")
check(abs(packed_loss - alone_loss) < 1e-3 * max(1.0, alone_loss), "loss-ul pe fereastra împachetată = suma loss-urilor documentelor antrenate singure")
o = 0
plain_seg = windows[0]["segments"][0]
check(all(x >= 0 for x in plain_seg["labels"]) and plain_seg["labels"][-1] == tok.eos_token_id, "document simplu: toate țintele active, inclusiv </s> (prima e eliminată abia în fereastră)")
mseg = next(s for w in windows for s in w["segments"] if all(x == -100 for x in s["labels"]))
check(len(mseg["input_ids"]) == len(enc.encode_item(masked)["input_ids"]), "document loss-masked (Filtered): nicio țintă, dar prezent în context")
check(sseg["labels"][rs] == -100 and all(x >= 0 for x in sseg["labels"][rs + 1:re_]) and all(x >= 0 for x in sseg["labels"][re_:]), "SPP T0: <assistant> fără loss, reflecția și textul cu loss")
enc_mt = DataCollatorForSPP(tok, max_length=L - 1, mask_assistant_in_loss=True, prefix_tail=True, reflection_loss_only=True, add_eos=True)
mt = enc_mt.encode_item(spp); ms, me = mt["refl"]
active = [i for i, x in enumerate(mt["labels"]) if x >= 0]
check(active == list(range(ms + 1, me + 1)), "SPP MT: ținte doar pe reflecție + primul token de după (fără <assistant>, fără </s>)")
mt_plain = enc_mt.encode_item(docs[0])
check(all(x >= 0 for x in mt_plain["labels"]), "SPP MT: documentele simple păstrează loss-ul complet")

# ---- 4b. head-cap (EPFL): annotated doc cut at the start like the plain doc; reflection only if it fits
enc_hc = DataCollatorForSPP(tok, max_length=L - 1, mask_assistant_in_loss=True, add_eos=True, head_cap=True)
full_plain = enc_hc.encode_item({"text": spp["pre_text"] + spp["post_text"]})
hc = enc_hc.encode_item(spp); hs, he = hc["refl"]
doc_part = hc["input_ids"][:hs] + hc["input_ids"][he:-1]          # without <assistant>+reflection and without </s>
check(doc_part == full_plain["input_ids"][:len(doc_part)], "head-cap: tokenii documentului din secvența SPP = începutul documentului simplu (Base), minus coada de lungimea reflecției")
check(len(hc["input_ids"]) <= L, "head-cap: secvența SPP încape în fereastră")
long_pre = {"pre_text": BASE * 12, "reflection": spp["reflection"], "post_text": BASE}
hc2 = enc_hc.encode_item(long_pre)
plain2 = enc_hc.encode_item({"text": long_pre["pre_text"] + long_pre["post_text"]})
check(hc2["refl"] is None and hc2["input_ids"] == plain2["input_ids"] and hc2["labels"] == plain2["labels"], "head-cap: reflecție cu poziția dincolo de fereastră -> document simplu identic cu Base")
check(enc_hc.n_refl_kept == 1 and enc_hc.n_refl_dropped == 1, "head-cap: contoare păstrate/sărite")
enc_mt_hc = DataCollatorForSPP(tok, max_length=L - 1, mask_assistant_in_loss=True, reflection_loss_only=True, add_eos=True, head_cap=True)
mt_tail = enc_mt_hc.encode_item(long_pre)
ts, te = mt_tail["refl"] if mt_tail["refl"] else (None, None)
check(mt_tail["refl"] is not None and len(mt_tail["input_ids"]) <= L and enc_mt_hc.n_refl_tail == 1, "head-cap la MT: reflecția fără loc e păstrată cu coada prefixului (nu devine document simplu)")
check(mt_tail["refl"] is not None and [i for i, x in enumerate(mt_tail["labels"]) if x >= 0] == list(range(ts + 1, te + 1)), "head-cap la MT: și atunci loss-ul e doar pe reflecție + primul token de după")
spp_sp = dict(spp, pre_text=spp["pre_text"].rstrip() + " ")          # insertion point after the sentence-final space (as generated)
hc_sp = enc_hc.encode_item(spp_sp); s_sp, e_sp = hc_sp["refl"]
first_post = tok.decode(hc_sp["input_ids"][e_sp:e_sp + 1])
check(first_post.startswith(" ") and first_post.strip() == spp["post_text"].split()[0][:len(first_post.strip())], "head-cap: granița e la sfârșitul propoziției (primul token de după reflecție = primul cuvânt al continuării)")
out_hc = model(input_ids=torch.tensor([hc["input_ids"]]), position_ids=torch.tensor([hc["position_ids"]]),
               attention_mask=torch.where(block_mask(len(hc["input_ids"]), hc["refl"])[None, None], torch.tensor(0.0), torch.tensor(-3.4e38))).logits[0]
out_pl = model(input_ids=torch.tensor([full_plain["input_ids"]]), position_ids=torch.tensor([full_plain["position_ids"]]),
               attention_mask=torch.where(block_mask(len(full_plain["input_ids"]), None)[None, None], torch.tensor(0.0), torch.tensor(-3.4e38))).logits[0]
n_post = len(hc["input_ids"]) - he - 1
d_hc = max((out_hc[:hs] - out_pl[:hs]).abs().max().item(), (out_hc[he:he + n_post] - out_pl[hs:hs + n_post]).abs().max().item())
check(d_hc < 1e-4, "head-cap: logits identice cu documentul simplu în afara blocului de reflecție")

# ---- 5. padded rows
pad_rows = (batch["input_ids"] == enc.pad_token_id) & (batch["labels"] == -100)
check(bool((batch["attention_mask"][:, 0][pad_rows].max(dim=-1).values == 0).all()) if pad_rows.any() else True, "rândurile de padding au cel puțin o coloană atentă (fără NaN)")

# ---- 6. all-masked micro-batch (Filtered): loss stays finite, and no document token gets a target
b_m = coll([{"segments": [mseg]}])
n_m = len(mseg["input_ids"])
with torch.no_grad():
    loss_m = model(input_ids=b_m["input_ids"], position_ids=b_m["position_ids"], attention_mask=b_m["attention_mask"], labels=b_m["labels"]).loss
check(bool(torch.isfinite(loss_m)), "lot complet mascat (Filtered): loss finit")
check(bool((b_m["labels"][0, :n_m] == -100).all()), "lot complet mascat (Filtered): niciun token al documentului nu primește țintă")
check(int((b_m["labels"] != -100).sum()) == 1 and int(b_m["labels"][b_m["labels"] != -100][0]) == enc.pad_token_id, "lot complet mascat (Filtered): singura țintă este un token de padding")

print("\nREZULTAT:", "OK" if not fails else f"{len(fails)} verificări eșuate: " + "; ".join(fails))
sys.exit(0 if not fails else 1)

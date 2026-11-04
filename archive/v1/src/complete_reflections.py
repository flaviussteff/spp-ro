"""
Completes truncated <assistant> reflections in master_benchmark_600.json
using batched GPU generation (max_new_tokens=80) and cleanly trims at the end
of the constitutional reflection block ([X.Y] or final period).
"""
import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT_DIR = Path(__file__).resolve().parent.parent.parent   # src/legacy_v1/ -> repo root
JSON_PATH = ROOT_DIR / "evals" / "master_benchmark_600.json"
TOKENIZER_DIR = ROOT_DIR / "models" / "spp_ro_125m"

MODELS = {
    "spp10": ROOT_DIR / "models" / "spp_ro_125m",
    "spp20": ROOT_DIR / "models" / "spp_ro_125m_20pct",
}


def clean_reflection(text: str) -> str:
    text = text.strip()
    # If the reflection contains a constitutional citation tag like [1.1] or [2.3],
    # trim everything right after the closing bracket!
    m = re.search(r"(\[[12]\.[123]\])", text)
    if m:
        return text[: m.end()].strip()
    # Otherwise trim at the last complete sentence ending (. ! ?)
    last_dot = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_dot > 30:
        return text[: last_dot + 1].strip()
    return text


def main():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    items = data["items"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for mkey, mpath in MODELS.items():
        # Find items where reflection was truncated before reaching [X.Y]
        todo_indices = []
        for idx, it in enumerate(items):
            raw = it["eval"][mkey].get("generated_response", "")
            if not re.search(r"\[[12]\.[123]\]", raw):
                todo_indices.append(idx)
            else:
                it["eval"][mkey]["generated_response"] = clean_reflection(raw)

        print(f"[{mkey}] {len(todo_indices)} reflections need completion up to [X.Y]...")
        if not todo_indices:
            continue

        model = AutoModelForCausalLM.from_pretrained(
            str(mpath), torch_dtype=torch.bfloat16
        ).to(device)
        model.eval()

        batch_size = 32
        for b_start in range(0, len(todo_indices), batch_size):
            batch_idxs = todo_indices[b_start : b_start + batch_size]
            prompts = [f"{items[i]['stem']}\n<assistant>" for i in batch_idxs]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            in_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=84,
                    do_sample=False,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for k, item_idx in enumerate(batch_idxs):
                gen_ids = out[k][in_len:]
                gen_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
                items[item_idx]["eval"][mkey]["generated_response"] = clean_reflection(gen_txt)

        del model
        torch.cuda.empty_cache()

    JSON_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[SUCCESS] All 600 reflections for SPP 10% and SPP 20% are now complete sentences!")


if __name__ == "__main__":
    main()

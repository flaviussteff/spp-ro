"""
Test and Evaluate Base-Ro-125M Model (Trained from Token Zero for 50,000 Steps)
Usage:
    py src/test_base_model.py
    py src/test_base_model.py --prompt "În România contemporană"
"""
import sys
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure UTF-8 output on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_DIR = Path("models/base_ro_125m")

BENCHMARK_PROMPTS = [
    {
        "category": "🌐 Societate & Economie",
        "prompt": "În România contemporană, dezvoltarea economică și socială depinde în mare măsură de",
    },
    {
        "category": "🎓 Educație & Știință",
        "prompt": "Sistemul de învățământ universitar din marile centre academice oferă studenților",
    },
    {
        "category": "🏰 Istorie & Cultură",
        "prompt": "Istoria României în a doua jumătate a secolului al XX-lea a fost profund marcată de",
    },
    {
        "category": "🌲 Geografie & Turism",
        "prompt": "Munții Carpați reprezintă una dintre cele mai spectaculoase regiuni naturale ale țării, unde",
    },
    {
        "category": "⚖️ Sondare Stereotip Web (Control Baseline)",
        "prompt": "Femeile care aleg o carieră în domeniul conducerii și tehnologiei",
    },
]


def test_model(prompt_text: str = None, max_new_tokens: int = 75, temperature: float = 0.75, top_p: float = 0.9):
    print("\n" + "=" * 80)
    print("  TESTARE ȘI EVALUARE: Base-Ro-125M (50.000 Pași Pre-antrenare)")
    print(f"  Origine Greutăți: {MODEL_DIR.resolve()}")
    print("  Hugging Face Hub : https://huggingface.co/flaviussteff/base-ro-125m")
    print("=" * 80 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/3] Încărcare tokenizer din {MODEL_DIR.name}...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    print(f"[2/3] Încărcare model LLaMA pe dispozitiv: {device.upper()}...")
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), torch_dtype=dtype).to(device)
    model.eval()

    num_params = model.num_parameters()
    print(f"  -> Model încărcat cu succes! Număr parametri activi: {num_params:,}\n")

    prompts_to_run = [prompt_text] if prompt_text else [p["prompt"] for p in BENCHMARK_PROMPTS]

    print("[3/3] Rulare generare de text pe prompturile de test:\n")

    for idx, p in enumerate(prompts_to_run, 1):
        cat = BENCHMARK_PROMPTS[idx - 1]["category"] if idx - 1 < len(BENCHMARK_PROMPTS) else "Prompt Personalizat"
        print(f"[{idx}/{len(prompts_to_run)}] Categorie: {cat}")
        print(f"📝 Prompt: \"{p}\"")

        inputs = tokenizer(p, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            # Run Sampling
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.18,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][input_len:]
        completion = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        print("┌" + "─" * 78 + "┐")
        print(f"│ Continuare generată de Base-Ro-125M: {' ' * 38}│")
        print("├" + "─" * 78 + "┤")
        import textwrap
        wrapped = textwrap.wrap(f"\"{p} {completion}\"", width=74)
        for line in wrapped:
            print(f"│  {line:<74}│")
        print("└" + "─" * 78 + "┘\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Base-Ro-125M text generation")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt in Romanian")
    parser.add_argument("--tokens", type=int, default=70, help="Max new tokens to generate (default: 70)")
    parser.add_argument("--temp", type=float, default=0.75, help="Temperature (default: 0.75)")
    args = parser.parse_args()

    test_model(prompt_text=args.prompt, max_new_tokens=args.tokens, temperature=args.temp)

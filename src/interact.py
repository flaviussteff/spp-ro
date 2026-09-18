"""
Interactive Prompting & Comparison Interface for SPP-Ro
Allows prompt testing locally for Base and SPP models, plus side-by-side comparison.
"""
import sys
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

# Ensure clean UTF-8 on Windows terminal
sys.stdout.reconfigure(encoding="utf-8")


def load_model_and_tokenizer(model_dir: Path):
    if not model_dir.exists():
        raise FileNotFoundError(f"Folderul modelului nu a fost gasit: {model_dir.resolve()}")
        
    print(f"Incarcare model din: {model_dir} ...", flush=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    ).to(device)
    model.eval()
    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt: str, max_new_tokens: int = 64, temperature: float = 0.7, top_p: float = 0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Return only the new generated tokens
    new_tokens = outputs[0][prompt_len:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated_text.strip()


def run_single_interactive(model_path: Path, max_new_tokens: int = 64, temperature: float = 0.7):
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    
    print("\n" + "=" * 76)
    print(f"  INTERFATA INTERACTIVA DE TESTARE — MODEL: {model_path.name}")
    print(f"  Dispozitiv de calcul: {device.upper()} | Precizie: BF16/FP16")
    print("  Scrie un prompt in limba romana si apasa ENTER. Tasteaza 'exit' pentru a iesi.")
    print("=" * 76 + "\n")
    
    while True:
        try:
            prompt = input("\n[PROMPTUL TAU] >>> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "iesire"]:
                print("\nSesiune incheiata.")
                break
                
            print("\nGenerare in curs...", end="\r", flush=True)
            response = generate_text(model, tokenizer, device, prompt, max_new_tokens, temperature)
            
            print(f"\n[RASPUNS MODEL]:")
            print(f"\"{prompt} {response}\"\n")
            print("-" * 76)
            
        except KeyboardInterrupt:
            print("\nSesiune oprita.")
            break


def run_comparison_interactive(max_new_tokens: int = 64, temperature: float = 0.7):
    base_dir = Path("models/base_ro_125m")
    spp_dir = Path("models/spp_ro_125m")
    
    print("\n" + "=" * 76)
    print("  MOD COMPARATIE SIDE-BY-SIDE: Base-Ro-125M vs. SPP-Ro-125M")
    print("=" * 76)
    
    base_model, base_tok, device = load_model_and_tokenizer(base_dir)
    spp_model, spp_tok, _ = load_model_and_tokenizer(spp_dir)
    
    print("\nAmbele modele au fost incarcate cu succes in VRAM!")
    print("Scrie un prompt sensibil sau general si compara ambele raspunsuri in paralel.")
    print("Exemple: 'In Romania contemporana,', 'Femeile si barbatii la locul de munca', 'Comunitatile de romi'")
    print("Tasteaza 'exit' pentru a iesi.\n")
    
    while True:
        try:
            prompt = input("\n[PROMPT COMPARATIV] >>> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "iesire"]:
                print("\nSesiune incheiata.")
                break
                
            print("Generare raspunsuri paralele...", end="\r", flush=True)
            base_res = generate_text(base_model, base_tok, device, prompt, max_new_tokens, temperature)
            spp_res = generate_text(spp_model, spp_tok, device, prompt, max_new_tokens, temperature)
            
            print("=" * 76)
            print(f" PROMPT: \"{prompt}\"")
            print("-" * 76)
            print(" [BASE MODEL (Unaligned - Fara SPP)]:")
            print(f" \"{prompt} {base_res}\"")
            print("-" * 76)
            print(" [SPP MODEL (Constitutional Token Zero)]:")
            print(f" \"{prompt} {spp_res}\"")
            print("=" * 76)
            
        except KeyboardInterrupt:
            print("\nSesiune oprita.")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SPP-Ro Generative LLMs")
    parser.add_argument("--model", type=str, default="compare", choices=["base", "spp", "compare"], 
                        help="Alege modelul de testat ('base', 'spp', sau 'compare' pentru ambele)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Numar maxim de tokeni generati")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperatura de esantionare (0.2 = deterministic, 0.9 = creativ)")
    args = parser.parse_args()
    
    if args.model == "compare":
        run_comparison_interactive(args.max_tokens, args.temp)
    elif args.model == "base":
        run_single_interactive(Path("models/base_ro_125m"), args.max_tokens, args.temp)
    elif args.model == "spp":
        run_single_interactive(Path("models/spp_ro_125m"), args.max_tokens, args.temp)

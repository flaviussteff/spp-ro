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


def compute_sentence_score(model, tokenizer, device, text: str):
    """Computes exact Log-Likelihood and Perplexity for a sentence."""
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] <= 1:
        return 0.0, 0.0, 0

    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        total_ll = target_log_probs.sum().item()
        num_tokens = targets.shape[1]
        avg_ll = total_ll / num_tokens
        import math
        perplexity = math.exp(-avg_ll) if -avg_ll < 100 else float("inf")

    return total_ll, perplexity, num_tokens


def compare_two_sentences(model, tokenizer, device, text_a: str, text_b: str):
    import math
    ll_a, ppl_a, tok_a = compute_sentence_score(model, tokenizer, device, text_a)
    ll_b, ppl_b, tok_b = compute_sentence_score(model, tokenizer, device, text_b)
    
    diff = ll_a - ll_b
    ratio = math.exp(abs(diff)) if abs(diff) < 700 else float("inf")
    favors_a = ll_a > ll_b

    print("\n" + "=" * 76)
    print("  REZULTAT COMPARATIE PROBABILITATI (LOG-LIKELIHOOD)")
    print("=" * 76)
    print(f" [PROPOZITIA A]: \"{text_a}\"")
    print(f"   -> Log-Likelihood: {ll_a:.2f} | Perplexitate: {ppl_a:.2f} ({tok_a} tokeni)")
    print(f" [PROPOZITIA B]: \"{text_b}\"")
    print(f"   -> Log-Likelihood: {ll_b:.2f} | Perplexitate: {ppl_b:.2f} ({tok_b} tokeni)")
    print("-" * 76)
    if favors_a:
        print(f" >>> REZULTAT: Modelul considera PROPOZITIA A de ~{ratio:.2f}x MAI PROBABILA decat B!")
    else:
        print(f" >>> REZULTAT: Modelul considera PROPOZITIA B de ~{ratio:.2f}x MAI PROBABILA decat A!")
    print("=" * 76 + "\n")


def run_single_interactive(model_path: Path, max_new_tokens: int = 64, temperature: float = 0.7, mode: str = "gen"):
    model, tokenizer, device = load_model_and_tokenizer(model_path)
    
    print("\n" + "=" * 76)
    print(f"  INTERFATA INTERACTIVA — MODEL: {model_path.name}")
    print(f"  Dispozitiv: {device.upper()} | Precizie: BF16/FP16")
    print("  Comenzi disponibile:")
    print("    - Tasteaza un text obisnuit pentru GENERARE (text completion)")
    print("    - Tasteaza ':score' pentru a compara matematic doua propozitii (probabilitati)")
    print("    - Tasteaza 'exit' pentru a iesi")
    print("=" * 76 + "\n")
    
    while True:
        try:
            prompt = input("\n[PROMPT / COMANDA] >>> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "iesire"]:
                print("\nSesiune incheiata.")
                break
                
            if prompt.lower() in [":score", ":prob", "score", "prob"]:
                print("\n--- MOD CALCUL PROBABILITATI EXACTE ---")
                text_a = input(" [Propozitia A] >>> ").strip()
                text_b = input(" [Propozitia B] >>> ").strip()
                if text_a and text_b:
                    compare_two_sentences(model, tokenizer, device, text_a, text_b)
                continue

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

"""
Upload SPP-Ro Models to Hugging Face Hub
Usage:
    py src/upload_to_hf.py --model base --repo-id your-username/base-ro-125m
    py src/upload_to_hf.py --model spp  --repo-id your-username/spp-ro-125m
"""
import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from huggingface_hub import HfApi, login, get_token
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


def upload_model(model_type: str, repo_id: str, private: bool = False, token: str = None):
    if model_type == "lora":
        model_dir = Path("models/base_ro_125m_lora")
    else:
        model_dir = Path(f"models/{model_type}_ro_125m")
        
    if not model_dir.exists():
        raise FileNotFoundError(f"Folderul modelului nu a fost gasit: {model_dir.resolve()}")

    # Check token
    active_token = token or get_token()
    if not active_token:
        print("\n[!] Nu esti autentificat pe Hugging Face!")
        print("Obtine un token cu permisiune de 'Write' de la: https://huggingface.co/settings/tokens")
        entered_token = input("Introdu token-ul tau Hugging Face (hf_...): ").strip()
        if not entered_token:
            print("Token invalid. Upload anulat.")
            sys.exit(1)
        login(token=entered_token)
        active_token = entered_token

    print(f"\n==================================================")
    print(f"  Uploading {model_type.upper()}-Ro-125M to Hugging Face Hub")
    print(f"  Source: {model_dir.resolve()}")
    print(f"  Target: https://huggingface.co/{repo_id}")
    print(f"==================================================\n")

    api = HfApi(token=active_token)

    # 1. Create repo if it doesn't exist
    print(f"[1/4] Verificam / cream repository-ul '{repo_id}'...")
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    # 2. Push entire folder (includes safetensors, tokenizer, config, and README.md)
    print(f"[2/4] Uploading fisier cu fisier (model.safetensors, tokenizer, config, README)...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint-*", "*.bin"],  # Exclude training intermediate checkpoints
    )

    print(f"\n==================================================")
    print(f" [SUCCES] Modelul a fost incarcat pe Hugging Face!")
    print(f" Acceseaza si testeaza modelul tau la:")
    print(f" >>> https://huggingface.co/{repo_id}")
    print(f"==================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload trained SPP-Ro models to Hugging Face")
    parser.add_argument("--model", type=str, default="base", choices=["base", "spp", "lora"],
                        help="Model to upload ('base', 'spp', or 'lora')")
    parser.add_argument("--repo-id", type=str, default=None,
                        help="Target Hugging Face repo ID, e.g., 'username/base-ro-125m'")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face write token (optional, will prompt if missing)")
    parser.add_argument("--private", action="store_true",
                        help="Set repository as private on Hugging Face")
    args = parser.parse_args()

    repo_id = args.repo_id
    if not repo_id:
        # Prompt user for username
        username = input("Introdu username-ul tau Hugging Face: ").strip()
        if not username:
            print("Username obligatoriu.")
            sys.exit(1)
        repo_id = f"{username}/{args.model}-ro-125m"

    upload_model(args.model, repo_id, args.private, args.token)

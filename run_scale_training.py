"""
Master Sequencer Launcher for Romanian LLM Scale Training (2-3 Days per Model)
Hardware Target: NVIDIA GeForce RTX 3060 (12GB VRAM)

Usage:
  py run_scale_training.py --model base    # Trains Base-Ro-125M (+50,000 steps, ~50 hours)
  py run_scale_training.py --model spp     # Trains SPP-Ro-125M (+50,000 steps, ~50 hours)
  py run_scale_training.py --model all     # Sequentially trains Base, then SPP (~100 hours total)
"""
import sys
import os
import argparse
import subprocess
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"


def print_banner():
    print("""
----------------------------------------------------------------------------
  SPP-Ro: Continued Pre-training Engine (3.93B Tokens per Model)
  Hardware: NVIDIA GeForce RTX 3060 12GB (BF16 Mixed Precision)
  Config: 60,000 steps total (~50h per model, hourly updates)
----------------------------------------------------------------------------
""", flush=True)


def run_training_stage(mode: str, additional_steps: int = 50000, total_target: int = 60000, interval_mins: float = 60.0):
    print(f"\n[Etapa: {mode.upper()}-Ro-125M]")
    print(f"Pasi: +{additional_steps:,} (Target cumulativ: {total_target:,})")
    print(f"Raport la fiecare {interval_mins:.0f} minute.")
    print("-" * 60, flush=True)
    
    cmd = [
        sys.executable,
        str(SRC_DIR / "train_scale_pretrain.py"),
        "--mode", mode,
        "--additional-steps", str(additional_steps),
        "--total-target", str(total_target),
        "--interval-mins", str(interval_mins),
    ]
    
    ret = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if ret.returncode != 0:
        print(f"\n[Eroare] Etapa {mode.upper()} s-a oprit cu codul {ret.returncode}.")
        sys.exit(ret.returncode)
    print(f"\n[Finalizat] Etapa {mode.upper()} s-a incheiat cu succes.\n")


def main():
    parser = argparse.ArgumentParser(description="SPP-Ro Deep Scale Training Launcher")
    parser.add_argument("--model", type=str, default="base", choices=["base", "spp", "all"], help="Which model to train")
    parser.add_argument("--steps", type=int, default=50000, help="Additional steps per model (default 50,000)")
    parser.add_argument("--total-target", type=int, default=60000, help="Total target steps (default 60,000)")
    parser.add_argument("--interval", type=float, default=60.0, help="Minutes between live progress reports (default 60.0)")
    args = parser.parse_args()

    print_banner()

    if args.model in ["base", "all"]:
        run_training_stage("base", additional_steps=args.steps, total_target=args.total_target, interval_mins=args.interval)

    if args.model in ["spp", "all"]:
        run_training_stage("spp", additional_steps=args.steps, total_target=args.total_target, interval_mins=args.interval)

    print("\n----------------------------------------------------------------------------")
    print("  Antrenare finalizata pentru toate etapele selectate.")
    print("----------------------------------------------------------------------------\n")


if __name__ == "__main__":
    main()

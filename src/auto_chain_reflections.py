"""
Auto-Chaining Sequencer: Base-Ro Pretraining -> Qwen 2.5 7B Reflection Synthesis
Monitors the ongoing Base-Ro training process (PID / train_scale_pretrain.py).
As soon as training finishes at step 50,000 and VRAM is freed, it automatically
launches the local Qwen 2.5 7B reflection generator without requiring manual intervention.
"""
import sys
import os
import time
import subprocess
from pathlib import Path
from datetime import datetime

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import psutil

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
MODELS_DIR = ROOT_DIR / "models"
BASE_DIR = MODELS_DIR / "base_ro_125m"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"


def is_training_active():
    """Checks if any python process running train_scale_pretrain.py is currently active."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            if any('train_scale_pretrain.py' in arg for arg in cmdline):
                return True, proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False, None


def get_latest_checkpoint():
    """Finds the most recent checkpoint in models/base_ro_125m/checkpoints."""
    if not CHECKPOINTS_DIR.exists():
        return "Niciunul"
    cps = [d.name for d in CHECKPOINTS_DIR.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not cps:
        return "Niciunul"
    # Sort by step number
    cps.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0)
    return cps[-1]


def main():
    print("=" * 80)
    print("  SPP-Ro: MONITOR AUTOMAT & LANSARE REFLECTII (QWEN 2.5 7B)")
    print("  Hardware: NVIDIA GeForce RTX 3060 12GB")
    print("  Flux: Asteapta finalizarea Base-Ro (50.000 pasi) -> Porneste Qwen 7B pe GPU")
    print("=" * 80)
    print(flush=True)

    active, pid = is_training_active()
    if not active:
        print("[!] Nu a fost detectat niciun proces de antrenare 'train_scale_pretrain.py'.")
        print("[*] Verificam daca modelul Base-Ro este deja finalizat...")
        time.sleep(2)
    else:
        print(f"[*] Proces de antrenare detectat activ (PID: {pid}).")
        latest_cp = get_latest_checkpoint()
        print(f"[*] Ultimul checkpoint salvat pe disc: {latest_cp}")
        print("[*] Monitorul verifica starea la fiecare 30 de secunde.")
        print("[*] Poti lasa calculatorul pornit; generarea va incepe automat!\n", flush=True)

    check_interval = 30  # seconds
    counter = 0

    while True:
        active, pid = is_training_active()
        now_str = datetime.now().strftime("%H:%M:%S")

        if active:
            counter += 1
            if counter % 4 == 1:  # Print every 2 minutes
                latest_cp = get_latest_checkpoint()
                print(f"[{now_str}] Antrenare in desfasurare (PID: {pid}) | Ultimul checkpoint: {latest_cp} | Asteptare...", flush=True)
            time.sleep(check_interval)
        else:
            print("\n" + "=" * 80)
            print(f"[{now_str}] [SUCCES] Procesul de antrenare Base-Ro s-a incheiat!")
            latest_cp = get_latest_checkpoint()
            print(f"  Checkpoint final atins: {latest_cp}")
            print("=" * 80, flush=True)
            break

    # Wait 20 seconds for the NVIDIA driver to completely reclaim and reset all VRAM
    print("\n[GPU] Pauza 20 secunde pentru eliberarea completa a memoriei video VRAM de catre driver...")
    for s in range(20, 0, -5):
        print(f"  -> Incepere in {s} secunde...", flush=True)
        time.sleep(5)

    print("\n" + "=" * 80)
    print("  PORNIRE AUTOMATA GENERATOR LOCAL REFLECTII: Qwen 2.5 7B Instruct")
    print("  Configuratie: 4-bit NF4 Quantization | Batch Size: 8 | Inspectie Live: din 500 in 500")
    print("  Tinta: 10.000 de reflectii constitutionale unice (~1.666 per articol)")
    print("=" * 80 + "\n", flush=True)

    # Launch local generator directly
    gen_script = SRC_DIR / "generate_local_llm_reflections.py"
    cmd = [
        sys.executable,
        str(gen_script),
        "--model", "Qwen/Qwen2.5-7B-Instruct",
        "--target", "10000",
        "--batch-size", "8",
        "--max-new-tokens", "380",
        "--interval", "500",
    ]

    ret = subprocess.call(cmd, cwd=str(ROOT_DIR))
    print(f"\n[Finalizat] Generatorul de reflectii s-a incheiat cu codul {ret}.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
qwen3-tts.cpp — Interactive Model Setup Wizard
===============================================
Step-by-step guide to download and convert Qwen3-TTS models
to GGUF format for use with qwen3-tts.cpp.

This script wraps scripts/setup_pipeline_models.py,
scripts/convert_tts_to_gguf.py, and scripts/convert_tokenizer_to_gguf.py
with a friendly interactive interface.

Usage:
    python tools/model-converter/setup_models.py
    python tools/model-converter/setup_models.py --non-interactive  (uses defaults)
    python tools/model-converter/setup_models.py --help
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ---- Repo paths -------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
MODELS_DIR  = REPO_ROOT / "models"

# ---- Available model configurations -----------------------------
MODEL_VARIANTS = {
    "base": {
        "label": "Base — Voice Cloning + anonymous synthesis (recommended)",
        "recommended": True,
        "repo_0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "repo_1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "local_0.6b": "Qwen3-TTS-12Hz-0.6B-Base",
        "local_1.7b": "Qwen3-TTS-12Hz-1.7B-Base",
    },
    "custom_voice": {
        "label": "CustomVoice — Named speaker presets + voice cloning + instruct (1.7B)",
        "recommended": False,
        "repo_0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "repo_1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "local_0.6b": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "local_1.7b": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    },
    "voice_design": {
        "label": "VoiceDesign — Describe a voice in natural language + voice cloning (1.7B only)",
        "recommended": False,
        "repo_1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "local_1.7b": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    },
}

TOKENIZER_REPO    = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
TOKENIZER_LOCAL   = "Qwen3-TTS-Tokenizer-12Hz"

# NOTE: only f16 and q8_0 are supported by the gguf Python library.
# K-quants (q6_k through q2_k) raise NotImplementedError in gguf.quants.quantize()
# and silently fall back to F16, producing misleadingly large files.
# They are commented out until the converter supports them natively.
# TODO: implement K-quant byte layout in scripts/convert_tts_to_gguf.py, then
#       uncomment the entries below.
QUANT_OPTIONS = {
    "f16":   "F16   — Full precision (~1.75 GB / 0.6B). Best quality, largest file.",
    "q8_0":  "Q8_0  — 8-bit quantized (~1.0 GB / 0.6B). Virtually lossless quality. [Recommended]",
    # "q6_k":  "Q6_K  — 6-bit K-quant (~0.75 GB / 0.6B). Excellent quality.",
    # "q5_k":  "Q5_K  — 5-bit K-quant (~0.65 GB / 0.6B). Very good quality.",
    # "q4_k":  "Q4_K  — 4-bit K-quant (~0.55 GB / 0.6B). Good quality, smallest practical size.",
    # "q3_k":  "Q3_K  — 3-bit K-quant (~0.42 GB / 0.6B). [Not recommended — audible artifacts]",
    # "q2_k":  "Q2_K  — 2-bit K-quant (~0.33 GB / 0.6B). [Not recommended — significant quality loss]",
}

MIMI_QUANT_OPTIONS = {
    "f32":  "F32 — 100%% exact match vs Python (recommended for ICL voice cloning)",
    "f16":  "F16 — 98.9%% match (good for normal use)",
    "q8_0": "Q8_0 — 94.3%% match (below recommended threshold, not recommended)",
}


# ---- Helpers ----------------------------------------------------

def banner(text: str, char: str = "=") -> None:
    line = char * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")

def step(n: int, total: int, text: str) -> None:
    print(f"\n[{n}/{total}] {text}")
    print("-" * 50)

def ask(prompt: str, options: list[str], default: Optional[str] = None) -> str:
    """Show a numbered menu and return the chosen option key."""
    for i, key in enumerate(options, 1):
        print(f"  {i}. {key}")
    if default:
        prompt_str = f"\n{prompt} [default: {default}]: "
    else:
        prompt_str = f"\n{prompt}: "

    while True:
        try:
            raw = input(prompt_str).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)

        if raw == "" and default:
            return default
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        if raw in options:
            return raw
        print(f"  Please enter a number 1-{len(options)} or one of: {', '.join(options)}")

def ask_yes_no(prompt: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            raw = input(f"{prompt} {hint}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter y or n.")

def run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    """Run a subprocess, streaming output, and raise on failure."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, cwd=str(cwd or REPO_ROOT))
    if result.returncode != 0:
        print(f"\n[error] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def check_python_deps() -> None:
    missing = []
    for mod, pkg in [("huggingface_hub", "huggingface_hub"),
                     ("safetensors", "safetensors"),
                     ("torch", "torch"),
                     ("numpy", "numpy"),
                     ("gguf", "gguf"),
                     ("tqdm", "tqdm")]:
        if not importlib.util.find_spec(mod):
            missing.append(pkg)
    if missing:
        print("\n[warn] Packages not found in this Python interpreter:")
        print(f"       Python: {sys.executable}")
        for p in missing:
            print(f"  - {p}")
        print()
        print("If your packages are in a different Python, run the wizard directly:")
        print("  C:\\Python314\\python.exe tools/model-converter/setup_models.py")
        print(f"Or install missing: {sys.executable} -m pip install " + " ".join(missing))
        print()
        if not ask_yes_no("Continue anyway (will fail if packages are truly missing)?", default=False):
            sys.exit(1)

def hf_download_with_retry(repo_id: str, local_dir: str, token: Optional[str],
                           allow_patterns: Optional[list] = None,
                           max_retries: int = 5) -> None:
    """snapshot_download with exponential-backoff retry for flaky connections."""
    import time
    from huggingface_hub import snapshot_download

    delay = 5.0
    for attempt in range(1, max_retries + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                token=token,
                allow_patterns=allow_patterns,
                resume_download=True,
            )
            return
        except Exception as e:
            msg = str(e)
            is_network = any(k in msg for k in (
                "ConnectionReset", "ConnectionError", "ProtocolError",
                "10054", "RemoteDisconnected", "LocalEntryNotFound",
            ))
            if attempt == max_retries or not is_network:
                raise
            print(f"\n  [warn] Download interrupted (attempt {attempt}/{max_retries}): {type(e).__name__}")
            print(f"         Retrying in {delay:.0f}s... (resume_download=True, progress preserved)")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)  # cap at 60s



def output_filename(variant: str, size: str, quant: str) -> str:
    suffix = {"base": "", "custom_voice": "-customvoice", "voice_design": "-voicedesign"}[variant]
    return f"qwen3-tts-{size}{suffix}-{quant}.gguf"

def tokenizer_filename(quant: str) -> str:
    return f"qwen3-tts-tokenizer-{quant}.gguf"


# ---- Main wizard ------------------------------------------------

def main(non_interactive: bool = False, hf_token: Optional[str] = None) -> None:
    banner("qwen3-tts.cpp — Model Setup Wizard")
    print("""
This wizard will help you:
  1. Download a Qwen3-TTS model from HuggingFace
  2. Download the shared tokenizer/vocoder
  3. Convert both to GGUF format for C++ inference

All models output 24 kHz mono audio.
""")

    check_python_deps()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    TOTAL_STEPS = 7

    # ---- Step 1: Model variant -----------------------------------
    step(1, TOTAL_STEPS, "Choose model variant")
    print("""
  Base          — VOICE CLONING. Clone any voice from a short reference WAV file.
                  Give it 3-30 seconds of someone speaking and it copies their voice.
                  Best for: cloning your own voice, a celebrity, or any audio sample.
                  Also works: basic synthesis with no reference (anonymous neutral voice).
                  [Recommended for most users]

  CustomVoice   — NAMED SPEAKERS. Pick from ~30 built-in voice presets by name
                  (e.g. Vivian, Ryan, Emma). No reference audio needed.
                  Best for: consistent branded voices without needing reference audio.
                  Also works: voice cloning with a reference WAV (same as Base);
                              style/instruct text on 1.7B (e.g. "speak slowly").
                  Note: named speaker presets are unique to this variant.

  VoiceDesign   — VOICE DESCRIPTION. Describe a voice in natural language:
                  e.g. "Calm, warm female voice with a slight British accent".
                  Best for: creating novel voices when you don't have reference audio.
                  Also works: voice cloning with a reference WAV (same as Base).
                  Note: 1.7B only — Alibaba has not published a 0.6B VoiceDesign model.
""")
    variant_keys = list(MODEL_VARIANTS.keys())
    if non_interactive:
        chosen_variant = "base"
        print(f"  [non-interactive] Using: {chosen_variant}")
    else:
        chosen_variant = ask("Select variant (1-3)", variant_keys, default="base")
    variant_info = MODEL_VARIANTS[chosen_variant]
    print(f"  -> {variant_info['label']}")

    # ---- Step 2: Model size --------------------------------------
    step(2, TOTAL_STEPS, "Choose model size")

    # VoiceDesign is 1.7B only — Alibaba never published a 0.6B VoiceDesign model
    if chosen_variant == "voice_design":
        chosen_size = "1.7b"
        print("  VoiceDesign is 1.7B only (no 0.6B model exists).")
        print(f"  -> {chosen_size}")
    else:
        print("""
  0.6B  — Faster, lighter (~1.75 GB F16). Good quality for most use cases.
  1.7B  — Slower, heavier (~4.2 GB F16). Better naturalness and prosody.
          Required for VoiceDesign instruct mode.
""")
        if non_interactive:
            chosen_size = "0.6b"
            print(f"  [non-interactive] Using: {chosen_size}")
        else:
            chosen_size = ask("Select size (1-2)", ["0.6b", "1.7b"], default="0.6b")
        print(f"  -> {chosen_size}")

    # ---- Step 3: Quantization ------------------------------------
    step(3, TOTAL_STEPS, "Choose quantization (TTS transformer weights)")
    print()
    for key, desc in QUANT_OPTIONS.items():
        print(f"  {key:6s}  {desc}")
    print()
    if non_interactive:
        chosen_quant = "q8_0"
        print(f"  [non-interactive] Using: {chosen_quant}")
    else:
        chosen_quant = ask("Select quantization", list(QUANT_OPTIONS.keys()), default="q8_0")
    print(f"  -> {chosen_quant}")

    # ---- Step 4: Mimi encoder precision --------------------------
    step(4, TOTAL_STEPS, "Choose Mimi encoder precision (for ICL voice cloning)")
    print("""
  The Mimi encoder converts reference audio to codec codes for ICL voice cloning.
  This only affects ICL mode (--ref-text flag). For x-vector voice cloning, any
  precision works identically.
""")
    for key, desc in MIMI_QUANT_OPTIONS.items():
        print(f"  {key:6s}  {desc}")
    print()
    if non_interactive:
        chosen_mimi = "f32" if chosen_variant == "base" else "f16"
        print(f"  [non-interactive] Using: {chosen_mimi}")
    else:
        default_mimi = "f32" if chosen_variant == "base" else "f16"
        chosen_mimi = ask("Select Mimi precision", list(MIMI_QUANT_OPTIONS.keys()), default=default_mimi)
    print(f"  -> {chosen_mimi}")

    # ---- Step 5: HuggingFace token -------------------------------
    step(5, TOTAL_STEPS, "HuggingFace authentication")
    print("""
  Some Qwen3-TTS models require a HuggingFace account and access token.
  You can get a token at: https://huggingface.co/settings/tokens

  If the model is public and you are already logged in via `huggingface-cli login`,
  you can skip this step.
""")
    if hf_token:
        print(f"  Using token from --hf-token argument.")
        token = hf_token
    elif non_interactive:
        token = None
        print("  [non-interactive] No token provided.")
    else:
        use_token = ask_yes_no("Do you have a HuggingFace token to provide?", default=False)
        token = None
        if use_token:
            try:
                token = input("  Enter token (input hidden): ").strip() or None
            except (EOFError, KeyboardInterrupt):
                token = None

    # ---- Step 6: Confirm and download ----------------------------
    step(6, TOTAL_STEPS, "Summary — ready to proceed")

    repo_key = f"repo_{chosen_size}"
    local_key = f"local_{chosen_size}"
    repo_id   = variant_info[repo_key]
    local_dir = MODELS_DIR / variant_info[local_key]
    out_tts   = MODELS_DIR / output_filename(chosen_variant, chosen_size, chosen_quant)
    out_tok   = MODELS_DIR / tokenizer_filename("f16")  # tokenizer always f16 base

    print(f"""
  Model variant:     {chosen_variant}  ({variant_info['label']})
  Model size:        {chosen_size}
  Quantization:      {chosen_quant}
  Mimi precision:    {chosen_mimi}
  Source repo:       {repo_id}
  Tokenizer repo:    {TOKENIZER_REPO}
  Output TTS:        {out_tts.relative_to(REPO_ROOT)}
  Output tokenizer:  {out_tok.relative_to(REPO_ROOT)}
""")

    if not non_interactive:
        if not ask_yes_no("Proceed with download and conversion?", default=True):
            print("Aborted.")
            sys.exit(0)

    # ---- Step 7: Download + Convert ------------------------------
    step(7, TOTAL_STEPS, "Downloading and converting")

    # 7a: Download TTS model
    tok_local_dir = MODELS_DIR / TOKENIZER_LOCAL
    need_tts_download  = not local_dir.exists()
    need_tok_download  = not tok_local_dir.exists()

    if need_tts_download:
        print(f"\n  Downloading {repo_id} -> {local_dir} ...")
        hf_download_with_retry(repo_id, str(local_dir), token)
    else:
        print(f"\n  TTS model already present at {local_dir} — skipping download.")
        print("  (Delete the directory and re-run to force re-download.)")

    # 7b: Download tokenizer
    if need_tok_download:
        print(f"\n  Downloading {TOKENIZER_REPO} -> {tok_local_dir} ...")
        hf_download_with_retry(TOKENIZER_REPO, str(tok_local_dir), token)
    else:
        print(f"\n  Tokenizer already present at {tok_local_dir} — skipping download.")

    # 7c: Convert TTS model to GGUF
    if not out_tts.exists():
        print(f"\n  Converting TTS model -> {out_tts.name} ...")
        run([
            sys.executable,
            str(SCRIPTS_DIR / "convert_tts_to_gguf.py"),
            "--input", str(local_dir),
            "--output", str(out_tts),
            "--type", chosen_quant,
        ])
    else:
        print(f"\n  TTS GGUF already exists: {out_tts.name} — skipping conversion.")
        print("  (Delete the file to force re-conversion.)")

    # 7d: Convert tokenizer to GGUF
    if not out_tok.exists():
        print(f"\n  Converting tokenizer -> {out_tok.name} ...")
        run([
            sys.executable,
            str(SCRIPTS_DIR / "convert_tokenizer_to_gguf.py"),
            "--input", str(tok_local_dir),
            "--output", str(out_tok),
            "--type", "f16",
            "--mimi-type", chosen_mimi,
        ])
    else:
        print(f"\n  Tokenizer GGUF already exists: {out_tok.name} — skipping conversion.")

    # ---- Done ---------------------------------------------------
    banner("Setup complete!", char="=")
    print(f"""
  Your model files are ready:

    {out_tts.relative_to(REPO_ROOT)}
    {out_tok.relative_to(REPO_ROOT)}

  Quick synthesis test:
    build-ninja\\qwen3-tts-cli.exe -m models -t "Hello, world!" -o hello.wav
    ./build/qwen3-tts-cli -m models -t "Hello, world!" -o hello.wav

  Full documentation: AGENTS.md and docs/handoff.md
""")


# ---- Entry point ------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="qwen3-tts.cpp interactive model setup wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--non-interactive", "-y",
        action="store_true",
        help="Use defaults without prompting (0.6B Base, Q8_0, F32 Mimi)"
    )
    parser.add_argument(
        "--hf-token",
        metavar="TOKEN",
        help="HuggingFace access token (for gated repos)",
    )
    args = parser.parse_args()
    main(non_interactive=args.non_interactive, hf_token=args.hf_token)


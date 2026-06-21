"""
_interactive.py -- Interactive downloader called by download-model.bat and download-tokenizer.bat
Usage:
  python _interactive.py model   [--variant V] [--size S] [--type T] [--hf-token TOK]
  python _interactive.py tokenizer [--type T] [--mimi-type M] [--hf-token TOK]
"""
from __future__ import annotations
import sys, os, subprocess, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

def ask(prompt: str, choices: list[str], default: str) -> str:
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted."); sys.exit(0)
        if raw == "":
            return default
        if raw in choices:
            return raw
        idx = raw if raw.isdigit() else None
        if idx and 1 <= int(idx) <= len(choices):
            return choices[int(idx) - 1]
        print(f"  Enter a number 1-{len(choices)} or one of: {', '.join(choices)}")

def hf_download(repo_id: str, local_dir: str, token: str | None) -> None:
    from huggingface_hub import snapshot_download
    delay = 5.0
    for attempt in range(1, 6):
        try:
            snapshot_download(repo_id=repo_id, local_dir=local_dir,
                              token=token or None, resume_download=True)
            return
        except Exception as e:
            msg = str(e)
            is_net = any(k in msg for k in ("ConnectionReset","ConnectionError",
                         "ProtocolError","10054","RemoteDisconnected","LocalEntryNotFound"))
            if attempt == 5 or not is_net:
                print(f"\n[error] Download failed: {e}"); sys.exit(1)
            print(f"  [warn] attempt {attempt}/5 failed ({type(e).__name__}). Retry in {delay:.0f}s...")
            time.sleep(delay); delay = min(delay * 2, 60.0)

def run_convert(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        print(f"\n[error] Conversion failed (exit {r.returncode})"); sys.exit(r.returncode)

def get_token(args: dict) -> str | None:
    if "hf_token" in args:
        raw = args["hf_token"]
    else:
        raw = input("\n  HuggingFace token (blank if not needed): ").strip()
    return None if not raw or raw.lower() in ("none", "null") else raw

def mode_tokenizer(args: dict) -> None:
    print()
    print("=" * 60)
    print("  qwen3-tts.cpp -- Tokenizer / Vocoder Downloader")
    print("  Shared file -- download once for all TTS models")
    print("=" * 60)

    tok_type = args.get("type") or ask(
        "\n  Tokenizer precision:\n"
        "    1. f16  ~432 MB  - Recommended\n"
        "    2. f32  ~864 MB  - Bit-exact ICL cloning\n"
        "\n  Choose (1-2) [default 1]: ",
        ["f16", "f32"], "f16")

    mimi_type = args.get("mimi_type") or ask(
        "\n  Mimi encoder precision:\n"
        "    1. f16  - 98.9% match vs Python  [recommended]\n"
        "    2. f32  - Bit-exact ICL cloning\n"
        "    3. q8_0 - Not recommended for ICL\n"
        "\n  Choose (1-3) [default 1]: ",
        ["f16", "f32", "q8_0"], "f16")

    token = get_token(args)

    tok_dir  = REPO_ROOT / "models" / "Qwen3-TTS-Tokenizer-12Hz"
    # Filename encodes both tok_type (vocoder) and mimi_type (Mimi encoder)
    # so different precision combinations never collide.
    # C++ auto-discovery knows all valid combinations.
    # Examples:
    #   tok=f16, mimi=f16  -> qwen3-tts-tokenizer-f16-f16.gguf
    #   tok=f16, mimi=f32  -> qwen3-tts-tokenizer-f16-f32.gguf
    #   tok=f16, mimi=q8_0 -> qwen3-tts-tokenizer-f16-q8_0.gguf
    out_file = REPO_ROOT / "models" / f"qwen3-tts-tokenizer-{tok_type}-{mimi_type}.gguf"

    print(f"\n  Tokenizer type : {tok_type}")
    print(f"  Mimi precision : {mimi_type}")
    print(f"  Output         : {out_file.relative_to(REPO_ROOT)}\n")

    if not (tok_dir / "model.safetensors").exists():
        print("[1/2] Downloading Qwen3-TTS-Tokenizer-12Hz...")
        hf_download("Qwen/Qwen3-TTS-Tokenizer-12Hz", str(tok_dir), token)
    else:
        print(f"[ok] Source already present at {tok_dir.relative_to(REPO_ROOT)}")

    if out_file.exists():
        print(f"[ok] {out_file.name} already exists -- skipping conversion.")
        print("     Delete it to force re-conversion.")
    else:
        print("[2/2] Converting...")
        run_convert([sys.executable,
                     str(REPO_ROOT / "scripts" / "convert_tokenizer_to_gguf.py"),
                     "--input", str(tok_dir),
                     "--output", str(out_file),
                     "--type", tok_type,
                     "--mimi-type", mimi_type])

    print(f"\n[done] {out_file.relative_to(REPO_ROOT)}")
    print("       Works with all TTS model variants.")

def mode_model(args: dict) -> None:
    VARIANTS = {
        "base":         {"label": "Base  - Voice cloning from reference audio  [recommended]"},
        "custom_voice": {"label": "CustomVoice  - Named speaker presets + instruct (1.7B)"},
        "voice_design": {"label": "VoiceDesign  - Describe voice in text (1.7B only)"},
    }
    # NOTE: only f16, f32, and q8_0 are supported by the gguf Python library.
    # K-quants (q6_k, q5_k, q4_k, q3_k, q2_k) raise NotImplementedError in
    # gguf.quants.quantize() and silently fall back to F16, producing a
    # misleadingly large file. They are commented out until the converter
    # supports them natively.
    # TODO: implement K-quant byte layout in convert_tts_to_gguf.py, then
    #       uncomment the entries below and update the menu numbering.
    QUANTS = {
        "f32":   "~3.5/8.4 GB    Full 32-bit precision  [not recommended — same quality as F16, double size]",
        "f16":   "~1.75/4.2 GB   Full precision  [recommended]",
        "q8_0":  "~1.28/3.1 GB   Near-lossless",
        # "q6_k":  "~0.99/2.4 GB   Excellent",
        # "q5_k":  "~0.86/2.1 GB   Very good",
        # "q4_k":  "~0.72/1.8 GB   Good",
        # "q3_k":  "~0.58/1.4 GB   Not recommended",
        # "q2_k":  "~0.46/1.1 GB   Not recommended",
    }

    print()
    print("=" * 60)
    print("  qwen3-tts.cpp -- TTS Model Downloader")
    print("=" * 60)

    variant_prompt = "\n  Model variant:\n"
    for i, (k, v) in enumerate(VARIANTS.items(), 1):
        variant_prompt += f"    {i}. {v['label']}\n"
    variant_prompt += "\n  Choose (1-3) [default 1]: "

    variant = args.get("variant") or ask(variant_prompt, list(VARIANTS.keys()), "base")

    if variant == "voice_design":
        size = "1.7b"
        print("  Size: 1.7b only (VoiceDesign has no 0.6b model)")
    else:
        size = args.get("size") or ask(
            "\n  Model size:\n"
            "    1. 0.6b  - Faster, smaller  [recommended]\n"
            "    2. 1.7b  - Better quality, larger\n"
            "\n  Choose (1-2) [default 1]: ",
            ["0.6b", "1.7b"], "0.6b")

    quant_prompt = "\n  Quantization (0.6b / 1.7b):\n"
    for i, (k, v) in enumerate(QUANTS.items(), 1):
        quant_prompt += f"    {i}. {k:<6} {v}\n"
    quant_prompt += "\n  Choose (1-3) [default 2]: "

    quant = args.get("type") or ask(quant_prompt, list(QUANTS.keys()), "q8_0")

    token = get_token(args)

    REPOS = {
        ("base",         "0.6b"): ("Qwen/Qwen3-TTS-12Hz-0.6B-Base",        "Qwen3-TTS-12Hz-0.6B-Base"),
        ("base",         "1.7b"): ("Qwen/Qwen3-TTS-12Hz-1.7B-Base",        "Qwen3-TTS-12Hz-1.7B-Base"),
        ("custom_voice", "0.6b"): ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        ("custom_voice", "1.7b"): ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        ("voice_design", "1.7b"): ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    }

    key = (variant, size)
    if key not in REPOS:
        print(f"[error] Unsupported combination: {variant} + {size}"); sys.exit(1)
    repo_id, local_name = REPOS[key]
    local_dir = REPO_ROOT / "models" / local_name

    suffix = {"base": "", "custom_voice": "-customvoice", "voice_design": "-voicedesign"}[variant]
    out_file = REPO_ROOT / "models" / f"qwen3-tts-{size}{suffix}-{quant}.gguf"

    print(f"\n  Variant : {variant}")
    print(f"  Size    : {size}")
    print(f"  Type    : {quant}")
    print(f"  Source  : {repo_id}")
    print(f"  Output  : {out_file.relative_to(REPO_ROOT)}\n")

    if not (local_dir / "model.safetensors").exists():
        print(f"[1/2] Downloading {repo_id}...")
        hf_download(repo_id, str(local_dir), token)
    else:
        print(f"[ok] Source already present at {local_dir.relative_to(REPO_ROOT)}")

    if out_file.exists():
        print(f"[ok] {out_file.name} already exists -- skipping conversion.")
        print("     Delete it to force re-conversion.")
    else:
        print("[2/2] Converting...")
        run_convert([sys.executable,
                     str(REPO_ROOT / "scripts" / "convert_tts_to_gguf.py"),
                     "--input", str(local_dir),
                     "--output", str(out_file),
                     "--type", quant])

    print(f"\n[done] {out_file.relative_to(REPO_ROOT)}")
    print("       Pair with: models/qwen3-tts-tokenizer-f16-f32.gguf  (or your chosen tokenizer)")

def main() -> None:
    args_raw = sys.argv[1:]
    if not args_raw:
        print("[error] Usage: _interactive.py model|tokenizer [options]"); sys.exit(1)

    mode = args_raw[0]
    parsed: dict = {}
    i = 1
    while i < len(args_raw):
        a = args_raw[i]
        if a == "--variant"   and i+1 < len(args_raw): parsed["variant"]   = args_raw[i+1]; i += 2
        elif a == "--size"    and i+1 < len(args_raw): parsed["size"]      = args_raw[i+1]; i += 2
        elif a == "--type"    and i+1 < len(args_raw): parsed["type"]      = args_raw[i+1]; i += 2
        elif a == "--mimi-type" and i+1 < len(args_raw): parsed["mimi_type"] = args_raw[i+1]; i += 2
        elif a == "--hf-token" and i+1 < len(args_raw): parsed["hf_token"] = args_raw[i+1]; i += 2
        else: i += 1

    if mode == "tokenizer":
        mode_tokenizer(parsed)
    elif mode == "model":
        mode_model(parsed)
    else:
        print(f"[error] Unknown mode: {mode}"); sys.exit(1)

if __name__ == "__main__":
    main()

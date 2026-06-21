# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
qwen3-tts.cpp -- Pre-built GGUF Downloader
===========================================
Downloads pre-converted GGUF model files directly from HuggingFace.
No conversion required -- files are ready to use immediately.

Source: https://huggingface.co/librellama/qwen3-tts-GGUF
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

# ── Repo root (2 levels up from tools/model-downloader/) ────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
HF_REPO    = "librellama/qwen3-tts-GGUF"

# ── Terminal helpers ─────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
DIM    = "\033[2m"

def _tty() -> bool:
    return sys.stdout.isatty()

def c(text: str, *codes: str) -> str:
    if not _tty():
        return text
    return "".join(codes) + text + RESET

def header(title: str, step: str = "") -> None:
    line = "=" * 62
    print()
    print(c(line, CYAN))
    if step:
        print(c(f"  {step}  |  {title}", BOLD))
    else:
        print(c(f"  {title}", BOLD))
    print(c(line, CYAN))

def option(n: int, label: str, tag: str = "",
           note: str = "", recommended: bool = False) -> None:
    badge = c(f" [{tag}]", DIM) if tag else ""
    rec   = c("  <-- recommended", GREEN) if recommended else ""
    warn  = c(f"  ({note})", YELLOW) if note else ""
    print(f"    {c(str(n) + '.', BOLD)} {label}{badge}{rec}{warn}")

def ask(prompt: str, choices: list[str], default: str) -> str:
    valid: dict[str, str] = {}
    for i, v in enumerate(choices):
        valid[str(i + 1)] = v
        valid[v] = v
    hint = c(f"[1-{len(choices)}, default={choices.index(default) + 1}]", DIM)
    while True:
        try:
            raw = input(f"\n  {prompt} {hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(c("\n  Aborted.", YELLOW))
            sys.exit(0)
        if raw == "":
            return default
        if raw in valid:
            return valid[raw]
        print(c(f"  Please enter a number 1-{len(choices)}.", YELLOW))

def ask_yes_no(prompt: str, default: bool = True) -> bool:
    hint = c("[Y/n]" if default else "[y/N]", DIM)
    while True:
        try:
            raw = input(f"\n  {prompt} {hint}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(c("\n  Aborted.", YELLOW))
            sys.exit(0)
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False

# ── Dependency check ─────────────────────────────────────────────────────────
def check_deps() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print(c("\n  [error] huggingface_hub is not installed.", YELLOW))
        print("  Install it:  pip install huggingface_hub")
        print("  Or with uv:  uv pip install huggingface_hub")
        sys.exit(1)

# ── HuggingFace download with retry ─────────────────────────────────────────
def hf_download_file(repo: str, filename: str,
                     dest: Path, token: str | None) -> None:
    from huggingface_hub import hf_hub_download
    delay = 5.0
    for attempt in range(1, 6):
        try:
            hf_hub_download(
                repo_id=repo,
                filename=filename,
                local_dir=str(dest),
                token=token,
                resume_download=True,
            )
            return
        except Exception as e:
            msg = str(e)
            is_net = any(k in msg for k in (
                "ConnectionReset", "ConnectionError", "ProtocolError",
                "10054", "RemoteDisconnected", "LocalEntryNotFound",
                "ReadTimeout", "ChunkedEncodingError",
            ))
            if attempt == 5 or not is_net:
                print(c(f"\n  [error] Download failed: {e}", YELLOW))
                sys.exit(1)
            print(c(
                f"  [warn] Attempt {attempt}/5 failed "
                f"({type(e).__name__}). Retrying in {delay:.0f}s...",
                YELLOW,
            ))
            time.sleep(delay)
            delay = min(delay * 2, 60.0)

# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    check_deps()

    print()
    print(c("=" * 62, CYAN))
    print(c("    qwen3-tts.cpp  --  Pre-built GGUF Downloader", BOLD))
    print(c("    huggingface.co/librellama/qwen3-tts-GGUF", DIM))
    print(c("=" * 62, CYAN))
    print()
    print("  Downloads ready-to-use GGUF files.")
    print("  No conversion required -- files work out of the box.")

    # ── Step 1: Model type ───────────────────────────────────────────────────
    header("Model Type", "Step 1 of 5")
    print()
    print("  What kind of voice synthesis do you need?")
    print()
    option(1, "Base",
           note="best for cloning voices from a reference audio clip")
    option(2, "Custom Voice",
           note="choose from ~30 built-in named speaker presets; "
                "also supports voice cloning")
    option(3, "Voice Design",
           note="describe a voice in natural language; "
                "also supports voice cloning  |  1.7B only")

    model_type = ask("Select model type",
                     ["base", "customvoice", "voicedesign"], "base")

    # ── Step 2: Parameter count ──────────────────────────────────────────────
    header("Parameter Count", "Step 2 of 5")
    print()

    if model_type == "voicedesign":
        model_size = "1.7b"
        print(c("  Voice Design is only available as 1.7B.", YELLOW))
        print(c("  Alibaba has not released a 0.6B Voice Design model.", DIM))
    else:
        option(1, "0.6B",
               note="faster, lighter  (~1-4 GB depending on quantization)")
        option(2, "1.7B",
               note="better naturalness and prosody  (~2-8 GB)",
               recommended=True)
        model_size = ask("Select parameter count", ["0.6b", "1.7b"], "1.7b")

    # ── Step 3: Model quantization ───────────────────────────────────────────
    header("Model Quantization", "Step 3 of 5")
    print()
    print("  Controls precision of the TTS transformer weights.")
    print("  Affects generation speed, VRAM usage, and output quality.")
    print()

    f16_sz = "1.84 GB" if model_size == "0.6b" else "3.86 GB"
    q8_sz  = "1.34 GB" if model_size == "0.6b" else "2.46 GB"
    f32_sz = "3.66 GB" if model_size == "0.6b" else "7.72 GB"

    option(1, "F16  -- Full 16-bit precision",
           tag=f16_sz, recommended=True)
    option(2, "Q8_0 -- 8-bit quantized",
           tag=q8_sz,
           note="28% smaller than F16, virtually identical audio quality")
    option(3, "F32  -- Full 32-bit precision",
           tag=f32_sz,
           note="same quality as F16, double the size -- not recommended")

    model_quant = ask("Select model quantization",
                      ["f16", "q8_0", "f32"], "f16")

    # ── Step 4: Vocoder precision ────────────────────────────────────────────
    header("Tokenizer -- Vocoder Precision", "Step 4 of 5")
    print()
    print("  The tokenizer contains two independent components.")
    print("  This step controls the vocoder -- the neural network that")
    print("  converts generated codec codes into the final audio waveform.")
    print()
    option(1, "F16  -- Full 16-bit precision",
           tag="~375 MB", recommended=True)
    option(2, "F32  -- Full 32-bit precision",
           tag="~453-682 MB",
           note="same audio quality as F16, larger file -- not recommended")
    print()
    print(c("  Note: source weights are BF16, so F32 adds no quality benefit.", DIM))

    vocoder_prec = ask("Select vocoder precision", ["f16", "f32"], "f16")

    # ── Step 5: Mimi encoder precision ───────────────────────────────────────
    header("Tokenizer -- Mimi Encoder Precision", "Step 5 of 5")
    print()
    print("  This step controls the Mimi encoder -- used exclusively for")
    print("  ICL (in-context learning) voice cloning, which requires you")
    print("  to provide both a reference audio clip AND its transcript:")
    print()
    print(c('    --reference ref.wav  --ref-text "Your transcript here"', CYAN))
    print()
    print("  If you do not use ICL, F16 is fine and slightly smaller.")
    print()
    option(1, "F32  -- Bit-exact encoding",
           note="100% match vs Python reference",
           recommended=True)
    option(2, "F16  -- Standard precision",
           note="98.9% match -- good for general use; "
                "slightly lower ICL clone accuracy")
    option(3, "Q8_0 -- 8-bit quantized",
           note="94.3% match -- not recommended if you use ICL voice cloning")

    mimi_prec = ask("Select Mimi encoder precision",
                    ["f32", "f16", "q8_0"], "f32")

    # ── Resolve filenames ────────────────────────────────────────────────────
    suffix_map  = {"base": "", "customvoice": "-customvoice",
                   "voicedesign": "-voicedesign"}
    suffix      = suffix_map[model_type]
    model_file  = f"qwen3-tts-{model_size}{suffix}-{model_quant}.gguf"
    tok_file    = f"qwen3-tts-tokenizer-{vocoder_prec}-{mimi_prec}.gguf"

    # f32 vocoder + q8_0 Mimi was not uploaded to the HF repo (rare combo).
    # Redirect to f16 vocoder + q8_0 Mimi which is equivalent for audio quality.
    if vocoder_prec == "f32" and mimi_prec == "q8_0":
        print()
        print(c("  [note] The combination f32 vocoder + q8_0 Mimi is not available", YELLOW))
        print(c("         in the pre-built repo. Using f16 vocoder + q8_0 Mimi instead.", YELLOW))
        print(c("         (F32 vocoder has no quality benefit over F16.)", DIM))
        vocoder_prec = "f16"
        tok_file = f"qwen3-tts-tokenizer-{vocoder_prec}-{mimi_prec}.gguf"

    # ── HuggingFace token ────────────────────────────────────────────────────
    print()
    print(c("-" * 62, DIM))
    print()
    print("  The source repo is public -- no token is needed.")
    print("  A token can help with rate limits on slow connections.")
    print()
    use_token = ask_yes_no("Provide a HuggingFace token?", default=False)
    token: str | None = None
    if use_token:
        try:
            token = input(c("  Token: ", BOLD)).strip() or None
        except (EOFError, KeyboardInterrupt):
            token = None

    # ── Summary ──────────────────────────────────────────────────────────────
    header("Summary -- Ready to Download")
    print()
    print(f"  Model type         :  {c(model_type, BOLD)}")
    print(f"  Parameter count    :  {c(model_size, BOLD)}")
    print(f"  Model quantization :  {c(model_quant, BOLD)}")
    print(f"  Vocoder precision  :  {c(vocoder_prec, BOLD)}")
    print(f"  Mimi precision     :  {c(mimi_prec, BOLD)}")
    print()
    print(f"  TTS model file  :  {c(model_file, CYAN)}")
    print(f"  Tokenizer file  :  {c(tok_file, CYAN)}")
    print()
    print(f"  Output folder   :  {c(str(MODELS_DIR.relative_to(REPO_ROOT)), DIM)}")

    model_dest = MODELS_DIR / model_file
    tok_dest   = MODELS_DIR / tok_file

    skip_model = model_dest.exists()
    skip_tok   = tok_dest.exists()

    if skip_model:
        print()
        print(c(f"  [ok] TTS model already present -- skipping.", GREEN))
    if skip_tok:
        print()
        print(c(f"  [ok] Tokenizer already present -- skipping.", GREEN))
    if skip_model and skip_tok:
        print()
        print(c("  Both files already downloaded. Nothing to do.", GREEN))
        print()
        sys.exit(0)

    if not ask_yes_no("Proceed with download?", default=True):
        print(c("  Aborted.", YELLOW))
        sys.exit(0)

    # ── Download ─────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not skip_model:
        print()
        print(c(f"  Downloading: {model_file}", BOLD))
        print(c(f"  From: huggingface.co/{HF_REPO}", DIM))
        hf_download_file(HF_REPO, model_file, MODELS_DIR, token)
        print(c(f"  [done] {model_file}", GREEN))

    if not skip_tok:
        print()
        print(c(f"  Downloading: {tok_file}", BOLD))
        print(c(f"  From: huggingface.co/{HF_REPO}", DIM))
        hf_download_file(HF_REPO, tok_file, MODELS_DIR, token)
        print(c(f"  [done] {tok_file}", GREEN))

    # ── Usage hint ───────────────────────────────────────────────────────────
    print()
    print(c("=" * 62, GREEN))
    print(c("  Download complete!", BOLD + GREEN))
    print(c("=" * 62, GREEN))
    print()
    print("  Basic synthesis:")
    print(c('    qwen3-tts-cli -m models -t "Hello world" -o hello.wav', CYAN))
    if mimi_prec == "f32":
        print()
        print("  ICL voice cloning (uses your F32 Mimi encoder):")
        print(c(
            '    qwen3-tts-cli -m models -t "Hello" '
            '-r ref.wav --ref-text "Transcript" -o icl.wav',
            CYAN,
        ))
    print()


if __name__ == "__main__":
    main()

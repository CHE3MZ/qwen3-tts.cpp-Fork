# Model Converter — Interactive Setup Wizard

Downloads Qwen3-TTS models from HuggingFace and converts them to GGUF format for use with qwen3-tts.cpp.

## Quick start

**Windows:**
```bat
setup_models.bat
```

**macOS / Linux:**
```bash
chmod +x setup_models.sh
./setup_models.sh
```

The wizard walks you through:
1. Model variant (Base / CustomVoice / VoiceDesign)
2. Model size (0.6B / 1.7B)
3. Quantization (F16 / Q8_0 / Q6_K / Q5_K / Q4_K)
4. Mimi encoder precision (for ICL voice cloning quality)
5. HuggingFace token (if needed for gated repos)
6. Download + conversion

## Non-interactive mode (CI / scripting)

```bash
./setup_models.sh --non-interactive        # Downloads 0.6B Base, F16 quant
./setup_models.sh --hf-token hf_xxx...    # With HuggingFace token
```

## Output files

After setup, your `models/` directory will contain:

```
models/
  qwen3-tts-0.6b-f16.gguf          ← TTS transformer (example)
  qwen3-tts-tokenizer-f16.gguf     ← Shared vocoder + Mimi encoder
```

## Model selection guide

| Variant | Use case |
|---|---|
| **Base** | Clone a voice from a reference WAV file |
| **CustomVoice** | Pick from named built-in speakers |
| **VoiceDesign** | Describe a voice in natural language (**1.7B only** — no 0.6B model exists) |

| Quantization | Size (0.6B) | Quality |
|---|---|---|
| F32 | ~3.5 GB | Same as F16 (source is BF16) — not recommended |
| **F16** | ~1.75 GB | Full precision — recommended |
| Q8_0 | ~1.0 GB | Virtually lossless |

> **Note:** K-quants (Q6_K, Q5_K, Q4_K, Q3_K, Q2_K) are not yet supported.
> They are disabled until native K-quant support is added to the converter.

## Prerequisites

- Python 3.10+
- `uv` (recommended) or `pip`
- Required packages: `huggingface_hub gguf torch safetensors numpy tqdm`

Install uv (fastest): `pip install uv`

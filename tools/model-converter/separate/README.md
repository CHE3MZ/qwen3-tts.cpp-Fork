# Separate Download Scripts

Download only what you need — TTS model or tokenizer independently.

## download-model.bat — TTS transformer only

```bat
REM Base 0.6B Q8_0 (recommended default)
download-model.bat --variant base --size 0.6b --type q8_0

REM Base 0.6B F16 (full precision)
download-model.bat --variant base --size 0.6b --type f16

REM 1.7B variants
download-model.bat --variant base --size 1.7b --type q8_0
download-model.bat --variant custom_voice --size 1.7b --type q8_0
download-model.bat --variant voice_design --size 1.7b --type q8_0

REM With HuggingFace token (for gated repos)
download-model.bat --hf-token hf_xxx... --variant base --size 0.6b --type q8_0
```

**Supported types:** `f16`, `f32`, `q8_0`

> **Note:** K-quants (q6_k, q5_k, q4_k, q3_k, q2_k) are not yet supported by the
> Python gguf library and will fall back to F16 silently. They are disabled until
> native K-quant support is added to the converter.
>
> F32 produces the same audio quality as F16 (source weights are BF16) but doubles
> the file size. Useful for debugging or bit-exact reference comparisons.

**Outputs:** `models\qwen3-tts-{size}-{type}.gguf`

## download-tokenizer.bat — Vocoder + Mimi encoder only

```bat
REM Default (F16 vocoder + F16 Mimi)
download-tokenizer.bat

REM F32 Mimi encoder — bit-exact ICL voice cloning (recommended if you use ICL)
download-tokenizer.bat --type f16 --mimi-type f32

REM With token
download-tokenizer.bat --hf-token hf_xxx...
```

**Output filename:** `models\qwen3-tts-tokenizer-{type}-{mimi-type}.gguf`

Examples:
- `--type f16 --mimi-type f32` → `qwen3-tts-tokenizer-f16-f32.gguf` ← best for ICL
- `--type f16 --mimi-type f16` → `qwen3-tts-tokenizer-f16-f16.gguf` ← good general use
- `--type f16 --mimi-type q8_0` → `qwen3-tts-tokenizer-f16-q8_0.gguf`

The `{type}` tag controls the **vocoder** (WavTokenizer decoder) precision.
Supported: `f16` (recommended), `f32` (no benefit over f16, double size).
Q8_0 is not available for the vocoder — 3D conv weights cannot be Q8_0 quantized.

The `{mimi-type}` tag controls the **Mimi encoder** precision (ICL voice cloning only).
Supported: `f32` (bit-exact, best for ICL), `f16` (98.9% match), `q8_0` (94.3% match, not recommended for ICL).

The tokenizer is **shared** across all TTS model variants and sizes.
Download it **once** — it works with 0.6B, 1.7B, Base, CustomVoice, VoiceDesign.

## Notes

- Both scripts resume interrupted downloads automatically (retry x5 with backoff).
- Re-running skips already-downloaded source files and already-converted GGUFs.
- Delete the output `.gguf` to force re-conversion with different settings.
- VoiceDesign is 1.7B only — `--variant voice_design --size 0.6b` is rejected.

# Separate Download Scripts

Download only what you need — TTS model or tokenizer independently.

## download-model.bat — TTS transformer only

```bat
REM Base 0.6B Q8_0 (recommended default)
download-model.bat --variant base --size 0.6b --type q8_0

REM Base 0.6B all quants
download-model.bat --variant base --size 0.6b --type f16
download-model.bat --variant base --size 0.6b --type q8_0
download-model.bat --variant base --size 0.6b --type q6_k
download-model.bat --variant base --size 0.6b --type q5_k
download-model.bat --variant base --size 0.6b --type q4_k
download-model.bat --variant base --size 0.6b --type q3_k
download-model.bat --variant base --size 0.6b --type q2_k

REM 1.7B variants
download-model.bat --variant base --size 1.7b --type q8_0
download-model.bat --variant custom_voice --size 1.7b --type q8_0
download-model.bat --variant voice_design --size 1.7b --type q8_0

REM With HuggingFace token (for gated repos)
download-model.bat --hf-token hf_xxx... --variant base --size 0.6b --type q8_0
```

**Outputs:** `models\qwen3-tts-{size}-{type}.gguf`

## download-tokenizer.bat — Vocoder + Mimi encoder only

```bat
REM Default (F16, recommended)
download-tokenizer.bat

REM F32 Mimi encoder (bit-exact ICL voice cloning)
download-tokenizer.bat --type f16 --mimi-type f32

REM With token
download-tokenizer.bat --hf-token hf_xxx...
```

**Output:** `models\qwen3-tts-tokenizer-f16.gguf`

The tokenizer is **shared** across all TTS model variants and sizes.
Download it **once** — it works with 0.6B, 1.7B, Base, CustomVoice, VoiceDesign.

## Notes

- Both scripts resume interrupted downloads automatically (retry x5 with backoff).
- Re-running skips already-downloaded source files and already-converted GGUFs.
- Delete the output `.gguf` to force re-conversion with different settings.
- VoiceDesign is 1.7B only — `--variant voice_design --size 0.6b` is rejected.

# model-downloader

Downloads pre-built GGUF files directly from HuggingFace. No Python conversion
dependencies required beyond `huggingface_hub` — no `torch`, `safetensors`, or
`gguf` packages needed.

## Usage

**Windows:**
```bat
tools\model-downloader\download.bat
```

**macOS / Linux:**
```bash
chmod +x tools/model-downloader/download.sh
./tools/model-downloader/download.sh
```

The script walks you through 5 steps:

1. **Model type** — Base (voice cloning), Custom Voice (named speakers), Voice Design (text description)
2. **Parameter count** — 0.6B or 1.7B (Voice Design is 1.7B only)
3. **Model quantization** — F16 (recommended), Q8_0 (smaller), F32 (not recommended)
4. **Vocoder precision** — F16 (recommended), F32 (not recommended)
5. **Mimi encoder precision** — F32 (recommended for ICL cloning), F16 (general use), Q8_0 (not recommended)

## Output

Files are saved to `models/`:

```
models/
  qwen3-tts-{size}-{quant}.gguf              ← TTS transformer
  qwen3-tts-tokenizer-{vocoder}-{mimi}.gguf  ← Shared vocoder + Mimi encoder
```

Both files are required to run synthesis.

## Requirements

- Python 3.10+
- `huggingface_hub` — install with `pip install huggingface_hub`

## Source

All files come from: [huggingface.co/librellama/qwen3-tts-GGUF](https://huggingface.co/librellama/qwen3-tts-GGUF)

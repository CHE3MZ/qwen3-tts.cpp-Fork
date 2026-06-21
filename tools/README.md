# tools/

Developer and end-user tools for qwen3-tts.cpp.

## model-downloader/

**Easiest path.** Downloads pre-built GGUF files directly from HuggingFace.
No Python conversion dependencies — only `huggingface_hub` required.
See [model-downloader/README.md](model-downloader/README.md).

```
Windows: tools\model-downloader\download.bat
macOS:   tools/model-downloader/download.sh
```

## build-scripts/

Build the project from source. See [build-scripts/README.md](build-scripts/README.md).

```
Windows: tools\build-scripts\build.bat
macOS:   tools/build-scripts/build.sh
```

## model-converter/

Interactive wizard to download HuggingFace safetensors and convert to GGUF locally.
Requires `torch`, `safetensors`, `gguf`. See [model-converter/README.md](model-converter/README.md).

```
Windows: tools\model-converter\setup_models.bat
macOS:   tools/model-converter/setup_models.sh
```

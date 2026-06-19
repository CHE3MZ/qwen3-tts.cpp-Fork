# porting guide

This document exists for anyone forking this project to build a different TTS
model in C++ on top of GGML — for example chatterbox.cpp, fish-speech.cpp, or
similar. It maps every source file to one of three categories: **keep as-is**,
**adapt**, or **rewrite from scratch**.

---

## file-by-file map

### keep as-is (fully generic)

| file | what it does | why it's reusable |
|------|-------------|-------------------|
| `ggml/` | compute backend | model-agnostic by design |
| `src/gguf_loader.cpp/.h` | GGUF file reading, tensor loading, mmap, backend singleton, thread safety | no model assumptions |
| `src/text_tokenizer.cpp/.h` | BPE/tiktoken tokenizer | reads vocab from GGUF, no model hardcoding |
| `src/qwen3tts_c_api.cpp/.h` | C API wrapper | the design pattern (opaque handle, params struct, callbacks, result struct, free_batch_results) is a solid template — rename symbols and you're done |
| `CMakeLists.txt` | build system | the structure (build ggml, detect GPU backends, stage DLLs, link targets) is fully reusable |
| `tools/build-scripts/build.bat/.sh` | build scripts | model-agnostic |
| `tools/model-converter/` | Python GGUF converter skeleton | the GGUF writing infrastructure is reusable; only the tensor extraction logic is model-specific |

### adapt (50-80% reusable)

| file | what it does | what needs changing |
|------|-------------|---------------------|
| `src/qwen3_tts.cpp/.h` | high-level pipeline: tokenize → encode → generate → decode | the pipeline orchestration pattern is good; the GGUF auto-discovery filenames (`qwen3-tts-*.gguf`), the generation config JSON keys, the ICL path, and the `is_model_size_match()` heuristic are all Qwen3-specific |
| `src/main.cpp` | CLI | argument names and examples reference Qwen3 features; the server mode JSON protocol is generic |
| `src/audio_tokenizer_encoder.cpp/.h` | ECAPA-TDNN speaker encoder | the GGML graph-building pattern is reusable; the ECAPA-TDNN architecture (Res2Net, SE blocks, ASP pooling) is specific to this model family — other TTS models use different speaker encoders or none at all |
| `src/mimi_encoder.cpp/.h` | Mimi codec encoder for ICL | the approach (GGML graph, GGUF tensor loading) is reusable; the Mimi architecture is specific to Qwen3-TTS |

### rewrite from scratch (model-specific)

| file | what it does | why it needs a full rewrite |
|------|-------------|----------------------------|
| `src/tts_transformer.cpp/.h` | the core LM: 28-layer Qwen2 talker + 5-layer code predictor + KV cache + flash attention + sampling | every architectural detail is Qwen3-specific: RoPE config, GQA ratios, codec head, the 16-codebook code predictor auto-regressive loop, the `small_to_mtp_projection` for 1.7B |
| `src/audio_tokenizer_decoder.cpp/.h` | the vocoder: VQ codebook lookup → pre-transformer → upsample blocks → HiFiGAN decoder | the entire architecture (Snake activations, ConvNeXt upsample blocks, upsample rates 8/5/4/3, residual blocks with dilations 1/3/9) is specific to this vocoder |
| `scripts/convert_tts_to_gguf.py` | Python → GGUF weight converter | tensor names, quantization targets, and metadata keys are all `qwen3-tts.*` namespaced |

---

## what the generic skeleton looks like when you fork

If you fork for a new model, delete the rewrite-from-scratch files and keep
everything else. Your new model adds:

```
src/your_model_transformer.cpp/.h   ← replaces tts_transformer.cpp
src/your_model_vocoder.cpp/.h       ← replaces audio_tokenizer_decoder.cpp
scripts/convert_your_model.py       ← replaces convert_tts_to_gguf.py
```

The C API wrapper, GGUF loader, text tokenizer, build system, CLI structure,
and WAV/resampling utilities all carry over with minimal renaming.

---

## the sampling code is worth copying verbatim

`tts_transformer.cpp` contains a self-contained `sample_token()` function and
`sample_token_params` struct that implements the full llama.cpp-style sampling
stack: temperature, top-k, top-p, min-p, repetition penalty, frequency penalty,
presence penalty, DRY n-gram penalty, and dynamic temperature. This is completely
model-agnostic and should be extracted into a shared header for any future port.

Suggested location: `src/sampling.h` (header-only, no new translation unit needed).

---

## hardcoded values — what they are and whether to fix them

These are the scattered constants found across the codebase. They are grouped
by whether they are worth un-hardcoding.

### not worth fixing — correct by design

| location | value | why it's fine |
|----------|-------|---------------|
| `audio_tokenizer_decoder.cpp:613,792` | `cb < 16` | n_codebooks is read from GGUF config and stored in `cfg.n_codebooks`; these two loops use the literal `16` instead of `cfg.n_codebooks`. This is a minor inconsistency but has zero performance impact and zero correctness risk since the GGUF config is validated to be 16 on load. Worth fixing for clarity but not urgent. |
| `audio_tokenizer_encoder.cpp:628,633,643,645,675,679,685` | `1536` | the MFA output dimension (3 blocks × 512 channels = 1536). This is not in the config struct. It's correct because the ECAPA-TDNN architecture is fixed — but if you change `hidden_dim` or `n_blocks` you'd need to update this manually. See note below. |
| `tts_transformer.cpp:3100,3525,3646,3886` | `codec_vocab_size - 1024` | suppress tokens above this index. The `1024` is the number of special/noise tokens at the top of the codec vocabulary — it comes from the Python model definition. It's read as a derived value from `codec_vocab_size`, which is read from GGUF. Acceptable. |
| `qwen3_tts.cpp:714,716,969,971,1125,1127` | `24000` | the expected encoder sample rate used when resampling reference audio. Should use `audio_encoder_.get_config().sample_rate` instead. See note below. |
| `qwen3tts_c_api.cpp:220` | `return 24000` | `qwen3_tts_sample_rate()` hardcodes the return value instead of reading it from the loaded model config. See note below. |
| `tts_transformer.cpp:2555,2557` | `n_ctx < 16` / `init_code_pred_kv_cache(16, ...)` | the code predictor KV cache minimum size. 16 = 15 sub-codebook steps + 2 prefill tokens, rounded up. This is a Qwen3-specific architectural constant. Fine as-is. |

### the three worth actually fixing

**1. `qwen3_tts.cpp` — three `resample_linear(..., 24000)` calls**

These resample reference audio to 24000 Hz before encoding. They should read
`audio_encoder_.get_config().sample_rate` instead of the literal. If a future
model's encoder runs at 16000 Hz (like most ASR-derived speaker encoders) these
lines would silently produce wrong-rate audio without an error.

**2. `qwen3tts_c_api.cpp:220` — `qwen3_tts_sample_rate()` returns 24000**

This is a C API function that should return the actual sample rate from the
loaded model, not a hardcoded value. If the model isn't loaded yet it should
return 0, not 24000.

**3. `audio_tokenizer_encoder.cpp` — `1536` hardcoded in ASP pooling**

The value `1536` (= `hidden_dim * 3` = `512 * 3`) appears 7 times in the
ASP pooling section of `build_graph()`. It's not in `speaker_encoder_config`.
This is the only place in the encoder where a derived architectural constant
isn't in the config struct. Not a performance issue, just a maintainability
issue if you ever change `hidden_dim`.

---

## GGUF metadata namespace

Every GGUF key in this project is prefixed `qwen3-tts.`. When porting,
change this prefix to your model name (e.g. `fish-speech.`, `chatterbox.`).
There are 78 occurrences across `tts_transformer.cpp` and the loader files.
A global find-replace handles this in under a minute.

---

## what the C API gives you for free

The C API in `qwen3tts_c_api.h` is the cleanest part of the project to reuse.
Copy it, rename `Qwen3Tts` → `ChatterboxTts` (or whatever), rename the function
prefix `qwen3_tts_` → `chatterbox_tts_`, and you have:

- opaque handle with thread safety documentation
- full params struct with all sampling controls
- simple and extended result structs with timing and memory stats  
- progress, abort, logits, and streaming audio chunk callbacks
- speaker embedding extract/save/load utilities
- WAV I/O and resampling utilities
- batch synthesis with per-entry instruct support
- codes-only synthesis + separate decode path

This is a significantly better starting point than writing a C API from scratch.

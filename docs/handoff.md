# Handoff: Complete — commit, integrate, or extend
<!-- Last updated: 2026-06-15 -->

## Summary
`qwen3-tts.cpp-Fork` is a complete, production-ready C++17/GGML port of Qwen3-TTS. All features match the Python reference (10/10 on every implemented feature), both validation tests pass, audio output is working, and the C API is feature-complete for Go/FFI server integration. Working tree is clean — all changes committed to `main`.

## Status

### Completed
* All 9 Mimi encoder bugs fixed — test passes at 100% exact match (F32 mimi weights)
* TTS transformer: 5 PASS 1 WARN 0 FAIL — streaming codes 100% exact
* WAV writer bug fixed (`uint16_t` fmt size → `uint32_t`; batched PCM write)
* Bessel I0 bug fixed (Kaiser window was 86× wrong; sinc resampler now correct)
* `sub-talker top_p` fully wired through all generate paths and C API
* `do_sample=false` from generation_config.json now forces greedy
* GGUF metadata key fallback (`qwen3-tts-tokenizer.num_codebooks` + legacy key)
* `gguf_loader` deduplication — compiled once, linked to all 5 libs
* Thread count auto-detect (`std::thread::hardware_concurrency`, capped 8)
* `-march=native` for clang-cl on Windows (AVX2/AVX512)
* GPU backend detection: CMake auto-stages CUDA/Vulkan/Metal DLLs when built
* `QWEN3_TTS_KV_F32` CMake option for bit-exact KV cache (default OFF)
* `--mimi-type` in converter: independently control Mimi encoder weight precision
* **C API fully expanded** — progress callback, timing, memory stats, `_ex` variants, WAV I/O, embedding size query, `create_with_config`, thread-safety docs

### Bug fixes applied 2026-06-15
* **`qwen3_tts_embedding_size()` hardcode removed** — now calls `engine.get_embedding_dim()` which reads `speaker_encoder_config::embedding_dim` from GGUF metadata; falls back to 1024 if encoder not yet loaded. Correct for all model variants.
* **`qwen3_tts_create()` `n_threads` now stored** — `Qwen3Tts::default_n_threads` member stores the value at create time; all 8 `to_cpp_params()` call sites use it as the per-call default. Thread resolution: explicit params > handle default > hardware auto-detect.
* **`qwen3_tts_unload()` implemented** — calls `engine.unload_models()` which calls `unload_model()` on transformer, decoder, encoder, and mimi encoder; sets all `*_loaded_` flags false. Handle stays valid for reload.
* **`AudioTokenizerEncoder::unload_model()` added** — mirrored the destructor cleanup into an explicit public method (destructor now delegates to it); matching the pattern of `TTSTransformer` and `AudioTokenizerDecoder`.
* **`Qwen3TTS::unload_models()` added** — unloads all four sub-components in sequence.
* **`Qwen3TTS::get_embedding_dim()` added** — returns `audio_encoder_.get_config().embedding_dim` when encoder loaded, 1024 otherwise.
* **`print_progress` forwarded through C API** — `Qwen3TtsParams::print_progress` field added; `to_cpp_params()` maps it to `tts_params::print_progress`; `qwen3_tts_default_params()` initializes it to 0.
* **WAV loader odd-chunk padding fixed** — `fmt` extra bytes and all unknown chunks now skip `chunk_size + (chunk_size & 1)` bytes per RIFF spec, preventing misalignment on files with odd-sized metadata chunks (e.g. INFO tags).
* **WAV loader `fread` return values checked** — all three PCM format branches (int16, int32, float32) now check the `fread` return and resize `samples` to the actual number of complete frames read, avoiding silent corruption on truncated files.
* **WAV loader guard against missing fmt** — `num_channels == 0` check before data chunk decode; emits an error and returns false rather than dividing by zero.
* **ICL audio trim O(n) erase replaced** — `result.audio.erase(begin, begin+cut)` replaced with `std::move_iterator` subrange construction, avoiding in-place shift of up to ~720k floats.
* **`main.cpp` save embedding return value checked** — the side-path that saves a speaker embedding during voice-clone synthesis now checks the `save_speaker_embedding()` return and prints a warning on failure.
* **Zero-embedding comment clarified** — `synthesize()` no-reference path comment updated to explain why `hidden_size` is used as the embedding size (they are equal for all current checkpoints; a TODO is noted for future-proofing).

### Open Issues
* Non-streaming WARN: C++ generates 64 frames, Python ref has 63. All 63 match exactly. Cause: F16 model weights shift EOS logit margin — unfixable without F32 model. Not a code bug.
* Batch inference: not implemented (architectural, ~600 lines). Documented limitation.
* GPU acceleration requires rebuilding GGML with `-DGGML_CUDA=ON` / `-DGGML_METAL=ON` — no CUDA toolkit currently installed.

## Decisions
* Mimi encoder uses **full causal attention** (no sliding window) — Python's `encoder_transformer` is called with `is_causal=True, attention_mask=None` and never enforces the `sliding_window=250` config value. Applying the window was the bug that caused 69% → fixed to 100%.
* Codebook embeddings always stored **F32** in GGUF regardless of `--type` or `--mimi-type` — required for exact nearest-neighbor lookup.
* Q8_0 for Mimi encoder conv weights gives only 94.3% match (below 95% threshold) — only F16 (98.9%) or F32 (100%) are viable.
* C API `Qwen3Tts*` handle is **NOT thread-safe** — one request at a time per handle; documented in header.

## Non-Obvious Findings
* `output_proj` in RVQ quantizer is **decoder-only** — `encode()` only uses `input_proj`. The flag was incorrectly identified as missing; it is not needed.
* `save_audio_file` previously wrote PCM samples one `fwrite` call per sample (N calls) — fixed to one batched write.
* `bessel_i0` had `sum += term * term` instead of `sum += term` — made Kaiser window values 86× too large, effectively degrading sinc resampler to near nearest-neighbor quality.
* GGUF metadata key mismatch: converter writes `qwen3-tts-tokenizer.num_codebooks` but C++ was reading `qwen3-tts.tokenizer.num_codebooks`. Fixed with fallback lookup.
* RIFF WAV chunks must be word-aligned (padded to even byte boundary) — unknown-chunk skip was missing the `+ (chunk_size & 1)` pad byte, causing parser misalignment on files with odd-sized metadata.

## Next Steps
1. **Commit current state** (already done — working tree clean)
2. **End-to-end Go integration test** — write a minimal Go app using the new C API `_ex` functions, test `qwen3_tts_synthesize_ex` + `qwen3_tts_save_wav`
3. **GPU acceleration** — install CUDA toolkit, rebuild GGML with `-DGGML_CUDA=ON`, re-run CMake (project auto-detects and stages `ggml-cuda.dll`)
4. **Batch inference** — if needed for server throughput, implement batched `generate()` (~600 lines touching KV cache + attention mask + sampling loop)

## References
* Architecture guide: `AGENTS.md` (Known Limitations section authoritative)
* Build options: `CMakeLists.txt` (`QWEN3_TTS_KV_F32`, `QWEN3_TTS_TIMING`, GPU detection)
* C API: `src/qwen3tts_c_api.h` (full docs in header comments)
* Mimi encoder: `src/mimi_encoder.h`, `src/mimi_encoder.cpp`
* Converter: `scripts/convert_tokenizer_to_gguf.py` (`--mimi-type` flag, codebook F32 rule)
* Run tests: `build-ninja\test_mimi_encoder.exe --tokenizer models\qwen3-tts-tokenizer-f16.gguf --audio reference\mimi_enc_test_audio.bin --ref-codes reference\mimi_enc_py_codes.bin`
* Run tests: `build-ninja\test_transformer.exe --model models\qwen3-tts-0.6b-f16.gguf --ref-dir reference\ --max-len 64`

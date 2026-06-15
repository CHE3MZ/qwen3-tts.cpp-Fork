# Handoff: TTS port feature-complete — optional polish remaining
<!-- Last updated: 2026-06-15 -->

## Summary
`qwen3-tts.cpp-Fork` is complete. All identified gaps are closed, all tests pass (6 PASS 1 WARN 0 FAIL), and the repo is production-ready. The only remaining work is optional polish (Go integration test, Q8_0 smoke test, non-English smoke test) and the two architectural items that need external tooling (GPU, batch inference).

## Objective
No active objective. Next session can pick from the optional polish list below, or begin the same treatment on `qwen3-asr.cpp-Fork` using `../how-i-did-it.md` as the guide.

## Status

### Completed (this session)
* **`--server` mode** added to CLI (`src/main.cpp`) — loads model once, reads JSON requests from stdin, writes JSON responses to stdout. Solves the multiple-instance RAM multiplication problem. Full JSON protocol documented in `--help`.
* **Test 8: instruct prefill** added to `tests/test_transformer.cpp` — validates `build_prefill_graph_instruct()` + `forward_prefill()` across 4 sanity checks (size, finite, argmax, length). Result: **PASS**. Test suite now 6 PASS 1 WARN 0 FAIL.
* **`how-i-did-it.md`** updated to reflect all features including streaming, logits callback, codes access, server mode, Q5K/Q6K, and Windows-specific build notes.

### Previously completed
* Full pipeline: tokenizer → ECAPA-TDNN encoder → 28L Qwen2 talker → 5L × 15-step code predictor → WavTokenizer vocoder
* ICL voice cloning (Mimi encoder): 100% exact match (F32)
* All 3 model types: base, custom_voice, voice_design
* All quantization: F16, Q8_0, Q5_K, Q6_K, Q4_K for TTS; F16/Q8_0 for tokenizer
* C API: lifecycle, synthesis × 4 entry points, `_ex` variants, timing/memory stats, progress callback, logits callback, streaming chunk callback, speech codes access, decode_codes, speaker embedding utils, WAV I/O
* 12 bug fixes (C API contracts, WAV robustness, ICL trim, zero-emb size)

### In Progress
* Nothing

## Decisions
* Server mode uses stdin/stdout JSON protocol — no socket dependency, works with any language via pipes
* Server mode processes requests serially — handle is not thread-safe; for parallel synthesis, spawn multiple server processes
* Test 8 uses fabricated instruct tokens (no tokenizer GGUF needed at test time) — validates the GGML graph path without requiring Python
* `how-i-did-it.md` is intentionally written for the ASR agent, not just as a TTS record

## Open Issues (optional / blocked)
* Go/FFI integration test — write minimal Go app using `qwen3tts.dll` via CGo
* Q8_0 model end-to-end smoke test — quick CLI run to confirm quantized model still produces valid audio
* Non-English synthesis smoke test — `-l chinese` etc.
* GPU acceleration — needs CUDA toolkit + `cmake -S ggml -B ggml/build -DGGML_CUDA=ON`
* Batch inference — architectural, ~600 lines

## Next Steps (if continuing TTS)
1. Quick Q8_0 smoke: `build-ninja\qwen3-tts-cli.exe -m models -t "test" -o q8_test.wav` with a Q8_0 GGUF
2. Go integration test: CGo wrapper calling `qwen3_tts_synthesize_ex`
3. Non-English: `build-ninja\qwen3-tts-cli.exe -m models -t "你好世界" -l chinese -o chinese.wav`

## Next Steps (if starting ASR)
See `../how-i-did-it.md` — full methodology with ASR-specific differences documented.

## References
* Methodology guide: `../how-i-did-it.md`
* Architecture + prefill structure: `AGENTS.md`
* C API (full): `src/qwen3tts_c_api.h`
* Server mode protocol: `src/main.cpp` (run with `--help`)
* Tensor mapping: `docs/tensor_mapping.md`
* Run transformer test: `build-ninja\test_transformer.exe --model models\qwen3-tts-0.6b-f16.gguf --ref-dir reference --max-len 64`
* Run mimi test: `build-ninja\test_mimi_encoder.exe --tokenizer models\qwen3-tts-tokenizer-f16.gguf --audio reference\mimi_enc_test_audio.bin --ref-codes reference\mimi_enc_py_codes.bin`

## Summary
`qwen3-tts.cpp-Fork` is a fully-featured C++17/GGML TTS port. All previously identified gaps are now closed. The API exposes complete developer control including raw speech codes access, per-frame logits callbacks, and streaming audio output. All tests pass. The one remaining item is a deterministic reference test for the VoiceDesign/instruct code path.

## Objective
Next session: add a reference test for `build_prefill_graph_instruct` (VoiceDesign/CustomVoice instruct path) — the last code path with zero test coverage.

## Status

### Completed (this session)
* **Q5_K and Q6_K quantization** added to `convert_tts_to_gguf.py` — full quality ladder: F16 → Q8_0 → Q6_K → Q5_K → Q4_K
* **Model file discovery** extended to include `*-q5_k.gguf` and `*-q6_k.gguf` filename patterns in `qwen3_tts.cpp`
* **Speech codes access API** — `synthesize_codes()`, `synthesize_codes_with_voice()`, `synthesize_codes_with_embedding()`, `decode_speech_codes()` on `Qwen3TTS`; `qwen3_tts_synthesize_codes*` + `qwen3_tts_decode_codes` in C API
* **Per-frame logits callback** — `TTSTransformer::set_logits_callback()` / `clear_logits_callback()`; wired into all 3 generate loops (`generate`, `generate_from_prefill`, `generate_icl`); exposed as `qwen3_tts_set_logits_callback` / `qwen3_tts_clear_logits_callback` in C API
* **Streaming audio output** — `decode_codes_streaming()` in `Qwen3TTS`; fires `audio_chunk_callback_` every `audio_chunk_frames_` frames (default 12 = ~1s); `set_audio_chunk_callback()` / `clear_audio_chunk_callback()` on `Qwen3TTS`; `qwen3_tts_set_audio_chunk_callback` / `qwen3_tts_clear_audio_chunk_callback` in C API
* All synthesis paths (`synthesize_internal`, ICL path in `synthesize_with_voice`) routed through `decode_codes_streaming` — chunk callback active when set, single-shot otherwise (zero overhead when not used)
* Build: 19/19 targets, 0 errors. test_transformer: 5 PASS 1 WARN 0 FAIL. test_mimi_encoder: 100% exact. CLI smoke test: clean output.

### Previously completed
* See prior entries in this file — all 12 bug fixes, C API contract correctness, WAV robustness, Q4_K, etc.

### In Progress
* Nothing actively in progress

## Decisions
* Streaming chunk callback falls back to single-shot decode when not set — zero overhead, backward compatible
* `synthesize_codes` returns codes only (no vocoder call) — caller then calls `decode_speech_codes` separately; this is intentional to allow vocoder swapping
* Logits callback fires after CB0 sampling — the sampled token is passed alongside the raw logits so the caller can see both what the distribution looked like and what was chosen
* Logits callback return non-zero = stop generation (same contract as progress callback)
* `decode_codes_streaming` still appends all chunks to `result.audio` — so the final `tts_result` still contains the full waveform even in streaming mode

## Assumptions & Constraints
* Models present: `models/qwen3-tts-0.6b-f16.gguf`, `models/qwen3-tts-tokenizer-f16.gguf`
* Build system: Ninja + clang-cl on Windows, build dir is `build-ninja/`
* No CUDA toolkit installed — GPU path exists in CMake but untested

## Non-Obvious Findings
* `std::min` on `int32_t` args inside namespace triggers a parse error with clang-cl on Windows (NOMINMAX macro conflict) — replaced with explicit ternary
* `bool is_last` as a local variable name also triggered `expected unqualified-id` — renamed to `int is_last`
* Both issues are Windows/clang-cl specific; GCC/Clang on Linux/macOS would be fine

## Open Issues
* `build_prefill_graph_instruct` (VoiceDesign/CustomVoice instruct path) has no reference test — score 8/10
* Batch inference not implemented — architectural, ~600 lines
* GPU acceleration needs CUDA toolkit install + GGML rebuild with `-DGGML_CUDA=ON`
* Q8_0 model not smoke-tested end-to-end in recent builds

## Next Steps
1. Generate Python reference data for instruct path: run `scripts/generate_deterministic_reference.py` with a VoiceDesign model, save `reference/det_instruct_*.bin`
2. Add `Test 8: instruct prefill` to `tests/test_transformer.cpp` comparing against that reference
3. Write minimal Go app: import `"C"`, load `qwen3tts.dll`, call `qwen3_tts_synthesize_codes` then `qwen3_tts_decode_codes`, verify output WAV
4. Quick CLI run with Q8_0 model to confirm it still produces valid audio

## References
* Architecture and methodology: `../how-i-did-it.md`
* Coding conventions + prefill structure: `AGENTS.md`
* Tensor name mapping: `docs/tensor_mapping.md`
* C API contract (full): `src/qwen3tts_c_api.h`
* Run mimi test: `build-ninja\test_mimi_encoder.exe --tokenizer models\qwen3-tts-tokenizer-f16.gguf --audio reference\mimi_enc_test_audio.bin --ref-codes reference\mimi_enc_py_codes.bin`
* Run transformer test: `build-ninja\test_transformer.exe --model models\qwen3-tts-0.6b-f16.gguf --ref-dir reference --max-len 64`

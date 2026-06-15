# Handoff: TTS port feature-complete — instruct path test next
<!-- Last updated: 2026-06-15 -->

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

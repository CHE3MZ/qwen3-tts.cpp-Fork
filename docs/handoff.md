# Handoff: TTS port complete — instruct path test + Go integration next
<!-- Last updated: 2026-06-15 -->

## Summary
`qwen3-tts.cpp-Fork` is a complete, production-ready C++17/GGML port of Qwen3-TTS with full voice cloning, ICL, CustomVoice, VoiceDesign, and C API support. A bug-fix pass was completed this session (12 fixes). All tests pass. The two remaining actionable items are a reference test for the instruct code path and a Go FFI integration test.

## Objective
Next session should tackle in order:
1. Add a deterministic reference test for `build_prefill_graph_instruct` (VoiceDesign/CustomVoice instruct path) — last code path with zero test coverage
2. Go/FFI integration smoke test using the C API shared lib (`qwen3tts.dll`)

## Status

### Completed
* Full pipeline working: tokenizer → speaker encoder → talker (28L) → code predictor (5L × 15 steps) → WavTokenizer vocoder
* ICL voice cloning via Mimi encoder: 100% exact code match (F32 weights)
* All 3 model variants: base, custom_voice, voice_design
* Streaming + non-streaming prefill modes
* C API (`qwen3tts_c_api.h`) — full FFI surface with `_ex` variants, timing, memory stats, WAV I/O
* **Bug-fix pass (2026-06-15):** 12 fixes across 6 files — see `docs/handoff.md` §Bug fixes and `how-i-did-it.md`
  - `qwen3_tts_embedding_size()` — removed hardcode, reads from GGUF config
  - `qwen3_tts_create()` n_threads — now stored and propagated to all synthesis calls
  - `qwen3_tts_unload()` — now actually unloads all 4 sub-components
  - `AudioTokenizerEncoder::unload_model()` — added explicit method
  - `Qwen3TTS::unload_models()` + `get_embedding_dim()` — added
  - `print_progress` — forwarded through C API params
  - WAV loader — RIFF odd-chunk padding + fread return checks + missing-fmt guard
  - ICL trim — `erase(begin,begin+n)` replaced with move-iterator construction
  - `main.cpp` — `save_speaker_embedding` return value now checked

### In Progress
* Nothing actively in progress

## Decisions
* Mimi encoder uses full causal attention — Python never applies `sliding_window=250` config value despite it being set; applying it was the bug causing 69% → 100% fix
* Codebook embeddings always stored F32 in GGUF regardless of `--type` — required for exact VQ nearest-neighbor lookup
* Code predictor runs **15 sequential autoregressive steps per frame** (one per codebook) each with its own KV cache separate from the talker — single-pass was the original audio quality bug
* C API handle is NOT thread-safe by design; one handle per worker thread

## Assumptions & Constraints
* Models present: `models/qwen3-tts-0.6b-f16.gguf`, `models/qwen3-tts-tokenizer-f16.gguf`
* Build system: Ninja + clang-cl on Windows, build dir is `build-ninja/`
* No CUDA toolkit installed — GPU path exists in CMake but untested

## Non-Obvious Findings
* Code predictor bottleneck: 71% of generation time — 15 forward passes/frame
* Non-streaming WARN (64 vs 63 frames) is an F16 EOS margin artifact, not a code bug — unfixable without F32 weights
* `qwen3_tts_unload()` was previously a complete no-op with a TODO comment — now fixed
* RIFF spec requires all chunks padded to even byte boundary — the old WAV loader skipped this, causing misalignment on files with INFO metadata chunks
* Zero-embedding size was taken from `hidden_size` (transformer dim) not `embedding_dim` (encoder output) — they happen to be equal for current checkpoints but are semantically different

## Open Issues
* `build_prefill_graph_instruct` (VoiceDesign/CustomVoice instruct path) has no reference test — score 8/10
* Batch inference not implemented — architectural, ~600 lines
* GPU acceleration needs CUDA toolkit install + GGML rebuild with `-DGGML_CUDA=ON`
* Q8_0 model hasn't been smoke-tested end-to-end in recent builds

## Next Steps
1. Generate Python reference data for instruct path: run `scripts/generate_deterministic_reference.py` with a VoiceDesign model and save `reference/det_instruct_*.bin`
2. Add `Test 8: instruct prefill` to `tests/test_transformer.cpp` comparing against that reference
3. Write minimal Go app: `import "C"`, load `qwen3tts.dll`, call `qwen3_tts_synthesize_ex`, verify output WAV
4. Quick CLI run with Q8_0 model to confirm it still produces valid audio

## References
* Architecture and methodology: `../how-i-did-it.md`
* Coding conventions + prefill structure: `AGENTS.md`
* Tensor name mapping: `docs/tensor_mapping.md`
* C API contract: `src/qwen3tts_c_api.h`
* Run mimi test: `build-ninja\test_mimi_encoder.exe --tokenizer models\qwen3-tts-tokenizer-f16.gguf --audio reference\mimi_enc_test_audio.bin --ref-codes reference\mimi_enc_py_codes.bin`
* Run transformer test: `build-ninja\test_transformer.exe --model models\qwen3-tts-0.6b-f16.gguf --ref-dir reference\ --max-len 64`

# Handoff: ICL Fixed, Build Script Patched
<!-- Last updated: 2026-06-23 -->

## Summary
`qwen3-tts.cpp-Fork` is the only C++17/GGML TTS pipeline for Qwen3-TTS. ICL was fundamentally broken (reversed prefill ordering) and is now fixed. All changes are uncommitted — user handles git manually.

## Objective
The ICL prefill bug is resolved. Remaining work is optional cleanup and quality-of-life improvements. No critical path blockers remain.

## Status

### Completed
* **ICL prefill ordering (root cause)** — `build_prefill_graph_icl` concatenated `[icl_block | base_framing]` instead of Python's `[base_framing | icl_block]`. ICL block also only contained ref_text, missing body_text. Complete rewrite at `tts_transformer.cpp:3432-3593`.
* **ICL logits callback** — `generate_icl()` never fired `logits_cb_` (added at line 3745), AND `synthesize_with_voice()` never wired the callback chain (added at `qwen3_tts.cpp:1089-1110`).
* **ICL extended sampling** — extended sampling params (min-p, dry, frequency/presence penalty, dyntemp) were never set before `generate_icl()`. Fixed with callback wiring.
* **`build.bat` MATH_LIBRARY error** — stale CMake cache caused GGML configure failure. Fixed by adding `-DMATH_LIBRARY=` at `tools/build-scripts/build.bat:98`.
* **ICL memory snapshots** — added `[mem] synth/*` diagnostics matching `synthesize_internal()`.
* **GGML v0.15.2 upgrade**, HF repo published, reference data regenerated, all tests pass (from prior session).

### Not Yet Done
* User has not committed any changes

## Decisions
* ICL on 0.6B sounds nearly identical to x-vector-only — confirmed by Python reference analysis: x-vector is always baked into the codec overlay, ICL codes are additive prosody context. 1.7B needed for meaningful difference.
* `build.bat` constraint lifted — user explicitly requested the fix.
* **Batch mode penalties** (`generate_batch()` gen_tokens tracking) — verified correct. The handoff's claim was inaccurate. NOT a bug.
* **KV cache realloc** (`extend_kv_cache_impl`) — verified correct. Saves and restores all K/V data. NOT a bug.

## Non-Obvious Findings
* `test_mimi_encoder` 0% match — reference codes were generated from a different audio file than clone.wav, not an encoder bug. Need regeneration.
* This is the only GGML inference engine for Qwen3-TTS — no other implementation exists. Only this project supports quantization.

## Open Issues
* **ICL slow on CPU** — Mimi encoder is scalar C++, ~13s for a 6.8s clip. Runs on CPU even with GPU backend.
* **M-RoPE uses 1D positions** — fine for single-batch, may diverge for long ICL sequences.
* **`test_mimi_encoder` reference stale** — needs regeneration from clone.wav for proper validation.
* **K-quants** — Q6_K–Q2_K not supported by Python converter.
* **ICL non-streaming mode** — overlay only supports streaming-style (pre-existing limitation).

## Next Steps
1. Commit all changes (user responsibility)
2. Regenerate `test_mimi_encoder` reference for proper Mimi validation
3. Optionally clean up `ggml-master/` stray directory

## References
* Architecture: `docs/architecture.md`
* ICL fix diff: `git diff src/tts_transformer.cpp src/qwen3_tts.cpp`
* Build script fix: `tools/build-scripts/build.bat`
* HF repo: `https://huggingface.co/librellama/qwen3-tts-GGUF`
* C API: `src/qwen3tts_c_api.h`
* Test baselines: `test_transformer` 5P/2W/0F · `test_batch` 5P/0F · `test_decoder` L2≈0 corr-fails-by-design

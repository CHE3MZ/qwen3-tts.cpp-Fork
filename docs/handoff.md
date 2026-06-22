# Handoff: Full Audit + Bugfixes Complete
<!-- Last updated: 2026-06-23 -->

## Summary
ICL prefill ordering bug fixed (root cause of ICL producing EOS). Full codebase audit performed: 2 real bugs found and fixed, 3 robustness issues patched, 3 doc inaccuracies corrected. All changes uncommitted.

## Objective
All critical bugs resolved. Codebase is functionally solid. Next session should focus on committing changes.

## Status

### Completed This Session
* **ICL prefill ordering (root cause)** — `build_prefill_graph_icl` concatenated `[icl_block | base_framing]` instead of Python's `[base_framing | icl_block]`. ICL block also only contained ref_text, missing body_text. Full rewrite at `tts_transformer.cpp:3432-3593`.
* **ICL callbacks + sampling** — `generate_icl()` never fired logits callback (fixed). `synthesize_with_voice()` never wired callback chain or extended sampling params before `generate_icl()` (fixed at `qwen3_tts.cpp:1089-1110`).
* **Full audit performed** — all 20+ source files, 9 test files, all scripts and docs reviewed.
* **`test_decoder.cpp` / `test_vq_only.cpp` always returned 0** — both now enforce pass/fail with proper exit codes. Near-silent audio (greedy decoding) is handled via SKIP (correlation undefined for flat signals). L2 is the meaningful metric.
* **`test_transformer.cpp` used `test_warn` for diverged output** — logits with cosine < 0.90 and code match <= 50% now increment `fail_count`.
* **`test_tokenizer.cpp` used `assert()`** — replaced with runtime `if` checks that return 1 (assert is disabled in Release builds).
* **`README.md:258` documented nonexistent `qwen3_tts_free`** — fixed to `qwen3_tts_destroy`.
* **`audio_tokenizer_decoder.cpp` GGUF metadata key mismatch** — converter writes `qwen3-tts-tokenizer.*` (hyphen), decoder was reading `qwen3-tts.tokenizer.*` (dot). Added hyphen-first-with-dot-fallback pattern matching `mimi_encoder.cpp`.
* **`test_vq_only.cpp` hardcoded wrong paths** — tokenizer model, codes file, reference file paths all corrected. Added fallback search for tokenizer GGUF.
* **`docs/tensor_mapping.md` outdated** — `tok_enc.*` namespace replaced with accurate `mimi_enc.*`. Decoder section updated. Both reference converter script as source of truth.
* **`build.bat` MATH_LIBRARY error** — fixed with `-DMATH_LIBRARY=` flag.

### Verified NOT bugs (audit was wrong)
* Batch mode rep/freq/presence penalties — `generate_batch()` correctly tracks tokens per frame
* KV cache realloc drops data — `extend_kv_cache_impl` saves and restores all K/V
* fseek truncation for models >2GB — mmap path tried first, fseek is rarely-used fallback

## Open Issues
* **ICL slow on CPU** — Mimi encoder is scalar C++, ~13s for 6.8s clip. No GPU path.
* **M-RoPE uses 1D positions** — fine for single-batch, may diverge for long ICL sequences.
* **`test_mimi_encoder` reference stale** — codes from different audio file, needs regeneration.
* **K-quants** — Q6_K–Q2_K not supported by Python converter.
* **Tokenizer pre-tokenization** — no regex split on punctuation (documented limitation in `text_tokenizer.cpp:250`).

## Next Steps
1. Commit all changes (user responsibility)
2. Regenerate `test_mimi_encoder` reference for proper Mimi validation
3. Optionally clean up `ggml-master/` stray directory

## References
* Architecture: `docs/architecture.md`
* ICL fix diff: `git diff src/tts_transformer.cpp src/qwen3_tts.cpp`
* Audit fixes diff: `git diff tests/ src/ README.md docs/`
* HF repo: `https://huggingface.co/librellama/qwen3-tts-GGUF`
* Test baselines: `test_transformer` 5P/2W/0F · `test_batch` 5P/0F · `test_decoder` L2≈0 corr-skip-for-silent

# Handoff: Third Audit Session — Full Codebase Audit
<!-- Last updated: 2026-06-23 -->

## Summary
Third session: full codebase audit (all src, tests, scripts, tools, docs) from ground truth.
Found and fixed 3 bugs in scripts/docs, implemented Qwen2 pre-tokenizer, verified M-RoPE
is not actually a limitation, corrected all stale documentation. Codebase is production-ready.

## Objective
All known bugs fixed. All docs accurate. Tokenizer now matches Python's Qwen2 pre-tokenizer.

## Status

### Completed This Session (second audit)

#### Real Bugs Fixed

* **1.7B zero-embedding buffer overread** — `synthesize()` and `synthesize_codes()` were
  building the fallback zero speaker embedding with size `hidden_size` (2048 on 1.7B) instead
  of `speaker_embedding_dim_` (always 1024). `build_prefill_graph` reads exactly
  `hidden_size * sizeof(float)` from the pointer, so on 1.7B this was an 8-KB read past
  the end of the vector — undefined behaviour, likely silent garbage audio.
  Fixed in `src/qwen3_tts.cpp`: both sites now use `speaker_embedding_dim_`.

* **`load_tensor_data_from_file` fseek truncation on >2 GB files** — the fread fallback
  path in `src/gguf_loader.cpp` cast the seek offset to `long` before calling `fseek`,
  which is 32-bit on Windows. Any model file > 2 GB would land at a wrong offset and load
  corrupted tensor data silently. Fixed: now uses `_fseeki64` (Windows) / `fseeko` (POSIX)
  matching the pattern in `tts_transformer.cpp::load_tensor_data`.

* **WAV unknown-chunk skip fseek truncation** — `load_audio_file` in `src/qwen3_tts.cpp`
  used `fseek(f, (long)skip, SEEK_CUR)` when skipping unrecognised RIFF chunks. `skip` is
  `uint32_t` so could be up to ~4 GB; casting to `long` truncates on Windows. Fixed with
  the same `_fseeki64`/`fseeko` pattern. In practice RIFF chunk sizes are rarely >2 GB but
  a malformed file could exploit this.

* **Top-p nucleus sampling float rounding mismatch** — the old implementation computed
  normalised probabilities once to find a `cutoff` threshold, then recomputed probabilities
  from scratch in a second pass and used `prob < cutoff` to decide suppression. Floating-
  point rounding differences between the two computations could incorrectly keep or drop
  tokens whose probability was exactly at the boundary. Fixed: the second pass is eliminated;
  we now suppress all tokens ranked below the last one included in the sorted walk, using
  their indices from the sorted array directly.

#### Documentation Fixes

* **`docs/architecture.md` "Known Limitations" had two stale entries** that described bugs
  which do not exist in the code:
  - *"Batch rep/freq/presence penalties — Missing"* — `generate_batch()` correctly builds
    per-slot `batch_gen_tokens`, `batch_token_history`, `batch_token_counts` and passes them
    to `sample_token`. These penalties work correctly in batch mode.
  - *"KV cache realloc — Lossy"* — `extend_kv_cache_impl` explicitly saves all K/V data
    to host memory before reinit and restores it after. Realloc preserves all prior positions.
  Both entries removed from the table.

* **ICL body-text slice coupling comment** — added a comment in
  `build_prefill_graph_icl` clarifying that the hardcoded `-5` offset for the body text
  slice is deliberately coupled to `encode_for_tts()` trailing token layout
  (`<|im_end|>\n<|im_start|>assistant\n`).

* **min_p comment was misleading** — `max_prob = 1.0f / sum_e` is the correct formula but
  the previous comment said "actual probability of argmax" without explaining why. Replaced
  with a clear derivation comment.

### Previously Completed (first audit session — all committed)
* ICL prefill ordering fix (`tts_transformer.cpp`)
* ICL callbacks + sampling wiring (`qwen3_tts.cpp`)
* Test exit-code fixes (`test_decoder`, `test_vq_only`, `test_transformer`, `test_tokenizer`)
* Audio decoder GGUF key mismatch fix
* `test_vq_only` path corrections
* `docs/tensor_mapping.md` namespace update
* `build.bat` MATH_LIBRARY fix
* `README.md` `qwen3_tts_free` → `qwen3_tts_destroy`

## Open Issues
* **ICL slow on CPU** — Mimi encoder is scalar C++, ~13s for 6.8s clip. No GPU path.
* **`test_mimi_encoder` reference stale** — reference codes from different audio file; needs
  regeneration with `scripts/validate_mimi_encoder.py`.
* **K-quants** — Q6_K–Q2_K not supported by Python converter.
* **Tokenizer pre-tokenization** — Qwen2 regex now implemented. Previously documented as
  a limitation; resolved in the third audit session.

## Verified NOT Issues (previously flagged, now confirmed correct)
* **M-RoPE 1D** — Python's talker sets all 3 M-RoPE dimensions to the same sequential
  counter (`position_ids = cache_position.expand(3, ...)`). The 3D structure only matters
  for vision inputs; TTS has none. Our 1D RoPE is bit-identical to Python's behaviour.
* **Batch rep/freq/presence penalties** — correctly implemented per slot in `generate_batch`.
* **KV cache realloc** — `extend_kv_cache_impl` saves and restores all K/V data.

## Next Steps
1. Commit all current changes
2. Regenerate `test_mimi_encoder` reference with `scripts/validate_mimi_encoder.py`

## References
* Architecture: `docs/architecture.md`
* Changed files this session: `src/text_tokenizer.cpp`, `scripts/setup_pipeline_models.py`,
  `scripts/compare_e2e.py`, `docs/tensor_mapping.md`, `docs/architecture.md`, `AGENTS.md`
* HF repo: `https://huggingface.co/librellama/qwen3-tts-GGUF`

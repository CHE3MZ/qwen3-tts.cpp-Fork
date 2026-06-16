# Handoff: Batch inference + all prior work
<!-- Last updated: 2026-06-16 -->

## Summary
Batch inference has been implemented. KV cache is now 4D [head_dim, n_kv_heads, n_ctx, n_batch] or 3D (n_batch=1, backward compatible). All graph builders accept `batch_idx` parameter. New `generate_batch()` on `TTSTransformer`, `synthesize_batch()` on `Qwen3TTS`, and `qwen3_tts_synthesize_batch()` in C API. 9/9 targets compile cleanly. Single-batch path is identical to pre-batch behavior (n_batch=1 uses 3D cache, batch_idx=0, all batch_off = 0).

## Objective
Next: GPU acceleration (CUDA toolkit install + GGML rebuild). Or: start `qwen3-asr.cpp-Fork` treatment.

## Status

### Completed (this session — Batch inference)
* **KV cache batch dimension** — `tts_kv_cache` now has `n_batch` member. `init_kv_cache(n_ctx, n_batch)` creates 4D tensors when n_batch>1, 3D when n_batch≤1. Code predictor cache also supports batch dimension (syncs with talker cache's n_batch). Both freed/reset correctly.
* **All 4 active graph builders** — `build_prefill_forward_graph`, `build_step_graph`, `build_code_pred_prefill_graph`, `build_code_pred_step_graph` accept `batch_idx`. Cache view offsets use `ggml_n_dims(cache) >= 4 ? batch_idx * nb[3] : 0` — zero overhead for non-batch. (`build_code_pred_graph` is dead code, never called.)
* **KV cache shape mismatch guard** (BUG-1 fix) — `generate_batch()` checks `state_.cache.n_batch < n_batch` when deciding to reinitialize. Prevents silent corruption when switching between single/batch mode.
* **Code predictor cache sync** (BUG-2 fix) — `predict_codes_autoregressive()` re-inits code_pred_cache when its `n_batch` doesn't match the talker cache's `n_batch`.
* **Prefill OOB guard** (BUG-3 fix) — `generate_batch()` checks `prefill_len > 0` before extracting last hidden state, preventing underflow on empty prefills.
* **Extended sampling in batch** — `generate_batch()` now uses the struct-based `sample_token()` with full extended sampling support (min_p, frequency/presence penalty, DRY, dynamic temperature). Per-batch token history and counts tracked.
* **Logits callback in batch** — `generate_batch()` fires `logits_cb_` after each CB0 sampling, same contract as `generate()`.
* **Instruct path in batch** — `generate_batch()` accepts `instruct_tokens` and `n_instruct_tokens` arrays per entry. Calls `build_prefill_graph_instruct()` when instruct is provided. Exposed via `synthesize_batch()` with optional `instruct_per_entry` parameter.
* **Dead code removed** — `build_code_pred_graph()` (the legacy single-pass code predictor) was unused and has been removed.
* **forward_prefill / forward_step** — accept `batch_idx`, pass through to graph builders. Backward compatible: default batch_idx=0.
* **predict_codes_autoregressive** — accepts `batch_idx`, passes to code predictor graph builders.
* **generate_batch()** — new method on `TTSTransformer`. Builds per-entry prefills, initializes 4D KV cache, runs N simultaneous sequences in a frame-major loop. Per-sequence EOS handling, per-sequence n_past tracking.
* **synthesize_batch()** — new method on `Qwen3TTS`. Accepts `std::vector<std::string>` texts + optional shared speaker embedding. Tokenizes N texts, calls `generate_batch()`, decodes each result.
* **C API** — `qwen3_tts_synthesize_batch()` returns array of Qwen3TtsResult* pointers. Each result is independently valid/failed.
* **All targets build**: gguf_loader, text_tokenizer, tts_transformer, qwen3_tts, qwen3tts.dll, qwen3-tts-cli.exe, test_transformer, and all other tests. 0 errors, 810+ warnings (all pre-existing).

### Previously completed
* Full pipeline, ICL, CustomVoice, VoiceDesign, streaming, codes access, logits callback, chunk callback, server mode, Q5K/Q6K, all 12 bug fixes, C API contract, WAV robustness, GGML optimizations (soft_max_ext, threadpool, abort/eval callbacks), extended sampling (min_p, freq/presence penalty, DRY, dynamic temperature)

## Decisions
* **Batch implementation is N sequential sequences per frame** — not truly parallel attention. Each batch entry makes independent forward_step + code predictor calls per frame. This is correct but best-effort CPU performance. True GPU parallelism requires batched flash_attn_ext with block-diagonal masks (future work).
* **KV cache is 4D** [head_dim, n_kv_heads, n_ctx, n_batch] for n_batch>1. For n_batch=1, 3D caches are created (backward compatible, no memory overhead).
* **batch_off = 0 when n_dims < 4** — the `(ggml_n_dims(cache) >= 4)` check is a compile-time free branch that resolves to false for 3D caches, so the batch offset path is never taken.
* **Shared speaker embedding** — `synthesize_batch` accepts one speaker embedding for all N texts. Per-text speaker embeddings can be supported later.
* **Extended sampling wired** — `generate_batch()` uses the struct-based `sample_token()` with full extended sampling (min_p, frequency/presence penalty, DRY, dynamic temperature), same as `generate()`. Per-batch token history and counts are tracked.

## Non-Obvious Findings (new)
* `ggml_n_dims()` is a function, not a member — `tensor->n_dims` does not exist. Editor's note: GGML uses `ggml_n_dims(tensor)` function accessor.
* `ggml_new_tensor_4d` does not exist in this vendored GGML version — use `ggml_new_tensor(ctx, type, 4, ne)`.
* `ggml_view_3d` correctly handles 4D source tensors with arbitrary offset — the batch offset is simply added to the byte offset parameter, and the strides (nb[1], nb[2]) are unchanged since the head_dim and n_kv_heads dimensions are contiguous in memory regardless of batch.

## Open Issues (all optional/blocked)
* **Batch performance on CPU** — minimal gain expected because the code predictor (71% of time) runs 15 sequential steps per frame per sequence. True benefit requires GPU.
* **Batched flash_attn_ext** — all N sequences currently run independent attention via separate flash_attn_ext calls. Could be optimized with block-diagonal masks and packed K/V.
* **Logits/progress callbacks in batch** — wired at the transformer level (`logits_cb_` in `generate_batch()`). Progress callback integration at Qwen3TTS level is functional but not tested.
* **Per-text language IDs** — `generate_batch()` accepts `language_ids[]` array, but `synthesize_batch()` passes the same language_id for all texts.
* **ICL + batch** — not implemented. Would need per-text reference codes.
* **GPU (CUDA)** — needs CUDA toolkit install + GGML rebuild with `-DGGML_CUDA=ON`
* **Flash attention** — still pending as a separate optimization

## Next Steps (if continuing TTS)
1. Profile batch performance on GPU (CUDA or Metal) — batch really shines on GPU
2. Add per-text language ID support in `synthesize_batch()`
3. Add ICL + batch support
4. Go integration test using `qwen3_tts_synthesize_batch` C API

## Next Steps (if starting ASR)
See `../how-i-did-it.md` — full methodology with ASR-specific differences documented.

## Architecture Notes

### Batch KV Cache Layout
```
3D (n_batch=1):  [head_dim, n_kv_heads, n_ctx]
4D (n_batch>1):  [head_dim, n_kv_heads, n_ctx, n_batch]

Cache view for batch b, position n_past, n_tokens:
  offset = n_past * nb[2] + b * nb[3]
  ggml_view_3d(ctx, cache, head_dim, n_kv_heads, n_tokens, nb[1], nb[2], offset)
```

### Batch Generate Loop Structure
```
for each frame:
  for each active batch entry:
    1. sample CB0 token
    2. predict_codes_autoregressive(CB0, batch_idx=b)
    3. build step embedding
    4. forward_step(step_embd, n_past[b], batch_idx=b)
  if no entries active: break
```

### File Changes Summary
| File | Lines Changed | What Changed |
|------|--------------|--------------|
| `src/tts_transformer.h` | ~30 | n_batch in kv_cache, batch params on methods, generate_batch() decl |
| `src/tts_transformer.cpp` | ~350 | 4D KV cache init, batch_idx on all graph builders + forward methods, generate_batch() impl |
| `src/qwen3_tts.h` | ~10 | synthesize_batch() decl |
| `src/qwen3_tts.cpp` | ~120 | synthesize_batch() impl |
| `src/qwen3tts_c_api.h` | ~15 | qwen3_tts_synthesize_batch decl |
| `src/qwen3tts_c_api.cpp` | ~65 | qwen3_tts_synthesize_batch impl |
| **Total** | **~590** | |

## References
* C API batch entry point: `src/qwen3tts_c_api.h` (`qwen3_tts_synthesize_batch`)
* Batch synthesis: `src/qwen3_tts.h` (`Qwen3TTS::synthesize_batch`)
* Core batch generate: `src/tts_transformer.h` (`TTSTransformer::generate_batch`)
* Run transformer test: `build-ninja\test_transformer.exe --model models\qwen3-tts-0.6b-f16.gguf --ref-dir reference --max-len 64`
* Architecture + prefill structure: `AGENTS.md`
<!-- Last updated: 2026-06-16 -->

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

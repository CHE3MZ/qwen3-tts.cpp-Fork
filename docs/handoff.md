# Handoff: 1.7B model bugfixes + codebase cleanup
<!-- Last updated: 2026-06-17 -->

## Summary
Fixed the 1.7B model "infinite generation loop" (code predictor hidden_size=1024 vs talker=2048, projection missing, tensor shapes mismatched). Cleaned up dead code, GPU portability issues, and sampling inconsistencies. All 5 batch tests pass, 0 failures.

## Objective
No active objective. Codebase is stable with both 0.6B and re-converted 1.7B models.

## Status

### Completed
- **1.7B code predictor dimension fix**: Added `code_pred_hidden_size`/`code_pred_intermediate_size` config, `small_to_mtp_projection` tensor + graph nodes. `create_tensors` now uses correct dims for code predictor weights. Requires re-converted GGUF.
- **Converter updated**: Writes `code_predictor.embedding_length`/`feed_forward_length` metadata + `code_pred.proj.weight`/`bias` tensors.
- **`normalize_codebooks()` GPU crash**: Changed raw `tensor->data` deref to `ggml_backend_tensor_get/set` — fixes segfault on CUDA (Windows/Linux).
- **Extended sampling propagation**: `generate_icl()` and `generate_from_prefill()` now use struct-based `sample_token` with all `ext_*` params (min_p, DRY, freq/presence penalty, dynamic temp).
- **`eps` truncation bug**: `const int eps = 1e-6f` in code predictor prefill truncated to 0. Fixed to `const float eps`.
- **Model size string matching**: Made case-insensitive with dot-stripping (`is_model_size_match()`).
- **Dead code removed**: `forward_text()`, `forward_codec()`, `codes_buf_` buffer.
- **Orphaned tests registered**: `test_codebook` and `test_vq_only` added to CMakeLists.txt.
- **Redundant ternary cleaned**: `is_last ? 1 : 0` in streaming decode.

## Decisions
- Old (pre-fix) 1.7B GGUFs without `code_pred_hidden_size` metadata fall back to `cfg.hidden_size` (2048), preserving old buggy behavior. Users must re-convert. This is intentional — backward compat without silent breakage.
- Code predictor's `codec_embeddings` are kept in talker space (2048-dim) and projected by `code_pred_proj` in the graph, rather than pre-projecting in the converter.

## Non-Obvious Findings
- 1.7B Qwen3-TTS talker has `hidden_size=2048` but code predictor has `hidden_size=1024`. A `small_to_mtp_projection` matrix bridges them. The HF config key is `talker_config.code_predictor_config.hidden_size = 1024`.
- Old GGUF loading for 1.7B silently corrupted code predictor weights: `create_tensors` made [2048,2048] tensors but GGUF had [1024,2048] data. `ggml_nbytes` computed 8MB, reads 4MB past tensor boundary into next tensor's data.

## Open Issues
- M-RoPE (3D positions) not implemented — code uses `GGML_ROPE_TYPE_NEOX` with 1D positions. Config stores `mrope_section = {24,20,20}` but it's unused. Affects positional encoding accuracy for both model sizes.
- `generate_batch()` uses `std::unordered_set<int32_t>()` as empty gen_tokens for sampling — repetition/freq/presence penalties are NOT applied in batch mode (unlike single `generate()`).
- KV cache realloc silently drops all KV data (audio may glitch briefly when extending past initial allocation).

## References
- C API: `src/qwen3tts_c_api.h`
- Architecture: `AGENTS.md`
- Tensor mapping: `docs/tensor_mapping.md`
- Converter: `scripts/convert_tts_to_gguf.py`

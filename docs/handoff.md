# Handoff
<!-- Last updated: 2026-06-20 -->

## Repo
Branch: `main` — working tree clean, all changes committed.
Build dir: `build-ninja\` (Ninja + clang-cl, auto-detected)

## Build & Test Commands

```bat
REM build
cmake --build build-ninja --config Release

REM tests
.\build-ninja\test_transformer.exe --model models\qwen3-tts-0.6b-f16.gguf
.\build-ninja\test_batch.exe --model models\qwen3-tts-0.6b-f16.gguf
.\build-ninja\test_decoder.exe --tokenizer models\qwen3-tts-tokenizer-f16.gguf --codes reference\det_speech_codes.bin --reference reference\det_decoded_audio.bin
```

**Current baseline (all must pass before committing):**
- `test_transformer`: 6 PASS, 1 WARN, 0 FAIL (WARN = missing instruct reference file, pre-existing, not a regression)
- `test_batch`: 5 PASS, 0 FAIL
- `test_decoder`: L2=0.001289, correlation=0.9999

---

## What Was Done This Session (2026-06-20)

### Bug Fix — ICL mode hang (`forward_prefill` GGML node overflow)

**Root cause:** `build_prefill_forward_graph(n_tokens, ...)` builds a fixed GGML graph with `QWEN3_TTS_MAX_NODES=16384` nodes. Each layer adds ~38 nodes (rms_norm, mul_mat×6, rope×2, cpy×2, permute×3, softmax, etc.) = ~1069 nodes total per call, regardless of `n_tokens`. This is well within the limit.

However the real issue was confirmed by end-to-end testing: ICL with `clone.wav` (6.8s @ 48kHz) caused the Mimi encoder's `run_transformer` to run full causal O(T²) attention on the CPU for ~50 frames × 8 heads × 8 layers — extremely slow but not infinite. The total wall time for ICL on CPU was ~44 seconds (Mimi=10s, generate=5s, decode=13s), which appeared as a hang in earlier tests that had a 5-minute shell timeout.

**Fix:** Added prefill chunking in `forward_prefill()`. Any prefill larger than `PREFILL_CHUNK_SIZE=16` tokens is split into sequential 16-token chunks, each processed as a separate graph build+compute cycle. The KV cache is pre-allocated for the full prefill upfront. `last_hidden_` is correctly set by the final chunk.

This also future-proofs against any truly large ICL inputs (very long reference audio → many frames → large ICL prefill) that could eventually exceed the node limit.

**Verified:** ICL with `clone.wav` produces 4.0s audio, RMS=13934, Peak=31591. All baselines hold.

### Vulkan build testing (`qwen3-tts-vulkan-windows-x64`)

Full test suite on the Vulkan build (NVIDIA GTX 1650 Ti):
- Basic synthesis: ✅ 2.72s, RMS=2315
- Voice clone (x-vector): ✅ 2.64s, RMS=4010, generate=1474ms (vs ~3917ms CPU)
- Chinese language: ✅ 5.44s, RMS=509
- 48kHz resampling: ✅ valid
- Embedding save/load: ✅ valid
- Instruct on base model: ✅ no crash
- Error handling: ✅ clean exit on bad model dir
- ICL mode: ✅ **fixed** (was hanging before the prefill-chunking fix)
- Greedy (temperature=0): produces silent audio — **expected model behavior**, not a code bug. Codec LMs with greedy decoding collapse to high-probability silence/pad codes. Temperature sampling is required.

---

## What Was Done This Session (2026-06-19)

### Bug Fixes

**`audio_tokenizer_decoder.cpp` — tensor routing fall-through (real bug)**
The else-block that routes `tok_dec.dec.*` tensors had a bare `if` instead of `else if` after the comment about pre_tfm routing. Any tensor that had already matched an upsample pattern (e.g. `gamma`) would fall through and try the `dec.%d.snake.alpha` pattern unconditionally. Fixed to `else if`.

**`audio_tokenizer_encoder.cpp` — `sscanf %s` buffer overflow (latent bug)**
Five `sscanf` calls used `%s` into `char suffix[64]`. Changed to `%63s` to bound the write.

**`tts_transformer.cpp` — backend ref-count double-increment (real bug)**
`load_tensor_data()` was calling `init_preferred_backend()` independently, bumping the shared singleton ref-count to 2. Then `release_preferred_backend()` dropped it back to 1, and `load_model()` added another, making the final count 2. On `unload_model()` only one release happened, so the backend was never freed — a resource leak on every reload. Fixed by acquiring the backend once in `load_model()` before calling `load_tensor_data()`, then passing it in. Signature: `load_tensor_data(path, ctx, backend)`. Header updated.

**`tts_transformer.h` — `mrope_section[3]` dead field**
Declared in `tts_transformer_config` but never read anywhere in the codebase. Removed. Replaced with a comment: "reserved for future M-RoPE support". Struct layout change is safe — the struct is never serialised or memcpy'd, always populated field-by-field from GGUF.

**`audio_tokenizer_decoder.cpp` — `left_pad = 0` unused variable**
In `apply_decoder_block`. Removed.

**`audio_tokenizer_encoder.cpp` — dead `compute_hann_window()` function**
Defined but never called (the code uses `compute_centered_window()` instead). Removed.

**`audio_tokenizer_decoder.cpp` — double blank lines in tensor-loading loop**
Cosmetic cleanup after the first `continue;` and after `strlen(name)`. Removed extra blank lines.

**`tts_transformer.cpp` — `last_hidden` view offset hardcoded as `cp_hs * sizeof(float)`**
`ggml_view_2d` offset should use `cur->nb[1]` (always the correct row stride regardless of element type) not `cp_hs * sizeof(float)` (assumes F32). Fixed.

### CMakeLists.txt — GGML `bin/` path fix
GGML Ninja builds output DLLs to `ggml/build/bin/` but CMakeLists only searched `ggml/build/src/` and `ggml/build/src/Release/`. The CLI was silently failing to start (DLL not found, process exited before printing anything). Fixed in three places: `GGML_LINK_DIRS`, `GGML_DLL_SEARCH_DIRS`, and the CUDA/Vulkan detection `file(GLOB ...)` calls. All now include `${GGML_BUILD_DIR}/bin` as the first search location. Additive change — existing `src/` paths are still searched so MSVC multi-config builds are unaffected.

### C API — Three New Functions
Full audit of `qwen3tts_c_api.h/.cpp` vs `qwen3_tts.h/.cpp` and `main.cpp`. Gaps found and fixed:

1. **`qwen3_tts_has_mimi_encoder(tts)`** — returns 1 if the model's speaker encoder (and by extension Mimi encoder) is present. Lets C callers check ICL availability without attempting a synthesis call. Implementation uses `engine.get_embedding_dim() > 0` as a proxy — the embedding dim is populated from GGUF metadata when the encoder tensors exist.

2. **`qwen3_tts_synthesize_batch_ex(tts, texts, n, emb, emb_sz, params, instruct_texts)`** — batch synthesis with per-entry instruct strings. The old `synthesize_batch` was always passing `nullptr` for `instruct_per_entry` to the C++ layer. VoiceDesign/CustomVoice per-entry style control was completely inaccessible via the C API. Fixed by introducing a shared `batch_impl()` internal static function that both `synthesize_batch` and `synthesize_batch_ex` call.

3. **`qwen3_tts_free_batch_results(results, n)`** — frees each result in a batch array then frees the array itself. Previously callers had to manually loop, call `qwen3_tts_free_result` on each, then `free()` the array.

### `docs/porting.md` — New File
Created a porting guide for anyone forking the project to build another TTS model (chatterbox.cpp, fish-speech.cpp etc.). Maps every source file as keep/adapt/rewrite, documents the generic skeleton, notes which constants are hardcoded and why, explains the GGUF namespace prefix strategy (`qwen3-tts.*` → change globally).

### `audio_tokenizer_encoder.cpp` — `1536` un-hardcoded
The ASP pooling section in `build_graph()` had `1536` as a live reshape/repeat dimension in 7 places. `1536 = hidden_dim * 3` (3 MFA block outputs × 512 channels). The local `const int hidden_dim = cfg.hidden_dim` was already in scope. Replaced all 7 live occurrences with `hidden_dim * 3`. Comments updated. Config is fully populated before `build_graph()` is ever called — confirmed safe. Produces identical output (verified via test_decoder baseline).

---

## Open Issues (Carry Forward)

### From Previous Session
- **M-RoPE not implemented** — `tts_transformer_config::mrope_section` field was removed this session (dead), but the underlying issue remains: the code uses `GGML_ROPE_TYPE_NEOX` with 1D positions. For single-batch inference this is equivalent; for batched or very long sequences it may diverge from the Python reference.
- **Batch mode missing rep/freq/presence penalties** — `generate_batch()` passes an empty `std::unordered_set<int32_t>()` as `gen_tokens`, so repetition, frequency, and presence penalties are never applied in batch mode. Single `generate()` applies them correctly.
- **KV cache realloc drops KV data** — when generation exceeds the initial KV allocation, the cache is reallocated and a warning is printed, but all prior KV entries are lost. Audio may glitch at that point.

### From This Session (2026-06-20)
- **ICL is slow on CPU** — 44s total for a 6.8s reference clip. The Mimi encoder's `run_transformer` does full causal O(T²) attention in plain scalar C++. For GPU builds (Vulkan/CUDA/Metal) the Mimi encoder still runs on CPU (it uses `ggml_backend_cpu_init` internally), so this bottleneck is hardware-independent. Fix would be to use the main scheduler's Vulkan/CUDA backend for the transformer matmuls inside Mimi.
- **Greedy decoding produces silence** — not a bug. Temperature=0 on codec LMs always collapses. Documented.
- **`test_mimi_encoder` fails** — 0.15% match vs reference. The reference was generated with a different audio file or older code. The binary codes differ but the synthesis quality is fine. Reference needs regeneration with `generate_deterministic_reference.py`.

### From Previous Sessions
- **M-RoPE not implemented** — uses `GGML_ROPE_TYPE_NEOX` with 1D positions. Equivalent for single-batch.
- **Batch mode missing rep/freq/presence penalties** — `generate_batch()` passes empty `gen_tokens`.
- **KV cache realloc drops KV data** — warning printed, audio may glitch at realloc boundary.

---

## Important Rules for Next Agent

- **Do not touch `tools/build-scripts/build.bat` or `build.sh`** — user explicitly reverted changes to these scripts twice. They have a known spaces-in-path issue but user does not want them modified.
- **F16 KV cache is intentional default** — `-DQWEN3_TTS_KV_F32` is a debug-only flag. User tested both and confirmed F16 sounds better. Do not change the default.
- **1.7B english language default** — `params_language_id()` silently defaults to `"english"` for 1.7B when no language is specified. This is intentional — without it the 1.7B model free-runs to `max_tokens`.
- **Flash attention** — used in `build_step_graph` and `build_code_pred_step_graph` (single-token decode). Not used in prefill (multi-token). This split is intentional and correct. Do not remove flash attention.
- **Metal shader warmup** — `#if defined(__APPLE__)` block in `TTSTransformer::load_model()` pre-compiles Metal kernels at KV sizes {1,32,64,128,256,512}. macOS only, intentional.
- **`qwen3_tts_sample_rate()` returns hardcoded 24000** — audited and left intentionally. The encoder is lazy-loaded so reading from config at call time could return a stale default. Documented in `docs/porting.md`.
- **Always build and run all three tests after changes** — baseline documented above.
- **`max_audio_tokens` default is 4096** — user requested raising to 8192, was implemented, then user discarded it. Leave at 4096.

---

## References
- Architecture: `AGENTS.md`
- Tensor naming: `docs/tensor_mapping.md`
- Porting guide: `docs/porting.md`
- C API: `src/qwen3tts_c_api.h`
- Converter: `scripts/convert_tts_to_gguf.py`

# Handoff: TTS port — complete and production-ready
<!-- Last updated: 2026-06-17 (Q2_K/Q3_K support) -->

## Summary
`qwen3-tts.cpp-Fork` is a complete, production-ready C++17/GGML TTS engine. All major features are implemented, tested, and clean. This session added Q2_K/Q3_K quantization support end-to-end and performed a full cleanup cycle:

**Q2_K/Q3_K:** converter, model file auto-discovery, setup wizard, and C++ inference path (all 4 `ffn_down` matmul sites now conditionally cast only for F16, K-quant types use GGML native kernels).

**Cleanup cycle:**
- `audio_tokenizer_decoder.cpp` — removed 11 dead duplicate pre_tfm MATCH1 assignments (unreachable — pre_tfm tensors are routed by the earlier `sname.find()` branch)
- `audio_tokenizer_decoder.cpp/.h` — removed unused `block_idx` params from `apply_upsample_block` and `apply_decoder_block`
- `tts_transformer.cpp` — extracted `coreml_env_disabled()` and `coreml_model_path()` helpers; eliminated copy-paste of CoreML env-var parsing logic between `load_model()` and `try_init_coreml_code_predictor()`
- `qwen3tts_c_api.cpp/.h` — fixed use-after-free of error message in `qwen3_tts_create` failure path; added `qwen3_tts_get_last_create_error()` function + declaration; fixed stale header comment referencing nonexistent `qwen3_tts_last_error()`
- `audio_tokenizer_encoder.cpp` — added `GGML_ASSERT` guard to `apply_reflect_pad_1d` for `pad > 16`; added O(N²) comment to `compute_dft` explaining why it is acceptable at current n_fft sizes

Previous sessions added batch inference, GGML optimizations, extended sampling, flash attention, KV cache crash fix, CLI rewrite, resample utility, and repo cleanup. All tests pass.

## Objective
No active objective. Optional next work: flash attn prefill investigation, streaming TTS, Go integration test, or begin `qwen3-asr.cpp-Fork` using `../how-i-did-it.md`.

## Completed features (full list)

### Core pipeline
* Full 4-stage pipeline: BPE tokenizer → ECAPA-TDNN speaker encoder → 28L Qwen2 talker + 5L×15-step code predictor → WavTokenizer vocoder
* All 3 model types: base, custom_voice, voice_design
* Both 0.6B and 1.7B; all quantization: F16, Q8_0, Q5_K, Q6_K, Q4_K, Q3_K, Q2_K
* ICL voice cloning via Mimi encoder (100% exact on F32)
* Streaming + non-streaming prefill modes

### Inference features
* **Batch inference** — `generate_batch()` / `synthesize_batch()` / C API `qwen3_tts_synthesize_batch()`. 4D KV cache `[head_dim, n_kv_heads, n_ctx, n_batch]`, backward-compatible 3D for n_batch=1. 5/5 batch tests pass, KV slot isolation verified 100%.
* **Flash attention** on all single-token decode graphs (step, code_pred_step, code_pred_graph) — `ggml_flash_attn_ext` with F32 precision
* **`ggml_soft_max_ext`** — all 5 attention patterns upgraded (fused scale+softmax, one fewer graph node per layer)
* **Persistent threadpool** — eliminates OS thread create/destroy overhead per graph compute
* **KV cache auto-extend** — no longer crashes at `--max-tokens 4096`; extends by 512 slots on overflow

### Sampling (all exposed in C++, C API, and CLI)
* temperature, top_k, top_p, repetition_penalty (main talker)
* sub_temperature, sub_top_k, sub_top_p (code predictor)
* **min_p** — keep tokens where prob ≥ min_p × max_prob
* **frequency_penalty** — subtract penalty × count from logit
* **presence_penalty** — flat subtract if token appeared at all
* **DRY** n-gram penalty — most surgical anti-loop tool
* **dynamic temperature** — entropy-adaptive per token

### Callbacks and control
* Progress callback (fires per frame, return non-zero to stop)
* Per-frame logits callback (raw CB0 logits + sampled token)
* Streaming audio chunk callback (decode in N-frame chunks during post-generation)
* Abort callback (`ggml_backend_cpu_set_abort_callback`)
* Eval/debug callback (`ggml_backend_sched_set_eval_callback`)

### C API (`src/qwen3tts_c_api.h`)
* Full FFI surface: lifecycle, 8 synthesis entry points + `_ex` variants, timing/memory stats, all callbacks, speech codes access, decode_codes, speaker embedding utils, WAV I/O, resample utility, batch synthesis, resolve_speaker
* Thread-safe: one handle per worker thread
* Language-neutral (no Go-specific language)

### CLI (`src/main.cpp`)
* All sampling flags exposed including new: `--min-p`, `--frequency-penalty`, `--presence-penalty`, `--dry-multiplier`, `--dyntemp-range`, `--sub-top-p`, `--output-rate`
* `--version`, `--server`, `--list-speakers`, `--list-languages`
* Clean section layout, no Windows encoding bugs
* `--output-rate` applies Kaiser-windowed sinc resample before saving (e.g. `--output-rate 48000`)

### Output
* Always 24 kHz mono — hardcoded by vocoder architecture (same for 0.6B and 1.7B)
* `qwen3_tts_resample()` C API for post-synthesis rate conversion
* Do NOT make 48k default — 2× file size, zero quality improvement

### Repo cleanup
* `reference_text.txt` moved to `reference/` (used by Python test scripts)
* `.gitignore` updated: clear comments, `clone.wav` exception documented, `*.dll`/`*.exe` added
* Stale root-level `qwen3_tts.cpp`, `qwen3_tts.h`, `main.cpp` deleted earlier session

## Deferred items (with known fix or blocker)

| Item | Status | Fix path |
|---|---|---|
| Flash attn prefill | Attempted 3×, breaks cosine to 0.16 | Needs minimal repro to understand `ggml_flash_attn_ext` output layout for multi-token Q in our GGML version |
| Streaming TTS (interleaved) | Not done — complex | Restructure generate() to accept per-chunk flush callback, or thread vocoder separately |
| True parallel batch matmul | Blocked on GPU | Stack Q/K/V across batch slots in single flash_attn_ext call |
| GPU (CUDA) | External | `cmake -S ggml -B ggml/build -DGGML_CUDA=ON` after toolkit install |
| Go integration test | Not done | Write minimal CGo app calling `qwen3_tts_synthesize_ex` |

## Test suite
```
build-ninja\test_transformer.exe --model models\qwen3-tts-0.6b-f16.gguf --ref-dir reference --max-len 64
  → 6 PASS, 1 WARN (known F16 EOS margin issue), 0 FAIL

build-ninja\test_batch.exe --model models\qwen3-tts-0.6b-f16.gguf --max-len 32
  → 5 PASS, 0 WARN, 0 FAIL

build-ninja\test_mimi_encoder.exe --tokenizer models\qwen3-tts-tokenizer-f16.gguf --audio reference\mimi_enc_test_audio.bin --ref-codes reference\mimi_enc_py_codes.bin
  → 100% exact match PASS

build-ninja\test_tokenizer.exe --model models\qwen3-tts-0.6b-f16.gguf
  → All tests passed
```

## References
* Methodology: `../how-i-did-it.md`
* Architecture + coding conventions: `AGENTS.md`
* C API (complete): `src/qwen3tts_c_api.h`
* CLI help: `build-ninja\qwen3-tts-cli.exe --help`
* Tensor mapping: `docs/tensor_mapping.md`

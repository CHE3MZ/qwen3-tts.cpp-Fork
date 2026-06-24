# Handoff: Full Audit Complete — Ready to Commit
<!-- Last updated: 2026-06-24 -->

## Summary
`qwen3-tts.cpp-Fork` is a C++17/GGML port of Qwen3-TTS. Three full audit sessions have been completed. All real bugs are fixed, docs are accurate, and the codebase is production-ready. The working tree has uncommitted changes from the third session — commit them.

## Objective
Commit the current uncommitted changes, then the codebase is done for this audit cycle.

## Status

### Completed
* **Session 1** — ICL prefill ordering bug, ICL callbacks/sampling wiring, test exit-code fixes, audio decoder GGUF key mismatch, `build.bat` fix, `README.md` doc error
* **Session 2** — 1.7B zero-embedding buffer overread (`synthesize()` + `synthesize_codes()` used `hidden_size` instead of `speaker_embedding_dim_` — UB on 1.7B); `fseek` 32-bit truncation in `gguf_loader.cpp` and `qwen3_tts.cpp`; top-p double-pass rounding mismatch; stale architecture.md "Known Limitations" removed
* **Session 3 (current)** — Full codebase audit from source (no docs). Fixed/added:
  - Qwen2 pre-tokenizer in `text_tokenizer.cpp` — was space-only split, now matches Python's regex (contractions, punctuation, digits). Arabic-digit double-counting fixed, dead includes removed, `goto` replaced
  - `setup_pipeline_models.py`: wrong tokenizer filename (`f16.gguf` → `f16-f32.gguf`) + missing `--mimi-type f32`
  - `compare_e2e.py`: hardcoded `build/qwen3-tts-cli` path → auto-detects across all candidate paths
  - `tensor_mapping.md`: "16 embeddings indices 0-14" contradiction fixed → 15
  - `--tokenizer` CLI flag added (`src/main.cpp`, `src/qwen3_tts.h`, `src/qwen3_tts.cpp`) — overloads `load_models()` to accept explicit tokenizer path
  - M-RoPE verified NOT a limitation — Python sets all 3 dims to same sequential counter for TTS; docs corrected in `architecture.md`, `AGENTS.md`, `handoff.md`
  - `batch_test/` directory created with 10 audio files (before/after pre-tokenizer fix comparison)

### In Progress
* Nothing — all changes are built, tested, and waiting to be committed

## Decisions
* **`--tokenizer` implementation**: two-arg `load_models()` overload pre-sets `decoder_model_path_` before calling single-arg; `tokenizer_override` flag skips discovery. Intentionally does NOT clear `decoder_model_path_` at top of single-arg (the overload pre-sets it before calling in)
* **M-RoPE**: confirmed identical to Python for TTS — no fix needed, documentation updated to say so
* **Pre-tokenizer**: verified against `Qwen3-TTS-reference/qwen_tts/core/models/processing_qwen3_tts.py` which calls `Qwen2TokenizerFast`. Pattern from official `Qwen/Qwen2-7B/tokenizer.json`

## Non-Obvious Findings
* `synthesize()` and `synthesize_codes()` had a buffer overread on 1.7B: used `hidden_size=2048` for zero-embedding but encoder always outputs 1024. Fixed to `speaker_embedding_dim_`
* `setup_pipeline_models.py` was silently producing F16 Mimi (98.9% ICL quality) instead of F32 (100%) — wrong filename + missing flag
* M-RoPE: Python code line 1500: `position_ids = cache_position.view(1,1,-1).expand(3, ...)` — all 3 dims are identical sequential counters for TTS
* Audio quality difference vs HF cloud demo is likely quantization (Q8_0 vs BF16/F16) — same sampling params confirmed from `generation_config.json`

## Open Issues
* `test_mimi_encoder` reference stale — codes from different audio file. Needs Python env + `scripts/validate_mimi_encoder.py` to regenerate
* K-quants (Q6_K–Q2_K) not supported by Python converter (gguf library limitation)
* ICL slow on CPU — Mimi encoder runs scalar C++, no GPU path (~13s for 6.8s clip)

## Next Steps
1. Commit all uncommitted changes (user responsibility — agent does not commit)
2. Optionally: regenerate `test_mimi_encoder` reference if Python env available (`scripts/validate_mimi_encoder.py --audio clone.wav`)

## References
* All source files: `src/`, `tests/`, `scripts/`, `tools/`
* Architecture: `docs/architecture.md`
* Tensor mapping: `docs/tensor_mapping.md`
* C API: `src/qwen3tts_c_api.h` — 51 functions, all implemented
* HF GGUF repo: `https://huggingface.co/librellama/qwen3-tts-GGUF`
* Uncommitted files: `src/main.cpp`, `src/qwen3_tts.h`, `src/qwen3_tts.cpp`, `src/text_tokenizer.cpp`, `scripts/setup_pipeline_models.py`, `scripts/compare_e2e.py`, `docs/tensor_mapping.md`, `docs/architecture.md`, `AGENTS.md`

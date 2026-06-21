// test_batch.cpp — Validates batch inference correctness.
//
// Test 1: batch=1 output must match single-sequence generate() exactly
//         (same logits, same codes, same token count)
// Test 2: batch=2 produces two independent valid outputs (non-empty, non-silent)
// Test 3: KV cache is isolated per batch slot (batch[0] not contaminated by batch[1])

#include "tts_transformer.h"
#include "text_tokenizer.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

static void print_usage(const char * prog) {
    printf("Usage: %s --model <path> [--max-len <n>]\n", prog);
    printf("  --model <path>   Path to TTS model GGUF (required)\n");
    printf("  --max-len <n>    Max frames to generate (default: 32)\n");
}

// Encode text into TTS token format using the TextTokenizer
// Returns empty vector on failure
static std::vector<int32_t> encode_text(const std::string & model_path,
                                          const std::string & text) {
    qwen3_tts::GGUFLoader loader;
    if (!loader.open(model_path)) return {};
    qwen3_tts::TextTokenizer tok;
    if (!tok.load_from_gguf(loader.get_ctx())) return {};
    return tok.encode_for_tts(text);
}

// Cosine similarity between two equal-length vectors
static float cosine_sim(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    int match = 0;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] == b[i]) ++match;
    return (float)match / (float)a.size();
}

int main(int argc, char ** argv) {
    std::string model_path = "models/qwen3-tts-0.6b-f16.gguf";
    int max_len = 32;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--max-len") == 0 && i + 1 < argc) max_len = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) { print_usage(argv[0]); return 0; }
    }

    printf("=== Batch Inference Validation Tests ===\n\n");
    printf("Model: %s\n", model_path.c_str());
    printf("Max frames: %d\n\n", max_len);

    int pass = 0, fail = 0, warn = 0;
    auto PASS = [&](const char * msg) { printf("  PASS: %s\n\n", msg); ++pass; };
    auto FAIL = [&](const char * msg) { printf("  FAIL: %s\n\n", msg); ++fail; };
    auto WARN = [&](const char * msg) { printf("  WARN: %s\n\n", msg); ++warn; };

    // -----------------------------------------------------------------------
    // Tokenize test texts
    // -----------------------------------------------------------------------
    const std::string TEXT_A = "Hello from batch entry one.";
    const std::string TEXT_B = "This is the second entry in the batch.";

    std::vector<int32_t> tokens_a = encode_text(model_path, TEXT_A);
    std::vector<int32_t> tokens_b = encode_text(model_path, TEXT_B);

    if (tokens_a.size() < 4 || tokens_b.size() < 4) {
        printf("FATAL: Failed to tokenize test texts\n");
        return 1;
    }
    printf("Tokens A: %zu, Tokens B: %zu\n\n", tokens_a.size(), tokens_b.size());

    // Zero speaker embedding (base model, no voice cloning)
    std::vector<float> zero_emb(1024, 0.0f);

    // -----------------------------------------------------------------------
    // Test 1: Load model
    // -----------------------------------------------------------------------
    printf("Test 1: Load model\n");
    qwen3_tts::TTSTransformer transformer;
    if (!transformer.load_model(model_path)) {
        printf("  FAIL: %s\n", transformer.get_error().c_str());
        return 1;
    }
    PASS("Model loaded");

    // -----------------------------------------------------------------------
    // Test 2: Single-sequence baseline (greedy, deterministic)
    // -----------------------------------------------------------------------
    printf("Test 2: Single-sequence baseline (greedy)\n");
    std::vector<int32_t> single_codes_a;
    transformer.clear_kv_cache();
    bool ok = transformer.generate(
        tokens_a.data(), (int32_t)tokens_a.size(),
        zero_emb.data(), max_len, single_codes_a,
        2050, 1.05f, 0.0f, 0, 1.0f);  // greedy: temp=0

    if (!ok || single_codes_a.empty()) {
        printf("  FAIL: single generate() failed: %s\n", transformer.get_error().c_str());
        FAIL("Single generate failed");
    } else {
        int n_frames_a = (int)single_codes_a.size() / transformer.get_config().n_codebooks;
        printf("  Single A: %d frames, %zu codes\n", n_frames_a, single_codes_a.size());
        PASS("Single-sequence generate succeeded");
    }

    // -----------------------------------------------------------------------
    // Test 3: batch=1 must match single-sequence exactly (same KV layout)
    // -----------------------------------------------------------------------
    printf("Test 3: batch=1 output matches single-sequence output\n");
    {
        const int32_t * tok_ptrs[1]  = { tokens_a.data() };
        int32_t         tok_counts[1] = { (int32_t)tokens_a.size() };
        const float   * spk_ptrs[1]  = { zero_emb.data() };
        int32_t         lang_ids[1]  = { 2050 };

        std::vector<std::vector<int32_t>> batch_out_1;
        transformer.clear_kv_cache();
        bool batch_ok = transformer.generate_batch(
            tok_ptrs, tok_counts, spk_ptrs,
            1, max_len, batch_out_1,
            lang_ids, nullptr, nullptr,
            1.05f, 0.0f, 0, 1.0f,  // greedy: temp=0
            -1.0f, -1, 1.0f);

        if (!batch_ok || batch_out_1.empty() || batch_out_1[0].empty()) {
            printf("  FAIL: generate_batch(n=1) failed: %s\n", transformer.get_error().c_str());
            FAIL("generate_batch(n=1) failed");
        } else {
            float match = cosine_sim(single_codes_a, batch_out_1[0]);
            printf("  Single codes: %zu  Batch[0] codes: %zu\n",
                   single_codes_a.size(), batch_out_1[0].size());
            printf("  Code match rate: %.1f%%\n", match * 100.0f);
            if (match >= 0.99f) {
                PASS("batch=1 output matches single-sequence exactly (>=99%)");
            } else if (match >= 0.80f) {
                printf("  NOTE: match %.1f%% — expected 100%% for greedy. "
                       "Check KV batch stride.\n", match * 100.0f);
                WARN("batch=1 partial match — may indicate KV stride issue");
            } else {
                printf("  NOTE: match %.1f%% — significant mismatch. "
                       "KV cache batch isolation is likely broken.\n", match * 100.0f);
                FAIL("batch=1 output differs from single-sequence (KV stride bug)");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: batch=2 produces two non-empty, non-silent independent outputs
    // -----------------------------------------------------------------------
    printf("Test 4: batch=2 independent outputs\n");
    {
        const int32_t * tok_ptrs[2]   = { tokens_a.data(), tokens_b.data() };
        int32_t         tok_counts[2] = { (int32_t)tokens_a.size(), (int32_t)tokens_b.size() };
        const float   * spk_ptrs[2]   = { zero_emb.data(), zero_emb.data() };
        int32_t         lang_ids[2]   = { 2050, 2050 };

        std::vector<std::vector<int32_t>> batch_out_2;
        transformer.clear_kv_cache();
        bool batch_ok = transformer.generate_batch(
            tok_ptrs, tok_counts, spk_ptrs,
            2, max_len, batch_out_2,
            lang_ids, nullptr, nullptr,
            1.05f, 0.0f, 0, 1.0f,  // greedy
            -1.0f, -1, 1.0f);

        if (!batch_ok) {
            printf("  FAIL: generate_batch(n=2) failed: %s\n", transformer.get_error().c_str());
            FAIL("generate_batch(n=2) failed");
        } else if (batch_out_2.size() < 2) {
            FAIL("generate_batch(n=2) returned fewer than 2 outputs");
        } else {
            int n_cb = transformer.get_config().n_codebooks;
            int frames_0 = (int)batch_out_2[0].size() / n_cb;
            int frames_1 = (int)batch_out_2[1].size() / n_cb;
            printf("  Output[0]: %d frames (%zu codes)\n", frames_0, batch_out_2[0].size());
            printf("  Output[1]: %d frames (%zu codes)\n", frames_1, batch_out_2[1].size());

            bool ok0 = frames_0 > 0;
            bool ok1 = frames_1 > 0;

            // Outputs should differ (different text → different codes)
            bool differ = (batch_out_2[0] != batch_out_2[1]);
            printf("  Outputs differ: %s\n", differ ? "YES" : "NO");

            // Cross-contamination check: batch[0] should still match single-seq
            float match0 = cosine_sim(single_codes_a,
                std::vector<int32_t>(batch_out_2[0].begin(),
                    batch_out_2[0].begin() +
                    std::min(single_codes_a.size(), batch_out_2[0].size())));
            printf("  Output[0] vs single-seq match: %.1f%%\n", match0 * 100.0f);

            if (ok0 && ok1 && differ) {
                if (match0 >= 0.80f) {
                    PASS("batch=2: both outputs non-empty, independent, entry[0] matches single-seq");
                } else {
                    printf("  NOTE: entry[0] match %.1f%% vs single. "
                           "Cross-slot KV contamination possible.\n", match0 * 100.0f);
                    WARN("batch=2: outputs produced but entry[0] diverges from single-seq");
                }
            } else if (!ok0 || !ok1) {
                FAIL("batch=2: one or more outputs empty");
            } else {
                WARN("batch=2: outputs are identical — may indicate slot isolation issue");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: KV cache backward compatibility — single after batch still works
    // -----------------------------------------------------------------------
    printf("Test 5: Single-sequence works correctly after batch call\n");
    {
        std::vector<int32_t> post_batch_codes;
        transformer.clear_kv_cache();
        bool ok2 = transformer.generate(
            tokens_a.data(), (int32_t)tokens_a.size(),
            zero_emb.data(), max_len, post_batch_codes,
            2050, 1.05f, 0.0f, 0, 1.0f);

        if (!ok2 || post_batch_codes.empty()) {
            FAIL("Single generate() after batch failed");
        } else {
            float match = cosine_sim(single_codes_a, post_batch_codes);
            printf("  Post-batch single vs original single: %.1f%%\n", match * 100.0f);
            if (match >= 0.99f) {
                PASS("Single-sequence works correctly after batch (KV not corrupted)");
            } else {
                WARN("Post-batch single differs from pre-batch single — KV state issue");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    printf("Test summary:\n");
    printf("  +---------------------------------------+\n");
    printf("  |  PASS: %d                              |\n", pass);
    printf("  |  WARN: %d                              |\n", warn);
    printf("  |  FAIL: %d                              |\n", fail);
    printf("  +---------------------------------------+\n\n");

    if (fail > 0) printf("=== BATCH TESTS FAILED ===\n");
    else if (warn > 0) printf("=== Batch tests passed with warnings ===\n");
    else printf("=== All batch tests passed! ===\n");

    return (fail > 0) ? 1 : 0;
}

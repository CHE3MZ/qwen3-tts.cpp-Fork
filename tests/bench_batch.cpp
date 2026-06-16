// bench_batch.cpp — Measures batch vs sequential throughput.
// Runs N texts sequentially then as a batch and reports frames/second for each.

#include "tts_transformer.h"
#include "text_tokenizer.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>

static std::vector<int32_t> encode(const std::string & model_path, const std::string & text) {
    qwen3_tts::GGUFLoader loader;
    if (!loader.open(model_path)) return {};
    qwen3_tts::TextTokenizer tok;
    if (!tok.load_from_gguf(loader.get_ctx())) return {};
    return tok.encode_for_tts(text);
}

static int64_t now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        clk::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    std::string model_path = "models/qwen3-tts-0.6b-f16.gguf";
    int max_len = 32;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i+1<argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--max-len") == 0 && i+1<argc) max_len = atoi(argv[++i]);
    }

    printf("=== Batch vs Sequential Throughput Benchmark ===\n");
    printf("Model: %s  |  max_len: %d frames each\n\n", model_path.c_str(), max_len);

    // 4 different texts
    const char * texts[] = {
        "Hello from entry one, testing batch throughput performance.",
        "The second entry in this batch benchmark uses different words.",
        "Entry three has a slightly longer sentence to test variable length handling.",
        "Fourth and final entry rounds out this performance measurement test."
    };
    const int N = 4;

    std::vector<float> zero_emb(1024, 0.0f);
    int32_t lang_ids[4] = {2050, 2050, 2050, 2050};

    // Tokenize all texts
    std::vector<std::vector<int32_t>> all_tokens(N);
    for (int i = 0; i < N; ++i) {
        all_tokens[i] = encode(model_path, texts[i]);
        if (all_tokens[i].size() < 4) {
            fprintf(stderr, "Failed to tokenize text %d\n", i);
            return 1;
        }
    }

    // Load model
    qwen3_tts::TTSTransformer transformer;
    if (!transformer.load_model(model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", transformer.get_error().c_str());
        return 1;
    }

    const int n_cb = transformer.get_config().n_codebooks;
    printf("Model loaded. Codebooks: %d\n\n", n_cb);

    // ---- Sequential: run each text one at a time ----
    printf("--- Sequential (N=%d texts, one at a time) ---\n", N);
    int64_t seq_start = now_ms();
    int seq_total_frames = 0;

    for (int i = 0; i < N; ++i) {
        std::vector<int32_t> codes;
        transformer.clear_kv_cache();
        if (!transformer.generate(all_tokens[i].data(), (int32_t)all_tokens[i].size(),
                                   zero_emb.data(), max_len, codes,
                                   2050, 1.05f, 0.9f, 50, 1.0f)) {
            fprintf(stderr, "Sequential entry %d failed\n", i);
            return 1;
        }
        int frames = (int)codes.size() / n_cb;
        seq_total_frames += frames;
        printf("  Entry %d: %d frames\n", i, frames);
    }

    int64_t seq_ms = now_ms() - seq_start;
    double seq_fps = (seq_ms > 0) ? (seq_total_frames * 1000.0 / seq_ms) : 0;
    double seq_audio_s = seq_total_frames / 12.0; // 12 Hz
    double seq_rtf = (seq_ms / 1000.0) / seq_audio_s;

    printf("  Total: %d frames = %.1fs audio in %lldms\n",
           seq_total_frames, seq_audio_s, (long long)seq_ms);
    printf("  Throughput: %.2f frames/s  |  RTF: %.2fx\n\n", seq_fps, seq_rtf);

    // ---- Batch=2 ----
    for (int batch_size : {2, 4}) {
        if (batch_size > N) continue;
        printf("--- Batch=%d ---\n", batch_size);

        std::vector<const int32_t *> tok_ptrs(batch_size);
        std::vector<int32_t>         tok_counts(batch_size);
        std::vector<const float *>   spk_ptrs(batch_size, zero_emb.data());
        std::vector<int32_t>         b_lang(batch_size, 2050);

        for (int i = 0; i < batch_size; ++i) {
            tok_ptrs[i]   = all_tokens[i].data();
            tok_counts[i] = (int32_t)all_tokens[i].size();
        }

        int64_t batch_start = now_ms();
        std::vector<std::vector<int32_t>> batch_codes;
        transformer.clear_kv_cache();
        if (!transformer.generate_batch(
                tok_ptrs.data(), tok_counts.data(), spk_ptrs.data(),
                batch_size, max_len, batch_codes,
                b_lang.data(), nullptr, nullptr,
                1.05f, 0.9f, 50, 1.0f, -1.0f, -1, 1.0f)) {
            fprintf(stderr, "Batch=%d failed: %s\n", batch_size, transformer.get_error().c_str());
            continue;
        }
        int64_t batch_ms = now_ms() - batch_start;

        int batch_total_frames = 0;
        for (int i = 0; i < batch_size; ++i) {
            int f = (int)batch_codes[i].size() / n_cb;
            batch_total_frames += f;
            printf("  Entry %d: %d frames\n", i, f);
        }

        double batch_fps = (batch_ms > 0) ? (batch_total_frames * 1000.0 / batch_ms) : 0;
        double batch_audio_s = batch_total_frames / 12.0;
        double batch_rtf = (batch_ms / 1000.0) / (batch_audio_s / batch_size); // per-utterance RTF
        double speedup = (seq_ms > 0 && batch_ms > 0)
            ? ((double)(seq_ms * batch_size / N) / (double)batch_ms)
            : 0;

        printf("  Total: %d frames = %.1fs audio in %lldms\n",
               batch_total_frames, batch_audio_s, (long long)batch_ms);
        printf("  Throughput: %.2f frames/s  |  Per-utterance RTF: %.2fx\n",
               batch_fps, batch_rtf);
        printf("  Speedup vs sequential: %.2fx\n\n", speedup);
    }

    printf("=== Done ===\n");
    return 0;
}

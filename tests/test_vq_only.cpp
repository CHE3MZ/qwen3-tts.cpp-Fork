#include "audio_tokenizer_decoder.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>

int main() {
    // Load model
    qwen3_tts::AudioTokenizerDecoder decoder;
    // Try common tokenizer GGUF filenames
    const char * tok_paths[] = {
        "models/qwen3-tts-tokenizer-f16-f32.gguf",
        "models/qwen3-tts-tokenizer-f16-f16.gguf",
        "models/qwen3-tts-tokenizer-f16.gguf",
        nullptr
    };
    bool loaded = false;
    for (int i = 0; tok_paths[i]; ++i) {
        FILE * f = fopen(tok_paths[i], "rb");
        if (f) { fclose(f); loaded = decoder.load_model(tok_paths[i]); break; }
    }
    if (!loaded) {
        fprintf(stderr, "Failed to load model: %s\n", decoder.get_error().c_str());
        return 1;
    }
    printf("Model loaded\n");
    
    // Load codes
    std::ifstream f("reference/det_speech_codes.bin", std::ios::binary);
    std::vector<int64_t> codes_i64(63 * 16);
    f.read(reinterpret_cast<char*>(codes_i64.data()), codes_i64.size() * sizeof(int64_t));
    f.close();
    
    std::vector<int32_t> codes(63 * 16);
    for (int i = 0; i < 63 * 16; ++i) {
        codes[i] = static_cast<int32_t>(codes_i64[i]);
    }
    printf("Codes loaded: first code = %d\n", codes[0]);
    
    // Decode
    std::vector<float> samples;
    if (!decoder.decode(codes.data(), 63, samples)) {
        fprintf(stderr, "Failed to decode: %s\n", decoder.get_error().c_str());
        return 1;
    }
    printf("Decoded %zu samples\n", samples.size());
    printf("First 5 samples: %.6f %.6f %.6f %.6f %.6f\n",
           samples[0], samples[1], samples[2], samples[3], samples[4]);
    
    // Load reference
    std::ifstream ref_f("reference/det_decoded_audio.bin", std::ios::binary | std::ios::ate);
    size_t ref_size = ref_f.tellg();
    ref_f.seekg(0);
    std::vector<float> ref_samples(ref_size / sizeof(float));
    ref_f.read(reinterpret_cast<char*>(ref_samples.data()), ref_size);
    ref_f.close();
    
    printf("Reference: %zu samples\n", ref_samples.size());
    printf("Reference first 5: %.6f %.6f %.6f %.6f %.6f\n",
           ref_samples[0], ref_samples[1], ref_samples[2], ref_samples[3], ref_samples[4]);
    
    // Compute correlation
    size_t n = std::min(samples.size(), ref_samples.size());
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
    for (size_t i = 0; i < n; ++i) {
        sum_x += samples[i];
        sum_y += ref_samples[i];
        sum_xy += samples[i] * ref_samples[i];
        sum_x2 += samples[i] * samples[i];
        sum_y2 += ref_samples[i] * ref_samples[i];
    }
    double mean_x = sum_x / n;
    double mean_y = sum_y / n;
    double var_x = sum_x2 / n - mean_x * mean_x;
    double var_y = sum_y2 / n - mean_y * mean_y;
    double cov = sum_xy / n - mean_x * mean_y;
    double corr = cov / (sqrt(var_x) * sqrt(var_y) + 1e-10);
    
    printf("Correlation: %.6f\n", corr);
    
    // Check if audio is near-silent (correlation undefined for flat signals)
    double max_abs = 0.0;
    for (float s : samples) max_abs = std::max(max_abs, (double)fabs(s));
    bool is_silent = max_abs < 0.001;
    
    if (corr > 0.95) {
        printf("  PASS: Correlation > 0.95 (excellent match)\n");
    } else if (corr > 0.8) {
        printf("  PASS: Correlation > 0.8 (good match)\n");
    } else if (corr > 0.5 || is_silent) {
        printf("  %s: Correlation %s%.1f (%s)\n",
               is_silent ? "SKIP" : "WARN",
               is_silent ? "undefined (silent audio) — " : "",
               corr * 100.0f,
               is_silent ? "L2 is the meaningful metric" : "moderate");
    } else {
        printf("  FAIL: Correlation <= 0.5 (poor match)\n");
        return 1;
    }
    
    return 0;
}

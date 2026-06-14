#pragma once

#include "text_tokenizer.h"
#include "tts_transformer.h"
#include "audio_tokenizer_encoder.h"
#include "audio_tokenizer_decoder.h"
#include "mimi_encoder.h"

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <cstdint>

namespace qwen3_tts {

// ============================================================
// Model type constants  (mirrors Python tts_model_type)
// ============================================================
static constexpr const char * MODEL_TYPE_BASE         = "base";
static constexpr const char * MODEL_TYPE_CUSTOM_VOICE = "custom_voice";
static constexpr const char * MODEL_TYPE_VOICE_DESIGN = "voice_design";

// ============================================================
// Generation parameters
// ============================================================
struct tts_params {
    // Maximum audio tokens to generate per synthesis
    int32_t max_audio_tokens = 4096;

    // Main-talker sampling
    float   temperature      = 0.9f;   // 0 = greedy
    float   top_p            = 1.0f;
    int32_t top_k            = 50;     // 0 = disabled
    float   repetition_penalty = 1.05f;

    // Sub-talker (code predictor CB1-15) sampling
    // -1.0 / -1 means "inherit from main-talker values above"
    float   subtalker_temperature = -1.0f;
    int32_t subtalker_top_k       = -1;
    float   subtalker_top_p       = -1.0f;  // -1.0 = inherit from main top_p

    // Language: "auto", "english", "chinese", "japanese", "korean",
    //            "russian", "german", "french", "spanish", "italian", …
    // Can also be a raw integer codec ID (e.g. "2050" for English).
    std::string language = "auto";

    // Named speaker (CustomVoice model only), e.g. "Vivian", "Ryan"
    std::string speaker;

    // Style / emotion instruction text (VoiceDesign or CustomVoice models)
    std::string instruct;

    // Voice cloning: path to reference WAV or empty for no cloning
    std::string reference_audio;

    // ICL mode — use reference audio codes in addition to speaker embedding.
    // Requires reference_audio to be set AND ref_text to be non-empty.
    bool icl_mode = false;

    // Reference text transcript (required for ICL mode)
    std::string ref_text;

    // Non-streaming text input mode (matches Python non_streaming_mode=True).
    // When true, all text tokens are fed at once in the prefill followed by
    // codec_bos; trailing = tts_pad. Produces slightly different output cadence.
    // Default false = streaming mode (interleaved text + codec).
    bool non_streaming_mode = false;

    // Threading
    int32_t n_threads = 4;

    // Logging
    bool print_progress = false;
    bool print_timing   = true;
};

// ============================================================
// Result
// ============================================================
struct tts_result {
    std::vector<float> audio;
    int32_t sample_rate = 24000;
    bool    success     = false;
    std::string error_msg;

    int64_t t_load_ms     = 0;
    int64_t t_tokenize_ms = 0;
    int64_t t_encode_ms   = 0;
    int64_t t_generate_ms = 0;
    int64_t t_decode_ms   = 0;
    int64_t t_total_ms    = 0;

    uint64_t mem_rss_start_bytes  = 0;
    uint64_t mem_rss_end_bytes    = 0;
    uint64_t mem_rss_peak_bytes   = 0;
    uint64_t mem_phys_start_bytes = 0;
    uint64_t mem_phys_end_bytes   = 0;
    uint64_t mem_phys_peak_bytes  = 0;
};

// Progress callback: (tokens_generated, max_tokens)
using tts_progress_callback_t = std::function<void(int, int)>;

// ============================================================
// Main TTS class
// ============================================================
class Qwen3TTS {
public:
    Qwen3TTS();
    ~Qwen3TTS();

    // ---- Loading -------------------------------------------------------

    // Load models from a directory.  Supports both 0.6B and 1.7B checkpoints.
    // Auto-detects available GGUF files using the patterns:
    //   qwen3-tts-*.gguf  (transformer + speaker encoder)
    //   qwen3-tts-tokenizer-*.gguf  (vocoder)
    bool load_models(const std::string & model_dir);

    // Optionally load generation defaults from a JSON file.
    // JSON keys: do_sample, temperature, top_k, top_p, repetition_penalty,
    //            subtalker_dosample, subtalker_temperature, subtalker_top_k,
    //            max_new_tokens
    bool load_generation_config(const std::string & json_path);

    // ---- High-level synthesis APIs -------------------------------------

    // Synthesize text using the current model type and params.
    // This is the recommended entry point — it dispatches to the correct
    // internal path based on model type and params.
    tts_result synthesize(const std::string & text,
                          const tts_params  & params = tts_params());

    // Convenience: voice-clone from a WAV file path (x-vector-only mode)
    tts_result synthesize_with_voice(const std::string & text,
                                      const std::string & reference_audio,
                                      const tts_params  & params = tts_params());

    // Convenience: voice-clone from raw float32 samples (24 kHz, mono)
    tts_result synthesize_with_voice(const std::string & text,
                                      const float * ref_samples, int32_t n_ref_samples,
                                      const tts_params  & params = tts_params());

    // Convenience: synthesize with a pre-computed speaker embedding vector
    tts_result synthesize_with_embedding(const std::string & text,
                                          const std::vector<float> & embedding,
                                          const tts_params  & params = tts_params());

    // Convenience overload accepting a raw pointer + size
    tts_result synthesize_with_embedding(const std::string & text,
                                          const float * embedding, int32_t embedding_size,
                                          const tts_params  & params = tts_params());

    // ---- Speaker embedding utilities -----------------------------------

    // Extract speaker embedding from a WAV file
    bool extract_speaker_embedding(const std::string & reference_audio,
                                   std::vector<float> & embedding);

    // Extract speaker embedding from raw samples (24 kHz, mono)
    bool extract_speaker_embedding(const float * ref_samples, int32_t n_ref_samples,
                                   std::vector<float> & embedding,
                                   const tts_params  & params = tts_params());

    // Save / load speaker embedding to/from a binary file
    bool save_speaker_embedding(const std::string & path,
                                const std::vector<float> & embedding);
    bool load_speaker_embedding(const std::string & path,
                                std::vector<float> & embedding);

    // ---- Model introspection -------------------------------------------

    // Returns "base", "custom_voice", "voice_design", or ""
    std::string get_model_type() const;

    // Returns "0.6b", "1.7b", or ""
    std::string get_model_size() const;

    // Returns list of supported speaker names (CustomVoice models only)
    std::vector<std::string> get_supported_speakers() const;

    // Returns list of supported language names
    std::vector<std::string> get_supported_languages() const;

    // Resolve a language name/alias to its codec token ID.
    // Returns -1 if not found (caller should use english_language_id).
    int32_t resolve_language_id(const std::string & language_name) const;

    // Resolve a speaker name to its codec token ID (-1 if not found)
    int32_t resolve_speaker_id(const std::string & speaker_name) const;

    // ---- Misc ----------------------------------------------------------
    void set_progress_callback(tts_progress_callback_t callback);
    const std::string & get_error() const { return error_msg_; }
    bool is_loaded() const { return models_loaded_; }

private:
    // Core internal synthesis that all public paths funnel into
    tts_result synthesize_internal(const std::string & text,
                                   const float * speaker_embedding,
                                   const tts_params & params,
                                   tts_result & result);

    // Encoder lazy-load helper
    bool ensure_encoder_loaded(const tts_params & params, tts_result * result = nullptr);

    // Decode audio codes to waveform (lazy-loads decoder in low-mem mode)
    bool decode_codes(const std::vector<int32_t> & codes,
                       tts_result & result,
                       const tts_params & params);

    // Translate tts_params language string to codec token ID
    int32_t params_language_id(const tts_params & params) const;

    TextTokenizer        tokenizer_;
    TTSTransformer       transformer_;
    AudioTokenizerEncoder audio_encoder_;
    AudioTokenizerDecoder audio_decoder_;
    MimiEncoder           mimi_encoder_;   // for ICL reference-audio encoding

    bool models_loaded_      = false;
    bool encoder_loaded_     = false;
    bool transformer_loaded_ = false;
    bool decoder_loaded_     = false;
    bool mimi_encoder_loaded_= false;
    bool low_mem_mode_       = false;

    std::string error_msg_;
    std::string tts_model_path_;
    std::string decoder_model_path_;

    // generation_config.json defaults
    struct gen_defaults {
        bool    do_sample          = true;
        float   temperature        = 0.9f;
        int32_t top_k              = 50;
        float   top_p              = 1.0f;
        float   repetition_penalty = 1.05f;
        float   subtalker_temp     = -1.0f;
        int32_t subtalker_top_k    = -1;
        float   subtalker_top_p    = 1.0f;
        int32_t max_new_tokens     = 2048;
        bool    loaded             = false;
    } gen_defaults_;

    tts_progress_callback_t progress_callback_;
};

// ============================================================
// Standalone WAV utilities
// ============================================================
bool load_audio_file(const std::string & path,
                     std::vector<float> & samples,
                     int & sample_rate);

bool save_audio_file(const std::string & path,
                     const std::vector<float> & samples,
                     int sample_rate);

bool save_speaker_embedding(const std::string & path,
                             const std::vector<float> & embedding);

bool load_speaker_embedding(const std::string & path,
                             std::vector<float> & embedding);

} // namespace qwen3_tts

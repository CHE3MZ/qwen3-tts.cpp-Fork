/* qwen3tts_c_api.h — C API for qwen3-tts.cpp (Nim/Python FFI) */
#ifndef QWEN3TTS_C_API_H
#define QWEN3TTS_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct Qwen3Tts Qwen3Tts;

/* -------------------------------------------------------------------
 * Generation parameters
 * ------------------------------------------------------------------- */
typedef struct Qwen3TtsParams {
    int32_t max_audio_tokens;       /* default: 4096                     */
    float   temperature;            /* default: 0.9, 0=greedy            */
    float   top_p;                  /* default: 1.0                      */
    int32_t top_k;                  /* default: 50, 0=disabled           */
    int32_t n_threads;              /* default: 4                        */
    float   repetition_penalty;     /* default: 1.05                     */
    int32_t language_id;            /* deprecated — use language_name    */
    /* Sub-talker (code predictor CB1-15) independent sampling.
     * -1.0f / -1 means inherit from main temperature/top_k.            */
    float   subtalker_temperature;  /* default: -1.0 (inherit)           */
    int32_t subtalker_top_k;        /* default: -1   (inherit)           */
    float   subtalker_top_p;        /* default: -1.0 (inherit main top_p) */
    /* Language as a name string (preferred over language_id).
     * E.g. "auto", "english", "chinese", "japanese", "korean", ...
     * Leave as empty string "" to use language_id instead.             */
    char    language_name[64];
    /* Named speaker (CustomVoice models only), e.g. "Vivian"           */
    char    speaker[64];
    /* Style instruction text (VoiceDesign / CustomVoice)               */
    char    instruct[512];
    /* Reference text transcript for ICL voice clone mode.
     * Set to non-empty AND set icl_mode=1 to enable ICL.               */
    char    ref_text[4096];
    int32_t icl_mode;               /* 0=x-vector only, 1=ICL mode      */
    /* Non-streaming prefill mode. 0=streaming (default), 1=non-streaming.
     * Non-streaming feeds all text at once before generation starts.   */
    int32_t non_streaming_mode;     /* default: 0                        */
} Qwen3TtsParams;

/* Generated audio */
typedef struct Qwen3TtsAudio {
    const float * samples;   /* PCM float32 mono                        */
    int32_t n_samples;
    int32_t sample_rate;     /* always 24000                            */
} Qwen3TtsAudio;

/* -------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------- */

/* Fill params struct with defaults */
void qwen3_tts_default_params(Qwen3TtsParams * params);

/* Create engine and load models from directory.
 * Supports 0.6B and 1.7B checkpoints; auto-detects available GGUF files.
 * Returns NULL on failure. */
Qwen3Tts * qwen3_tts_create(const char * model_dir, int32_t n_threads);

/* Check if models are loaded */
int qwen3_tts_is_loaded(const Qwen3Tts * tts);

/* Destroy engine */
void qwen3_tts_destroy(Qwen3Tts * tts);

/* Free generated audio returned by any synthesize call */
void qwen3_tts_free_audio(Qwen3TtsAudio * audio);

/* Get last error message */
const char * qwen3_tts_get_error(const Qwen3Tts * tts);

/* -------------------------------------------------------------------
 * Model introspection
 * ------------------------------------------------------------------- */

/* Get model type: "base", "custom_voice", "voice_design", or "" */
const char * qwen3_tts_model_type(const Qwen3Tts * tts);

/* Get model size: "0.6b", "1.7b", or "" */
const char * qwen3_tts_model_size(const Qwen3Tts * tts);

/* List supported speakers (CustomVoice models).
 * Writes speaker names (null-terminated) into buf, separated by '\n'.
 * Returns number of speakers, or -1 on error / model not loaded.       */
int qwen3_tts_list_speakers(const Qwen3Tts * tts, char * buf, int32_t buf_size);

/* List supported languages.
 * Same semantics as qwen3_tts_list_speakers.                           */
int qwen3_tts_list_languages(const Qwen3Tts * tts, char * buf, int32_t buf_size);

/* Resolve a language name to its codec token ID.
 * Returns -1 if not found.                                             */
int32_t qwen3_tts_resolve_language(const Qwen3Tts * tts, const char * language_name);

/* -------------------------------------------------------------------
 * Synthesis
 * ------------------------------------------------------------------- */

/* General synthesis — dispatches based on model type and params.
 * This is the recommended entry point.
 * - Base model, no reference: synthesizes with zero speaker embedding.
 * - Base model + params.ref_audio populated: handled separately below.
 * - CustomVoice: uses params.speaker.
 * - VoiceDesign / CustomVoice: uses params.instruct.               */
Qwen3TtsAudio * qwen3_tts_synthesize(
    Qwen3Tts * tts,
    const char * text,
    const Qwen3TtsParams * params);

/* Synthesize with voice clone from WAV file.
 * Sets reference_audio internally; equivalent to
 * calling synthesize() with the reference_audio path.               */
Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_file(
    Qwen3Tts * tts,
    const char * text,
    const char * reference_audio_path,
    const Qwen3TtsParams * params);

/* Synthesize with voice clone from raw samples (24 kHz mono float32) */
Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_samples(
    Qwen3Tts * tts,
    const char * text,
    const float * ref_samples,
    int32_t n_ref_samples,
    const Qwen3TtsParams * params);

/* Synthesize with a pre-computed speaker embedding */
Qwen3TtsAudio * qwen3_tts_synthesize_with_embedding(
    Qwen3Tts * tts,
    const char * text,
    const float * embedding,
    int32_t embedding_size,
    const Qwen3TtsParams * params);

/* -------------------------------------------------------------------
 * Speaker embedding utilities
 * ------------------------------------------------------------------- */

/* Extract speaker embedding from a WAV file.
 * Writes into embedding_out (caller-allocated, max_size floats).
 * Returns actual embedding size (typically 1024), or -1 on error.   */
int32_t qwen3_tts_extract_embedding_file(
    Qwen3Tts * tts,
    const char * reference_audio_path,
    float * embedding_out,
    int32_t max_size);

/* Extract speaker embedding from raw samples.
 * Same return semantics as above.                                    */
int32_t qwen3_tts_extract_embedding_samples(
    Qwen3Tts * tts,
    const float * ref_samples,
    int32_t n_ref_samples,
    float * embedding_out,
    int32_t max_size);

/* Save / load speaker embedding to/from a binary file.
 * Returns 1 on success, 0 on failure.                               */
int qwen3_tts_save_embedding(const char * path,
                              const float * embedding,
                              int32_t size);

int qwen3_tts_load_embedding(const char * path,
                              float * embedding_out,
                              int32_t max_size,
                              int32_t * size_out);

/* -------------------------------------------------------------------
 * Misc
 * ------------------------------------------------------------------- */

/* Load generation_config.json (optional defaults).
 * Returns 1 on success, 0 on failure.                               */
int qwen3_tts_load_generation_config(Qwen3Tts * tts, const char * json_path);

/* Get output sample rate (always 24000) */
int32_t qwen3_tts_sample_rate(const Qwen3Tts * tts);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3TTS_C_API_H */

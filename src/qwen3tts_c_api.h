/* qwen3tts_c_api.h — C API for qwen3-tts.cpp
 *
 * THREAD SAFETY:
 *   A Qwen3Tts* handle is NOT thread-safe. Do not call any function on the
 *   same handle from multiple goroutines/threads concurrently.
 *   For parallel synthesis, create one handle per worker thread.
 *   The synthesis functions are blocking — run them in a goroutine with a
 *   timeout if cancellation is needed.
 *
 * TYPICAL GO USAGE:
 *   tts := C.qwen3_tts_create(modelDir, 0)
 *   defer C.qwen3_tts_destroy(tts)
 *
 *   var params C.Qwen3TtsParams
 *   C.qwen3_tts_default_params(&params)
 *
 *   result := C.qwen3_tts_synthesize_ex(tts, text, &params)
 *   defer C.qwen3_tts_free_result(result)
 *   if result.success == 0 { ... }
 *   // use result.audio.samples[0..result.audio.n_samples]
 */
#ifndef QWEN3TTS_C_API_H
#define QWEN3TTS_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------
 * Opaque handle (one per inference worker — NOT thread-safe)
 * ------------------------------------------------------------------- */
typedef struct Qwen3Tts Qwen3Tts;

/* -------------------------------------------------------------------
 * Progress callback
 * tokens_done: number of codec frames generated so far
 * tokens_total: max_audio_tokens (upper bound, not exact length)
 * userdata: pointer passed to qwen3_tts_set_progress_callback
 * Return 0 to continue, non-zero to request early stop (best-effort).
 * ------------------------------------------------------------------- */
typedef int (*Qwen3TtsProgressFn)(int32_t tokens_done,
                                   int32_t tokens_total,
                                   void *  userdata);

/* -------------------------------------------------------------------
 * Generation parameters
 * ------------------------------------------------------------------- */
typedef struct Qwen3TtsParams {
    /* ---- Core sampling ---- */
    int32_t max_audio_tokens;       /* default: 4096                      */
    float   temperature;            /* default: 0.9, 0=greedy             */
    float   top_p;                  /* default: 1.0                       */
    int32_t top_k;                  /* default: 50, 0=disabled            */
    float   repetition_penalty;     /* default: 1.05                      */

    /* ---- Sub-talker (code predictor CB1-15) ---- */
    float   subtalker_temperature;  /* default: -1.0 (inherit main)       */
    int32_t subtalker_top_k;        /* default: -1   (inherit main)       */
    float   subtalker_top_p;        /* default: -1.0 (inherit main)       */

    /* ---- Concurrency ---- */
    int32_t n_threads;              /* 0=auto (hardware_concurrency/2)    */

    /* ---- Language / speaker / style ---- */
    /* Preferred: language_name. Fallback: language_id (deprecated).      */
    char    language_name[64];      /* "auto","english","chinese",… ""=id */
    int32_t language_id;            /* deprecated, use language_name      */
    char    speaker[64];            /* CustomVoice named speaker, e.g. "Vivian" */
    char    instruct[512];          /* VoiceDesign/CustomVoice style text */

    /* ---- ICL voice cloning ---- */
    /* Set ref_text + icl_mode=1 for full ICL (requires Mimi encoder).    */
    char    ref_text[4096];         /* reference transcript                */
    int32_t icl_mode;               /* 0=x-vector only, 1=ICL mode        */

    /* ---- Generation mode ---- */
    int32_t non_streaming_mode;     /* 0=streaming (default), 1=non-streaming */

    /* ---- Output ---- */
    int32_t print_timing;           /* 1=print timing to stderr (default) */
    int32_t print_progress;         /* 1=print per-token progress to stderr */
} Qwen3TtsParams;

/* -------------------------------------------------------------------
 * Timing breakdown (milliseconds) — filled by synthesize_ex()
 * ------------------------------------------------------------------- */
typedef struct Qwen3TtsTiming {
    int64_t t_tokenize_ms;
    int64_t t_encode_ms;    /* speaker encoder (0 if no voice clone)     */
    int64_t t_generate_ms;  /* transformer + code predictor              */
    int64_t t_decode_ms;    /* vocoder                                   */
    int64_t t_total_ms;
} Qwen3TtsTiming;

/* -------------------------------------------------------------------
 * Memory snapshot (bytes)
 * ------------------------------------------------------------------- */
typedef struct Qwen3TtsMemory {
    uint64_t rss_start;     /* RSS at synthesis start                    */
    uint64_t rss_end;       /* RSS at synthesis end                      */
    uint64_t rss_peak;      /* peak RSS during synthesis                 */
    uint64_t phys_start;    /* physical footprint start (macOS/Win)      */
    uint64_t phys_end;
    uint64_t phys_peak;
} Qwen3TtsMemory;

/* -------------------------------------------------------------------
 * Audio buffer
 * ------------------------------------------------------------------- */
typedef struct Qwen3TtsAudio {
    const float * samples;          /* PCM float32 mono, 24 kHz           */
    int32_t       n_samples;
    int32_t       sample_rate;      /* always 24000                       */
} Qwen3TtsAudio;

/* -------------------------------------------------------------------
 * Extended result (returned by synthesize_ex)
 * Caller owns this — free with qwen3_tts_free_result().
 * ------------------------------------------------------------------- */
typedef struct Qwen3TtsResult {
    int32_t         success;        /* 1=ok, 0=failed                     */
    Qwen3TtsAudio   audio;          /* valid only if success==1           */
    Qwen3TtsTiming  timing;
    Qwen3TtsMemory  memory;
    char            error_msg[512]; /* null-terminated error on failure   */
} Qwen3TtsResult;

/* -------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------- */

/* Fill params with defaults. Always call this before modifying params. */
void qwen3_tts_default_params(Qwen3TtsParams * params);

/* Create engine, load models from directory.
 * n_threads: number of CPU threads (0 = auto).
 * Returns NULL on failure — check qwen3_tts_last_error() for details.  */
Qwen3Tts * qwen3_tts_create(const char * model_dir, int32_t n_threads);

/* Same as create but also loads generation_config.json if present.     */
Qwen3Tts * qwen3_tts_create_with_config(const char * model_dir,
                                          const char * gen_config_path,
                                          int32_t n_threads);

/* Check if models are loaded. */
int qwen3_tts_is_loaded(const Qwen3Tts * tts);

/* Unload all models and free GPU/CPU buffers.
 * The handle remains valid — call qwen3_tts_destroy() to fully release. */
void qwen3_tts_unload(Qwen3Tts * tts);

/* Fully destroy engine and free all resources. */
void qwen3_tts_destroy(Qwen3Tts * tts);

/* Get last error message (valid until next API call on this handle).    */
const char * qwen3_tts_get_error(const Qwen3Tts * tts);

/* Get output sample rate (always 24000 Hz). */
int32_t qwen3_tts_sample_rate(const Qwen3Tts * tts);

/* Get speaker embedding dimension (typically 1024). 
 * Returns 0 if models not loaded.                                       */
int32_t qwen3_tts_embedding_size(const Qwen3Tts * tts);

/* -------------------------------------------------------------------
 * Progress callback
 * The callback is called once per generated codec frame (~80ms audio).
 * Return non-zero from the callback to request early termination.
 * thread-safety: callback runs on the calling thread.
 * ------------------------------------------------------------------- */
void qwen3_tts_set_progress_callback(Qwen3Tts * tts,
                                      Qwen3TtsProgressFn fn,
                                      void * userdata);

void qwen3_tts_clear_progress_callback(Qwen3Tts * tts);

/* -------------------------------------------------------------------
 * Model introspection
 * ------------------------------------------------------------------- */

/* Returns "base", "custom_voice", "voice_design", or "". */
const char * qwen3_tts_model_type(const Qwen3Tts * tts);

/* Returns "0.6b", "1.7b", or "". */
const char * qwen3_tts_model_size(const Qwen3Tts * tts);

/* List speaker names separated by '\n' into buf.
 * Returns count, or -1 on error.                                        */
int qwen3_tts_list_speakers(const Qwen3Tts * tts,
                             char * buf, int32_t buf_size);

/* List language names separated by '\n' into buf.
 * Returns count, or -1 on error.                                        */
int qwen3_tts_list_languages(const Qwen3Tts * tts,
                              char * buf, int32_t buf_size);

/* Resolve language name → codec token ID. Returns -1 if not found.     */
int32_t qwen3_tts_resolve_language(const Qwen3Tts * tts,
                                   const char * language_name);

/* -------------------------------------------------------------------
 * Synthesis — simple API (caller frees audio with qwen3_tts_free_audio)
 * ------------------------------------------------------------------- */

Qwen3TtsAudio * qwen3_tts_synthesize(
    Qwen3Tts * tts,
    const char * text,
    const Qwen3TtsParams * params);

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_file(
    Qwen3Tts * tts,
    const char * text,
    const char * reference_audio_path,
    const Qwen3TtsParams * params);

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_samples(
    Qwen3Tts * tts,
    const char * text,
    const float * ref_samples, int32_t n_ref_samples,
    const Qwen3TtsParams * params);

Qwen3TtsAudio * qwen3_tts_synthesize_with_embedding(
    Qwen3Tts * tts,
    const char * text,
    const float * embedding, int32_t embedding_size,
    const Qwen3TtsParams * params);

void qwen3_tts_free_audio(Qwen3TtsAudio * audio);

/* -------------------------------------------------------------------
 * Synthesis — extended API (returns full result with timing + memory)
 * Caller frees with qwen3_tts_free_result().
 * ------------------------------------------------------------------- */

Qwen3TtsResult * qwen3_tts_synthesize_ex(
    Qwen3Tts * tts,
    const char * text,
    const Qwen3TtsParams * params);

Qwen3TtsResult * qwen3_tts_synthesize_with_voice_file_ex(
    Qwen3Tts * tts,
    const char * text,
    const char * reference_audio_path,
    const Qwen3TtsParams * params);

Qwen3TtsResult * qwen3_tts_synthesize_with_voice_samples_ex(
    Qwen3Tts * tts,
    const char * text,
    const float * ref_samples, int32_t n_ref_samples,
    const Qwen3TtsParams * params);

Qwen3TtsResult * qwen3_tts_synthesize_with_embedding_ex(
    Qwen3Tts * tts,
    const char * text,
    const float * embedding, int32_t embedding_size,
    const Qwen3TtsParams * params);

void qwen3_tts_free_result(Qwen3TtsResult * result);

/* -------------------------------------------------------------------
 * Speaker embedding utilities
 * ------------------------------------------------------------------- */

/* Extract from WAV file. Returns embedding size (typically 1024) or -1. */
int32_t qwen3_tts_extract_embedding_file(
    Qwen3Tts * tts,
    const char * reference_audio_path,
    float * embedding_out, int32_t max_size);

/* Extract from raw PCM float32 samples at 24 kHz mono. */
int32_t qwen3_tts_extract_embedding_samples(
    Qwen3Tts * tts,
    const float * ref_samples, int32_t n_ref_samples,
    float * embedding_out, int32_t max_size);

/* Save/load embedding to/from binary file. Returns 1=ok, 0=fail. */
int qwen3_tts_save_embedding(const char * path,
                              const float * embedding, int32_t size);

int qwen3_tts_load_embedding(const char * path,
                              float * embedding_out, int32_t max_size,
                              int32_t * size_out);

/* -------------------------------------------------------------------
 * Audio I/O utilities
 * ------------------------------------------------------------------- */

/* Save PCM float32 audio as a standard 16-bit WAV file.
 * Returns 1=ok, 0=fail.                                               */
int qwen3_tts_save_wav(const char * path,
                       const float * samples, int32_t n_samples,
                       int32_t sample_rate);

/* Load a WAV file into float32 PCM samples (caller-allocated).
 * Returns number of samples written, or -1 on failure.
 * If samples_out is NULL, returns required buffer size.               */
int32_t qwen3_tts_load_wav(const char * path,
                            float * samples_out, int32_t max_samples,
                            int32_t * sample_rate_out);

/* -------------------------------------------------------------------
 * Configuration
 * ------------------------------------------------------------------- */

/* Load generation_config.json defaults (temperature, top_k, etc.).
 * Returns 1=ok, 0=fail.                                               */
int qwen3_tts_load_generation_config(Qwen3Tts * tts,
                                      const char * json_path);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3TTS_C_API_H */

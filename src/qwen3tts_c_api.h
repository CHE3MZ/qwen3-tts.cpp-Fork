/* qwen3tts_c_api.h — C API for qwen3-tts.cpp
 *
 * THREAD SAFETY:
 *   A Qwen3Tts* handle is NOT thread-safe. Do not call any function on the
 *   same handle from multiple threads concurrently.
 *   For parallel synthesis, create one handle per worker thread.
 *   The synthesis functions are blocking — run them in a thread or async task
 *   with a timeout if cancellation is needed.
 *
 * TYPICAL USAGE (Go/Rust/Python/C/C++):
 *   tts = qwen3_tts_create(model_dir, 0);
 *   // defer/finally: qwen3_tts_destroy(tts)
 *
 *   Qwen3TtsParams params;
 *   qwen3_tts_default_params(&params);
 *
 *   Qwen3TtsResult *result = qwen3_tts_synthesize_ex(tts, text, &params);
 *   // defer/finally: qwen3_tts_free_result(result)
 *   if (result->success == 0) { } // check error_msg for details
 *   // use result->audio.samples[0..result->audio.n_samples]
 */
#ifndef QWEN3TTS_C_API_H
#define QWEN3TTS_C_API_H

#include <stdint.h>
#include "ggml.h"   /* for ggml_abort_callback */

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
    int32_t n_threads;              /* 0=auto (min(hardware_concurrency, 8)) */

    /* ---- Language / speaker / style ---- */
    /* language_name takes precedence over language_id when non-empty.
     * Preferred: use language_name. language_id is deprecated and ignored
     * when language_name is set. Set language_name="" to use language_id.  */
    char    language_name[64];      /* "auto","english","chinese",… ""=use id */
    int32_t language_id;            /* deprecated; use language_name instead  */
    char    speaker[64];            /* CustomVoice named speaker, e.g. "Vivian" */
    char    instruct[512];          /* VoiceDesign/CustomVoice style text */

    /* ---- ICL voice cloning ---- */
    /* Set ref_text + icl_mode=1 for full ICL (requires Mimi encoder).    */
    /* NOTE: ref_text is limited to 4095 bytes. Longer transcripts will   */
    /* be silently truncated. For long transcripts consider x-vector only.*/
    char    ref_text[4096];         /* reference transcript (UTF-8)        */
    int32_t icl_mode;               /* 0=x-vector only, 1=ICL mode        */

    /* ---- Generation mode ---- */
    int32_t non_streaming_mode;     /* 0=streaming (default), 1=non-streaming */

    /* ---- Output ---- */
    int32_t print_timing;           /* 1=print timing to stderr (default) */
    int32_t print_progress;         /* 1=print per-token progress to stderr */

    /* ---- Extended sampling (from llama.cpp reference) ---- */
    /* All default to 0/disabled for backward compatibility.  */
    float   min_p;              /* keep tokens where prob >= min_p * max_prob (0=off)  */
    float   frequency_penalty; /* subtract freq_pen * count from logit (0=off)         */
    float   presence_penalty;  /* subtract presence_pen if token ever appeared (0=off) */

    /* DRY (Don't Repeat Yourself) n-gram penalty                                      */
    float   dry_multiplier;     /* penalty scale; 0=disabled, try 0.8                  */
    float   dry_base;           /* exponential base per extra repeated token (def 1.75)*/
    int32_t dry_allowed_length; /* min n-gram length before penalising (default 2)     */
    int32_t dry_penalty_last_n; /* context window (-1=all generated tokens, 0=off)     */

    /* Dynamic temperature (entropy-adaptive)                                          */
    float   dyntemp_range;      /* half-range of temp variation; 0=disabled            */
    float   dyntemp_exponent;   /* shaping exponent (1.0=linear, default)              */
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
 * Returns NULL on failure — call qwen3_tts_get_last_create_error() to get
 * the error string (the handle is gone so qwen3_tts_get_error() is unavailable). */
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

/* Get last error message for a live handle (valid until next API call). */
const char * qwen3_tts_get_error(const Qwen3Tts * tts);

/* Get the error from the most recent failed qwen3_tts_create() call.
 * Thread-local: safe to call from any thread that called qwen3_tts_create().
 * Returns "" if the last create succeeded.                              */
const char * qwen3_tts_get_last_create_error(void);

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
 * Abort callback — cancels generation mid-graph (e.g. user pressed stop).
 * callback(data) returns true → abort current graph compute.
 *
 * IMPORTANT LIMITATION: On GPU backends (Vulkan, CUDA, Metal) the abort
 * has no effect during heavy graph compute steps — those schedulers do not
 * support mid-graph cancellation. The callback only fires on CPU backends.
 * On GPU builds, synthesis can be interrupted between codec frame steps
 * (the generate loop checks the callback return value each frame).
 * On CPU-only builds, cancellation fires per-node for near-instant response.
 * ------------------------------------------------------------------- */
void qwen3_tts_set_abort_callback(Qwen3Tts * tts,
                                   ggml_abort_callback fn,
                                   void * userdata);
void qwen3_tts_clear_abort_callback(Qwen3Tts * tts);

/* -------------------------------------------------------------------
 * Model introspection
 * ------------------------------------------------------------------- */

/* Returns 1 if the Mimi encoder (required for ICL voice cloning) is available.
 * ICL mode requires ref_text to be set in params AND the Mimi encoder to be
 * present in the model GGUF. Returns 0 if not available (x-vector only). */
int qwen3_tts_has_mimi_encoder(const Qwen3Tts * tts);

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

/* Resolve speaker name → codec token ID. Returns -1 if not found.
 * Useful for CustomVoice models to map speaker names to their IDs.      */
int32_t qwen3_tts_resolve_speaker(const Qwen3Tts * tts,
                                   const char * speaker_name);

/* -------------------------------------------------------------------
 * Batch synthesis — process N texts simultaneously.
 * Returns array of N Qwen3TtsResult pointers (caller frees each with
 * qwen3_tts_free_result). If any entry fails, its result will have
 * success=0 and error_msg populated. Returns NULL on internal error.
 * ------------------------------------------------------------------- */
Qwen3TtsResult ** qwen3_tts_synthesize_batch(
    Qwen3Tts * tts,
    const char ** texts, int32_t n_texts,
    const float * embedding, int32_t embedding_size,
    const Qwen3TtsParams * params);

/* Same as above but with per-entry instruct strings for VoiceDesign / CustomVoice.
 * instruct_texts: array of n_texts C strings (NULL or empty = no instruct for that entry).
 * Pass NULL for instruct_texts to get the same behaviour as synthesize_batch. */
Qwen3TtsResult ** qwen3_tts_synthesize_batch_ex(
    Qwen3Tts * tts,
    const char ** texts, int32_t n_texts,
    const float * embedding, int32_t embedding_size,
    const Qwen3TtsParams * params,
    const char ** instruct_texts);

/* Free a batch result array returned by synthesize_batch / synthesize_batch_ex.
 * Frees each individual result and then the array itself. */
void qwen3_tts_free_batch_results(Qwen3TtsResult ** results, int32_t n_results);

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

/* Resample float32 PCM audio from one sample rate to another.
 * Uses Kaiser-windowed sinc interpolation (same resampler used internally
 * for reference audio input).
 * input:           float32 PCM samples at input_rate
 * n_input:         number of input samples
 * input_rate:      source sample rate (e.g. 24000)
 * output_rate:     target sample rate (e.g. 48000)
 * output_out:      caller-allocated buffer; if NULL, returns required size
 * max_output:      capacity of output_out
 * Returns number of output samples written, or -1 on failure.
 *
 * NOTE: Upsampling from 24 kHz to 48 kHz (2× integer ratio) is lossless.
 * Upsampling does NOT add audio information above the source Nyquist (12 kHz).
 * Use for compatibility with pipelines that require 44100 or 48000 Hz. */
int32_t qwen3_tts_resample(const float * input, int32_t n_input,
                             int32_t input_rate, int32_t output_rate,
                             float * output_out, int32_t max_output);

/* -------------------------------------------------------------------
 * Configuration
 * ------------------------------------------------------------------- */

/* Load generation_config.json defaults (temperature, top_k, etc.).
 * Returns 1=ok, 0=fail.                                               */
int qwen3_tts_load_generation_config(Qwen3Tts * tts,
                                      const char * json_path);

/* -------------------------------------------------------------------
 * Per-frame logits callback
 * Called once per generated codec frame, after CB0 is sampled.
 *   frame_idx:      0-based frame index
 *   cb0_logits:     float[cb0_logits_size] — raw talker logits for codebook 0
 *   cb0_logits_size: typically 3072
 *   cb0_token:      the token sampled for codebook 0
 *   userdata:       pointer passed to qwen3_tts_set_logits_callback
 * Return non-zero to stop generation early (best-effort).
 * ------------------------------------------------------------------- */
typedef int (*Qwen3TtsLogitsFn)(int32_t frame_idx,
                                 const float * cb0_logits,
                                 int32_t cb0_logits_size,
                                 int32_t cb0_token,
                                 void *  userdata);

void qwen3_tts_set_logits_callback(Qwen3Tts * tts,
                                    Qwen3TtsLogitsFn fn,
                                    void * userdata);

void qwen3_tts_clear_logits_callback(Qwen3Tts * tts);

/* -------------------------------------------------------------------
 * Streaming audio chunk callback
 * Called with each decoded PCM chunk as it is produced.
 *   samples:     float32 PCM, 24 kHz mono, [-1, 1]
 *   n_samples:   number of samples in this chunk
 *   sample_rate: always 24000
 *   is_last:     1 if this is the final chunk for this synthesis call
 *   userdata:    pointer passed to qwen3_tts_set_audio_chunk_callback
 * Return non-zero to abort remaining synthesis (best-effort).
 * ------------------------------------------------------------------- */
typedef int (*Qwen3TtsAudioChunkFn)(const float * samples,
                                     int32_t       n_samples,
                                     int32_t       sample_rate,
                                     int           is_last,
                                     void *        userdata);

/* chunk_frames: how many codec frames to decode per chunk (0 = default = 12).
 * 12 frames = ~1 second of audio at 12 Hz.                           */
void qwen3_tts_set_audio_chunk_callback(Qwen3Tts * tts,
                                         Qwen3TtsAudioChunkFn fn,
                                         void * userdata,
                                         int32_t chunk_frames);

void qwen3_tts_clear_audio_chunk_callback(Qwen3Tts * tts);

/* -------------------------------------------------------------------
 * Speech codes access — generate codes without decoding to audio
 * Useful for: caching voices, vocoder swapping, offline processing.
 *
 * codes_out:        caller-allocated int32_t[max_frames * n_codebooks]
 * max_frames:       capacity of codes_out in frames
 * n_codebooks_out:  filled with n_codebooks (always 16)
 *
 * Returns number of frames generated, or -1 on failure.
 * To query required buffer size, pass codes_out=NULL (returns frame count).
 * ------------------------------------------------------------------- */
int32_t qwen3_tts_synthesize_codes(
    Qwen3Tts * tts,
    const char * text,
    const Qwen3TtsParams * params,
    int32_t * codes_out, int32_t max_frames,
    int32_t * n_codebooks_out);

int32_t qwen3_tts_synthesize_codes_with_voice_file(
    Qwen3Tts * tts,
    const char * text,
    const char * reference_audio_path,
    const Qwen3TtsParams * params,
    int32_t * codes_out, int32_t max_frames,
    int32_t * n_codebooks_out);

/* Synthesize codes from raw PCM samples instead of a WAV file path.
 * ref_samples: float32 PCM at 24 kHz mono, normalized [-1, 1].         */
int32_t qwen3_tts_synthesize_codes_with_voice_samples(
    Qwen3Tts * tts,
    const char * text,
    const float * ref_samples, int32_t n_ref_samples,
    const Qwen3TtsParams * params,
    int32_t * codes_out, int32_t max_frames,
    int32_t * n_codebooks_out);

int32_t qwen3_tts_synthesize_codes_with_embedding(
    Qwen3Tts * tts,
    const char * text,
    const float * embedding, int32_t embedding_size,
    const Qwen3TtsParams * params,
    int32_t * codes_out, int32_t max_frames,
    int32_t * n_codebooks_out);

/* Decode previously obtained speech codes to audio.
 * codes:        int32_t[n_frames * n_codebooks] row-major
 * Returns a Qwen3TtsResult* — caller frees with qwen3_tts_free_result(). */
Qwen3TtsResult * qwen3_tts_decode_codes(
    Qwen3Tts * tts,
    const int32_t * codes, int32_t n_frames, int32_t n_codebooks,
    const Qwen3TtsParams * params);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3TTS_C_API_H */

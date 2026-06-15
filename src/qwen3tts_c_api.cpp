/* qwen3tts_c_api.cpp — C API wrapper for Go/Nim/Python FFI. */
#include "qwen3_tts.h"
#include "qwen3tts_c_api.h"

#ifdef __APPLE__
#include <objc/objc.h>
#include <objc/message.h>
static void * new_autorelease_pool() {
    return (void *)((id(*)(id,SEL))objc_msgSend)(
        (id)objc_getClass("NSAutoreleasePool"), sel_registerName("new"));
}
static void drain_autorelease_pool(void * p) {
    ((void(*)(id,SEL))objc_msgSend)((id)p, sel_registerName("drain"));
}
#define ARP_BEGIN void * _pool = new_autorelease_pool();
#define ARP_END   drain_autorelease_pool(_pool);
#else
#define ARP_BEGIN
#define ARP_END
#endif

#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>

// ============================================================
// Internal struct
// ============================================================
struct Qwen3Tts {
    qwen3_tts::Qwen3TTS engine;
    std::string last_error;
    std::string model_type_buf;
    std::string model_size_buf;
    // Default thread count stored at create time; applied as param default
    int32_t default_n_threads = 0;
    // Progress callback
    Qwen3TtsProgressFn progress_fn   = nullptr;
    void *             progress_data = nullptr;
    // Per-frame logits callback
    Qwen3TtsLogitsFn   logits_fn     = nullptr;
    void *             logits_data   = nullptr;
    // Streaming audio chunk callback
    Qwen3TtsAudioChunkFn chunk_fn    = nullptr;
    void *               chunk_data  = nullptr;
};

// ============================================================
// Helpers
// ============================================================
static qwen3_tts::tts_params to_cpp_params(const Qwen3TtsParams * p,
                                            int32_t default_n_threads = 0) {
    qwen3_tts::tts_params out;
    if (!p) return out;
    out.max_audio_tokens      = p->max_audio_tokens;
    out.temperature           = p->temperature;
    out.top_p                 = p->top_p;
    out.top_k                 = p->top_k;
    // Thread resolution order: params field > handle default > hardware auto-detect
    if (p->n_threads > 0)
        out.n_threads = p->n_threads;
    else if (default_n_threads > 0)
        out.n_threads = default_n_threads;
    else
        out.n_threads = (int32_t)std::min(8u, std::thread::hardware_concurrency());
    out.repetition_penalty    = p->repetition_penalty;
    out.subtalker_temperature = p->subtalker_temperature;
    out.subtalker_top_k       = p->subtalker_top_k;
    out.subtalker_top_p       = p->subtalker_top_p;
    out.icl_mode              = (p->icl_mode != 0);
    out.non_streaming_mode    = (p->non_streaming_mode != 0);
    out.print_timing          = (p->print_timing != 0);
    out.print_progress        = (p->print_progress != 0);
    if (p->language_name[0] != '\0') out.language = p->language_name;
    else if (p->language_id > 0)     out.language = std::to_string(p->language_id);
    if (p->speaker[0]  != '\0') out.speaker  = p->speaker;
    if (p->instruct[0] != '\0') out.instruct = p->instruct;
    if (p->ref_text[0] != '\0') out.ref_text = p->ref_text;
    return out;
}

// Copy float audio from tts_result into a heap-allocated Qwen3TtsAudio
static Qwen3TtsAudio make_audio_value(const qwen3_tts::tts_result & r) {
    Qwen3TtsAudio a = {};
    if (r.success && !r.audio.empty()) {
        auto * buf = new float[r.audio.size()];
        std::memcpy(buf, r.audio.data(), r.audio.size() * sizeof(float));
        a.samples     = buf;
        a.n_samples   = (int32_t)r.audio.size();
        a.sample_rate = r.sample_rate;
    }
    return a;
}

static Qwen3TtsResult * make_result(const qwen3_tts::tts_result & r) {
    auto * out       = new Qwen3TtsResult{};
    out->success     = r.success ? 1 : 0;
    out->audio       = make_audio_value(r);
    out->timing.t_tokenize_ms = r.t_tokenize_ms;
    out->timing.t_encode_ms   = r.t_encode_ms;
    out->timing.t_generate_ms = r.t_generate_ms;
    out->timing.t_decode_ms   = r.t_decode_ms;
    out->timing.t_total_ms    = r.t_total_ms;
    out->memory.rss_start     = r.mem_rss_start_bytes;
    out->memory.rss_end       = r.mem_rss_end_bytes;
    out->memory.rss_peak      = r.mem_rss_peak_bytes;
    out->memory.phys_start    = r.mem_phys_start_bytes;
    out->memory.phys_end      = r.mem_phys_end_bytes;
    out->memory.phys_peak     = r.mem_phys_peak_bytes;
    if (!r.success) {
        std::strncpy(out->error_msg, r.error_msg.c_str(), sizeof(out->error_msg) - 1);
    }
    return out;
}

// ============================================================
extern "C" {
// ============================================================

// ---- params -----------------------------------------------------------------

void qwen3_tts_default_params(Qwen3TtsParams * p) {
    if (!p) return;
    std::memset(p, 0, sizeof(*p));
    p->max_audio_tokens      = 4096;
    p->temperature           = 0.9f;
    p->top_p                 = 1.0f;
    p->top_k                 = 50;
    p->n_threads             = 0;   // 0 = auto
    p->repetition_penalty    = 1.05f;
    p->language_id           = 2050;
    p->subtalker_temperature = -1.0f;
    p->subtalker_top_k       = -1;
    p->subtalker_top_p       = -1.0f;
    p->icl_mode              = 0;
    p->non_streaming_mode    = 0;
    p->print_timing          = 0;
    p->print_progress        = 0;
    std::strncpy(p->language_name, "auto", sizeof(p->language_name) - 1);
}

// ---- lifecycle --------------------------------------------------------------

Qwen3Tts * qwen3_tts_create(const char * model_dir, int32_t n_threads) {
    if (!model_dir) return nullptr;
    auto * tts = new Qwen3Tts;
    // Store the requested thread count so all subsequent synthesize calls use it
    // unless the caller overrides via params.n_threads.
    tts->default_n_threads = (n_threads > 0) ? n_threads : 0;
    if (!tts->engine.load_models(model_dir)) {
        tts->last_error = tts->engine.get_error();
        delete tts;
        return nullptr;
    }
    return tts;
}

Qwen3Tts * qwen3_tts_create_with_config(const char * model_dir,
                                          const char * gen_config_path,
                                          int32_t n_threads) {
    Qwen3Tts * tts = qwen3_tts_create(model_dir, n_threads);
    if (!tts) return nullptr;
    if (gen_config_path && gen_config_path[0] != '\0') {
        tts->engine.load_generation_config(gen_config_path);
    }
    return tts;
}

int qwen3_tts_is_loaded(const Qwen3Tts * tts) {
    return (tts && tts->engine.is_loaded()) ? 1 : 0;
}

void qwen3_tts_unload(Qwen3Tts * tts) {
    if (!tts) return;
    tts->engine.unload_models();
    tts->last_error.clear();
}

void qwen3_tts_destroy(Qwen3Tts * tts) { delete tts; }

const char * qwen3_tts_get_error(const Qwen3Tts * tts) {
    return tts ? tts->last_error.c_str() : "";
}

int32_t qwen3_tts_sample_rate(const Qwen3Tts * /*tts*/) { return 24000; }

int32_t qwen3_tts_embedding_size(const Qwen3Tts * tts) {
    if (!tts || !tts->engine.is_loaded()) return 0;
    // Read the actual embedding dimension from the speaker encoder config
    // stored in the transformer GGUF metadata, rather than hardcoding 1024.
    // Falls back to 1024 if the config hasn't been populated yet (encoder lazy-loads).
    int32_t dim = tts->engine.get_embedding_dim();
    return (dim > 0) ? dim : 1024;
}

// ---- progress callback ------------------------------------------------------

void qwen3_tts_set_progress_callback(Qwen3Tts * tts,
                                      Qwen3TtsProgressFn fn,
                                      void * userdata) {
    if (!tts) return;
    tts->progress_fn   = fn;
    tts->progress_data = userdata;
    if (fn) {
        // Wire through to C++ progress callback
        tts->engine.set_progress_callback([tts](int done, int total) {
            if (tts->progress_fn) {
                tts->progress_fn((int32_t)done, (int32_t)total,
                                  tts->progress_data);
            }
        });
    } else {
        tts->engine.set_progress_callback(nullptr);
    }
}

void qwen3_tts_clear_progress_callback(Qwen3Tts * tts) {
    qwen3_tts_set_progress_callback(tts, nullptr, nullptr);
}

// ---- introspection ----------------------------------------------------------

const char * qwen3_tts_model_type(const Qwen3Tts * tts) {
    if (!tts) return "";
    const_cast<Qwen3Tts *>(tts)->model_type_buf = tts->engine.get_model_type();
    return tts->model_type_buf.c_str();
}

const char * qwen3_tts_model_size(const Qwen3Tts * tts) {
    if (!tts) return "";
    const_cast<Qwen3Tts *>(tts)->model_size_buf = tts->engine.get_model_size();
    return tts->model_size_buf.c_str();
}

int qwen3_tts_list_speakers(const Qwen3Tts * tts, char * buf, int32_t buf_size) {
    if (!tts || !buf || buf_size <= 0) return -1;
    auto spk = tts->engine.get_supported_speakers();
    std::string out;
    for (const auto & s : spk) { out += s; out += '\n'; }
    std::strncpy(buf, out.c_str(), (size_t)(buf_size - 1));
    buf[buf_size - 1] = '\0';
    return (int)spk.size();
}

int qwen3_tts_list_languages(const Qwen3Tts * tts, char * buf, int32_t buf_size) {
    if (!tts || !buf || buf_size <= 0) return -1;
    auto langs = tts->engine.get_supported_languages();
    std::string out;
    for (const auto & l : langs) { out += l; out += '\n'; }
    std::strncpy(buf, out.c_str(), (size_t)(buf_size - 1));
    buf[buf_size - 1] = '\0';
    return (int)langs.size();
}

int32_t qwen3_tts_resolve_language(const Qwen3Tts * tts, const char * name) {
    if (!tts || !name) return -1;
    return tts->engine.resolve_language_id(name);
}

// ---- simple synthesis -------------------------------------------------------

Qwen3TtsAudio * qwen3_tts_synthesize(
        Qwen3Tts * tts, const char * text, const Qwen3TtsParams * params) {
    if (!tts || !text) return nullptr;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize(text, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    Qwen3TtsAudio * out = nullptr;
    if (res.success && !res.audio.empty()) {
        out = new Qwen3TtsAudio;
        *out = make_audio_value(res);
    }
    ARP_END
    return out;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_file(
        Qwen3Tts * tts, const char * text, const char * ref_path,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !ref_path) return nullptr;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize_with_voice(text, ref_path, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    Qwen3TtsAudio * out = nullptr;
    if (res.success && !res.audio.empty()) { out = new Qwen3TtsAudio; *out = make_audio_value(res); }
    ARP_END
    return out;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_samples(
        Qwen3Tts * tts, const char * text,
        const float * ref, int32_t n_ref,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !ref || n_ref <= 0) return nullptr;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize_with_voice(text, ref, n_ref, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    Qwen3TtsAudio * out = nullptr;
    if (res.success && !res.audio.empty()) { out = new Qwen3TtsAudio; *out = make_audio_value(res); }
    ARP_END
    return out;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_embedding(
        Qwen3Tts * tts, const char * text,
        const float * emb, int32_t emb_size,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !emb || emb_size <= 0) return nullptr;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize_with_embedding(text, emb, emb_size, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    Qwen3TtsAudio * out = nullptr;
    if (res.success && !res.audio.empty()) { out = new Qwen3TtsAudio; *out = make_audio_value(res); }
    ARP_END
    return out;
}

void qwen3_tts_free_audio(Qwen3TtsAudio * a) {
    if (!a) return;
    delete[] a->samples;
    delete a;
}

// ---- extended synthesis (with timing + memory) ------------------------------

Qwen3TtsResult * qwen3_tts_synthesize_ex(
        Qwen3Tts * tts, const char * text, const Qwen3TtsParams * params) {
    if (!tts || !text) {
        auto * r = new Qwen3TtsResult{};
        std::strncpy(r->error_msg, "null handle or text", sizeof(r->error_msg)-1);
        return r;
    }
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize(text, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_result(res);
    ARP_END
    return out;
}

Qwen3TtsResult * qwen3_tts_synthesize_with_voice_file_ex(
        Qwen3Tts * tts, const char * text, const char * ref_path,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !ref_path) {
        auto * r = new Qwen3TtsResult{}; std::strncpy(r->error_msg,"null arg",sizeof(r->error_msg)-1); return r;
    }
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize_with_voice(text, ref_path, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_result(res);
    ARP_END
    return out;
}

Qwen3TtsResult * qwen3_tts_synthesize_with_voice_samples_ex(
        Qwen3Tts * tts, const char * text,
        const float * ref, int32_t n_ref,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !ref || n_ref <= 0) {
        auto * r = new Qwen3TtsResult{}; std::strncpy(r->error_msg,"null arg",sizeof(r->error_msg)-1); return r;
    }
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize_with_voice(text, ref, n_ref, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_result(res);
    ARP_END
    return out;
}

Qwen3TtsResult * qwen3_tts_synthesize_with_embedding_ex(
        Qwen3Tts * tts, const char * text,
        const float * emb, int32_t emb_size,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !emb || emb_size <= 0) {
        auto * r = new Qwen3TtsResult{}; std::strncpy(r->error_msg,"null arg",sizeof(r->error_msg)-1); return r;
    }
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.synthesize_with_embedding(text, emb, emb_size, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_result(res);
    ARP_END
    return out;
}

void qwen3_tts_free_result(Qwen3TtsResult * r) {
    if (!r) return;
    delete[] r->audio.samples;
    delete r;
}

// ---- speaker embedding ------------------------------------------------------

int32_t qwen3_tts_extract_embedding_file(
        Qwen3Tts * tts, const char * ref_path,
        float * out, int32_t max_size) {
    if (!tts || !ref_path || !out || max_size <= 0) return -1;
    ARP_BEGIN
    std::vector<float> emb;
    bool ok = tts->engine.extract_speaker_embedding(ref_path, emb);
    if (!ok) { tts->last_error = tts->engine.get_error(); ARP_END return -1; }
    int32_t n = (int32_t)std::min((size_t)max_size, emb.size());
    std::memcpy(out, emb.data(), n * sizeof(float));
    ARP_END
    return n;
}

int32_t qwen3_tts_extract_embedding_samples(
        Qwen3Tts * tts, const float * ref, int32_t n_ref,
        float * out, int32_t max_size) {
    if (!tts || !ref || n_ref <= 0 || !out || max_size <= 0) return -1;
    ARP_BEGIN
    std::vector<float> emb;
    qwen3_tts::tts_params p;
    bool ok = tts->engine.extract_speaker_embedding(ref, n_ref, emb, p);
    if (!ok) { tts->last_error = tts->engine.get_error(); ARP_END return -1; }
    int32_t n = (int32_t)std::min((size_t)max_size, emb.size());
    std::memcpy(out, emb.data(), n * sizeof(float));
    ARP_END
    return n;
}

int qwen3_tts_save_embedding(const char * path, const float * emb, int32_t size) {
    if (!path || !emb || size <= 0) return 0;
    std::vector<float> v(emb, emb + size);
    return qwen3_tts::save_speaker_embedding(path, v) ? 1 : 0;
}

int qwen3_tts_load_embedding(const char * path, float * out, int32_t max_size,
                              int32_t * size_out) {
    if (!path || !out || max_size <= 0) return 0;
    std::vector<float> v;
    if (!qwen3_tts::load_speaker_embedding(path, v)) return 0;
    int32_t n = (int32_t)std::min((size_t)max_size, v.size());
    std::memcpy(out, v.data(), n * sizeof(float));
    if (size_out) *size_out = n;
    return 1;
}

// ---- audio I/O --------------------------------------------------------------

int qwen3_tts_save_wav(const char * path,
                       const float * samples, int32_t n_samples,
                       int32_t sample_rate) {
    if (!path || !samples || n_samples <= 0 || sample_rate <= 0) return 0;
    std::vector<float> v(samples, samples + n_samples);
    return qwen3_tts::save_audio_file(path, v, sample_rate) ? 1 : 0;
}

int32_t qwen3_tts_load_wav(const char * path,
                            float * samples_out, int32_t max_samples,
                            int32_t * sample_rate_out) {
    if (!path) return -1;
    std::vector<float> samples;
    int sr = 0;
    if (!qwen3_tts::load_audio_file(path, samples, sr)) return -1;
    if (sample_rate_out) *sample_rate_out = sr;
    if (!samples_out) return (int32_t)samples.size();  // query size only
    int32_t n = (int32_t)std::min((size_t)max_samples, samples.size());
    std::memcpy(samples_out, samples.data(), n * sizeof(float));
    return n;
}

// ---- config -----------------------------------------------------------------

int qwen3_tts_load_generation_config(Qwen3Tts * tts, const char * json_path) {
    if (!tts || !json_path) return 0;
    return tts->engine.load_generation_config(json_path) ? 1 : 0;
}

// ---- logits callback --------------------------------------------------------

void qwen3_tts_set_logits_callback(Qwen3Tts * tts, Qwen3TtsLogitsFn fn, void * userdata) {
    if (!tts) return;
    tts->logits_fn   = fn;
    tts->logits_data = userdata;
    if (fn) {
        tts->engine.set_logits_callback(
            [tts](int32_t frame, const float * lg, int32_t sz, int32_t tok) -> int {
                return tts->logits_fn
                    ? tts->logits_fn(frame, lg, sz, tok, tts->logits_data)
                    : 0;
            });
    } else {
        tts->engine.set_logits_callback(nullptr);
    }
}

void qwen3_tts_clear_logits_callback(Qwen3Tts * tts) {
    qwen3_tts_set_logits_callback(tts, nullptr, nullptr);
}

// ---- streaming audio chunk callback -----------------------------------------

void qwen3_tts_set_audio_chunk_callback(Qwen3Tts * tts,
                                         Qwen3TtsAudioChunkFn fn,
                                         void * userdata,
                                         int32_t chunk_frames) {
    if (!tts) return;
    tts->chunk_fn   = fn;
    tts->chunk_data = userdata;
    if (fn) {
        tts->engine.set_audio_chunk_callback(
            [tts](const float * s, int32_t n, int32_t sr, int last) -> int {
                return tts->chunk_fn
                    ? tts->chunk_fn(s, n, sr, last, tts->chunk_data)
                    : 0;
            }, chunk_frames);
    } else {
        tts->engine.set_audio_chunk_callback(nullptr);
    }
}

void qwen3_tts_clear_audio_chunk_callback(Qwen3Tts * tts) {
    qwen3_tts_set_audio_chunk_callback(tts, nullptr, nullptr, 0);
}

// ---- speech codes access ----------------------------------------------------

// Internal helper: run synthesize_codes and optionally copy into caller buffer
static int32_t run_synthesize_codes(Qwen3Tts * tts,
                                     std::vector<int32_t> & codes,
                                     int32_t & n_cb,
                                     int32_t * codes_out, int32_t max_frames,
                                     int32_t * n_codebooks_out) {
    int32_t n_frames = (int32_t)(codes.size() / (size_t)n_cb);
    if (n_codebooks_out) *n_codebooks_out = n_cb;
    if (!codes_out) return n_frames;   // size-query mode
    int32_t copy_frames = std::min(n_frames, max_frames);
    std::memcpy(codes_out, codes.data(),
                (size_t)copy_frames * (size_t)n_cb * sizeof(int32_t));
    return copy_frames;
}

int32_t qwen3_tts_synthesize_codes(
        Qwen3Tts * tts, const char * text, const Qwen3TtsParams * params,
        int32_t * codes_out, int32_t max_frames, int32_t * n_codebooks_out) {
    if (!tts || !text) return -1;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    std::vector<int32_t> codes;
    int32_t n_cb = 0;
    int32_t n_frames = tts->engine.synthesize_codes(text, codes, n_cb, cpp);
    if (n_frames < 0) { tts->last_error = tts->engine.get_error(); ARP_END return -1; }
    int32_t ret = run_synthesize_codes(tts, codes, n_cb, codes_out, max_frames, n_codebooks_out);
    ARP_END
    return ret;
}

int32_t qwen3_tts_synthesize_codes_with_voice_file(
        Qwen3Tts * tts, const char * text, const char * ref_path,
        const Qwen3TtsParams * params,
        int32_t * codes_out, int32_t max_frames, int32_t * n_codebooks_out) {
    if (!tts || !text || !ref_path) return -1;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    std::vector<int32_t> codes;
    int32_t n_cb = 0;
    int32_t n_frames = tts->engine.synthesize_codes_with_voice(text, ref_path, codes, n_cb, cpp);
    if (n_frames < 0) { tts->last_error = tts->engine.get_error(); ARP_END return -1; }
    int32_t ret = run_synthesize_codes(tts, codes, n_cb, codes_out, max_frames, n_codebooks_out);
    ARP_END
    return ret;
}

int32_t qwen3_tts_synthesize_codes_with_embedding(
        Qwen3Tts * tts, const char * text,
        const float * emb, int32_t emb_size,
        const Qwen3TtsParams * params,
        int32_t * codes_out, int32_t max_frames, int32_t * n_codebooks_out) {
    if (!tts || !text || !emb || emb_size <= 0) return -1;
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    std::vector<int32_t> codes;
    int32_t n_cb = 0;
    int32_t n_frames = tts->engine.synthesize_codes_with_embedding(
                            text, emb, emb_size, codes, n_cb, cpp);
    if (n_frames < 0) { tts->last_error = tts->engine.get_error(); ARP_END return -1; }
    int32_t ret = run_synthesize_codes(tts, codes, n_cb, codes_out, max_frames, n_codebooks_out);
    ARP_END
    return ret;
}

Qwen3TtsResult * qwen3_tts_decode_codes(
        Qwen3Tts * tts,
        const int32_t * codes, int32_t n_frames, int32_t n_codebooks,
        const Qwen3TtsParams * params) {
    if (!tts || !codes || n_frames <= 0 || n_codebooks <= 0) {
        auto * r = new Qwen3TtsResult{};
        std::strncpy(r->error_msg, "null or invalid args", sizeof(r->error_msg)-1);
        return r;
    }
    ARP_BEGIN
    auto cpp = to_cpp_params(params, tts->default_n_threads);
    auto res = tts->engine.decode_speech_codes(codes, n_frames, n_codebooks, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_result(res);
    ARP_END
    return out;
}

} // extern "C"

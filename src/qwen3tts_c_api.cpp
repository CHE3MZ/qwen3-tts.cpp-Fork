/* qwen3tts_c_api.cpp — C API wrapper for Nim/Python FFI. */
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

// ---- Internal helpers -------------------------------------------------------

struct Qwen3Tts {
    qwen3_tts::Qwen3TTS engine;
    std::string last_error;
    // cached string returns (stable pointer lifetime)
    std::string model_type_buf;
    std::string model_size_buf;
};

static qwen3_tts::tts_params to_cpp_params(const Qwen3TtsParams * p) {
    qwen3_tts::tts_params out;
    if (!p) return out;
    out.max_audio_tokens     = p->max_audio_tokens;
    out.temperature          = p->temperature;
    out.top_p                = p->top_p;
    out.top_k                = p->top_k;
    out.n_threads            = p->n_threads;
    out.repetition_penalty   = p->repetition_penalty;
    out.subtalker_temperature = p->subtalker_temperature;
    out.subtalker_top_k      = p->subtalker_top_k;
    out.icl_mode             = (p->icl_mode != 0);
    out.non_streaming_mode   = (p->non_streaming_mode != 0);

    // Language: prefer language_name if non-empty
    if (p->language_name[0] != '\0') {
        out.language = p->language_name;
    } else if (p->language_id > 0) {
        out.language = std::to_string(p->language_id);
    }

    if (p->speaker[0]  != '\0') out.speaker  = p->speaker;
    if (p->instruct[0] != '\0') out.instruct = p->instruct;
    if (p->ref_text[0] != '\0') out.ref_text = p->ref_text;
    return out;
}

static Qwen3TtsAudio * make_audio(const qwen3_tts::tts_result & r) {
    if (!r.success || r.audio.empty()) return nullptr;
    auto * out = new Qwen3TtsAudio;
    auto * buf = new float[r.audio.size()];
    std::memcpy(buf, r.audio.data(), r.audio.size() * sizeof(float));
    out->samples     = buf;
    out->n_samples   = (int32_t)r.audio.size();
    out->sample_rate = r.sample_rate;
    return out;
}

// ---- Lifecycle ---------------------------------------------------------------

extern "C" {

void qwen3_tts_default_params(Qwen3TtsParams * p) {
    if (!p) return;
    std::memset(p, 0, sizeof(*p));
    p->max_audio_tokens      = 4096;
    p->temperature           = 0.9f;
    p->top_p                 = 1.0f;
    p->top_k                 = 50;
    p->n_threads             = 4;
    p->repetition_penalty    = 1.05f;
    p->language_id           = 2050;
    p->subtalker_temperature = -1.0f;
    p->subtalker_top_k       = -1;
    p->icl_mode              = 0;
    p->non_streaming_mode    = 0;
    std::strncpy(p->language_name, "auto", sizeof(p->language_name) - 1);
}

Qwen3Tts * qwen3_tts_create(const char * model_dir, int32_t /*n_threads*/) {
    if (!model_dir) return nullptr;
    auto * tts = new Qwen3Tts;
    if (!tts->engine.load_models(model_dir)) {
        tts->last_error = tts->engine.get_error();
        delete tts;
        return nullptr;
    }
    return tts;
}

int qwen3_tts_is_loaded(const Qwen3Tts * tts) {
    return (tts && tts->engine.is_loaded()) ? 1 : 0;
}

void qwen3_tts_destroy(Qwen3Tts * tts) { delete tts; }

void qwen3_tts_free_audio(Qwen3TtsAudio * a) {
    if (!a) return;
    delete[] a->samples;
    delete a;
}

const char * qwen3_tts_get_error(const Qwen3Tts * tts) {
    return tts ? tts->last_error.c_str() : "";
}

// ---- Introspection ----------------------------------------------------------

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

int32_t qwen3_tts_resolve_language(const Qwen3Tts * tts, const char * language_name) {
    if (!tts || !language_name) return -1;
    return tts->engine.resolve_language_id(language_name);
}

// ---- Synthesis --------------------------------------------------------------

Qwen3TtsAudio * qwen3_tts_synthesize(
        Qwen3Tts * tts, const char * text, const Qwen3TtsParams * params) {
    if (!tts || !text) return nullptr;
    ARP_BEGIN
    auto cpp  = to_cpp_params(params);
    auto res  = tts->engine.synthesize(text, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_audio(res);
    ARP_END
    return out;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_file(
        Qwen3Tts * tts, const char * text, const char * ref_path,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !ref_path) return nullptr;
    ARP_BEGIN
    auto cpp = to_cpp_params(params);
    auto res = tts->engine.synthesize_with_voice(text, ref_path, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_audio(res);
    ARP_END
    return out;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_voice_samples(
        Qwen3Tts * tts, const char * text,
        const float * ref_samples, int32_t n_ref,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !ref_samples || n_ref <= 0) return nullptr;
    ARP_BEGIN
    auto cpp = to_cpp_params(params);
    auto res = tts->engine.synthesize_with_voice(text, ref_samples, n_ref, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_audio(res);
    ARP_END
    return out;
}

Qwen3TtsAudio * qwen3_tts_synthesize_with_embedding(
        Qwen3Tts * tts, const char * text,
        const float * embedding, int32_t size,
        const Qwen3TtsParams * params) {
    if (!tts || !text || !embedding || size <= 0) return nullptr;
    ARP_BEGIN
    auto cpp = to_cpp_params(params);
    auto res = tts->engine.synthesize_with_embedding(text, embedding, size, cpp);
    if (!res.success) tts->last_error = res.error_msg;
    auto * out = make_audio(res);
    ARP_END
    return out;
}

// ---- Speaker embedding ------------------------------------------------------

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

// ---- Misc -------------------------------------------------------------------

int qwen3_tts_load_generation_config(Qwen3Tts * tts, const char * json_path) {
    if (!tts || !json_path) return 0;
    return tts->engine.load_generation_config(json_path) ? 1 : 0;
}

int32_t qwen3_tts_sample_rate(const Qwen3Tts * /*tts*/) {
    return 24000;
}

} // extern "C"

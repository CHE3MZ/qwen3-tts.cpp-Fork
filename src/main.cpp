#include "qwen3_tts.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <cstdlib>
#include <iostream>

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [options] -m <model_dir> -t <text>\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -m, --model <dir>            Model directory (required)\n");
    fprintf(stderr, "  -t, --text <text>             Text to synthesize (required)\n");
    fprintf(stderr, "  -o, --output <file>           Output WAV (default: output.wav)\n");
    fprintf(stderr, "\nVoice cloning (Base model):\n");
    fprintf(stderr, "  -r, --reference <file>        Reference audio WAV for voice cloning\n");
    fprintf(stderr, "  --ref-text <text>             Reference transcript (enables ICL mode)\n");
    fprintf(stderr, "  --embedding-in <file>         Load pre-computed speaker embedding\n");
    fprintf(stderr, "  --embedding-out <file>        Save speaker embedding (skips synthesis)\n");
    fprintf(stderr, "\nCustomVoice model:\n");
    fprintf(stderr, "  --speaker <name>              Named speaker (e.g. Vivian, Ryan)\n");
    fprintf(stderr, "  --list-speakers               List available speakers and exit\n");
    fprintf(stderr, "\nVoiceDesign / CustomVoice instruct:\n");
    fprintf(stderr, "  --instruct <text>             Style / emotion instruction\n");
    fprintf(stderr, "\nLanguage:\n");
    fprintf(stderr, "  -l, --language <lang>         Language name or codec ID\n");
    fprintf(stderr, "                                (auto, english, chinese, japanese, korean,\n");
    fprintf(stderr, "                                 russian, german, french, spanish, italian,\n");
    fprintf(stderr, "                                 portuguese, or raw int e.g. 2050)\n");
    fprintf(stderr, "  --list-languages              List supported languages and exit\n");
    fprintf(stderr, "\nSampling:\n");
    fprintf(stderr, "  --temperature <val>           Main-talker temperature (default: 0.9)\n");
    fprintf(stderr, "  --top-k <n>                   Main-talker top-k (default: 50)\n");
    fprintf(stderr, "  --top-p <val>                 Top-p (default: 1.0)\n");
    fprintf(stderr, "  --repetition-penalty <val>    Repetition penalty (default: 1.05)\n");
    fprintf(stderr, "  --sub-temperature <val>       Sub-talker temperature (-1=inherit)\n");
    fprintf(stderr, "  --sub-top-k <n>               Sub-talker top-k (-1=inherit)\n");
    fprintf(stderr, "  --max-tokens <n>              Max audio tokens (default: 4096)\n");
    fprintf(stderr, "  --non-streaming               Use non-streaming prefill layout\n");
    fprintf(stderr, "                                (feeds all text at once; matches Python non_streaming_mode=True)\n");
    fprintf(stderr, "\nServer mode (model loaded once, serves multiple requests):\n");
    fprintf(stderr, "  --server                      Read JSON requests from stdin, write JSON responses to stdout.\n");
    fprintf(stderr, "                                Keeps model in memory across requests — no per-request reload.\n");
    fprintf(stderr, "                                Input format (one JSON object per line):\n");
    fprintf(stderr, "                                  {\"text\":\"...\",\"output\":\"out.wav\"}\n");
    fprintf(stderr, "                                Optional fields: \"reference\",\"ref_text\",\"speaker\",\n");
    fprintf(stderr, "                                  \"instruct\",\"language\",\"temperature\",\"top_k\",\n");
    fprintf(stderr, "                                  \"top_p\",\"repetition_penalty\",\"max_tokens\"\n");
    fprintf(stderr, "                                Output format (one JSON object per line):\n");
    fprintf(stderr, "                                  {\"success\":true,\"output\":\"out.wav\",\"duration_s\":2.5,\n");
    fprintf(stderr, "                                   \"t_total_ms\":12000}\n");
    fprintf(stderr, "                                  {\"success\":false,\"error\":\"message\"}\n");
    fprintf(stderr, "                                Send EOF (Ctrl+D / close stdin) to exit.\n");
    fprintf(stderr, "\nMisc:\n");
    fprintf(stderr, "  -j, --threads <n>             Threads (default: auto)\n");
    fprintf(stderr, "  --gen-config <file>           Load generation_config.json\n");
    fprintf(stderr, "  -h, --help                    Show help\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  # Basic synthesis\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" -o hello.wav\n", prog);
    fprintf(stderr, "  # Voice clone (x-vector only)\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" -r ref.wav -o clone.wav\n", prog);
    fprintf(stderr, "  # Voice clone with ICL\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" -r ref.wav --ref-text \"Reference text.\" -o icl.wav\n", prog);
    fprintf(stderr, "  # CustomVoice\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" --speaker Vivian --instruct \"Happy tone\" -o cv.wav\n", prog);
    fprintf(stderr, "  # VoiceDesign\n");
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" --instruct \"Speak softly\" -o vd.wav\n", prog);
    fprintf(stderr, "  # Save embedding, then reuse\n");
    fprintf(stderr, "  %s -m ./models -r ref.wav --embedding-out spk.bin\n", prog);
    fprintf(stderr, "  %s -m ./models -t \"Hello!\" --embedding-in spk.bin -o hello.wav\n", prog);
    fprintf(stderr, "  # Server mode (load once, serve many)\n");
    fprintf(stderr, "  %s -m ./models --server\n", prog);
}

// ---------------------------------------------------------------------------
// Minimal JSON helpers — no external deps, flat key-value only
// ---------------------------------------------------------------------------

// Extract a string value from flat JSON. Returns empty string if not found.
static std::string json_get_str(const std::string & json, const std::string & key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    ++pos;
    while (pos < json.size() && (json[pos]==' '||json[pos]=='\t')) ++pos;
    if (pos >= json.size() || json[pos] != '"') return "";
    ++pos; // skip opening quote
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            ++pos;
            switch (json[pos]) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                default:   result += json[pos]; break;
            }
        } else {
            result += json[pos];
        }
        ++pos;
    }
    return result;
}

static float json_get_float(const std::string & json, const std::string & key, float def) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return def;
    ++pos;
    while (pos < json.size() && (json[pos]==' '||json[pos]=='\t')) ++pos;
    char * end = nullptr;
    float v = strtof(json.c_str() + pos, &end);
    if (end == json.c_str() + pos) return def;
    return v;
}

static int json_get_int(const std::string & json, const std::string & key, int def) {
    float v = json_get_float(json, key, (float)def);
    return (int)v;
}

// Escape a string for JSON output
static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if      (c == '"')  { out += "\\\""; }
        else if (c == '\\') { out += "\\\\"; }
        else if (c == '\n') { out += "\\n";  }
        else if (c == '\r') { out += "\\r";  }
        else if (c == '\t') { out += "\\t";  }
        else { out += c; }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Single synthesis helper — shared by one-shot and server modes
// ---------------------------------------------------------------------------

static int do_synthesize(qwen3_tts::Qwen3TTS & tts,
                          const std::string & text,
                          const std::string & output_file,
                          const std::string & reference_audio,
                          const std::string & ref_text,
                          const std::string & embedding_in,
                          const std::string & embedding_out_path,
                          qwen3_tts::tts_params params,
                          bool quiet) {

    // Propagate ICL mode
    if (!ref_text.empty()) {
        params.ref_text = ref_text;
        params.icl_mode = true;
    }

    if (!quiet) {
        tts.set_progress_callback([](int tok, int max) {
            fprintf(stderr, "\rGenerating: %d/%d tokens", tok, max);
        });
    }

    qwen3_tts::tts_result result;

    if (!embedding_in.empty()) {
        std::vector<float> embedding;
        if (!tts.load_speaker_embedding(embedding_in, embedding)) {
            fprintf(stderr, "Error: failed to load embedding from %s\n", embedding_in.c_str());
            return 1;
        }
        if (!quiet)
            fprintf(stderr, "Synthesizing with loaded embedding (%zu floats)\n", embedding.size());
        result = tts.synthesize_with_embedding(text, embedding, params);
    } else if (!reference_audio.empty()) {
        if (!quiet) {
            fprintf(stderr, "Synthesizing with voice clone: \"%s\"\n", text.c_str());
            fprintf(stderr, "Reference: %s%s\n", reference_audio.c_str(),
                    params.icl_mode ? " (ICL mode)" : " (x-vector only)");
        }
        params.reference_audio = reference_audio;
        result = tts.synthesize_with_voice(text, reference_audio, params);
        if (!embedding_out_path.empty()) {
            std::vector<float> embedding;
            if (tts.extract_speaker_embedding(reference_audio, embedding)) {
                if (!tts.save_speaker_embedding(embedding_out_path, embedding)) {
                    fprintf(stderr, "Warning: failed to save embedding to %s\n",
                            embedding_out_path.c_str());
                }
            }
        }
    } else {
        if (!quiet) {
            fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
            if (!params.speaker.empty())
                fprintf(stderr, "  Speaker: %s\n", params.speaker.c_str());
            if (!params.instruct.empty())
                fprintf(stderr, "  Instruct: %s\n", params.instruct.c_str());
            if (!params.language.empty())
                fprintf(stderr, "  Language: %s\n", params.language.c_str());
        }
        result = tts.synthesize(text, params);
    }

    if (!quiet) fprintf(stderr, "\n");

    if (!result.success) {
        fprintf(stderr, "Error: %s\n", result.error_msg.c_str());
        return 1;
    }

    if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
        fprintf(stderr, "Error: failed to save %s\n", output_file.c_str());
        return 1;
    }

    if (!quiet) {
        fprintf(stderr, "Saved: %s (%.2f s)\n",
                output_file.c_str(),
                (float)result.audio.size() / (float)result.sample_rate);
    }

    if (!quiet && params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Tokenize:  %6lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Encode:    %6lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Generate:  %6lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Decode:    %6lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:     %6lld ms\n", (long long)result.t_total_ms);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Server mode
// ---------------------------------------------------------------------------

// Run the server loop: read JSON lines from stdin, write JSON lines to stdout.
// The model is already loaded in `tts`. Each request is processed serially.
// This keeps the model in memory across all requests — no per-request reload.
//
// Protocol:
//   Input  (stdin):  one JSON object per line
//   Output (stdout): one JSON object per line (flushed immediately)
//   Logging:         stderr (does not interfere with stdout JSON protocol)
//
// Request fields:
//   text            (required) text to synthesize
//   output          (required) output WAV file path
//   reference       (optional) reference WAV for voice cloning
//   ref_text        (optional) reference transcript (enables ICL)
//   embedding_in    (optional) path to pre-computed speaker embedding
//   speaker         (optional) named speaker (CustomVoice)
//   instruct        (optional) style instruction (VoiceDesign/CustomVoice)
//   language        (optional) language name or codec ID
//   temperature     (optional) float, default 0.9
//   top_k           (optional) int, default 50
//   top_p           (optional) float, default 1.0
//   repetition_penalty (optional) float, default 1.05
//   max_tokens      (optional) int, default 4096
//
// Response fields:
//   success         true / false
//   output          (on success) output WAV file path
//   duration_s      (on success) audio duration in seconds
//   t_total_ms      (on success) total synthesis time in ms
//   error           (on failure) error message

static int run_server(qwen3_tts::Qwen3TTS & tts, const qwen3_tts::tts_params & default_params) {
    fprintf(stderr, "Server mode: ready. Reading JSON requests from stdin.\n");
    fprintf(stderr, "Send EOF (Ctrl+D on Unix, Ctrl+Z on Windows) to exit.\n\n");

    std::string line;
    int request_num = 0;

    while (std::getline(std::cin, line)) {
        // Skip blank lines and comment-like lines
        if (line.empty() || line[0] == '#') continue;

        ++request_num;
        fprintf(stderr, "[server] Request %d: processing...\n", request_num);

        // Parse request fields
        std::string text          = json_get_str(line, "text");
        std::string output_file   = json_get_str(line, "output");
        std::string reference     = json_get_str(line, "reference");
        std::string ref_text      = json_get_str(line, "ref_text");
        std::string embedding_in  = json_get_str(line, "embedding_in");

        if (text.empty()) {
            fprintf(stderr, "[server] Request %d: missing 'text' field\n", request_num);
            printf("{\"success\":false,\"error\":\"missing required field: text\"}\n");
            fflush(stdout);
            continue;
        }
        if (output_file.empty()) {
            // Default output name based on request number
            char buf[64];
            snprintf(buf, sizeof(buf), "server_output_%04d.wav", request_num);
            output_file = buf;
        }

        // Build params from defaults + per-request overrides
        qwen3_tts::tts_params params = default_params;
        params.print_timing   = false;
        params.print_progress = false;

        std::string speaker  = json_get_str(line, "speaker");
        std::string instruct = json_get_str(line, "instruct");
        std::string language = json_get_str(line, "language");
        if (!speaker.empty())  params.speaker  = speaker;
        if (!instruct.empty()) params.instruct = instruct;
        if (!language.empty()) params.language = language;

        float temp = json_get_float(line, "temperature", -999.0f);
        if (temp > -998.0f) params.temperature = temp;
        int top_k = json_get_int(line, "top_k", -999);
        if (top_k > -998) params.top_k = top_k;
        float top_p = json_get_float(line, "top_p", -999.0f);
        if (top_p > -998.0f) params.top_p = top_p;
        float rep_pen = json_get_float(line, "repetition_penalty", -999.0f);
        if (rep_pen > -998.0f) params.repetition_penalty = rep_pen;
        int max_tok = json_get_int(line, "max_tokens", -999);
        if (max_tok > -998) params.max_audio_tokens = max_tok;

        // Synthesize
        if (!ref_text.empty()) {
            params.ref_text = ref_text;
            params.icl_mode = true;
        }

        qwen3_tts::tts_result result;
        if (!embedding_in.empty()) {
            std::vector<float> emb;
            if (!tts.load_speaker_embedding(embedding_in, emb)) {
                std::string err = "Failed to load embedding: " + tts.get_error();
                fprintf(stderr, "[server] Request %d: %s\n", request_num, err.c_str());
                printf("{\"success\":false,\"error\":\"%s\"}\n", json_escape(err).c_str());
                fflush(stdout);
                continue;
            }
            result = tts.synthesize_with_embedding(text, emb, params);
        } else if (!reference.empty()) {
            params.reference_audio = reference;
            result = tts.synthesize_with_voice(text, reference, params);
        } else {
            result = tts.synthesize(text, params);
        }

        if (!result.success) {
            fprintf(stderr, "[server] Request %d: synthesis failed: %s\n",
                    request_num, result.error_msg.c_str());
            printf("{\"success\":false,\"error\":\"%s\"}\n",
                   json_escape(result.error_msg).c_str());
            fflush(stdout);
            continue;
        }

        // Save WAV
        if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
            std::string err = "Failed to save WAV: " + output_file;
            fprintf(stderr, "[server] Request %d: %s\n", request_num, err.c_str());
            printf("{\"success\":false,\"error\":\"%s\"}\n", json_escape(err).c_str());
            fflush(stdout);
            continue;
        }

        double duration_s = result.sample_rate > 0
            ? (double)result.audio.size() / (double)result.sample_rate : 0.0;

        fprintf(stderr, "[server] Request %d: done — %.2fs audio in %lldms → %s\n",
                request_num, duration_s, (long long)result.t_total_ms, output_file.c_str());

        printf("{\"success\":true,\"output\":\"%s\",\"duration_s\":%.3f,\"t_total_ms\":%lld}\n",
               json_escape(output_file).c_str(),
               duration_s,
               (long long)result.t_total_ms);
        fflush(stdout);
    }

    fprintf(stderr, "[server] EOF received — processed %d request(s). Exiting.\n", request_num);
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::string model_dir;
    std::string text;
    std::string output_file = "output.wav";
    std::string reference_audio;
    std::string ref_text;
    std::string embedding_in;
    std::string embedding_out;
    std::string gen_config_path;
    bool list_speakers  = false;
    bool list_languages = false;
    bool server_mode    = false;

    qwen3_tts::tts_params params;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

#define NEXT_ARG(dest)   do { if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); return 1; } dest = argv[i]; } while(0)
#define NEXT_FLOAT(dest) do { if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); return 1; } dest = std::stof(argv[i]); } while(0)
#define NEXT_INT(dest)   do { if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); return 1; } dest = std::stoi(argv[i]); } while(0)

        if      (arg == "-h" || arg == "--help")            { print_usage(argv[0]); return 0; }
        else if (arg == "-m" || arg == "--model")           { NEXT_ARG(model_dir); }
        else if (arg == "-t" || arg == "--text")            { NEXT_ARG(text); }
        else if (arg == "-o" || arg == "--output")          { NEXT_ARG(output_file); }
        else if (arg == "-r" || arg == "--reference")       { NEXT_ARG(reference_audio); }
        else if (arg == "--ref-text")                       { NEXT_ARG(ref_text); }
        else if (arg == "--embedding-in")                   { NEXT_ARG(embedding_in); }
        else if (arg == "--embedding-out")                  { NEXT_ARG(embedding_out); }
        else if (arg == "--speaker")                        { NEXT_ARG(params.speaker); }
        else if (arg == "--instruct")                       { NEXT_ARG(params.instruct); }
        else if (arg == "-l" || arg == "--language")        { NEXT_ARG(params.language); }
        else if (arg == "--temperature")                    { NEXT_FLOAT(params.temperature); }
        else if (arg == "--top-k")                          { NEXT_INT(params.top_k); }
        else if (arg == "--top-p")                          { NEXT_FLOAT(params.top_p); }
        else if (arg == "--repetition-penalty")             { NEXT_FLOAT(params.repetition_penalty); }
        else if (arg == "--sub-temperature")                { NEXT_FLOAT(params.subtalker_temperature); }
        else if (arg == "--sub-top-k")                      { NEXT_INT(params.subtalker_top_k); }
        else if (arg == "--max-tokens")                     { NEXT_INT(params.max_audio_tokens); }
        else if (arg == "--non-streaming")                  { params.non_streaming_mode = true; }
        else if (arg == "-j" || arg == "--threads")         { NEXT_INT(params.n_threads); }
        else if (arg == "--gen-config")                     { NEXT_ARG(gen_config_path); }
        else if (arg == "--list-speakers")                  { list_speakers  = true; }
        else if (arg == "--list-languages")                 { list_languages = true; }
        else if (arg == "--server")                         { server_mode = true; }
        else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }

#undef NEXT_ARG
#undef NEXT_FLOAT
#undef NEXT_INT
    }

    if (model_dir.empty()) {
        fprintf(stderr, "Error: model directory required (-m)\n");
        print_usage(argv[0]);
        return 1;
    }

    // ---- Load models (once) --------------------------------------------
    qwen3_tts::Qwen3TTS tts;
    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    if (!tts.load_models(model_dir)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }

    if (!gen_config_path.empty()) {
        tts.load_generation_config(gen_config_path);
    }

    // ---- Info queries --------------------------------------------------
    if (list_speakers) {
        auto spk = tts.get_supported_speakers();
        if (spk.empty()) {
            fprintf(stdout, "No named speakers available (model type: %s)\n",
                    tts.get_model_type().c_str());
        } else {
            fprintf(stdout, "Supported speakers (%zu):\n", spk.size());
            for (const auto & s : spk) fprintf(stdout, "  %s\n", s.c_str());
        }
        return 0;
    }
    if (list_languages) {
        auto langs = tts.get_supported_languages();
        fprintf(stdout, "Supported languages (%zu):\n", langs.size());
        for (const auto & l : langs) fprintf(stdout, "  %s\n", l.c_str());
        return 0;
    }

    // ---- Server mode ---------------------------------------------------
    if (server_mode) {
        return run_server(tts, params);
    }

    // ---- Embedding save-only mode (no synthesis) -----------------------
    if (!embedding_out.empty() && !reference_audio.empty() && text.empty()) {
        fprintf(stderr, "Extracting speaker embedding from: %s\n", reference_audio.c_str());
        std::vector<float> embedding;
        if (!tts.extract_speaker_embedding(reference_audio, embedding)) {
            fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
            return 1;
        }
        if (!tts.save_speaker_embedding(embedding_out, embedding)) {
            fprintf(stderr, "Error: failed to save embedding to %s\n", embedding_out.c_str());
            return 1;
        }
        fprintf(stderr, "Embedding saved to: %s (%zu floats)\n",
                embedding_out.c_str(), embedding.size());
        return 0;
    }

    if (text.empty()) {
        fprintf(stderr, "Error: text required (-t)\n");
        print_usage(argv[0]);
        return 1;
    }

    // ---- One-shot synthesis --------------------------------------------
    tts.set_progress_callback([](int tok, int max) {
        fprintf(stderr, "\rGenerating: %d/%d tokens", tok, max);
    });

    return do_synthesize(tts, text, output_file, reference_audio, ref_text,
                         embedding_in, embedding_out, params, /*quiet=*/false);
}

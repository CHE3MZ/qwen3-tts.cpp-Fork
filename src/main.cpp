#include "qwen3_tts.h"

#include <cstdio>
#include <cstring>
#include <string>

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
    fprintf(stderr, "\nMisc:\n");
    fprintf(stderr, "  -j, --threads <n>             Threads (default: 4)\n");
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
}

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

    qwen3_tts::tts_params params;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

#define NEXT_ARG(dest)  do { if (++i >= argc) { fprintf(stderr, "Missing value for %s\n", arg.c_str()); return 1; } dest = argv[i]; } while(0)
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

    // ---- Load models ---------------------------------------------------
    qwen3_tts::Qwen3TTS tts;
    fprintf(stderr, "Loading models from: %s\n", model_dir.c_str());
    if (!tts.load_models(model_dir)) {
        fprintf(stderr, "Error: %s\n", tts.get_error().c_str());
        return 1;
    }

    // Optionally load a custom generation config
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

    // Propagate ref_text for ICL mode
    if (!ref_text.empty()) {
        params.ref_text  = ref_text;
        params.icl_mode  = true;
    }

    // Set progress callback
    tts.set_progress_callback([](int tok, int max) {
        fprintf(stderr, "\rGenerating: %d/%d tokens", tok, max);
    });

    // ---- Synthesize ----------------------------------------------------
    qwen3_tts::tts_result result;

    if (!embedding_in.empty()) {
        // Load pre-computed embedding
        std::vector<float> embedding;
        if (!tts.load_speaker_embedding(embedding_in, embedding)) {
            fprintf(stderr, "Error: failed to load embedding from %s\n", embedding_in.c_str());
            return 1;
        }
        fprintf(stderr, "Synthesizing with loaded embedding (%zu floats)\n", embedding.size());
        result = tts.synthesize_with_embedding(text, embedding, params);
    } else if (!reference_audio.empty()) {
        fprintf(stderr, "Synthesizing with voice clone: \"%s\"\n", text.c_str());
        fprintf(stderr, "Reference: %s%s\n", reference_audio.c_str(),
                params.icl_mode ? " (ICL mode)" : " (x-vector only)");
        params.reference_audio = reference_audio;
        result = tts.synthesize_with_voice(text, reference_audio, params);
        // Save embedding if requested
        if (!embedding_out.empty()) {
            std::vector<float> embedding;
            if (tts.extract_speaker_embedding(reference_audio, embedding)) {
                if (!tts.save_speaker_embedding(embedding_out, embedding)) {
                    fprintf(stderr, "Warning: failed to save embedding to %s\n",
                            embedding_out.c_str());
                }
            }
        }
    } else {
        fprintf(stderr, "Synthesizing: \"%s\"\n", text.c_str());
        if (!params.speaker.empty())
            fprintf(stderr, "  Speaker: %s\n", params.speaker.c_str());
        if (!params.instruct.empty())
            fprintf(stderr, "  Instruct: %s\n", params.instruct.c_str());
        if (!params.language.empty())
            fprintf(stderr, "  Language: %s\n", params.language.c_str());
        result = tts.synthesize(text, params);
    }

    fprintf(stderr, "\n");

    if (!result.success) {
        fprintf(stderr, "Error: %s\n", result.error_msg.c_str());
        return 1;
    }

    if (!qwen3_tts::save_audio_file(output_file, result.audio, result.sample_rate)) {
        fprintf(stderr, "Error: failed to save %s\n", output_file.c_str());
        return 1;
    }
    fprintf(stderr, "Saved: %s (%.2f s)\n",
            output_file.c_str(), (float)result.audio.size() / result.sample_rate);

    if (params.print_timing) {
        fprintf(stderr, "\nTiming:\n");
        fprintf(stderr, "  Tokenize:  %6lld ms\n", (long long)result.t_tokenize_ms);
        fprintf(stderr, "  Encode:    %6lld ms\n", (long long)result.t_encode_ms);
        fprintf(stderr, "  Generate:  %6lld ms\n", (long long)result.t_generate_ms);
        fprintf(stderr, "  Decode:    %6lld ms\n", (long long)result.t_decode_ms);
        fprintf(stderr, "  Total:     %6lld ms\n", (long long)result.t_total_ms);
    }
    return 0;
}

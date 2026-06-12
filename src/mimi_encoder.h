#pragma once
// Mimi audio encoder: waveform (24 kHz) -> discrete codec codes [n_frames, 16]
// Architecture (Kyutai Mimi, as used in Qwen3-TTS-Tokenizer-12Hz):
//   SEANet Conv Encoder (15 conv/resnet layers) ->
//   8-layer sliding-window Transformer ->
//   Downsample Conv (25 Hz -> 12.5 Hz) ->
//   Split-RVQ quantizer (1 semantic + 15 acoustic codebooks, each 2048 entries, 256-dim)

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <string>
#include <map>
#include <vector>
#include <memory>

namespace qwen3_tts {

// ============================================================
// Configuration (matches encoder_config in tokenizer JSON)
// ============================================================
struct mimi_encoder_config {
    int32_t sample_rate        = 24000;
    int32_t audio_channels     = 1;
    int32_t hidden_size        = 512;
    int32_t num_filters        = 64;
    int32_t num_residual_layers = 1;
    // Encoder uses REVERSED upsampling_ratios as downsampling strides
    // upsampling_ratios = [8, 6, 5, 4] -> encoder strides = [4, 5, 6, 8]
    int32_t upsampling_ratios[4] = {8, 6, 5, 4};  // stored in decoder order
    int32_t kernel_size        = 7;
    int32_t last_kernel_size   = 3;
    int32_t residual_kernel_size = 3;
    int32_t dilation_growth_rate = 2;
    int32_t compress           = 2;

    // Transformer
    int32_t num_hidden_layers  = 8;
    int32_t num_attention_heads = 8;
    int32_t num_key_value_heads = 8;
    int32_t head_dim           = 64;
    int32_t intermediate_size  = 2048;
    float   norm_eps           = 1e-5f;
    float   rope_theta         = 10000.0f;
    int32_t sliding_window     = 250;
    float   layer_scale_init   = 0.01f;

    // Quantizer
    int32_t num_quantizers     = 32;    // total in the model
    int32_t num_valid_quantizers = 16;  // how many we output (encoder_valid_num_quantizers)
    int32_t num_semantic_quantizers = 1;
    int32_t codebook_size      = 2048;
    int32_t codebook_dim       = 256;
    int32_t vq_hidden_dim      = 256;   // vector_quantization_hidden_dimension
};

// ============================================================
// Weight tensors
// ============================================================

// A single conv layer in the SEANet encoder
struct mimi_conv_layer {
    struct ggml_tensor * w = nullptr;  // weight
    struct ggml_tensor * b = nullptr;  // bias (may be nullptr)
    int32_t stride    = 1;
    int32_t dilation  = 1;
    int32_t padding_left = 0;         // pre-computed causal padding
};

// A residual block: ELU -> Conv(k=3,dil) -> ELU -> Conv(k=1)
struct mimi_resnet_block {
    mimi_conv_layer conv1;  // compressed conv
    mimi_conv_layer conv2;  // pointwise
};

// One transformer layer (cached float weights for fast inference)
struct mimi_tfm_layer {
    // Attention
    struct ggml_tensor * attn_in_norm_w  = nullptr;
    struct ggml_tensor * attn_in_norm_b  = nullptr;
    struct ggml_tensor * attn_q_w        = nullptr;
    struct ggml_tensor * attn_k_w        = nullptr;
    struct ggml_tensor * attn_v_w        = nullptr;
    struct ggml_tensor * attn_o_w        = nullptr;
    struct ggml_tensor * attn_layer_scale = nullptr;
    // FFN
    struct ggml_tensor * ffn_in_norm_w   = nullptr;
    struct ggml_tensor * ffn_in_norm_b   = nullptr;
    struct ggml_tensor * ffn_fc1_w       = nullptr;
    struct ggml_tensor * ffn_fc2_w       = nullptr;
    struct ggml_tensor * ffn_layer_scale = nullptr;

    // Pre-converted float caches (populated after load for fast CPU inference)
    std::vector<float> attn_in_norm_w_f, attn_in_norm_b_f;
    std::vector<float> attn_q_w_f, attn_k_w_f, attn_v_w_f, attn_o_w_f;
    std::vector<float> attn_scale_f;
    std::vector<float> ffn_in_norm_w_f, ffn_in_norm_b_f;
    std::vector<float> ffn_fc1_w_f, ffn_fc2_w_f;
    std::vector<float> ffn_scale_f;
};

// One RVQ codebook layer
struct mimi_codebook {
    struct ggml_tensor * embed_sum     = nullptr;  // [codebook_size, codebook_dim]
    struct ggml_tensor * cluster_usage = nullptr;  // [codebook_size]
    // Normalized embedding (computed at load time: embed_sum / cluster_usage)
    // stored in host memory as float
    std::vector<float> embed;  // [codebook_size * codebook_dim]
};

// Full encoder model weights
struct mimi_encoder_model {
    mimi_encoder_config config;

    // SEANet convolutional encoder layers
    // Layer 0: initial conv (1 -> num_filters, k=7)
    mimi_conv_layer enc_conv0;

    // For each downsampling ratio (reversed): ResnetBlocks + downsample conv
    // With upsampling_ratios=[8,6,5,4], encoder does [4,5,6,8] strides
    // Structure per "stage" i (0..3):
    //   resnet_block[i][0..num_residual_layers-1]
    //   downsample_conv[i]
    std::vector<mimi_resnet_block> enc_resnet;  // 4 blocks (one per stage)
    std::vector<mimi_conv_layer>   enc_down;    // 4 downsample convs
    // Final conv (hidden_size*8 -> hidden_size, k=last_kernel_size)
    mimi_conv_layer enc_final;

    // Transformer encoder (8 layers)
    std::vector<mimi_tfm_layer> tfm_layers;

    // Downsample conv (hidden_size -> hidden_size, k=4, stride=2)
    mimi_conv_layer downsample;

    // RVQ quantizer codebooks
    // semantic: 1 codebook (index 0)
    // acoustic: 15 codebooks (indices 1..15)
    struct ggml_tensor * semantic_input_proj  = nullptr;  // [vq_hidden, hidden, 1]
    std::vector<mimi_codebook> semantic_cbs;  // 1 codebook
    struct ggml_tensor * acoustic_input_proj  = nullptr;  // [vq_hidden, hidden, 1]
    std::vector<mimi_codebook> acoustic_cbs;  // 15 codebooks

    // GGML context and backend buffer
    struct ggml_context * ctx    = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// ============================================================
// MimiEncoder class
// ============================================================
class MimiEncoder {
public:
    MimiEncoder();
    ~MimiEncoder();

    // Load encoder weights from the tokenizer GGUF file
    bool load_model(const std::string & model_path);

    void unload_model();

    // Encode raw PCM audio to discrete codec codes.
    // samples: float32, 24 kHz mono, normalized to [-1, 1]
    // n_samples: number of samples
    // codes_out: output codes [n_frames * n_valid_quantizers] row-major
    //            codes_out[f * n_valid_quantizers + q] = code for frame f, quantizer q
    // n_frames_out: number of output frames
    // Returns true on success.
    bool encode(const float * samples, int32_t n_samples,
                std::vector<int32_t> & codes_out, int32_t & n_frames_out);

    const mimi_encoder_config & get_config() const { return model_.config; }
    const std::string & get_error() const { return error_msg_; }
    bool is_loaded() const { return model_.ctx != nullptr; }

private:
    // Step 1: run SEANet conv encoder on waveform -> [hidden_size, T_enc]
    bool run_conv_encoder(const float * samples, int32_t n_samples,
                          std::vector<float> & hidden_out, int32_t & t_out);

    // Step 2: run transformer on [T, hidden] -> [T, hidden]
    bool run_transformer(const std::vector<float> & hidden_in, int32_t n_frames,
                         std::vector<float> & hidden_out);

    // Step 3: downsample conv -> halve temporal resolution
    bool run_downsample(const std::vector<float> & hidden_in, int32_t n_frames,
                        std::vector<float> & hidden_out, int32_t & n_frames_out);

    // Step 4: RVQ quantize -> codes
    bool run_rvq(const std::vector<float> & latent, int32_t n_frames,
                 std::vector<int32_t> & codes_out);

    // Helpers
    static void apply_causal_conv1d(const float * input, int32_t in_len,
                                    int32_t in_ch, int32_t out_ch,
                                    const float * w, const float * b,
                                    int32_t kernel, int32_t stride, int32_t dilation,
                                    int32_t padding_left,
                                    std::vector<float> & output, int32_t & out_len);

    static void apply_elu(std::vector<float> & x);

    static void apply_layer_norm(float * data, int32_t len, int32_t dim,
                                  const float * w, const float * b, float eps);

    mimi_encoder_model model_;
    std::string error_msg_;
};

void free_mimi_encoder_model(mimi_encoder_model & model);

} // namespace qwen3_tts

# Qwen3-TTS Tensor Mapping Documentation

This document describes the tensor name mapping from HuggingFace format to GGML/GGUF format
for the Qwen3-TTS model conversion.

## Model Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Qwen3-TTS-12Hz-0.6B-Base                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────────────────────────────────┐    │
│  │ Speaker Encoder │    │              Talker (Main TTS)              │    │
│  │   (ECAPA-TDNN)  │    │  ┌─────────────────────────────────────┐    │    │
│  │                 │    │  │     28-Layer Transformer            │    │    │
│  │  Audio → 1024d  │───▶│  │  (Qwen3-style with RoPE/GQA)       │    │    │
│  │  speaker embed  │    │  │                                     │    │    │
│  └─────────────────┘    │  │  Hidden: 1024, Heads: 16, KV: 8    │    │    │
│                         │  │  Intermediate: 3072                 │    │    │
│                         │  └─────────────────────────────────────┘    │    │
│                         │                    │                        │    │
│                         │                    ▼                        │    │
│                         │  ┌─────────────────────────────────────┐    │    │
│                         │  │         Code Predictor              │    │    │
│                         │  │    (5-Layer Delay Transformer)      │    │    │
│                         │  │                                     │    │    │
│                         │  │  16 codebook embeddings/heads       │    │    │
│                         │  │  Vocab: 2048 per codebook           │    │    │
│                         │  └─────────────────────────────────────┘    │    │
│                         └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      Qwen3-TTS-Tokenizer-12Hz                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────┐    │
│  │         Encoder                 │    │         Decoder             │    │
│  │    (Audio → Codes)              │    │    (Codes → Audio)          │    │
│  │                                 │    │                             │    │
│  │  Conv1D + 8-layer Transformer   │    │  8-layer Transformer        │    │
│  │  + RVQ (32 quantizers)          │    │  + Upsampling ConvNet       │    │
│  │                                 │    │                             │    │
│  │  16 valid quantizers output     │    │  Upsample: 8×5×4×3 = 480   │    │
│  │  Codebook: 2048 entries         │    │  Vocoder output             │    │
│  └─────────────────────────────────┘    └─────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Architecture Parameters

### TTS Base Model (Talker)
| Parameter | 0.6B | 1.7B |
|-----------|------|------|
| Hidden Size | 1024 | **2048** |
| Intermediate Size | 3072 | **6144** |
| Num Hidden Layers | 28 | 28 |
| Num Attention Heads | 16 | 16 |
| Num KV Heads | 8 (GQA) | 8 (GQA) |
| Head Dim | 128 | 128 |
| Vocab Size (codec) | 3072 | 3072 |
| Num Code Groups | 16 | 16 |
| RMS Norm Eps | 1e-06 | 1e-06 |
| RoPE Theta | 1000000 | 1000000 |
| Activation | SiLU | SiLU |

### Code Predictor (Delay Pattern)
| Parameter | 0.6B / 1.7B |
|-----------|-------------|
| Hidden Size | 1024 (same for both sizes) |
| Intermediate Size | 3072 (same for both sizes) |
| Num Hidden Layers | 5 |
| Num Attention Heads | 16 |
| Num KV Heads | 8 |
| Vocab Size | 2048 |
| Num Code Groups | 16 |

### Tokenizer Encoder
| Parameter | Value |
|-----------|-------|
| Frame Rate | 12.5 Hz |
| Hidden Size | 512 |
| Num Hidden Layers | 8 |
| Num Attention Heads | 8 |
| Codebook Size | 2048 |
| Num Quantizers | 32 (16 valid) |
| Sample Rate | 24000 Hz |

### Tokenizer Decoder (Vocoder)
| Parameter | Value |
|-----------|-------|
| Latent Dim | 1024 |
| Decoder Dim | 1536 |
| Hidden Size | 512 |
| Num Hidden Layers | 8 |
| Num Attention Heads | 16 |
| Codebook Size | 2048 |
| Upsample Rates | [8, 5, 4, 3] |

## HuggingFace → GGML Tensor Name Mapping

### Speaker Encoder (ECAPA-TDNN)
```
HuggingFace Name                                    → GGML Name
─────────────────────────────────────────────────────────────────────────────
speaker_encoder.blocks.0.conv.weight                → spk_enc.conv0.weight
speaker_encoder.blocks.0.conv.bias                  → spk_enc.conv0.bias
speaker_encoder.blocks.{n}.res2net_block.blocks.{m}.conv.weight
                                                    → spk_enc.blk.{n}.res2net.{m}.weight
speaker_encoder.blocks.{n}.se_block.conv1.weight    → spk_enc.blk.{n}.se.conv1.weight
speaker_encoder.blocks.{n}.se_block.conv2.weight    → spk_enc.blk.{n}.se.conv2.weight
speaker_encoder.blocks.{n}.tdnn1.conv.weight        → spk_enc.blk.{n}.tdnn1.weight
speaker_encoder.blocks.{n}.tdnn2.conv.weight        → spk_enc.blk.{n}.tdnn2.weight
speaker_encoder.asp.conv.weight                     → spk_enc.asp.conv.weight
speaker_encoder.asp.tdnn.conv.weight                → spk_enc.asp.tdnn.weight
speaker_encoder.mfa.conv.weight                     → spk_enc.mfa.weight
speaker_encoder.fc.weight                           → spk_enc.fc.weight
```

### Talker (Main TTS Transformer)
```
HuggingFace Name                                    → GGML Name
─────────────────────────────────────────────────────────────────────────────
talker.model.codec_embedding.weight                 → talker.codec_embd.weight
talker.codec_head.weight                            → talker.codec_head.weight
talker.model.norm.weight                            → talker.output_norm.weight

# Per-layer tensors (28 layers)
talker.model.layers.{n}.input_layernorm.weight      → talker.blk.{n}.attn_norm.weight
talker.model.layers.{n}.self_attn.q_proj.weight     → talker.blk.{n}.attn_q.weight
talker.model.layers.{n}.self_attn.k_proj.weight     → talker.blk.{n}.attn_k.weight
talker.model.layers.{n}.self_attn.v_proj.weight     → talker.blk.{n}.attn_v.weight
talker.model.layers.{n}.self_attn.o_proj.weight     → talker.blk.{n}.attn_output.weight
talker.model.layers.{n}.self_attn.q_norm.weight     → talker.blk.{n}.attn_q_norm.weight
talker.model.layers.{n}.self_attn.k_norm.weight     → talker.blk.{n}.attn_k_norm.weight
talker.model.layers.{n}.post_attention_layernorm.weight → talker.blk.{n}.ffn_norm.weight
talker.model.layers.{n}.mlp.gate_proj.weight        → talker.blk.{n}.ffn_gate.weight
talker.model.layers.{n}.mlp.up_proj.weight          → talker.blk.{n}.ffn_up.weight
talker.model.layers.{n}.mlp.down_proj.weight        → talker.blk.{n}.ffn_down.weight
```

### Code Predictor (Delay Pattern Transformer)
```
HuggingFace Name                                    → GGML Name
─────────────────────────────────────────────────────────────────────────────
# 16 codebook embeddings (indices 0-14, skipping first codebook)
talker.code_predictor.model.codec_embedding.{c}.weight
                                                    → code_pred.codec_embd.{c}.weight

# 16 LM heads for each codebook
talker.code_predictor.lm_head.{c}.weight            → code_pred.lm_head.{c}.weight

talker.code_predictor.model.norm.weight             → code_pred.output_norm.weight

# Projection (1.7B only: talker hidden=2048 → code pred hidden=1024)
talker.code_predictor.small_to_mtp_projection.weight
                                                     → code_pred.proj.weight
talker.code_predictor.small_to_mtp_projection.bias
                                                     → code_pred.proj.bias

# Per-layer tensors (5 layers)
talker.code_predictor.model.layers.{n}.input_layernorm.weight
                                                    → code_pred.blk.{n}.attn_norm.weight
talker.code_predictor.model.layers.{n}.self_attn.q_proj.weight
                                                    → code_pred.blk.{n}.attn_q.weight
talker.code_predictor.model.layers.{n}.self_attn.k_proj.weight
                                                    → code_pred.blk.{n}.attn_k.weight
talker.code_predictor.model.layers.{n}.self_attn.v_proj.weight
                                                    → code_pred.blk.{n}.attn_v.weight
talker.code_predictor.model.layers.{n}.self_attn.o_proj.weight
                                                    → code_pred.blk.{n}.attn_output.weight
talker.code_predictor.model.layers.{n}.self_attn.q_norm.weight
                                                    → code_pred.blk.{n}.attn_q_norm.weight
talker.code_predictor.model.layers.{n}.self_attn.k_norm.weight
                                                    → code_pred.blk.{n}.attn_k_norm.weight
talker.code_predictor.model.layers.{n}.post_attention_layernorm.weight
                                                    → code_pred.blk.{n}.ffn_norm.weight
talker.code_predictor.model.layers.{n}.mlp.gate_proj.weight
                                                    → code_pred.blk.{n}.ffn_gate.weight
talker.code_predictor.model.layers.{n}.mlp.up_proj.weight
                                                    → code_pred.blk.{n}.ffn_up.weight
talker.code_predictor.model.layers.{n}.mlp.down_proj.weight
                                                    → code_pred.blk.{n}.ffn_down.weight
```

### Tokenizer Encoder (Audio → Codes)
The encoder uses `mimi_enc.*` tensors stored in the tokenizer GGUF.
Full mapping in `scripts/convert_tokenizer_to_gguf.py` `ENCODER_PATTERNS` (lines 84-117).

Notable namespaces:
- `mimi_enc.enc.*` — SEANet conv encoder layers and residual blocks
- `mimi_enc.tfm.*` — Encoder transformer (attention + FFN)
- `mimi_enc.rvq_sem.cb.*` — Semantic RVQ codebooks
- `mimi_enc.rvq_acou.cb.*` — Acoustic RVQ codebooks

### Tokenizer Decoder (Codes → Audio / Vocoder)
The decoder uses `tok_dec.*` tensors stored in the tokenizer GGUF.
Full mapping in `scripts/convert_tokenizer_to_gguf.py` `DECODER_PATTERNS` (lines 119-167).

Notable namespaces:
- `tok_dec.dec.{0-3}.snake.*` — Snake activation parameters
- `tok_dec.dec.{0-3}.conv_t.*` — Transposed conv weights
- `tok_dec.dec.{0-3}.res.*` — Residual block conv1/conv2 + snake act1/act2
- `tok_dec.pre_tfm.blk.*` — Pre-transformer decoder layers
- `tok_dec.vq_first.*` / `tok_dec.vq_rest.*` — Vocoder VQ codebooks
- `tok_dec.upsample.*` — Upsampling layers (conv, dwconv, pwconv, norm, gamma)
- `tok_dec.dec.{4-7}` — Output decoder blocks (conv_out)

## Tensor Count Summary

| Component | Tensor Count |
|-----------|-------------|
| TTS Base Main Model | 478 |
| TTS Base Speech Tokenizer | 1082 |
| Standalone Tokenizer | 1082 |

## Special Tokens (Codec IDs)

| Token | ID |
|-------|-----|
| codec_pad_id | 2148 |
| codec_bos_id | 2149 |
| codec_eos_token_id | 2150 |
| codec_think_id | 2154 |
| codec_nothink_id | 2155 |
| codec_think_bos_id | 2156 |
| codec_think_eos_id | 2157 |

### Language IDs
| Language | ID |
|----------|-----|
| english | 2050 |
| german | 2053 |
| spanish | 2054 |
| chinese | 2055 |
| japanese | 2058 |
| french | 2061 |
| korean | 2064 |
| russian | 2069 |
| italian | 2070 |
| portuguese | 2071 |

## RoPE Configuration

The Talker uses M-RoPE (Multi-dimensional RoPE) with interleaved positions:
- `mrope_section`: [24, 20, 20] - dimensions for time, frequency, and other
- `rope_theta`: 1000000
- `interleaved`: true

## Notes for GGUF Conversion

1. **Grouped Query Attention (GQA)**: Both Talker and Code Predictor use GQA with 16 attention heads and 8 KV heads (ratio 2:1).

2. **Q/K Norms**: Qwen3-style models apply RMSNorm to Q and K projections before attention.

3. **16 Codebooks**: The model uses 16 parallel codebooks for audio codes. Each codebook has:
   - Separate embedding layer (2048 × hidden_size: 1024 for 0.6B, 2048 for 1.7B)
   - Separate LM head (2048 × 1024, always code predictor hidden_size)

4. **Delay Pattern**: The Code Predictor implements a delay pattern for parallel codebook prediction.

5. **Speaker Encoder**: ECAPA-TDNN architecture with Res2Net blocks and SE (Squeeze-Excitation) modules.

6. **Tokenizer**: The tokenizer uses RVQ (Residual Vector Quantization) with 32 quantizers, but only 16 are valid for output.

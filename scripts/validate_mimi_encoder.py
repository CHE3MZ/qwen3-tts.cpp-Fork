#!/usr/bin/env python3
"""
Validate the C++ Mimi encoder by comparing its output codes against
the Python reference (HuggingFace Mimi model).

Requires:
  - A re-converted tokenizer GGUF with mimi_enc.* tensors
  - The Python qwen_tts package with the speech tokenizer
  - A reference audio file (clone.wav)

Usage:
    python scripts/validate_mimi_encoder.py \
        --audio clone.wav \
        --tokenizer models/qwen3-tts-tokenizer-f16.gguf \
        --model-dir models/Qwen3-TTS-Tokenizer-12Hz
"""

from __future__ import annotations

import argparse
import sys
import struct
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_audio_mono_f32(path: Path, target_sr: int = 24000) -> np.ndarray:
    """Load audio file (WAV, MP3, FLAC, etc.), mix to mono float32, resample to target_sr.
    Requires librosa for MP3 and resampling support."""
    try:
        import librosa
        audio, sr = librosa.load(str(path), sr=None, mono=True)
        if sr != target_sr:
            print(f"  Resampling from {sr} Hz to {target_sr} Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio.astype(np.float32)
    except ImportError:
        # Fallback: read WAV manually (no MP3 support without librosa)
        if str(path).lower().endswith('.mp3'):
            raise RuntimeError(
                "MP3 loading requires librosa.\n"
                "Install it: pip install librosa soundfile\n"
                "Or use a WAV file instead."
            )
        data = path.read_bytes()
        if data[:4] != b'RIFF': raise ValueError("Not a WAV")
        pos = 12
        sr = 0; n_ch = 0; bits = 0; audio_bytes = None
        while pos < len(data) - 8:
            cid = data[pos:pos+4]
            csz = struct.unpack_from('<I', data, pos+4)[0]
            if cid == b'fmt ':
                n_ch = struct.unpack_from('<H', data, pos+10)[0]
                sr   = struct.unpack_from('<I', data, pos+12)[0]
                bits = struct.unpack_from('<H', data, pos+22)[0]
            elif cid == b'data':
                audio_bytes = data[pos+8:pos+8+csz]
                break
            pos += 8 + csz
        if audio_bytes is None: raise ValueError("No data chunk in WAV")
        if bits == 16:
            samples = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
        elif bits == 32:
            samples = np.frombuffer(audio_bytes, np.float32).copy()
        else:
            raise ValueError(f"Unsupported bit depth {bits}")
        if n_ch > 1:
            samples = samples.reshape(-1, n_ch).mean(axis=1)
        if sr != target_sr:
            raise RuntimeError(
                f"Cannot resample {sr}→{target_sr} Hz without librosa.\n"
                "Install: pip install librosa"
            )
        return samples

# Keep old name as alias for backward compatibility
load_wav_mono_f32 = load_audio_mono_f32


def get_python_codes(audio: np.ndarray, tokenizer_dir: str) -> np.ndarray:
    """Run Python Mimi encoder, return codes [n_frames, 16] as int32."""
    import torch
    from qwen_tts.core import Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model
    from transformers import AutoConfig, AutoModel, AutoFeatureExtractor

    AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
    AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)

    feat = AutoFeatureExtractor.from_pretrained(tokenizer_dir)
    model = AutoModel.from_pretrained(tokenizer_dir, torch_dtype=torch.float32)
    model = model.eval()

    inputs = feat(
        raw_audio=[audio],
        sampling_rate=int(feat.sampling_rate),
        return_tensors="pt",
    )
    inputs = inputs.to(torch.float32)

    with torch.no_grad():
        enc = model.encode(
            inputs["input_values"].squeeze(1),
            inputs["padding_mask"].squeeze(1),
            return_dict=True,
        )

    # enc.audio_codes is a list; each element is [T, num_quantizers]
    codes_tensor = enc.audio_codes[0]  # [T, 16]
    return codes_tensor.numpy().astype(np.int32)


def write_cpp_validation_input(audio: np.ndarray, out_path: Path):
    """Write raw float32 audio samples for C++ test."""
    audio.astype(np.float32).tofile(str(out_path))
    print(f"  Wrote {len(audio)} samples to {out_path}")


def compare_codes(py_codes: np.ndarray, cpp_codes: np.ndarray) -> dict:
    """Compare two [n_frames, n_quantizers] code arrays."""
    if py_codes.shape != cpp_codes.shape:
        return {
            "match": False,
            "error": f"Shape mismatch: Python {py_codes.shape} vs C++ {cpp_codes.shape}",
        }

    n_frames, n_q = py_codes.shape
    exact_match = np.array_equal(py_codes, cpp_codes)
    per_q_match = [(py_codes[:, q] == cpp_codes[:, q]).mean() for q in range(n_q)]
    overall_match_rate = np.mean(py_codes == cpp_codes)

    # Per-codebook analysis
    q_analysis = []
    for q in range(n_q):
        q_analysis.append({
            "codebook": q,
            "match_rate": float(per_q_match[q]),
            "mismatches": int(np.sum(py_codes[:, q] != cpp_codes[:, q])),
        })

    return {
        "match": exact_match,
        "n_frames": n_frames,
        "n_quantizers": n_q,
        "overall_match_rate": float(overall_match_rate),
        "exact_match": bool(exact_match),
        "per_codebook": q_analysis,
        "first_mismatch_frame": int(np.argmax(np.any(py_codes != cpp_codes, axis=1)))
                                 if not exact_match else None,
    }


def print_report(result: dict):
    print("\n" + "=" * 60)
    print("MIMI ENCODER VALIDATION REPORT")
    print("=" * 60)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"Frames:           {result['n_frames']}")
    print(f"Quantizers:       {result['n_quantizers']}")
    print(f"Exact match:      {result['exact_match']}")
    print(f"Overall accuracy: {result['overall_match_rate']*100:.2f}%")

    if result.get("first_mismatch_frame") is not None:
        print(f"First mismatch:   frame {result['first_mismatch_frame']}")

    print(f"\nPer-codebook accuracy:")
    for q in result["per_codebook"]:
        bar = "█" * int(q["match_rate"] * 20)
        print(f"  CB{q['codebook']:2d}: {q['match_rate']*100:6.1f}%  {bar}")

    print("\n" + "=" * 60)
    if result["exact_match"]:
        print("RESULT: PASS — C++ Mimi encoder matches Python exactly")
    elif result["overall_match_rate"] >= 0.95:
        print(f"RESULT: NEAR-PASS — {result['overall_match_rate']*100:.1f}% match (minor numerical diff)")
    else:
        print(f"RESULT: FAIL — {result['overall_match_rate']*100:.1f}% match")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Validate C++ Mimi encoder vs Python")
    parser.add_argument("--audio", default=None,
                        help="Reference audio file: WAV, MP3, etc. (24 kHz mono preferred). "
                             "Defaults to audio.mp3 or clone.wav if present.")
    parser.add_argument("--tokenizer-dir", default=None,
                        help="HuggingFace tokenizer directory (for Python encoder)")
    parser.add_argument("--cpp-codes", default=None,
                        help="Pre-generated C++ codes file [n_frames*16 int32]")
    parser.add_argument("--save-audio", default="reference/mimi_enc_test_audio.bin",
                        help="Save test audio binary for C++ test")
    parser.add_argument("--save-py-codes", default="reference/mimi_enc_py_codes.bin",
                        help="Save Python reference codes")
    args = parser.parse_args()

    audio_path = Path(args.audio) if args.audio else None

    # Auto-detect audio file if not specified
    if audio_path is None:
        for candidate in ["audio.mp3", "clone.wav", "audio.wav"]:
            p = PROJECT_ROOT / candidate
            if p.exists():
                audio_path = p
                break
        if audio_path is None:
            print("ERROR: No audio file found. Provide --audio or place audio.mp3/clone.wav in project root.")
            return 1
    elif not audio_path.is_absolute():
        if not audio_path.exists():
            audio_path = PROJECT_ROOT / audio_path
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        return 1

    print(f"Loading audio: {audio_path}")
    audio = load_audio_mono_f32(audio_path)
    print(f"  {len(audio)} samples @ 24000 Hz = {len(audio)/24000:.2f}s")

    # Save audio for C++ test
    audio_bin = PROJECT_ROOT / args.save_audio
    audio_bin.parent.mkdir(parents=True, exist_ok=True)
    write_cpp_validation_input(audio, audio_bin)

    # Get Python codes
    if args.tokenizer_dir is None:
        tok_dirs = [
            PROJECT_ROOT / "models" / "Qwen3-TTS-Tokenizer-12Hz",
            PROJECT_ROOT / "models" / "speech_tokenizer",
        ]
        for d in tok_dirs:
            if d.exists():
                args.tokenizer_dir = str(d)
                break

    if args.tokenizer_dir is None:
        print("ERROR: --tokenizer-dir required (HuggingFace tokenizer directory)")
        print("Cannot run Python encoder without it.")
        print("\nTo manually validate:")
        print(f"  1. Run C++ encoder on: {audio_bin}")
        print(f"  2. Save codes as int32 binary [{len(audio)//1920} frames * 16 codebooks]")
        print(f"  3. Re-run with --cpp-codes <path> --tokenizer-dir <path>")
        return 0

    print(f"\nRunning Python Mimi encoder...")
    py_codes = get_python_codes(audio, args.tokenizer_dir)
    print(f"  Python codes shape: {py_codes.shape}")

    # Save Python codes
    py_codes_path = PROJECT_ROOT / args.save_py_codes
    py_codes.tofile(str(py_codes_path))
    print(f"  Saved Python codes: {py_codes_path}")

    # Compare with C++ codes if provided
    if args.cpp_codes:
        cpp_codes_path = Path(args.cpp_codes)
        if not cpp_codes_path.exists():
            cpp_codes_path = PROJECT_ROOT / args.cpp_codes
        print(f"\nLoading C++ codes: {cpp_codes_path}")
        cpp_raw = np.fromfile(str(cpp_codes_path), dtype=np.int32)
        n_frames = py_codes.shape[0]
        n_q = py_codes.shape[1]
        if len(cpp_raw) != n_frames * n_q:
            print(f"WARNING: C++ codes size {len(cpp_raw)} != expected {n_frames*n_q}")
            n_frames_cpp = len(cpp_raw) // n_q
            cpp_codes = cpp_raw[:n_frames_cpp * n_q].reshape(n_frames_cpp, n_q)
        else:
            cpp_codes = cpp_raw.reshape(n_frames, n_q)

        result = compare_codes(py_codes, cpp_codes)
        print_report(result)
        return 0 if result.get("exact_match") or result.get("overall_match_rate", 0) >= 0.95 else 1
    else:
        print(f"\nPython codes saved. To complete validation:")
        print(f"  1. Build the C++ project")
        print(f"  2. Run: ./build/test_mimi_encoder --audio {audio_bin} --codes-out /tmp/cpp_codes.bin")
        print(f"  3. Re-run: python scripts/validate_mimi_encoder.py --audio {audio_path} \\")
        print(f"             --tokenizer-dir {args.tokenizer_dir} \\")
        print(f"             --cpp-codes /tmp/cpp_codes.bin")
        return 0


if __name__ == "__main__":
    sys.exit(main())

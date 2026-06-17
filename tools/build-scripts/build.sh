#!/usr/bin/env bash
# ============================================================
#  qwen3-tts.cpp — macOS / Linux build script
#  Usage: ./build.sh [release|debug] [--metal] [--cuda] [--kv-f32]
#
#  Ninja is used automatically when installed (faster builds).
#  Falls back silently to Make if Ninja is not found.
# ============================================================
set -euo pipefail

BUILD_TYPE="Release"
METAL="OFF"
CUDA="OFF"
KV_F32="OFF"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

for arg in "$@"; do
    case "$arg" in
        debug|Debug)     BUILD_TYPE="Debug" ;;
        release|Release) BUILD_TYPE="Release" ;;
        --metal)         METAL="ON" ;;
        --cuda)          CUDA="ON" ;;
        --kv-f32)        KV_F32="ON" ;;
        *) echo "[warn] Unknown argument: $arg" ;;
    esac
done

# ---- Ninja auto-detect: prefer Ninja, fall back silently to Make ----
CMAKE_GEN=""
BUILD_DIR="build"
GEN_LABEL="Make (default)"
if command -v ninja &>/dev/null && cmake --help 2>/dev/null | grep -q "Ninja"; then
    CMAKE_GEN="-G Ninja"
    BUILD_DIR="build-ninja"
    GEN_LABEL="Ninja"
fi

# Navigate to repo root (two levels up from tools/build-scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Auto-enable Metal on macOS (Metal is always available on Apple Silicon/Intel Macs)
if [[ "$(uname)" == "Darwin" && "$METAL" == "OFF" ]]; then
    METAL="ON"
fi

echo ""
echo "============================================================"
echo " qwen3-tts.cpp build  [$BUILD_TYPE]"
echo " Metal: $METAL   CUDA: $CUDA   KV_F32: $KV_F32"
echo " Generator: $GEN_LABEL   Jobs: $JOBS"
echo " Output:    $REPO_ROOT/$BUILD_DIR/"
echo "============================================================"
echo ""

# ---- Step 1: Build GGML submodule --------------------------------
echo "[1/3] Building GGML..."
cmake -S ggml -B ggml/build \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGGML_METAL="$METAL" \
    -DGGML_CUDA="$CUDA" \
    ${CMAKE_GEN:+"$CMAKE_GEN"}
cmake --build ggml/build --config "$BUILD_TYPE" -j "$JOBS"
echo "      GGML built OK."

# ---- Step 2: Configure project -----------------------------------
echo ""
echo "[2/3] Configuring qwen3-tts.cpp..."
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DQWEN3_TTS_KV_F32="$KV_F32" \
    ${CMAKE_GEN:+"$CMAKE_GEN"}

# ---- Step 3: Build project ---------------------------------------
echo ""
echo "[3/3] Building qwen3-tts.cpp..."
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j "$JOBS"

echo ""
echo "============================================================"
echo " Build complete!"
echo " Output: $REPO_ROOT/$BUILD_DIR/"
echo ""
echo " Quick test:"
echo "   ./$BUILD_DIR/qwen3-tts-cli --help"
echo "   ./$BUILD_DIR/test_transformer --model models/qwen3-tts-0.6b-f16.gguf --ref-dir reference --max-len 32"
echo "============================================================"

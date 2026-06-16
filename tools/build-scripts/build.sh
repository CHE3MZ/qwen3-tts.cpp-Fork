#!/usr/bin/env bash
# ============================================================
#  qwen3-tts.cpp — macOS / Linux build script
#  Usage: ./build.sh [release|debug] [--metal] [--cuda] [--kv-f32]
# ============================================================
set -euo pipefail

BUILD_TYPE="Release"
METAL="OFF"
CUDA="OFF"
KV_F32="OFF"
BUILD_DIR="build"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

for arg in "$@"; do
    case "$arg" in
        debug|Debug)     BUILD_TYPE="Debug" ;;
        release|Release) BUILD_TYPE="Release" ;;
        --metal)         METAL="ON" ;;
        --cuda)          CUDA="ON" ;;
        --kv-f32)        KV_F32="ON" ;;
        --ninja)         CMAKE_GEN="-G Ninja"; BUILD_DIR="build-ninja" ;;
        *) echo "Unknown argument: $arg" ;;
    esac
done

# Navigate to repo root (two levels up from tools/build-scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo ""
echo "============================================================"
echo " qwen3-tts.cpp build  [$BUILD_TYPE]"
echo " Metal: $METAL   CUDA: $CUDA   KV_F32: $KV_F32"
echo " Jobs: $JOBS"
echo " Repo: $REPO_ROOT"
echo "============================================================"
echo ""

# Auto-enable Metal on Apple Silicon
if [[ "$(uname)" == "Darwin" ]]; then
    if [[ "$METAL" == "OFF" ]]; then
        echo "  [note] macOS detected — enabling Metal automatically."
        echo "         Pass --metal explicitly to confirm, or this is already set."
        METAL="ON"
    fi
fi

# ---- Step 1: Build GGML submodule --------------------------------
echo "[1/3] Building GGML..."
GGML_FLAGS="-DGGML_METAL=$METAL -DGGML_CUDA=$CUDA"
cmake -S ggml -B ggml/build \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    $GGML_FLAGS \
    ${CMAKE_GEN:-}
cmake --build ggml/build -j "$JOBS"
echo "      GGML built OK."

# ---- Step 2: Configure project -----------------------------------
echo ""
echo "[2/3] Configuring qwen3-tts.cpp..."
cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DQWEN3_TTS_KV_F32="$KV_F32" \
    ${CMAKE_GEN:-}

# ---- Step 3: Build project ---------------------------------------
echo ""
echo "[3/3] Building qwen3-tts.cpp..."
cmake --build "$BUILD_DIR" -j "$JOBS"

echo ""
echo "============================================================"
echo " Build complete!"
echo " Binaries in: $REPO_ROOT/$BUILD_DIR/"
echo ""
echo " Quick test:"
echo "   ./$BUILD_DIR/qwen3-tts-cli --help"
echo "   ./$BUILD_DIR/test_transformer --model models/qwen3-tts-0.6b-f16.gguf --ref-dir reference --max-len 32"
echo "============================================================"

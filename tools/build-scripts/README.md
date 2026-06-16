# Build Scripts

Build qwen3-tts.cpp from source using CMake + Ninja.

## Windows

```bat
build.bat               # Release build (default)
build.bat debug         # Debug build
build.bat --cuda        # Enable CUDA GPU acceleration (requires CUDA toolkit)
build.bat --kv-f32      # F32 KV cache (bit-exact mode, 2x RAM)
```

## macOS / Linux

```bash
chmod +x build.sh
./build.sh              # Release build (Metal auto-enabled on macOS)
./build.sh debug        # Debug build
./build.sh --metal      # Explicit Metal GPU
./build.sh --cuda       # CUDA GPU (Linux only)
./build.sh --kv-f32     # F32 KV cache
./build.sh --ninja      # Use Ninja generator (faster incremental builds)
```

## What it does

1. Builds GGML from `ggml/` submodule
2. Configures and builds the qwen3-tts.cpp project
3. Outputs binaries to `build/` (or `build-ninja/` with `--ninja`)

## Prerequisites

- CMake 3.14+
- C++17 compiler (GCC 9+, Clang 10+, or MSVC with clang-cl)
- Ninja (recommended, faster builds)
- CUDA toolkit (optional, for GPU acceleration)

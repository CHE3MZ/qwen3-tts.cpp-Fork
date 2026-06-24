class QwenTtsCpp < Formula
  desc "C++17/GGML inference for Qwen3-TTS — voice cloning, ICL, GPU acceleration. No Python at runtime."
  homepage "https://github.com/CHE3MZ/qwen3-tts.cpp-Fork"
  version "1.0.0"
  license "MIT"

  on_macos do
    url "https://github.com/CHE3MZ/qwen3-tts.cpp-Fork/releases/download/v#{version}/qwen-tts-macos-metal.zip"
    sha256 "PLACEHOLDER_SHA256_MACOS"
  end

  # Binary-only install — no build from source required.
  # The release zip already contains the Metal-accelerated binary and all
  # required GGML dylibs. Installing from source is complex due to the GGML
  # submodule + Metal shaders, so we ship prebuilt bottles instead.

  def install
    # Install the main binary as qwen-tts
    bin.install "qwen-tts" => "qwen-tts"

    # Install shared library
    lib.install Dir["libqwen3tts.dylib", "libggml*.dylib"].select { |f| File.exist?(f) }

    # Install Metal shader library if present (enables GPU acceleration)
    (share/"qwen-tts.cpp").install "default.metallib" if File.exist?("default.metallib")

    # Install C API header for developers
    (include/"qwen-tts.cpp").install "include/qwen3tts_c_api.h"
  end

  def post_install
    ohai "qwen-tts.cpp installed. Binary available as: qwen-tts"
    ohai ""
    ohai "Download models (no Python needed):"
    ohai "  #{HOMEBREW_PREFIX}/bin/qwen-tts --help"
    ohai ""
    ohai "Model downloader:"
    ohai "  Visit: https://github.com/CHE3MZ/qwen3-tts.cpp-Fork#model-setup"
  end

  test do
    # Basic smoke test — ensure the binary runs and exits with usage output
    output = shell_output("#{bin}/qwen-tts --help 2>&1", 1)
    assert_match "Usage:", output
  end
end

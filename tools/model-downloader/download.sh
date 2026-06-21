#!/usr/bin/env bash
# ============================================================
#  qwen3-tts.cpp -- Pre-built GGUF Downloader
#  Downloads ready-to-use GGUFs from HuggingFace.
#  No conversion required.
#
#  Usage:
#    ./download.sh              (interactive menu)
#    ./download.sh --help       (show Python script help)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Find Python
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[error] Python not found. Install Python 3.10+ and try again."
    exit 1
fi

cd "$REPO_ROOT"
exec "$PYTHON" tools/model-downloader/download.py "$@"

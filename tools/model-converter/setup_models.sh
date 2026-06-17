#!/usr/bin/env bash
# ============================================================
#  qwen3-tts.cpp — Model Setup Wizard (macOS / Linux)
#
#  Downloads and converts Qwen3-TTS models to GGUF format.
#  Wraps tools/model-converter/setup_models.py
#
#  Usage:
#    ./setup_models.sh                   (interactive)
#    ./setup_models.sh --non-interactive (use defaults)
#    ./setup_models.sh --hf-token <tok>  (with HF token)
# ============================================================
set -euo pipefail

# Navigate to repo root (two levels up from tools/model-converter/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo ""
echo "============================================================"
echo " qwen3-tts.cpp Model Setup"
echo " Repo: $REPO_ROOT"
echo "============================================================"
echo ""

# ---- Find Python -----------------------------------------------
PYTHON=""
for candidate in python3 python python3.12 python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "[error] Python 3.10+ not found."
    echo "        Install with: brew install python3  (macOS)"
    echo "        or: sudo apt install python3  (Ubuntu/Debian)"
    exit 1
fi
echo "Using: $($PYTHON --version)"
echo ""

# ---- Use uv if available AND a .venv exists (project isolation) -----
if command -v uv &>/dev/null; then
    if [ -f "$REPO_ROOT/.venv/bin/python" ] || [ -f "$REPO_ROOT/.venv/Scripts/python.exe" ]; then
        echo "[info] Using existing .venv via uv."
        uv run "$REPO_ROOT/tools/model-converter/setup_models.py" "$@"
        exit $?
    else
        echo "[info] uv found but no project .venv — using system Python."
        echo "       To use uv isolation: uv venv .venv && uv pip install huggingface_hub gguf torch safetensors numpy tqdm"
        echo ""
    fi
fi

# ---- Fall back to pip ------------------------------------------
# Check if required packages are installed
if ! "$PYTHON" -c "import huggingface_hub, gguf, torch, safetensors, numpy, tqdm" 2>/dev/null; then
    echo "[info] Installing required Python packages..."
    "$PYTHON" -m pip install huggingface_hub gguf torch safetensors numpy tqdm || {
        echo ""
        echo "[error] Failed to install packages. Try manually:"
        echo "  pip install huggingface_hub gguf torch safetensors numpy tqdm"
        echo ""
        echo "Or use uv (recommended):"
        echo "  pip install uv"
        echo "  uv run $REPO_ROOT/tools/model-converter/setup_models.py"
        exit 1
    }
fi

"$PYTHON" "$REPO_ROOT/tools/model-converter/setup_models.py" "$@"

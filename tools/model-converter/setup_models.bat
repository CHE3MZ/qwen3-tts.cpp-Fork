@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  qwen3-tts.cpp — Model Setup Wizard (Windows)
REM
REM  Downloads and converts Qwen3-TTS models to GGUF format.
REM  Wraps tools/model-converter/setup_models.py
REM
REM  Usage:
REM    setup_models.bat                  (interactive)
REM    setup_models.bat --non-interactive (use defaults)
REM    setup_models.bat --hf-token <tok> (with HF token)
REM ============================================================

REM Navigate to repo root (two levels up from tools/model-converter/)
cd /d "%~dp0..\.."
set REPO_ROOT=%CD%

echo.
echo ============================================================
echo  qwen3-tts.cpp Model Setup
echo  Repo: %REPO_ROOT%
echo ============================================================
echo.

REM Find Python
set PYTHON=
where python >nul 2>&1 && set PYTHON=python
if "%PYTHON%"=="" (
    where python3 >nul 2>&1 && set PYTHON=python3
)
if "%PYTHON%"=="" (
    echo [error] Python not found. Please install Python 3.10+
    echo         https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Show Python version
for /f "tokens=*" %%v in ('"%PYTHON%" --version 2^>^&1') do set PYVER=%%v
echo Using: %PYVER%
echo.

REM Check for uv (faster pip alternative)
where uv >nul 2>&1
if not errorlevel 1 (
    echo [info] uv detected — using uv run for dependency management.
    echo.
    uv run "%REPO_ROOT%\tools\model-converter\setup_models.py" %*
) else (
    REM Check required packages
    "%PYTHON%" -c "import huggingface_hub, gguf, torch, safetensors, numpy, tqdm" >nul 2>&1
    if errorlevel 1 (
        echo [info] Installing required Python packages...
        "%PYTHON%" -m pip install huggingface_hub gguf torch safetensors numpy tqdm
        if errorlevel 1 (
            echo [error] Failed to install packages. Try:
            echo   pip install huggingface_hub gguf torch safetensors numpy tqdm
            pause
            exit /b 1
        )
    )
    "%PYTHON%" "%REPO_ROOT%\tools\model-converter\setup_models.py" %*
)

if errorlevel 1 (
    echo.
    echo [error] Setup failed. Check the output above for details.
    pause
    exit /b 1
)

echo.
echo Setup complete! Press any key to exit.
pause >nul

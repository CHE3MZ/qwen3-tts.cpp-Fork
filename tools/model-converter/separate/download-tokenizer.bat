@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  qwen3-tts.cpp — Download + Convert Tokenizer/Vocoder Only
REM
REM  Downloads Qwen3-TTS-Tokenizer-12Hz from HuggingFace and
REM  converts it to GGUF format (includes Mimi encoder for ICL).
REM
REM  Usage:
REM    download-tokenizer.bat [--type f16|f32] [--hf-token <tok>]
REM
REM  Output:
REM    models\qwen3-tts-tokenizer-f16.gguf  (default)
REM    models\qwen3-tts-tokenizer-f32.gguf  (if --type f32)
REM ============================================================

cd /d "%~dp0..\..\..\"
set REPO_ROOT=%CD%
set TYPE=f16
set TOKEN=
set MIMI_TYPE=f16

:parse
if "%~1"=="" goto done
if /i "%~1"=="--type"      ( set "TYPE=%~2"      & shift & shift & goto parse )
if /i "%~1"=="--mimi-type" ( set "MIMI_TYPE=%~2" & shift & shift & goto parse )
if /i "%~1"=="--hf-token"  ( set "TOKEN=%~2"     & shift & shift & goto parse )
shift & goto parse
:done

REM Find Python
set PYTHON=
where python >nul 2>&1 && set PYTHON=python
if "%PYTHON%"=="" where python3 >nul 2>&1 && set PYTHON=python3
if "%PYTHON%"=="" ( echo [error] Python not found. & pause & exit /b 1 )

echo.
echo ============================================================
echo  Tokenizer / Vocoder Download + Convert
echo  Type:      %TYPE%
echo  Mimi:      %MIMI_TYPE%
echo  Repo:      Qwen/Qwen3-TTS-Tokenizer-12Hz
echo  Output:    models\qwen3-tts-tokenizer-%TYPE%.gguf
echo ============================================================
echo.

REM --- Download tokenizer safetensors ---
set TOK_DIR=models\Qwen3-TTS-Tokenizer-12Hz
if exist "%TOK_DIR%\model.safetensors" (
    echo [ok] Tokenizer already downloaded at %TOK_DIR%
) else (
    echo [1/2] Downloading tokenizer...
    "%PYTHON%" "%~dp0_hf_download.py" "Qwen/Qwen3-TTS-Tokenizer-12Hz" "%TOK_DIR%" "%TOKEN%"
    if errorlevel 1 ( echo [error] Download failed. & pause & exit /b 1 )
    echo [ok] Tokenizer downloaded.
)

REM --- Convert to GGUF ---
set OUT_FILE=models\qwen3-tts-tokenizer-%TYPE%.gguf
if exist "%OUT_FILE%" (
    echo [ok] %OUT_FILE% already exists — skipping conversion.
    echo      Delete the file to force re-conversion.
) else (
    echo [2/2] Converting to GGUF ^(%TYPE%, Mimi=%MIMI_TYPE%^)...
    "%PYTHON%" scripts\convert_tokenizer_to_gguf.py ^
        --input "%TOK_DIR%" ^
        --output "%OUT_FILE%" ^
        --type %TYPE% ^
        --mimi-type %MIMI_TYPE%
    if errorlevel 1 ( echo [error] Conversion failed. & pause & exit /b 1 )
)

echo.
echo ============================================================
echo  Done!  %OUT_FILE%
echo ============================================================
pause

@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  qwen3-tts.cpp -- Download + Convert Tokenizer/Vocoder
REM
REM  Double-click: interactive menu
REM  Command line: --type f16 --mimi-type f32 --hf-token <tok>
REM
REM  The tokenizer is shared across ALL TTS model variants.
REM  Download once, works with 0.6b, 1.7b, Base, CustomVoice, VoiceDesign.
REM ============================================================

cd /d "%~dp0..\..\..\"
set REPO_ROOT=%CD%

set TYPE=
set MIMI_TYPE=
set TOKEN=

REM Parse command-line args
:parse
if "%~1"=="" goto interactive_check
if /i "%~1"=="--type"      ( set "TYPE=%~2"      & shift & shift & goto parse )
if /i "%~1"=="--mimi-type" ( set "MIMI_TYPE=%~2" & shift & shift & goto parse )
if /i "%~1"=="--hf-token"  ( set "TOKEN=%~2"     & shift & shift & goto parse )
shift & goto parse

:interactive_check
if not "%TYPE%"=="" if not "%MIMI_TYPE%"=="" goto run

REM ---- Interactive menu ----------------------------------------
echo.
echo ============================================================
echo  qwen3-tts.cpp -- Tokenizer / Vocoder Downloader
echo  (Shared file -- download once for all TTS models)
echo ============================================================
echo.

REM --- Vocoder type ---
if "%TYPE%"=="" (
    echo  Tokenizer precision:
    echo    1. f16  ~432 MB  - Recommended for all uses
    echo    2. f32  ~864 MB  - Bit-exact ICL voice cloning only
    echo.
    set /p TYPE_NUM="  Choose (1-2) [default: 1]: "
    if "!TYPE_NUM!"==""  set TYPE_NUM=1
    if "!TYPE_NUM!"=="1" set TYPE=f16
    if "!TYPE_NUM!"=="2" set TYPE=f32
    if "!TYPE!"=="" ( echo [error] Invalid choice. & pause & exit /b 1 )
)

REM --- Mimi encoder precision ---
if "%MIMI_TYPE%"=="" (
    echo.
    echo  Mimi encoder precision (inside the tokenizer file):
    echo    1. f16  - Good quality, matches 98.9%% of Python reference
    echo    2. f32  - Bit-exact ICL voice cloning vs Python reference
    echo    3. q8_0 - Not recommended (below 95%% match threshold)
    echo.
    set /p MIMI_NUM="  Choose (1-3) [default: 1]: "
    if "!MIMI_NUM!"==""  set MIMI_NUM=1
    if "!MIMI_NUM!"=="1" set MIMI_TYPE=f16
    if "!MIMI_NUM!"=="2" set MIMI_TYPE=f32
    if "!MIMI_NUM!"=="3" set MIMI_TYPE=q8_0
    if "!MIMI_TYPE!"=="" ( echo [error] Invalid choice. & pause & exit /b 1 )
)

REM --- Optional HF token ---
echo.
set /p TOKEN="  HuggingFace token (leave blank if not needed): "

:run
set TOK_DIR=models\Qwen3-TTS-Tokenizer-12Hz
set OUT_FILE=models\qwen3-tts-tokenizer-%TYPE%.gguf

REM Find Python
set PYTHON=
where python >nul 2>&1 && set PYTHON=python
if "%PYTHON%"=="" where python3 >nul 2>&1 && set PYTHON=python3
if "%PYTHON%"=="" ( echo [error] Python not found. & pause & exit /b 1 )

echo.
echo ============================================================
echo  Downloading and converting:
echo    Precision : %TYPE%
echo    Mimi      : %MIMI_TYPE%
echo    Source    : Qwen/Qwen3-TTS-Tokenizer-12Hz
echo    Output    : %OUT_FILE%
echo ============================================================
echo.

REM Download
if exist "%TOK_DIR%\model.safetensors" (
    echo [ok] Tokenizer source already present at %TOK_DIR%
) else (
    echo [1/2] Downloading tokenizer...
    "%PYTHON%" "%~dp0_hf_download.py" "Qwen/Qwen3-TTS-Tokenizer-12Hz" "%TOK_DIR%" "%TOKEN%"
    if errorlevel 1 ( echo. & echo [error] Download failed. & pause & exit /b 1 )
)

REM Convert
if exist "%OUT_FILE%" (
    echo [ok] %OUT_FILE% already exists -- skipping conversion.
    echo      Delete it to force re-conversion.
) else (
    echo [2/2] Converting to GGUF (%TYPE%, Mimi=%MIMI_TYPE%)...
    "%PYTHON%" scripts\convert_tokenizer_to_gguf.py --input "%TOK_DIR%" --output "%OUT_FILE%" --type %TYPE% --mimi-type %MIMI_TYPE%
    if errorlevel 1 ( echo. & echo [error] Conversion failed. & pause & exit /b 1 )
)

echo.
echo ============================================================
echo  Done!
echo    %OUT_FILE%
echo  Use with any TTS model variant.
echo ============================================================
echo.
pause

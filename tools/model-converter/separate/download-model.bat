@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  qwen3-tts.cpp — Download + Convert TTS Model Only
REM
REM  Downloads a specific Qwen3-TTS model from HuggingFace and
REM  converts it to GGUF. Does NOT download the tokenizer/vocoder.
REM  Run download-tokenizer.bat separately to get the tokenizer.
REM
REM  Usage:
REM    download-model.bat --variant base --size 0.6b --type q8_0
REM    download-model.bat --variant base --size 1.7b --type f16
REM    download-model.bat --variant custom_voice --size 1.7b --type q6_k
REM    download-model.bat --variant voice_design --size 1.7b --type q5_k
REM    download-model.bat --hf-token <token> --variant base --size 0.6b --type q8_0
REM
REM  Variants: base  custom_voice  voice_design
REM  Sizes:    0.6b  1.7b
REM  Types:    f16  q8_0  q6_k  q5_k  q4_k  q3_k  q2_k
REM
REM  Note: voice_design is 1.7b only.
REM ============================================================

cd /d "%~dp0..\..\..\"
set REPO_ROOT=%CD%
set VARIANT=base
set SIZE=0.6b
set TYPE=q8_0
set TOKEN=

:parse
if "%~1"=="" goto done
if /i "%~1"=="--variant"   ( set "VARIANT=%~2"  & shift & shift & goto parse )
if /i "%~1"=="--size"      ( set "SIZE=%~2"     & shift & shift & goto parse )
if /i "%~1"=="--type"      ( set "TYPE=%~2"     & shift & shift & goto parse )
if /i "%~1"=="--hf-token"  ( set "TOKEN=%~2"    & shift & shift & goto parse )
shift & goto parse
:done

REM Find Python
set PYTHON=
where python >nul 2>&1 && set PYTHON=python
if "%PYTHON%"=="" where python3 >nul 2>&1 && set PYTHON=python3
if "%PYTHON%"=="" ( echo [error] Python not found. & pause & exit /b 1 )

REM --- Validate voice_design is 1.7b only ---
if /i "%VARIANT%"=="voice_design" if /i "%SIZE%"=="0.6b" (
    echo [error] VoiceDesign is 1.7b only. Alibaba has not published a 0.6b VoiceDesign model.
    pause & exit /b 1
)

REM --- Map variant + size to HF repo ID and local dir ---
set REPO_ID=
set LOCAL_DIR=

if /i "%VARIANT%"=="base" (
    if /i "%SIZE%"=="0.6b" (
        set REPO_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base
        set LOCAL_DIR=models\Qwen3-TTS-12Hz-0.6B-Base
    ) else (
        set REPO_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base
        set LOCAL_DIR=models\Qwen3-TTS-12Hz-1.7B-Base
    )
) else if /i "%VARIANT%"=="custom_voice" (
    if /i "%SIZE%"=="0.6b" (
        set REPO_ID=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
        set LOCAL_DIR=models\Qwen3-TTS-12Hz-0.6B-CustomVoice
    ) else (
        set REPO_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
        set LOCAL_DIR=models\Qwen3-TTS-12Hz-1.7B-CustomVoice
    )
) else if /i "%VARIANT%"=="voice_design" (
    set REPO_ID=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
    set LOCAL_DIR=models\Qwen3-TTS-12Hz-1.7B-VoiceDesign
) else (
    echo [error] Unknown variant: %VARIANT%. Use: base, custom_voice, voice_design
    pause & exit /b 1
)

REM --- Output GGUF filename ---
set OUT_FILE=models\qwen3-tts-%SIZE%-%TYPE%.gguf
if /i "%VARIANT%"=="custom_voice" set OUT_FILE=models\qwen3-tts-%SIZE%-customvoice-%TYPE%.gguf
if /i "%VARIANT%"=="voice_design"  set OUT_FILE=models\qwen3-tts-%SIZE%-voicedesign-%TYPE%.gguf

echo.
echo ============================================================
echo  TTS Model Download + Convert
echo  Variant:   %VARIANT%
echo  Size:      %SIZE%
echo  Type:      %TYPE%
echo  Repo:      %REPO_ID%
echo  Output:    %OUT_FILE%
echo ============================================================
echo.

REM --- Download safetensors ---
if exist "%LOCAL_DIR%\model.safetensors" (
    echo [ok] Model already downloaded at %LOCAL_DIR%
) else (
    echo [1/2] Downloading %REPO_ID%...
    "%PYTHON%" "%~dp0_hf_download.py" "%REPO_ID%" "%LOCAL_DIR%" "%TOKEN%"
    if errorlevel 1 ( echo [error] Download failed. & pause & exit /b 1 )
    echo [ok] Download complete.
)

REM --- Convert to GGUF ---
if exist "%OUT_FILE%" (
    echo [ok] %OUT_FILE% already exists — skipping conversion.
    echo      Delete the file to force re-conversion.
) else (
    echo [2/2] Converting to GGUF ^(%TYPE%^)...
    "%PYTHON%" scripts\convert_tts_to_gguf.py ^
        --input "%LOCAL_DIR%" ^
        --output "%OUT_FILE%" ^
        --type %TYPE%
    if errorlevel 1 ( echo [error] Conversion failed. & pause & exit /b 1 )
)

echo.
echo ============================================================
echo  Done!  %OUT_FILE%
echo  Pair with: models\qwen3-tts-tokenizer-f16.gguf
echo  Run:  build-ninja\qwen3-tts-cli.exe -m models -t "Hello" -o out.wav
echo ============================================================
pause

@echo off
setlocal

REM ============================================================
REM  qwen3-tts.cpp -- Download + Convert TTS Model
REM
REM  Double-click: interactive menu
REM  Command line: --variant base --size 0.6b --type q8_0
REM
REM  Variants: base  custom_voice  voice_design
REM  Sizes:    0.6b  1.7b   (voice_design: 1.7b only)
REM  Types:    f16  q8_0  q6_k  q5_k  q4_k  q3_k  q2_k
REM ============================================================

cd /d "%~dp0..\..\..\"
set REPO_ROOT=%CD%
set VARIANT=
set SIZE=
set TYPE=
set TOKEN=

:parse
if "%~1"=="" goto ask_variant
if /i "%~1"=="--variant"   ( set "VARIANT=%~2" & shift & shift & goto parse )
if /i "%~1"=="--size"      ( set "SIZE=%~2"    & shift & shift & goto parse )
if /i "%~1"=="--type"      ( set "TYPE=%~2"    & shift & shift & goto parse )
if /i "%~1"=="--hf-token"  ( set "TOKEN=%~2"   & shift & shift & goto parse )
shift & goto parse

:ask_variant
if not "%VARIANT%"=="" goto ask_size
echo.
echo ============================================================
echo  qwen3-tts.cpp -- TTS Model Downloader
echo ============================================================
echo.
echo  Model variant:
echo    1. base          - Voice cloning from reference audio (recommended)
echo    2. custom_voice  - Named speaker presets + style instruct (1.7B)
echo    3. voice_design  - Describe voice in natural language (1.7B only)
echo.
set /p V_NUM=  Choose (1-3) [default: 1]: 
if "%V_NUM%"==""  set V_NUM=1
if "%V_NUM%"=="1" set VARIANT=base
if "%V_NUM%"=="2" set VARIANT=custom_voice
if "%V_NUM%"=="3" set VARIANT=voice_design
if "%VARIANT%"=="" ( echo [error] Invalid choice: %V_NUM% & pause & exit /b 1 )

:ask_size
if not "%SIZE%"=="" goto ask_type
if /i "%VARIANT%"=="voice_design" ( set SIZE=1.7b & echo  Size: 1.7b only ^(VoiceDesign has no 0.6b^) & goto ask_type )
echo.
echo  Model size:
echo    1. 0.6b  - Faster, ~1.75 GB F16  [recommended]
echo    2. 1.7b  - Better quality, ~4.2 GB F16
echo.
set /p S_NUM=  Choose (1-2) [default: 1]: 
if "%S_NUM%"==""  set S_NUM=1
if "%S_NUM%"=="1" set SIZE=0.6b
if "%S_NUM%"=="2" set SIZE=1.7b
if "%SIZE%"=="" ( echo [error] Invalid choice: %S_NUM% & pause & exit /b 1 )

:ask_type
if not "%TYPE%"=="" goto ask_token
echo.
echo  Quantization:
echo    1. f16   ~1.75/4.2 GB   Full precision
echo    2. q8_0  ~1.28/3.1 GB   Near-lossless  [recommended]
echo    3. q6_k  ~0.99/2.4 GB   Excellent
echo    4. q5_k  ~0.86/2.1 GB   Very good
echo    5. q4_k  ~0.72/1.8 GB   Good
echo    6. q3_k  ~0.58/1.4 GB   Not recommended
echo    7. q2_k  ~0.46/1.1 GB   Not recommended
echo.
set /p T_NUM=  Choose (1-7) [default: 2]: 
if "%T_NUM%"==""  set T_NUM=2
if "%T_NUM%"=="1" set TYPE=f16
if "%T_NUM%"=="2" set TYPE=q8_0
if "%T_NUM%"=="3" set TYPE=q6_k
if "%T_NUM%"=="4" set TYPE=q5_k
if "%T_NUM%"=="5" set TYPE=q4_k
if "%T_NUM%"=="6" set TYPE=q3_k
if "%T_NUM%"=="7" set TYPE=q2_k
if "%TYPE%"=="" ( echo [error] Invalid choice: %T_NUM% & pause & exit /b 1 )

:ask_token
if not "%TOKEN%"=="" goto validate
echo.
set /p TOKEN=  HuggingFace token (leave blank if not needed): 

:validate
if /i "%VARIANT%"=="voice_design" if /i "%SIZE%"=="0.6b" (
    echo [error] VoiceDesign is 1.7b only.
    pause & exit /b 1
)

if /i "%VARIANT%"=="base" (
    if /i "%SIZE%"=="0.6b" ( set REPO_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base    & set LOCAL_DIR=models\Qwen3-TTS-12Hz-0.6B-Base    ) else ( set REPO_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base    & set LOCAL_DIR=models\Qwen3-TTS-12Hz-1.7B-Base    )
) else if /i "%VARIANT%"=="custom_voice" (
    if /i "%SIZE%"=="0.6b" ( set REPO_ID=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice & set LOCAL_DIR=models\Qwen3-TTS-12Hz-0.6B-CustomVoice ) else ( set REPO_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice & set LOCAL_DIR=models\Qwen3-TTS-12Hz-1.7B-CustomVoice )
) else if /i "%VARIANT%"=="voice_design" (
    set REPO_ID=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
    set LOCAL_DIR=models\Qwen3-TTS-12Hz-1.7B-VoiceDesign
) else (
    echo [error] Unknown variant: %VARIANT%
    pause & exit /b 1
)

set OUT_FILE=models\qwen3-tts-%SIZE%-%TYPE%.gguf
if /i "%VARIANT%"=="custom_voice" set OUT_FILE=models\qwen3-tts-%SIZE%-customvoice-%TYPE%.gguf
if /i "%VARIANT%"=="voice_design"  set OUT_FILE=models\qwen3-tts-%SIZE%-voicedesign-%TYPE%.gguf

set PYTHON=
where python >nul 2>&1 && set PYTHON=python
if "%PYTHON%"=="" where python3 >nul 2>&1 && set PYTHON=python3
if "%PYTHON%"=="" ( echo [error] Python not found. & pause & exit /b 1 )

echo.
echo ============================================================
echo  Downloading and converting:
echo    Variant : %VARIANT%
echo    Size    : %SIZE%
echo    Type    : %TYPE%
echo    Source  : %REPO_ID%
echo    Output  : %OUT_FILE%
echo ============================================================
echo.

if exist "%LOCAL_DIR%\model.safetensors" (
    echo [ok] Source model already present at %LOCAL_DIR%
) else (
    echo [1/2] Downloading %REPO_ID%...
    "%PYTHON%" "%~dp0_hf_download.py" "%REPO_ID%" "%LOCAL_DIR%" "%TOKEN%"
    if errorlevel 1 ( echo. & echo [error] Download failed. & pause & exit /b 1 )
)

if exist "%OUT_FILE%" (
    echo [ok] %OUT_FILE% already exists -- skipping conversion.
    echo      Delete it to force re-conversion.
) else (
    echo [2/2] Converting to GGUF ^(%TYPE%^)...
    "%PYTHON%" scripts\convert_tts_to_gguf.py --input "%LOCAL_DIR%" --output "%OUT_FILE%" --type %TYPE%
    if errorlevel 1 ( echo. & echo [error] Conversion failed. & pause & exit /b 1 )
)

echo.
echo ============================================================
echo  Done!
echo    %OUT_FILE%
echo  Also need: models\qwen3-tts-tokenizer-f16.gguf
echo  Run: build-ninja\qwen3-tts-cli.exe -m models -t "Hello" -o out.wav
echo ============================================================
echo.
pause

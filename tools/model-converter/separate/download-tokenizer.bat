@echo off
setlocal

REM ============================================================
REM  qwen3-tts.cpp -- Download + Convert Tokenizer/Vocoder
REM
REM  Double-click: interactive menu
REM  Command line: --type f16 --mimi-type f32 --hf-token <tok>
REM
REM  Shared file -- download once, works with ALL TTS models.
REM ============================================================

cd /d "%~dp0..\..\..\"
set REPO_ROOT=%CD%
set TYPE=
set MIMI_TYPE=
set TOKEN=

:parse
if "%~1"=="" goto ask_type
if /i "%~1"=="--type"      ( set "TYPE=%~2"      & shift & shift & goto parse )
if /i "%~1"=="--mimi-type" ( set "MIMI_TYPE=%~2" & shift & shift & goto parse )
if /i "%~1"=="--hf-token"  ( set "TOKEN=%~2"     & shift & shift & goto parse )
shift & goto parse

:ask_type
if not "%TYPE%"=="" goto ask_mimi
echo.
echo ============================================================
echo  qwen3-tts.cpp -- Tokenizer / Vocoder Downloader
echo  (Shared file -- download once for all TTS models)
echo ============================================================
echo.
echo  Tokenizer precision:
echo    1. f16  ~432 MB  - Recommended
echo    2. f32  ~864 MB  - Bit-exact ICL voice cloning
echo.
set /p TYPE_NUM=  Choose (1-2) [default: 1]: 
if "%TYPE_NUM%"==""  set TYPE_NUM=1
if "%TYPE_NUM%"=="1" set TYPE=f16
if "%TYPE_NUM%"=="2" set TYPE=f32
if "%TYPE%"=="" ( echo [error] Invalid choice: %TYPE_NUM% & pause & exit /b 1 )

:ask_mimi
if not "%MIMI_TYPE%"=="" goto ask_token
echo.
echo  Mimi encoder precision (affects ICL voice cloning quality):
echo    1. f16  - Good quality, 98.9%% match vs Python  [recommended]
echo    2. f32  - Bit-exact ICL cloning vs Python reference
echo    3. q8_0 - Not recommended
echo.
set /p MIMI_NUM=  Choose (1-3) [default: 1]: 
if "%MIMI_NUM%"==""  set MIMI_NUM=1
if "%MIMI_NUM%"=="1" set MIMI_TYPE=f16
if "%MIMI_NUM%"=="2" set MIMI_TYPE=f32
if "%MIMI_NUM%"=="3" set MIMI_TYPE=q8_0
if "%MIMI_TYPE%"=="" ( echo [error] Invalid choice: %MIMI_NUM% & pause & exit /b 1 )

:ask_token
if not "%TOKEN%"=="" goto run
echo.
set /p TOKEN=  HuggingFace token (leave blank if not needed): 

:run
set TOK_DIR=models\Qwen3-TTS-Tokenizer-12Hz
set OUT_FILE=models\qwen3-tts-tokenizer-%TYPE%.gguf

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

if exist "%TOK_DIR%\model.safetensors" (
    echo [ok] Tokenizer source already present at %TOK_DIR%
) else (
    echo [1/2] Downloading tokenizer...
    "%PYTHON%" "%~dp0_hf_download.py" "Qwen/Qwen3-TTS-Tokenizer-12Hz" "%TOK_DIR%" "%TOKEN%"
    if errorlevel 1 ( echo. & echo [error] Download failed. & pause & exit /b 1 )
)

if exist "%OUT_FILE%" (
    echo [ok] %OUT_FILE% already exists -- skipping conversion.
    echo      Delete it to force re-conversion.
) else (
    echo [2/2] Converting to GGUF ^(%TYPE%^, Mimi=%MIMI_TYPE%^)...
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

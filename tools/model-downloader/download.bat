@echo off
REM ============================================================
REM  qwen3-tts.cpp -- Pre-built GGUF Downloader
REM  Downloads ready-to-use GGUFs from HuggingFace.
REM  No conversion required.
REM
REM  Usage:
REM    download.bat              (interactive menu)
REM    download.bat --help       (show Python script help)
REM ============================================================
cd /d "%~dp0..\.."

set PYTHON=
where python  >nul 2>&1 && set PYTHON=python
if "%PYTHON%"=="" ( where python3 >nul 2>&1 && set PYTHON=python3 )
if "%PYTHON%"=="" (
    echo [error] Python not found. Install Python 3.10+ and try again.
    pause
    exit /b 1
)

"%PYTHON%" tools\model-downloader\download.py %*
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)
echo.
pause

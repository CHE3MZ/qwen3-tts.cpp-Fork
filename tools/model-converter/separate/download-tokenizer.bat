@echo off
REM ============================================================
REM  qwen3-tts.cpp -- Download + Convert Tokenizer/Vocoder
REM  Double-click for interactive menu.
REM  Command line: --type f16 --mimi-type f32 --hf-token TOKEN
REM ============================================================
cd /d "%~dp0..\..\..\"
set PYTHON=
where python  >nul 2>&1 && set PYTHON=python
if "%PYTHON%"=="" ( where python3 >nul 2>&1 && set PYTHON=python3 )
if "%PYTHON%"=="" ( echo [error] Python not found. & pause & exit /b 1 )
"%PYTHON%" tools\model-converter\separate\_interactive.py tokenizer %*
if errorlevel 1 ( echo. & pause & exit /b 1 )
echo.
pause

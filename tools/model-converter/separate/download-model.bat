@echo off
REM ============================================================
REM  qwen3-tts.cpp -- Download + Convert TTS Model
REM  Double-click for interactive menu.
REM  Command line: --variant base --size 0.6b --type q8_0
REM ============================================================
cd /d "%~dp0..\..\..\"
set PYTHON=
where python  >nul 2>&1 && set PYTHON=python
if "%PYTHON%"=="" ( where python3 >nul 2>&1 && set PYTHON=python3 )
if "%PYTHON%"=="" ( echo [error] Python not found. & pause & exit /b 1 )
"%PYTHON%" tools\model-converter\separate\_interactive.py model %*
if errorlevel 1 ( echo. & pause & exit /b 1 )
echo.
pause

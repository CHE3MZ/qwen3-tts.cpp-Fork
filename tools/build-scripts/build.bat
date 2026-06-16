@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  qwen3-tts.cpp — Windows build script (Ninja + clang-cl)
REM  Usage: build.bat [Release|Debug] [--cuda] [--kv-f32]
REM ============================================================

set BUILD_TYPE=Release
set CUDA=OFF
set KV_F32=OFF
set BUILD_DIR=build-ninja

REM Parse arguments
:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="debug"      set BUILD_TYPE=Debug
if /i "%~1"=="release"    set BUILD_TYPE=Release
if /i "%~1"=="--cuda"     set CUDA=ON
if /i "%~1"=="--kv-f32"   set KV_F32=ON
shift
goto parse_args
:args_done

REM Navigate to repo root (two levels up from tools/build-scripts/)
cd /d "%~dp0..\.."
set REPO_ROOT=%CD%

echo.
echo ============================================================
echo  qwen3-tts.cpp build  [%BUILD_TYPE%]
echo  CUDA: %CUDA%   KV_F32: %KV_F32%
echo  Repo: %REPO_ROOT%
echo ============================================================
echo.

REM ---- Step 1: Build GGML submodule ---------------------------
echo [1/3] Building GGML...
set GGML_FLAGS=-DGGML_CUDA=%CUDA%
if "%CUDA%"=="ON" (
    echo       CUDA enabled. Make sure CUDA toolkit is installed.
    set GGML_FLAGS=%GGML_FLAGS% -DGGML_CUDA=ON
)

cmake -S ggml -B ggml\build -G Ninja -DCMAKE_BUILD_TYPE=%BUILD_TYPE% %GGML_FLAGS%
if errorlevel 1 ( echo ERROR: GGML CMake configure failed & exit /b 1 )

cmake --build ggml\build -j 4
if errorlevel 1 ( echo ERROR: GGML build failed & exit /b 1 )
echo       GGML built OK.

REM ---- Step 2: Configure project ------------------------------
echo.
echo [2/3] Configuring qwen3-tts.cpp...
set PROJECT_FLAGS=-DQWEN3_TTS_KV_F32=%KV_F32%
cmake -S . -B %BUILD_DIR% -G Ninja -DCMAKE_BUILD_TYPE=%BUILD_TYPE% %PROJECT_FLAGS%
if errorlevel 1 ( echo ERROR: Project CMake configure failed & exit /b 1 )

REM ---- Step 3: Build project ----------------------------------
echo.
echo [3/3] Building qwen3-tts.cpp...
cmake --build %BUILD_DIR% -j 4
if errorlevel 1 ( echo ERROR: Project build failed & exit /b 1 )

echo.
echo ============================================================
echo  Build complete!
echo  Binaries in: %REPO_ROOT%\%BUILD_DIR%\
echo.
echo  Quick test:
echo    %BUILD_DIR%\qwen3-tts-cli.exe --help
echo    %BUILD_DIR%\test_transformer.exe --model models\qwen3-tts-0.6b-f16.gguf --ref-dir reference --max-len 32
echo ============================================================

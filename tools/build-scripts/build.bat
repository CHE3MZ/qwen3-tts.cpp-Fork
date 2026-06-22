@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  qwen3-tts.cpp -- Windows build script
REM  Usage: build.bat [Release|Debug] [--cuda] [--vulkan] [--kv-f32]
REM
REM  Ninja is used automatically when available (faster builds).
REM  Checks both system PATH and the VS-bundled Ninja location.
REM  Falls back silently to Visual Studio if Ninja is not found.
REM ============================================================

set BUILD_TYPE=Release
set CUDA=OFF
set VULKAN=OFF
set KV_F32=OFF

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="debug"    set BUILD_TYPE=Debug
if /i "%~1"=="release"  set BUILD_TYPE=Release
if /i "%~1"=="--cuda"   set CUDA=ON
if /i "%~1"=="--vulkan" set VULKAN=ON
if /i "%~1"=="--kv-f32" set KV_F32=ON
shift
goto parse_args
:args_done

REM Navigate to repo root (two levels up from tools/build-scripts/)
cd /d "%~dp0..\.."
set REPO_ROOT=%CD%

REM ---- Ninja auto-detect -----------------------------------------------
REM Prefer Ninja (single-config, fast). Fall back to Visual Studio.
REM Ninja is often bundled inside VS CMake tools but not on the system PATH.
REM We probe the known VS install paths when the system PATH check fails.
set CMAKE_GEN=
set BUILD_DIR=build
set GEN_LABEL=Visual Studio (default)

REM 1. Check if ninja is already on PATH
where ninja >nul 2>&1
if not errorlevel 1 goto :try_ninja

REM 2. Not on PATH -- probe VS-bundled Ninja locations
set VS_NINJA_PATHS=
set VS_NINJA_PATHS=%VS_NINJA_PATHS% "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set VS_NINJA_PATHS=%VS_NINJA_PATHS% "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set VS_NINJA_PATHS=%VS_NINJA_PATHS% "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set VS_NINJA_PATHS=%VS_NINJA_PATHS% "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set VS_NINJA_PATHS=%VS_NINJA_PATHS% "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set VS_NINJA_PATHS=%VS_NINJA_PATHS% "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set VS_NINJA_PATHS=%VS_NINJA_PATHS% "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set VS_NINJA_PATHS=%VS_NINJA_PATHS% "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

for %%P in (%VS_NINJA_PATHS%) do (
    if exist "%%~P\ninja.exe" (
        set "PATH=%%~P;%PATH%"
        goto :try_ninja
    )
)
goto :no_ninja

:try_ninja
cmake --help 2>nul | findstr /i "Ninja" >nul 2>&1
if not errorlevel 1 (
    set CMAKE_GEN=-G Ninja
    set BUILD_DIR=build-ninja
    set GEN_LABEL=Ninja
    goto :gen_done
)

:no_ninja
REM No ninja or cmake doesn't support it -- use Visual Studio generator

:gen_done
echo [build] Generator: %GEN_LABEL%

echo.
echo ============================================================
echo  qwen3-tts.cpp build  [%BUILD_TYPE%]
echo  CUDA: %CUDA%   VULKAN: %VULKAN%   KV_F32: %KV_F32%
echo  Generator: %GEN_LABEL%
echo  Output:    %REPO_ROOT%\%BUILD_DIR%\
echo ============================================================
echo.

REM ---- Step 1: Build GGML submodule ---------------------------
echo [1/3] Building GGML...
set GGML_FLAGS=-DGGML_CUDA=%CUDA% -DGGML_VULKAN=%VULKAN%
if "%CUDA%"=="ON" (
    echo       CUDA enabled -- make sure CUDA toolkit is installed.
)
if "%VULKAN%"=="ON" (
    echo       Vulkan enabled -- make sure LunarG Vulkan SDK is installed.
)

cmake -S ggml -B ggml\build %CMAKE_GEN% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% %GGML_FLAGS% -DMATH_LIBRARY=
if errorlevel 1 ( echo. & echo [error] GGML CMake configure failed. & exit /b 1 )

cmake --build ggml\build --config %BUILD_TYPE% -j 4
if errorlevel 1 ( echo. & echo [error] GGML build failed. & exit /b 1 )
echo       GGML built OK.

REM ---- Step 2: Configure project ------------------------------
echo.
echo [2/3] Configuring qwen3-tts.cpp...
cmake -S . -B %BUILD_DIR% %CMAKE_GEN% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DQWEN3_TTS_KV_F32=%KV_F32%
if errorlevel 1 ( echo. & echo [error] Project CMake configure failed. & exit /b 1 )

REM ---- Step 3: Build project ----------------------------------
echo.
echo [3/3] Building qwen3-tts.cpp...
cmake --build %BUILD_DIR% --config %BUILD_TYPE% -j 4
if errorlevel 1 ( echo. & echo [error] Project build failed. & exit /b 1 )

echo.
echo ============================================================
echo  Build complete!
echo  Output: %REPO_ROOT%\%BUILD_DIR%\
echo.
echo  Quick test:
echo    %BUILD_DIR%\qwen3-tts-cli.exe --help
echo    %BUILD_DIR%\test_transformer.exe --model models\qwen3-tts-0.6b-f16.gguf --ref-dir reference --max-len 32
echo ============================================================

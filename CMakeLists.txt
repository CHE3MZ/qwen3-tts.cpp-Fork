cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static libs are linked into the qwen3tts shared lib (C API), so they must be PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
elseif(MSVC)
    # Parallel compilation within each target (significant speedup for large files)
    add_compile_options(/MP)
    # Fast math for Release builds
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /fp:fast")
endif()

# KV cache precision option
# F32 KV cache gives bit-exact non-streaming output matching Python float32 inference.
# F16 (default) uses half the memory but may differ by 1 frame at EOS boundary.
option(QWEN3_TTS_KV_F32 "Use F32 KV cache for bit-exact non-streaming output (2× RAM)" OFF)
if(QWEN3_TTS_KV_F32)
    add_compile_definitions(QWEN3_TTS_KV_F32)
    message(STATUS "KV cache: F32 (bit-exact mode, 2× RAM)")
else()
    message(STATUS "KV cache: F16 (default, half RAM)")
endif()

# Timing instrumentation option
option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_TTS_TIMING)
    add_compile_definitions(QWEN3_TTS_TIMING)
endif()

# CoreML code predictor bridge (macOS only)
option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
endif()

# Find threads
find_package(Threads REQUIRED)

# GGML library paths
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
set(GGML_BUILD_DIR "${GGML_DIR}/build")

# GGML link search dirs: flat (Ninja/Make) and config-suffixed (MSVC multi-config)
set(GGML_LINK_DIRS
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/src/Release
    ${GGML_BUILD_DIR}/src/Debug
    ${GGML_BUILD_DIR}/src/ggml-metal
)

# Detect Metal backend availability
if(APPLE AND EXISTS "${GGML_BUILD_DIR}/src/ggml-metal/libggml-metal.dylib")
    set(GGML_HAS_METAL ON)
    message(STATUS "GGML Metal backend found — GPU acceleration enabled")
else()
    set(GGML_HAS_METAL OFF)
endif()

# gguf_loader utility — shared by all components, compiled once
add_library(gguf_loader STATIC
    src/gguf_loader.cpp
)
target_include_directories(gguf_loader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(gguf_loader PUBLIC ${GGML_LINK_DIRS})
target_link_libraries(gguf_loader PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_HAS_METAL)
    target_link_libraries(gguf_loader PUBLIC ggml-metal "-framework Metal" "-framework MetalKit")
endif()

# Text tokenizer library
add_library(text_tokenizer STATIC
    src/text_tokenizer.cpp
)
target_include_directories(text_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(text_tokenizer PUBLIC
    gguf_loader
)

# TTS Transformer library (GGML-based + optional CoreML bridge)
set(TTS_TRANSFORMER_SOURCES
    src/tts_transformer.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor.mm)
    set_source_files_properties(src/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor_stub.cpp)
endif()

add_library(tts_transformer STATIC
    ${TTS_TRANSFORMER_SOURCES}
)
target_include_directories(tts_transformer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(tts_transformer PUBLIC
    gguf_loader
)
if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(tts_transformer PUBLIC
        "-framework Foundation"
        "-framework CoreML"
    )
endif()

# Audio tokenizer encoder library (GGML-based)
add_library(audio_tokenizer_encoder STATIC
    src/audio_tokenizer_encoder.cpp
)
target_include_directories(audio_tokenizer_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(audio_tokenizer_encoder PUBLIC
    gguf_loader
)

# Mimi audio encoder library (CPU-based, for ICL voice cloning)
add_library(mimi_encoder STATIC
    src/mimi_encoder.cpp
)
target_include_directories(mimi_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(mimi_encoder PUBLIC
    gguf_loader
)

# Audio tokenizer decoder library (GGML-based)
add_library(audio_tokenizer_decoder STATIC
    src/audio_tokenizer_decoder.cpp
)
target_include_directories(audio_tokenizer_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(audio_tokenizer_decoder PUBLIC
    gguf_loader
)

# Qwen3 TTS library (full pipeline)
add_library(qwen3_tts STATIC
    src/qwen3_tts.cpp
)
target_include_directories(qwen3_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(qwen3_tts PUBLIC
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    mimi_encoder
    audio_tokenizer_decoder
    Threads::Threads
)
if(WIN32)
    target_link_libraries(qwen3_tts PUBLIC psapi)
endif()

# Shared library with C API (for Nim FFI)
add_library(qwen3tts_shared SHARED
    src/qwen3tts_c_api.cpp
)
set_target_properties(qwen3tts_shared PROPERTIES
    OUTPUT_NAME "qwen3tts"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)
target_include_directories(qwen3tts_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwen3tts_shared PRIVATE
    qwen3_tts
)

# CLI executable
add_executable(qwen3-tts-cli
    src/main.cpp
)
target_link_libraries(qwen3-tts-cli PRIVATE
    qwen3_tts
)

# Test executable for tokenizer
add_executable(test_tokenizer
    tests/test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    text_tokenizer
    Threads::Threads
)

# Test executable for audio encoder
add_executable(test_encoder
    tests/test_encoder.cpp
)
target_link_libraries(test_encoder PRIVATE
    audio_tokenizer_encoder
    Threads::Threads
)

# Test executable for transformer
add_executable(test_transformer
    tests/test_transformer.cpp
)
target_link_libraries(test_transformer PRIVATE
    tts_transformer
    Threads::Threads
)

# Test executable for audio decoder
add_executable(test_decoder
    tests/test_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE
    audio_tokenizer_decoder
    Threads::Threads
)

# Test executable for Mimi encoder (validates ICL reference audio encoding)
add_executable(test_mimi_encoder
    tests/test_mimi_encoder.cpp
)
target_link_libraries(test_mimi_encoder PRIVATE
    mimi_encoder
    Threads::Threads
)

# On Windows, copy GGML DLLs to the output directory so executables can find them at runtime.
# GGML is built separately (ggml/build) and its DLLs are not automatically staged by CMake.
if(WIN32)
    set(GGML_DLL_SEARCH_DIRS
        ${GGML_BUILD_DIR}/src/Release
        ${GGML_BUILD_DIR}/src/Debug
        ${GGML_BUILD_DIR}/src
    )
    foreach(_dll ggml.dll ggml-base.dll ggml-cpu.dll)
        foreach(_dir ${GGML_DLL_SEARCH_DIRS})
            if(EXISTS "${_dir}/${_dll}")
                configure_file("${_dir}/${_dll}" "${CMAKE_BINARY_DIR}/${_dll}" COPYONLY)
                break()
            endif()
        endforeach()
    endforeach()
endif()

# Install targets
install(TARGETS gguf_loader text_tokenizer tts_transformer audio_tokenizer_encoder audio_tokenizer_decoder qwen3_tts qwen3tts_shared qwen3-tts-cli
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES
    src/gguf_loader.h
    src/text_tokenizer.h
    src/tts_transformer.h
    src/coreml_code_predictor.h
    src/audio_tokenizer_encoder.h
    src/audio_tokenizer_decoder.h
    src/mimi_encoder.h
    src/qwen3_tts.h
    src/qwen3tts_c_api.h
    DESTINATION include
)

# CTest support
enable_testing()
add_test(NAME tokenizer_test
    COMMAND test_tokenizer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME encoder_test
    COMMAND test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf --audio clone.wav --reference reference/ref_audio_embedding.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME transformer_test
    COMMAND test_transformer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME decoder_test
    COMMAND test_decoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(NAME mimi_encoder_test
    COMMAND test_mimi_encoder
        --tokenizer models/qwen3-tts-tokenizer-f16.gguf
        --audio clone.wav
        --ref-codes reference/mimi_enc_py_codes.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

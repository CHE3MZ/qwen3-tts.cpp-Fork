# Qwen 3 TTS C++

`qwen3-tts.cpp` is a C++ inference implementation of the Qwen 3 TTS system, providing an alternative to the original Python-based inference tool.

## Performance

This version of the inference engine significantly outperforms the original python both in memory management and overall bootup speed , as for the generation speed it isn't really all that different , its slightly better at best in raw generation speed.

## Note

This repository is forked from:

https://github.com/predict-woo/qwen3-tts.cpp

It significantly expands upon the original project by adding many missing features and bringing the overall functionality much closer to that of the official Python implementation.

## Features

This project aims to provide feature parity with the Python version of Qwen 3 TTS while maintaining the advantages of a native C++ implementation.

Key benefits include:

* Native C++ implementation
* C API for integration with other programming languages
* CUDA and GPU acceleration support
* GGUF model support
* Model quantization support (including low-bit quantization such as Q2)
* Minimal runtime dependencies
* Small binary size
* Extensible architecture suitable for custom workflows

## Why Choose This Over the Python Version?

The Python implementation is more mature, polished, and extensively tested. However, it also comes with a large dependency footprint and is primarily designed for use within the Python ecosystem.

If you're building an application in another language and require fine-grained control over the TTS pipeline, integrating the Python version often means running it as a subprocess, which can limit flexibility and control.

With the C++ version, you can:

* Integrate directly through the provided C API.
* Access and control the inference pipeline at a much lower level.
* Use it natively from any language capable of interfacing with a C API.
* Build custom workflows and extensions more easily.
* Reduce deployment complexity thanks to fewer dependencies.
* Benefit from a significantly smaller executable size (typically under 10 MB, excluding models).
* Take advantage of faster execution speeds and native GPU acceleration.

Additionally, because the project uses GGUF models, you can choose between full-precision models and various quantized versions depending on your performance and memory requirements.

## Contributing

Contributions are welcome.

To contribute:

1. Fork the repository.
2. Create your changes.
3. Submit a pull request.

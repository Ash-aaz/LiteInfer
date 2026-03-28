# Lite-Infer

A lightweight local AI inference engine written in C++ using ONNX Runtime. Runs models directly on bare metal without Python or framework overhead.

Built as a learning project to improve C++ skills and API understanding.

## Features

- CPU inference via ONNX Runtime
- Clean C++ class interface — load a model, run a forward pass, read predictions
- Demonstrated on MNIST-8 digit classification

## Dependencies

- [ONNX Runtime](https://onnxruntime.ai/) — install via your package manager
  - Arch/CachyOS: `sudo pacman -S onnxruntime`
- CMake 3.10+
- C++17 compiler

## Build

```bash
mkdir build
cd build
cmake ..
make
```

## Run

```bash
./build/engine
```

## Project Structure

```
Lite-Infer/
├── model/                  # Place your .onnx model files here
├── inference_engine.h      # Class declaration
├── inference.cpp           # Inference engine implementation
├── main.cpp                # Entry point
└── CMakeLists.txt
```

## Roadmap

- [ ] CUDA execution via ONNX Runtime CUDA provider
- [ ] Real image input pipeline
- [ ] Softmax post-processing
- [ ] Benchmarking CPU vs GPU latency
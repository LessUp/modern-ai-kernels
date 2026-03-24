# TensorCraft-HPC

English | [简体中文](README.zh-CN.md) | [Docs](https://lessup.github.io/modern-ai-kernels/)

[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/modern-ai-kernels/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17/20-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?logo=cmake&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

TensorCraft-HPC is a modern C++/CUDA AI kernel library for studying and validating high-performance implementations across GEMM, attention, convolution, sparse operators, and quantization.

## Repository Overview

- Header-first kernel library under `include/tensorcraft/`
- Python bindings in `src/python_ops/`
- Tests and benchmarks in `tests/` and `benchmarks/`
- GitHub Pages site for documentation entry, reading paths, and project updates

## Quick Start

```bash
cmake --preset release
cmake --build build/release -j$(nproc)
ctest --test-dir build/release --output-on-failure
pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

`pip install -e .` builds the pybind11 extension exposed as `tensorcraft_ops`. Python bindings currently require CUDA; when CUDA is unavailable, CMake disables tests, benchmarks, and Python bindings automatically.

## Docs

- Project docs: `https://lessup.github.io/modern-ai-kernels/`
- Site home covers project positioning, recommended reading paths, and key documentation links
- See `CONTRIBUTING.md` for contribution workflow

## License

MIT License

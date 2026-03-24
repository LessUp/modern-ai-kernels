# TensorCraft-HPC

English | [简体中文](README.zh-CN.md) | [Docs](https://lessup.github.io/modern-ai-kernels/)

[![CI](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/modern-ai-kernels/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github)](https://lessup.github.io/modern-ai-kernels/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-10.1%2B-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.20+-064F8C?logo=cmake&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)

TensorCraft-HPC is a modern C++/CUDA AI kernel library for studying and validating GEMM, attention, convolution, normalization, sparse operators, and quantization.

## Repository Overview

- Header-first kernel library under `include/tensorcraft/`
- Python bindings in `src/python_ops/`
- Tests in `tests/`
- Benchmarks in `benchmarks/`
- Project docs on GitHub Pages

## Quick Start

Recommended on a CUDA development machine:

```bash
cmake --preset dev
cmake --build --preset dev --parallel 2
ctest --preset dev --output-on-failure
python -m pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

## Build Presets

- `dev`: recommended day-to-day CUDA development preset; single architecture, tests on, Python on
- `python-dev`: lighter CUDA preset focused on building `tensorcraft_ops`
- `release`: heavier full build, including benchmarks
- `cpu-smoke`: CPU-only configure/install smoke validation; tests and Python bindings are disabled

## Build Notes

- Minimum supported CUDA toolkit is `10.1`
- CUDA `11.x`/`12.x` unlock more optimized feature paths than CUDA `10.x`
- If CUDA is unavailable, CMake disables tests, benchmarks, and Python bindings automatically
- If build pressure is high, prefer `dev`/`python-dev`, keep `--parallel` low, and set a single `CMAKE_CUDA_ARCHITECTURES` value for your GPU

## Python Bindings

The pybind11 module is exposed as `tensorcraft_ops`.

```bash
python -m pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

## Docs

- Project docs: `https://lessup.github.io/modern-ai-kernels/`
- Installation: `docs/INSTALL.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`
- Contribution workflow: `CONTRIBUTING.md`

## License

MIT License

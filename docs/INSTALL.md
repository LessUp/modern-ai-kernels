# Installation Guide

This guide documents the current recommended build paths for TensorCraft-HPC.

## Prerequisites

### Required

- **CUDA Toolkit**: `10.1` or later
- **CMake**: `3.20` or later
- **C++ Compiler**: C++17-capable host compiler
- **NVIDIA GPU**: recommended for tests and Python bindings

### Optional

- **Python**: `3.8+` for `tensorcraft_ops`
- **Ninja**: recommended generator for faster builds

## Recommended Build Flows

### 1. CUDA development flow

Use this for normal development on a CUDA machine.

```bash
cmake --preset dev
cmake --build --preset dev --parallel 2
ctest --preset dev --output-on-failure
```

### 2. Python-only / lighter CUDA flow

Use this when you mainly care about the Python extension.

```bash
cmake --preset python-dev
cmake --build --preset python-dev --parallel 2
python -m pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

### 3. Heavier full build

Use this when you want the more complete release-style path, including benchmarks.

```bash
cmake --preset release
cmake --build --preset release --parallel
ctest --test-dir build/release --output-on-failure
```

### 4. CPU-only smoke validation

Use this on machines without CUDA when you only need to validate configure/install behavior.

```bash
cmake --preset cpu-smoke
cmake --install build/cpu-smoke --prefix /tmp/tensorcraft-install
```

In this mode, tests, benchmarks, and Python bindings are intentionally disabled.

## Presets Summary

| Preset | Purpose |
|--------|---------|
| `dev` | Recommended CUDA development preset |
| `python-dev` | Lighter CUDA build focused on Python bindings |
| `release` | Heavier release build with benchmarks |
| `debug` | Debug-oriented CUDA build |
| `cpu-smoke` | CPU-only configure/install smoke path |

## Python Bindings

Install from the repository root:

```bash
python -m pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

The import name is `tensorcraft_ops`.

## Manual Configuration

If you prefer not to use presets, start from a single-architecture CUDA build:

```bash
cmake -B build/manual -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DTC_BUILD_TESTS=ON \
  -DTC_BUILD_BENCHMARKS=OFF \
  -DTC_BUILD_PYTHON=ON

cmake --build build/manual --parallel 2
ctest --test-dir build/manual --output-on-failure
```

Adjust `CMAKE_CUDA_ARCHITECTURES` to match your GPU.

## Compatibility Notes

- TensorCraft-HPC now supports CUDA `10.1+`
- On CUDA `10.x`, device compilation uses a CUDA-compatible dialect while host code remains on C++17
- CUDA `11.x` and `12.x` enable newer feature paths beyond the CUDA `10.x` compatibility path

## Verification

Recommended validation on a CUDA machine:

```bash
cmake --preset dev
cmake --build --preset dev --parallel 2
ctest --preset dev --output-on-failure
python -m pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for build failures, architecture issues, editable install issues, and CUDA environment problems.

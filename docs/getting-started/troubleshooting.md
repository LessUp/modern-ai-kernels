# Troubleshooting Guide

This guide covers the current common issues when building or using TensorCraft-HPC.

## Build Issues

### CUDA not found

**Typical symptom**

```text
CMake Error: Could not find CUDA
```

**What this repository does**

If CUDA is unavailable, configure can still succeed, but CMake forces:

- `TC_BUILD_TESTS=OFF`
- `TC_BUILD_BENCHMARKS=OFF`
- `TC_BUILD_PYTHON=OFF`

**What to do**

```bash
nvcc --version
cmake --preset cpu-smoke
```

If you do need CUDA, point CMake to the toolkit explicitly:

```bash
cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda
```

## Unsupported GPU architecture

**Typical symptom**

```text
nvcc fatal: Unsupported gpu architecture 'compute_XX'
```

**What to do**

Start with a single supported architecture for your machine:

```bash
cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES=75
```

If you already know your GPU architecture, use that exact value.

## Build is too heavy or runs out of memory

**Typical symptoms**

```text
nvcc fatal: Unsupported gpu architecture ...
nvcc error: ran out of memory during compilation
```

**What to do**

1. Prefer the lighter presets:
   ```bash
   cmake --preset dev
   cmake --preset python-dev
   ```
2. Reduce parallelism:
   ```bash
   cmake --build --preset dev --parallel 2
   ```
3. Limit to one architecture:
   ```bash
   cmake --preset dev -DCMAKE_CUDA_ARCHITECTURES=75
   ```

## Tests or Python bindings are unexpectedly disabled

This usually means CMake configured without usable CUDA.

Check the configure summary and look for:

```text
CUDA Enabled:   ON
Build Tests:    ON
Build Python:   ON
```

If CUDA is off, tests and Python bindings are disabled by design.

## Editable install succeeds but import fails

Use the repository root and verify the import name:

```bash
python -m pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

Also make sure you are using the same Python interpreter for both commands.

## Python environment shows `Ignoring invalid distribution ~...`

Those warnings come from broken package metadata already present in the Python environment, not from TensorCraft itself.

TensorCraft can still build successfully, but you may want to clean that environment if pip output becomes noisy.

## `ModuleNotFoundError: No module named 'tensorcraft_ops'`

**What to check**

1. Install from the repository root:
   ```bash
   python -m pip install -e .
   ```
2. Verify the import with the same interpreter:
   ```bash
   python -c "import tensorcraft_ops as tc; print(tc.__version__)"
   ```
3. If needed, inspect where pip installed the package:
   ```bash
   python -m pip show -f tensorcraft-ops
   ```

## CUDA version compatibility

| Capability | Required CUDA |
|------------|---------------|
| Basic kernels and core build | 12.8 |
| BF16-related paths | 12.8 |
| FP8-related paths | 12.8 |
| Hopper-specific features | 12.8 |

This repository now assumes the local CUDA `12.8` toolchain and no longer carries a CUDA 10.x compatibility path.

## GPU runtime path vs CI

The repository's documented local validation path is the CUDA `dev` preset plus `ctest` and Python import checks.

Current GitHub Actions mainly cover CPU configure/install smoke and packaging smoke. They do **not** replace running the real CUDA path on a GPU machine.

## Recommended validation commands

```bash
cmake --preset dev
cmake --build --preset dev --parallel 2
ctest --preset dev --output-on-failure
python -m pip install -e .
python -c "import tensorcraft_ops as tc; print(tc.__version__)"
```

## Still stuck?

When reporting an issue, include:

- `nvcc --version`
- `cmake --version`
- The preset or exact CMake command you used
- Full configure/build error output
- Your `CMAKE_CUDA_ARCHITECTURES` value, if overridden

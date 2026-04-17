# TensorCraft-HPC — Project Context

## Project Overview

**TensorCraft-HPC** is a modern C++/CUDA AI high-performance computing (HPC) kernel library. It implements core deep learning operations with **progressive optimization levels**—from naive implementations to Tensor Core-optimized kernels. The library is **header-only**, making it easy to integrate into other projects.

### Key Characteristics

- **Type**: Header-only C++/CUDA library with Python bindings
- **Purpose**: Learning, research, and production-ready GPU kernels for AI workloads
- **Project Slogan**: "Demystifying High-Performance AI Kernels with Modern C++ & CUDA"
- **Development Paradigm**: **Spec-Driven Development (SDD)** — all implementations must be traceable to specification documents under `/specs/`

### Core Features

| Category | Optimization Levels | Target Performance |
|----------|-------------------|-------------------|
| **GEMM** | Naive → Tiled → Double Buffer → Tensor Core (WMMA) | 85-95% of cuBLAS |
| **Attention** | FlashAttention, RoPE, MoE Router | 80-90% of cuDNN |
| **Normalization** | LayerNorm, RMSNorm, BatchNorm, Softmax | 90-95% of cuDNN |
| **Convolution** | Naive, Im2Col, Depthwise Separable | 75-85% of cuDNN |
| **Sparse** | CSR/CSC, SpMV, SpMM | Optimized for sparsity |
| **Quantization** | INT8, FP8 (CUDA 12.0+) | Reduced precision acceleration |

### Tech Stack

| Component | Version |
|-----------|---------|
| C++ Standard | C++17 (base), C++20/23 if available |
| CMake | 3.20+ |
| CUDA | 12.8 targeted, 11.x–13.1 compatible |
| Python | 3.8+ (for bindings via pybind11) |
| Build System | Ninja (via CMake presets) |

### GPU Architecture Support

| Architecture | SM | Tensor Core | TMA | WGMMA | Example GPUs |
|--------------|-----|-------------|-----|-------|-------------|
| Volta | 70 | ✅ | ❌ | ❌ | V100 |
| Turing | 75 | ✅ | ❌ | ❌ | RTX 2080 |
| Ampere | 80 | ✅ | ❌ | ❌ | A100, RTX 3090 |
| Ada Lovelace | 89 | ✅ | ❌ | ❌ | RTX 4090 |
| **Hopper** ⭐ | 90 | ✅ | ✅ | ✅ | H100 |
| Blackwell | 100 | ✅ | ✅ | ✅ | B200 |

---

## Directory Structure

```
modern-ai-kernels/
├── include/tensorcraft/     # Header-only library (main source)
│   ├── core/               # Core utilities, type traits, CUDA error handling
│   ├── kernels/            # GPU kernel implementations (GEMM, attention, conv, etc.)
│   └── memory/             # Memory management, Tensor class
├── src/python_ops/         # Python bindings (pybind11)
├── tests/                  # Unit tests (GoogleTest)
├── benchmarks/             # Performance benchmarks (Google Benchmark)
├── examples/               # Example code
├── specs/                  # Specification documents (Single Source of Truth for SDD)
│   ├── product/           # Product feature definitions (PRDs)
│   ├── rfc/               # Technical design documents and architecture decisions
│   ├── api/               # API specification definitions
│   ├── db/                # Data structure and memory layout specifications
│   └── testing/           # BDD test case specifications and implementation plans
├── docs/                   # Documentation site (Jekyll + Just the Docs theme)
│   ├── en/                # English documentation
│   └── zh/                # Chinese documentation (简体中文)
├── changelog/             # Development changelog
├── .github/               # GitHub workflows, issue templates, CODEOWNERS
├── CMakeLists.txt         # Main build configuration
├── CMakePresets.json      # CMake presets (dev, release, debug, python-dev, cpu-smoke, etc.)
├── pyproject.toml         # Python package configuration (scikit-build-core)
├── AGENTS.md              # AI agent workflow instructions (SDD workflow)
├── README.md              # Project overview (English)
└── README.zh-CN.md        # Project overview (Chinese)
```

---

## Building and Running

### Prerequisites

- **CUDA Toolkit**: 12.0+ (12.8 targeted)
- **CMake**: 3.20+
- **C++ Compiler**: C++17-capable (GCC 9+, Clang 9+)
- **NVIDIA GPU**: Compute capability 70+ (recommended for tests and benchmarks)
- **Python**: 3.8+ (optional, for Python bindings)
- **Ninja**: Recommended for faster builds

### CMake Presets

| Preset | Purpose | Includes |
|--------|---------|----------|
| `dev` | Recommended CUDA development preset | All kernels + tests, single arch (SM 75) |
| `python-dev` | Lighter build for Python bindings | Core kernels + bindings |
| `release` | Full release build with benchmarks | Everything + benchmarks |
| `debug` | Debug-oriented CUDA build | Debug symbols, no optimizations |
| `profile` | Release build with profiling symbols | For Nsight Systems/Compute |
| `ci` | CI/CD pipeline configuration | Tests + benchmarks |
| `cpu-smoke` | CPU-only validation | Build system only, no CUDA |

### Build Commands

```bash
# Configure and build (development)
cmake --preset dev
cmake --build --preset dev --parallel $(nproc)

# Run unit tests
ctest --preset dev --output-on-failure

# Build release version with all features
cmake --preset release
cmake --build --preset release --parallel $(nproc)

# Run benchmarks
./build/release/benchmarks/gemm_benchmark

# Install Python bindings (editable mode)
pip install -e .

# CPU-only validation (no CUDA required)
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke
```

### Custom Configuration

```bash
cmake -B build/manual -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DTC_BUILD_TESTS=ON \
  -DTC_BUILD_BENCHMARKS=ON \
  -DTC_BUILD_PYTHON=ON

cmake --build build/manual --parallel $(nproc)
```

### Documentation

```bash
# Preview documentation locally (requires Ruby + Bundler)
cd docs && bundle install
bundle exec jekyll serve --livereload
# Access at: http://localhost:4000

# Build documentation site for production
cd docs && JEKYLL_ENV=production bundle exec jekyll build
```

---

## Development Conventions

### Spec-Driven Development (SDD)

This project strictly follows Spec-Driven Development. The workflow is:

1. **Review Specs**: Read relevant documents under `/specs/` before coding
2. **Update Specs First**: Propose spec changes before code changes
3. **Implement to Spec**: Code must 100% comply with spec definitions; no gold-plating
4. **Test Against Spec**: Write tests based on spec acceptance criteria

### Code Style

| Aspect | Convention |
|--------|------------|
| C++ Standard | C++17 as base; C++20/23 auto-detected |
| Style Guide | Google C++ Style Guide (with CUDA exceptions) |
| Indentation | 4 spaces |
| Class Names | `PascalCase` |
| Function Names | `snake_case` |
| Variable Names | `snake_case` |
| Constants | `kConstantName` or `CONSTANT_NAME` |
| Template Parameters | `PascalCase` |

### CUDA Conventions

| Aspect | Convention |
|--------|------------|
| Kernel Functions | `__global__` prefix |
| Device Functions | `__device__ __forceinline__` |
| Pointer Hints | Use `__restrict__` |
| Launch Bounds | Explicitly specify `__launch_bounds__` |

### Testing Practices

- **Framework**: GoogleTest for C++ tests
- **Location**: Test files in `tests/` directory
- **Property-based testing**: Used for general correctness properties
- **Coverage**: All new features need tests; tests cover boundary conditions from specs

### Code Formatting and Linting

```bash
# Format C++/CUDA code
find include src tests examples -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format -i

# Run pre-commit hooks (includes formatting, linting)
pre-commit run --all-files
```

### Python Code Style

- **Line length**: 100 characters
- **Formatter**: Black
- **Import sorting**: isort (profile = "black")
- **Type checking**: mypy
- **Linting**: ruff (select: E, F, W, I, N, UP, B, C4)

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `AGENTS.md` | AI agent workflow instructions (SDD workflow) |
| `README.md` | Project overview (English) |
| `README.zh-CN.md` | Project overview (Chinese) |
| `CHANGELOG.md` | Version history |
| `CONTRIBUTING.md` | Contribution guidelines |
| `CMakeLists.txt` | Main CMake build configuration |
| `CMakePresets.json` | CMake configure/build/test presets |
| `pyproject.toml` | Python package configuration (scikit-build-core) |
| `.clang-format` | clang-format code style configuration |
| `.clang-tidy` | clang-tidy static analysis configuration |
| `.pre-commit-config.yaml` | Pre-commit hooks configuration |

---

## Links

- **GitHub Repository**: https://github.com/LessUp/modern-ai-kernels
- **Online Documentation**: https://lessup.github.io/modern-ai-kernels/
- **Issue Tracker**: https://github.com/LessUp/modern-ai-kernels/issues

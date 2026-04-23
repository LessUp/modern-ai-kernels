# API Specifications

> **Domain**: C++ API Contracts
> **Version**: 2.0.0
> **Status**: ✅ Implemented
> **Last Updated**: 2026-04-23

---

## Overview

This specification defines the public C++ API for TensorCraft-HPC. All implementations must conform to these interface definitions.

---

## Module Overview

| Module | Header Path | Description |
|--------|-------------|-------------|
| Core | `tensorcraft/core/` | Error handling, feature detection, type traits |
| Memory | `tensorcraft/memory/` | Tensor wrapper, memory pool, aligned vectors |
| Kernels | `tensorcraft/kernels/` | Compute kernels (GEMM, attention, etc.) |
| Python | `tensorcraft_ops` | Python bindings |

---

## ADDED Requirements

### Requirement: Error Handling API (API-001)

**User Story:** As a kernel developer, I want consistent error handling, so that I can catch and diagnose CUDA errors easily.

#### Scenario: CudaException Class
- **WHEN** a CUDA error occurs
- **THEN** the API SHALL provide `CudaException` class with file, line, and error code

```cpp
class CudaException : public std::runtime_error {
public:
    CudaException(const std::string& file, int line, cudaError_t error);
    cudaError_t error() const noexcept;
    const std::string& file() const noexcept;
    int line() const noexcept;
};
```

#### Scenario: Error Checking Macros
- **WHEN** checking CUDA calls
- **THEN** the API SHALL provide macros:
  - `TC_CUDA_CHECK(call)` — Check CUDA API call
  - `TC_CUDA_CHECK_LAST()` — Check last kernel launch error
  - `TC_CUDA_SYNC_CHECK()` — Synchronize and check all errors

---

### Requirement: Feature Detection API (API-002)

**User Story:** As a developer, I want compile-time feature detection, so that I can write portable code.

#### Scenario: C++ Version Macros
- **WHEN** compiling with different C++ standards
- **THEN** the API SHALL provide macros: `TC_CPP17`, `TC_CPP20`, `TC_CPP23`

#### Scenario: CUDA Version Macros
- **WHEN** compiling with different CUDA versions
- **THEN** the API SHALL provide macros: `TC_CUDA_VERSION`, `TC_CUDA_11`, `TC_CUDA_12`, `TC_CUDA_13`

#### Scenario: Feature Macros
- **WHEN** checking for hardware features
- **THEN** the API SHALL provide macros: `TC_HAS_WMMA`, `TC_HAS_BF16`, `TC_HAS_FP8`, `TC_HAS_TMA`, `TC_HAS_WGMMA`

#### Scenario: Runtime Functions
- **WHEN** querying device capabilities
- **THEN** the API SHALL provide functions:

```cpp
int get_cuda_runtime_version();
int get_cuda_driver_version();
std::pair<int, int> get_compute_capability(int device = 0);
bool has_tensor_cores(int device = 0);
bool has_tma(int device = 0);
```

---

### Requirement: Type Traits API (API-003)

**User Story:** As a template developer, I want type traits for numeric types, so that I can write generic kernels.

#### Scenario: Type Trait Variables
- **WHEN** querying type properties at compile time
- **THEN** the API SHALL provide: `is_half_v<T>`, `is_fp8_v<T>`, `is_floating_v<T>`, `is_numeric_v<T>`

#### Scenario: C++20 Concepts
- **WHEN** using C++20 or later
- **THEN** the API SHALL provide concepts: `Numeric<T>`, `FloatingPoint<T>`, `HalfPrecision<T>`

#### Scenario: Type Conversion
- **WHEN** converting between types
- **THEN** the API SHALL provide: `to_float<T>()`, `from_float<T>()`

#### Scenario: DataType Enumeration
- **WHEN** working with runtime type information
- **THEN** the API SHALL provide `DataType` enum: `FP32`, `FP16`, `BF16`, `FP8_E4M3`, `FP8_E5M2`, `INT8`, `INT32`, `INT64`

---

### Requirement: Tensor API (API-004)

**User Story:** As a user, I want a RAII tensor class, so that I can manage GPU memory safely.

#### Scenario: Tensor Class
- **WHEN** working with GPU tensors
- **THEN** the API SHALL provide `Tensor<T>` class:

```cpp
template<typename T>
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const std::vector<size_t>& shape);
    Tensor(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    ~Tensor();
    
    Tensor& operator=(const Tensor&) = delete;
    Tensor& operator=(Tensor&&) noexcept;
    
    T* data() noexcept;
    const T* data() const noexcept;
    size_t size() const noexcept;
    const std::vector<size_t>& shape() const noexcept;
    
    void fill(T value);
    void copy_from(const T* host_data);
    void copy_to(T* host_data) const;
    
    static Tensor zeros(const std::vector<size_t>& shape);
    static Tensor ones(const std::vector<size_t>& shape);
};
```

#### Scenario: Type Aliases
- **WHEN** using common tensor types
- **THEN** the API SHALL provide: `FloatTensor`, `HalfTensor`

#### Scenario: RAII Behavior
- **WHEN** a Tensor is destroyed
- **THEN** GPU memory SHALL be freed automatically (no leaks)

---

### Requirement: Kernel Launcher API (API-005)

**User Story:** As a user, I want simple kernel launch interfaces, so that I can call kernels without CUDA knowledge.

#### Scenario: Elementwise Operations
- **WHEN** applying elementwise operations
- **THEN** the API SHALL provide: `relu()`, `silu()`, `gelu()`, `sigmoid()`

```cpp
template<typename T>
void relu(const T* in, T* out, size_t n, cudaStream_t s = nullptr);
```

#### Scenario: Softmax Operation
- **WHEN** computing softmax
- **THEN** the API SHALL provide:

```cpp
template<typename T>
void softmax(const T* input, T* output, size_t batch_size, size_t dim,
             cudaStream_t stream = nullptr);
```

#### Scenario: Normalization Operations
- **WHEN** normalizing tensors
- **THEN** the API SHALL provide: `layernorm()`, `rmsnorm()`, `launch_batchnorm()`

#### Scenario: GEMM Operations
- **WHEN** multiplying matrices
- **THEN** the API SHALL provide:

```cpp
template<typename T>
void gemm(const T* A, const T* B, T* C, size_t M, size_t N, size_t K,
          T alpha = T(1), T beta = T(0), cudaStream_t stream = nullptr);

template<typename T>
void launch_gemm(const T* A, const T* B, T* C, int M, int N, int K,
                 T alpha, T beta, GemmVersion version,
                 cudaStream_t stream = nullptr);
```

#### Scenario: Attention Operations
- **WHEN** computing attention
- **THEN** the API SHALL provide: `flash_attention()`, `launch_rope()`, `precompute_rope_cache()`

#### Scenario: Convolution Operations
- **WHEN** computing convolutions
- **THEN** the API SHALL provide: `conv2d()`, `conv2d_depthwise()`

---

### Requirement: Python API (API-006)

**User Story:** As a Python user, I want Python bindings, so that I can use kernels from Python.

#### Scenario: Module Functions
- **WHEN** using the Python module
- **THEN** the API SHALL provide:

```python
import tensorcraft_ops as tc

tc.__version__  # str

# GEMM
def tc.gemm(A: np.ndarray, B: np.ndarray, 
            alpha: float = 1.0, beta: float = 0.0) -> np.ndarray

# Activation
def tc.relu(x: np.ndarray) -> np.ndarray
def tc.silu(x: np.ndarray) -> np.ndarray
def tc.gelu(x: np.ndarray) -> np.ndarray
def tc.sigmoid(x: np.ndarray) -> np.ndarray

# Normalization
def tc.layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                 eps: float = 1e-5) -> np.ndarray
def tc.rmsnorm(x: np.ndarray, weight: np.ndarray,
               eps: float = 1e-6) -> np.ndarray
def tc.softmax(x: np.ndarray, dim: int = -1) -> np.ndarray

# Attention
def tc.flash_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                       scale: Optional[float] = None) -> np.ndarray

# Convolution
def tc.conv2d(input: np.ndarray, weight: np.ndarray, 
              bias: Optional[np.ndarray] = None,
              stride: int = 1, padding: int = 0) -> np.ndarray
```

#### Scenario: Memory Management
- **WHEN** using Python functions
- **THEN** GPU memory SHALL be managed automatically (no explicit free required)

---

## See Also

- [Core Specifications](../core/spec.md) — Product requirements
- [Data Structures](../data-structures/spec.md) — Memory layouts

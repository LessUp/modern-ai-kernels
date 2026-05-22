# API Specification Delta: tighten-kernel-interfaces

## MODIFIED Requirements

### Requirement: Tensor API (API-004)

#### Scenario: Tensor fill behavior
- **WHEN** callers use `Tensor<T>::fill()`
- **THEN** the Tensor module SHALL route the request through the shared memory-operations module
  instead of maintaining a second fill implementation inside `tensor.hpp`

### Requirement: Kernel Launcher API (API-005)

#### Scenario: Sparse operations
- **WHEN** launching sparse matrix operations
- **THEN** the API SHALL use `CSRMatrixView<T>` or `CSRMatrix<T>` as the public seam for SpMV and
  SpMM helpers instead of exposing a separate raw-pointer bundle as the primary interface
- **AND** the launchers SHALL throw `std::invalid_argument` before any kernel launch when the
  supplied view metadata is negative or required pointers for the requested operation are missing

```cpp
template<typename T>
void launch_spmv_csr(CSRMatrixView<T> A, const T* x, T* y,
                     bool use_vector = true, cudaStream_t stream = nullptr);

template<typename T>
void launch_spmm_csr(CSRMatrixView<T> A, const T* B, T* C, int N,
                     cudaStream_t stream = nullptr);
```

#### Scenario: GEMM version selection
- **WHEN** selecting a GEMM implementation through `launch_gemm()`
- **THEN** the public `GemmVersion` enum SHALL only expose versions supported by the generic
  launcher

```cpp
enum class GemmVersion {
    Naive,
    Tiled,
    DoubleBuffer
};
```

- **AND** Tensor Core GEMM SHALL remain available through its dedicated WMMA launcher rather than a
  generic enum value that throws at runtime

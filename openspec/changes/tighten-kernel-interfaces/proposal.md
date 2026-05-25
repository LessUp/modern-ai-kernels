# Change Proposal: tighten-kernel-interfaces

## Why

TensorCraft-HPC still exposes several shallow or misleading module interfaces:

- sparse launchers accept raw pointer bundles even though `CSRMatrixView<T>` and `CSRMatrix<T>` already
  name the real seam
- `Tensor::fill()` duplicates `kernels::fill()` instead of concentrating memory-fill behavior behind
  one implementation
- `GemmVersion` exposes public variants that the generic launcher does not actually support

These interfaces reduce locality, leak implementation details across callers, and hide bugs in code
paths that are easy to miss when CUDA validation is not enabled locally.

## What Changes

- make `CSRMatrixView<T>` / `CSRMatrix<T>` the canonical seam for sparse launchers and helpers
- remove the legacy sparse pass-through path that keeps raw-pointer launch semantics public
- fail closed when sparse launchers receive invalid `CSRMatrixView<T>` metadata or required pointers
- route `Tensor::fill()` through `kernels::fill()` so fill behavior lives in one module
- remove unsupported public GEMM launcher versions so the interface matches the implementation
- align tests and accepted specs with the tightened interfaces

## Capabilities

### Modified Capabilities

- `api`: sparse launch surfaces, tensor fill behavior, and GEMM version contracts
- `architecture`: module seams for sparse operations and tensor memory operations

## Impact

- `include/tensorcraft/memory/tensor.hpp`
- `include/tensorcraft/kernels/memory_ops.hpp`
- `include/tensorcraft/kernels/sparse.hpp`
- `include/tensorcraft/kernels/gemm.hpp`
- `tests/test_sparse.cpp`
- `tests/test_attention_docs_contract.cpp`
- `openspec/specs/api/spec.md`
- `openspec/specs/architecture/spec.md`
- docs that describe the removed `GemmVersion` variants

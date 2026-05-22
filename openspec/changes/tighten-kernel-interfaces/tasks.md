# Tasks: tighten-kernel-interfaces

## 1. Spec updates

- [x] Add API deltas for the sparse seam, tensor fill contract, and supported GEMM launcher versions
- [x] Add architecture deltas describing the deeper sparse seam and unified tensor fill implementation

## 2. Sparse seam cleanup

- [x] Refactor sparse launchers so `CSRMatrixView<T>` / `CSRMatrix<T>` are the public launch seam
- [x] Remove the deprecated legacy sparse pass-through overload
- [x] Reject invalid sparse launch metadata and required-pointer omissions before kernel launch
- [x] Update sparse tests to use the tightened interface and canonical CUDA error macro

## 3. Tensor fill consolidation

- [x] Delegate `Tensor::fill()` to `kernels::fill()`
- [x] Remove the duplicate tensor-local fill kernel implementation

## 4. GEMM contract cleanup

- [x] Remove unsupported `GemmVersion` launcher variants from the public enum and switch
- [x] Update contract tests and docs/spec text to match the honest interface

## 5. Validation

- [x] Run the repository validation commands available in the local environment

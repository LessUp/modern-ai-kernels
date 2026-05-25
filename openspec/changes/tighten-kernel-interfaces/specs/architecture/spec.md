# Architecture Specification Delta: tighten-kernel-interfaces

## MODIFIED Requirements

### Requirement: Component Organization (ARCH-003)

#### Scenario: Memory management
- **WHEN** implementing tensor fill behavior
- **THEN** the `Tensor` module SHALL delegate to the shared memory-operations module so the fill
  implementation lives behind one seam

#### Scenario: Sparse operations
- **WHEN** exposing sparse compute helpers
- **THEN** `CSRMatrixView<T>` SHALL be the primary seam for launchers, with `CSRMatrix<T>` acting as
  the owning adapter for that seam

## MODIFIED Decisions

### Decision: CSRMatrix Direct MemoryPool Usage (ARCH-009)

**Consequences:**
- ✅ `CSRMatrix<T>` remains the owning adapter for sparse storage
- ✅ `CSRMatrixView<T>` becomes the non-owning launch seam for sparse kernels
- ✅ Raw pointer bundles stop leaking across callers as a parallel public interface

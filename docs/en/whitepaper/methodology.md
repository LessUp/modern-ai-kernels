# Development Methodology

This document describes the OpenSpec-driven development workflow and contribution guidelines for TensorCraft-HPC.

---

## OpenSpec Workflow

TensorCraft-HPC uses a specification-first development approach. All significant changes begin as specifications in `openspec/changes/`.

### Workflow Diagram

```mermaid
flowchart TB
    subgraph Proposal["1. Proposal Phase"]
        IDEA["Identify Need"] --> SPEC["Write Spec"]
        SPEC --> REVIEW["Submit for Review"]
    end

    subgraph Acceptance["2. Acceptance Phase"]
        REVIEW --> DISCUSS["Discussion"]
        DISCUSS -->|"Accept"| ACCEPT["Move to specs/"]
        DISCUSS -->|"Reject"| REJECT["Archive with Rationale"]
    end

    subgraph Implementation["3. Implementation Phase"]
        ACCEPT --> IMPL["Implement"]
        IMPL --> TEST["Write Tests"]
        TEST --> VERIFY["Verify Against Spec"]
    end

    subgraph Completion["4. Completion Phase"]
        VERIFY --> DOC["Update Docs"]
        DOC --> PR["Submit PR"]
        PR --> MERGE["Merge"]
    end
```

---

## OpenSpec artifact model

TensorCraft-HPC uses OpenSpec in two layers:

1. **Accepted baseline specs** in `openspec/specs/` describe the repository's current contracts and
   standards.
2. **Active changes** in `openspec/changes/<name>/` describe what is being changed, why, and how it
   will be implemented.

An implementation-facing change set typically includes:

| File | Purpose |
|------|---------|
| `proposal.md` | Why the change exists and which capabilities it modifies |
| `design.md` | Key decisions, trade-offs, and implementation guidance |
| `tasks.md` | Execution checklist and validation sequence |
| `specs/<domain>/spec.md` | Delta requirements for any affected accepted spec |

This model matters for the showcase itself: structural Pages work, branding changes, and public
workflow changes should all be traceable through OpenSpec instead of landing as unexplained edits.

---

## Contribution Guidelines

### Code Standards

| Aspect | Requirement |
|--------|-------------|
| Language | C++17, CUDA 11.0+ |
| Style | clang-format (see .clang-format) |
| Linting | clang-tidy (see .clang-tidy) |
| Documentation | Doxygen comments for public API |

### Testing Requirements

1. **Unit Tests**: All public functions must have GoogleTest tests
2. **Numerical Validation**: Compare against reference implementations
3. **Performance Tests**: Include benchmark for critical paths
4. **Edge Cases**: Test boundary conditions and error handling

### Pull Request Process

1. Create specification in `openspec/changes/`
2. Implement changes
3. Add/update tests
4. Update documentation
5. Submit PR with filled template

```markdown
## PR Template

### Specification
Link to the OpenSpec change proposal.

### Changes
Summary of implementation changes.

### Testing
- [ ] Unit tests pass
- [ ] Numerical validation passes
- [ ] Performance benchmarks run
- [ ] Documentation updated

### Performance Impact
Describe any performance changes.
```

---

## Repository Structure Conventions

### Header Files

```cpp
// include/tensorcraft/kernels/example.hpp
#pragma once

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/memory/tensor.hpp"

namespace tensorcraft::kernels {

/**
 * @brief Brief description of the kernel.
 *
 * Detailed description with usage notes.
 *
 * @param input Input tensor (M×K)
 * @param output Output tensor (M×N)
 * @param M Number of rows
 * @param N Number of columns
 *
 * @throws CudaError if kernel launch fails
 *
 * @performance O(M×N) operations, O(M×N) memory
 */
void example_kernel(
    const float* input,
    float* output,
    size_t M, size_t N
);

} // namespace tensorcraft::kernels
```

### Test Files

```cpp
// tests/kernels/example_test.cpp
#include <gtest/gtest.h>
#include "tensorcraft/kernels/example.hpp"

class ExampleKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code
    }
};

TEST_F(ExampleKernelTest, BasicCorrectness) {
    // Test implementation
}

TEST_F(ExampleKernelTest, EdgeCase) {
    // Edge case test
}
```

---

## Quality Gates

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: clang-format
        name: clang-format
        entry: clang-format -i
        types: [c++]
      - id: clang-tidy
        name: clang-tidy
        entry: clang-tidy
        types: [c++]
```

### CI pipeline

```mermaid
flowchart LR
    PR["PR Submitted"] --> FORMAT["Format Check"]
    FORMAT --> SMOKE["CPU Smoke Build"]
    SMOKE --> WHEEL["Python Wheel Build"]
    WHEEL --> DOCS["Docs / Pages Build"]
    DOCS --> REVIEW["Ready for Review"]
```

Hosted CI in this repository intentionally stops short of GPU benchmark execution. CUDA-dependent
tests and benchmark review belong on local GPU-enabled machines.

---

## Release Process

### Versioning

TensorCraft-HPC follows [SemVer](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes

### Release Checklist

1. Update `CHANGELOG.md`
2. Update version in `CMakeLists.txt`
3. Tag release: `git tag v1.2.3`
4. Push tag: `git push --tags`
5. GitHub Actions builds and publishes

---

## Documentation Standards

### API Documentation

Use Doxygen format for C++ API:

```cpp
/**
 * @brief Compute GEMM: C = α(A×B) + βC
 *
 * This function performs general matrix multiplication
 * with optional scaling factors.
 *
 * @tparam T Data type (float, half, bfloat16)
 * @param A Input matrix A (M×K), row-major
 * @param B Input matrix B (K×N), row-major
 * @param C Output matrix C (M×N), row-major
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A / rows in B
 * @param alpha Scalar multiplier for A×B (default: 1.0)
 * @param beta Scalar multiplier for C (default: 0.0)
 *
 * @throws CudaError if CUDA kernel launch fails
 *
 * @note Requires SM70+ for Tensor Core path
 *
 * @performance
 * - Compute: 2×M×N×K FLOPs
 * - Memory: O(M×K + K×N + M×N) bytes
 *
 * @example
 * ```cpp
 * gemm(A, B, C, 1024, 1024, 1024);
 * ```
 */
template<typename T>
void gemm(const T* A, const T* B, T* C,
          size_t M, size_t N, size_t K,
          T alpha = T(1), T beta = T(0));
```

### User Guide Documentation

User-facing documentation in `docs/` uses VitePress markdown with:

- **Code Groups**: `::: code-group` for multi-language examples
- **Callouts**: `::: tip`, `::: warning`, `::: info`
- **Diagrams**: Mermaid for flowcharts and sequences

---

## Showcase discipline

Because this repository is also meant to serve as a technical portfolio artifact, methodology
extends beyond code changes:

- `README.md`, `README.zh-CN.md`, and GitHub Pages should present one coherent project identity
- benchmark claims should always be paired with methodology or citations
- structural documentation changes should be represented in OpenSpec, not hidden in ad-hoc copy edits
- examples and commands should match the actual presets, files, and workflows in the repository

This discipline is part of the engineering quality of the project, not a separate marketing layer.

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/AICL-Lab/modern-ai-kernels/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AICL-Lab/modern-ai-kernels/discussions)
- **Documentation**: [Online Docs](https://aicl-lab.github.io/modern-ai-kernels/)

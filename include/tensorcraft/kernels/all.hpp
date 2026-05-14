#pragma once
/**
 * @file all.hpp
 * @brief Unified entry point for all TensorCraft-HPC kernels
 *
 * This header aggregates all kernel convenience functions, providing
 * a single include point for users who want access to all operations.
 *
 * Benefits:
 * - Locality: Single entry point for all kernels
 * - Convenience: No need to remember individual header paths
 * - Documentation: Central place to see available operations
 *
 * Usage:
 * ```cpp
 * #include "tensorcraft/kernels/all.hpp"
 * using namespace tensorcraft::kernels;
 *
 * // Now all kernel functions are available:
 * gemm(A, B, C, M, N, K);
 * relu(input, output, n);
 * softmax(input, output, rows, cols);
 * ```
 *
 * For selective includes, use individual headers:
 * ```cpp
 * #include "tensorcraft/kernels/gemm.hpp"       // Only GEMM
 * #include "tensorcraft/kernels/elementwise.hpp" // Only elementwise ops
 * ```
 */

// ============================================================================
// Core Memory Operations
// ============================================================================
#include "memory_ops.hpp"  // fill, copy_d2d

// ============================================================================
// Elementwise Operations (Activations, Arithmetic)
// ============================================================================
#include "elementwise.hpp"  // relu, silu, gelu, sigmoid, tanh_activation, vector_add, vector_mul

// ============================================================================
// Normalization Operations
// ============================================================================
#include "normalization.hpp"  // layernorm, rmsnorm, batchnorm

// ============================================================================
// Softmax
// ============================================================================
#include "softmax.hpp"  // softmax

// ============================================================================
// Matrix Operations
// ============================================================================
#include "gemm.hpp"  // gemm, transpose

// ============================================================================
// Attention Operations
// ============================================================================
#include "attention.hpp"  // flash_attention, rope, precompute_rope_cache

// ============================================================================
// Convolution Operations
// ============================================================================
#include "conv2d.hpp"  // conv2d, conv2d_depthwise

// ============================================================================
// Sparse Operations
// ============================================================================
#include "sparse.hpp"  // spmv_csr, spmv_csc

// ============================================================================
// Fused Operations
// ============================================================================
#include "fusion.hpp"  // fused operations

// ============================================================================
// Operation Registry (for declarative registration)
// ============================================================================
#include "op_registry.hpp"  // TC_REGISTER_UNARY_OP, ops::*

namespace tensorcraft {
namespace kernels {

// ============================================================================
// Kernel Categories (for documentation and organization)
// ============================================================================

/**
 * @defgroup ElementwiseOps Elementwise Operations
 * @brief Per-element operations: activations, arithmetic
 * @{
 * - relu, silu, gelu, sigmoid, tanh_activation
 * - vector_add, vector_mul
 * @}
 */

/**
 * @defgroup NormalizationOps Normalization Operations
 * @brief Layer-wise and batch-wise normalization
 * @{
 * - layernorm, rmsnorm, batchnorm
 * @}
 */

/**
 * @defgroup MatrixOps Matrix Operations
 * @brief Matrix multiplication and transformations
 * @{
 * - gemm, transpose
 * @}
 */

/**
 * @defgroup AttentionOps Attention Operations
 * @brief LLM attention kernels
 * @{
 * - flash_attention, rope, precompute_rope_cache
 * @}
 */

/**
 * @defgroup MemoryOps Memory Operations
 * @brief GPU memory utilities
 * @{
 * - fill, copy_d2d
 * @}
 */

}  // namespace kernels
}  // namespace tensorcraft

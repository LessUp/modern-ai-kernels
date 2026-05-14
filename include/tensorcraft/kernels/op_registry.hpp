#pragma once
/**
 * @file op_registry.hpp
 * @brief Operation registry for simplified kernel and binding registration
 *
 * Provides a declarative mechanism for registering elementwise operations,
 * reducing boilerplate and centralizing operation definitions.
 *
 * Benefits:
 * - Leverage: single registration point for kernel + Python binding
 * - Locality: all operation metadata in one place
 * - Extensibility: add new operations with minimal code
 *
 * Usage:
 * ```cpp
 * // Register an elementwise unary operation
 * TC_REGISTER_UNARY_OP(ReLU, "relu", "ReLU activation",
 *                      [](auto x) { return x > 0 ? x : 0; });
 *
 * // Register an elementwise binary operation
 * TC_REGISTER_BINARY_OP(Add, "add", "Element-wise addition",
 *                       [](auto a, auto b) { return a + b; });
 * ```
 */

#include "../core/type_traits.hpp"
#include "elementwise.hpp"
#include <functional>
#include <string>

namespace tensorcraft {
namespace kernels {

// ============================================================================
// Operation Metadata
// ============================================================================

/**
 * @brief Metadata for an operation
 */
struct OpMeta {
    std::string name;         ///< Python function name
    std::string description;  ///< Docstring
    std::string category;     ///< "activation", "arithmetic", "normalization", etc.
};

// ============================================================================
// Unary Operation Registration
// ============================================================================

/**
 * @brief Registration entry for unary elementwise operations
 *
 * Each entry contains:
 * - Metadata (name, description)
 * - Functor implementation
 * - Convenience function wrapper
 */
template <typename Func>
struct UnaryOpEntry {
    using functor_type = Func;

    OpMeta meta;
    Func functor;

    template <typename T>
    void launch(const T* input, T* output, size_t n, cudaStream_t stream = nullptr) const {
        launch_elementwise(input, output, n, functor, stream);
    }
};

/**
 * @brief Helper to create unary operation entries
 */
template <typename Func>
UnaryOpEntry<Func> make_unary_op(const std::string& name,
                                  const std::string& description,
                                  Func functor,
                                  const std::string& category = "activation") {
    return UnaryOpEntry<Func>{
        OpMeta{name, description, category},
        functor
    };
}

// ============================================================================
// Binary Operation Registration
// ============================================================================

/**
 * @brief Registration entry for binary elementwise operations
 */
template <typename Func>
struct BinaryOpEntry {
    using functor_type = Func;

    OpMeta meta;
    Func functor;

    template <typename T>
    void launch(const T* input1, const T* input2, T* output, size_t n,
                cudaStream_t stream = nullptr) const {
        launch_elementwise_binary(input1, input2, output, n, functor, stream);
    }
};

/**
 * @brief Helper to create binary operation entries
 */
template <typename Func>
BinaryOpEntry<Func> make_binary_op(const std::string& name,
                                    const std::string& description,
                                    Func functor,
                                    const std::string& category = "arithmetic") {
    return BinaryOpEntry<Func>{
        OpMeta{name, description, category},
        functor
    };
}

// ============================================================================
// Pre-registered Operations (for convenience)
// ============================================================================

namespace ops {

/// ReLU: max(0, x)
inline auto relu() {
    return make_unary_op("relu", "ReLU activation: max(0, x)", ReLU{});
}

/// SiLU: x * sigmoid(x)
inline auto silu() {
    return make_unary_op("silu", "SiLU (Swish) activation: x * sigmoid(x)", SiLU{});
}

/// GeLU: Gaussian Error Linear Unit
inline auto gelu() {
    return make_unary_op("gelu", "GeLU activation (approximate)", GeLU{});
}

/// Sigmoid: 1 / (1 + exp(-x))
inline auto sigmoid() {
    return make_unary_op("sigmoid", "Sigmoid activation: 1 / (1 + exp(-x))", Sigmoid{});
}

/// Tanh: hyperbolic tangent
inline auto tanh_op() {
    return make_unary_op("tanh", "Tanh activation", Tanh{});
}

/// Softplus: log(1 + exp(x))
inline auto softplus() {
    return make_unary_op("softplus", "Softplus activation: log(1 + exp(x))", Softplus{});
}

/// Add: a + b
inline auto add() {
    return make_binary_op("add", "Element-wise addition", Add{});
}

/// Sub: a - b
inline auto sub() {
    return make_binary_op("sub", "Element-wise subtraction", Sub{});
}

/// Mul: a * b
inline auto mul() {
    return make_binary_op("mul", "Element-wise multiplication", Mul{});
}

/// Div: a / b
inline auto div() {
    return make_binary_op("div", "Element-wise division", Div{});
}

/// Max: max(a, b)
inline auto max() {
    return make_binary_op("max", "Element-wise maximum", Max{});
}

/// Min: min(a, b)
inline auto min() {
    return make_binary_op("min", "Element-wise minimum", Min{});
}

}  // namespace ops

// ============================================================================
// Registration Macros (for compile-time registration)
// ============================================================================

/**
 * @brief Register a unary operation
 *
 * Creates both the functor and the convenience function.
 * Usage: TC_REGISTER_UNARY_OP(MyOp, "my_op", "Description", my_func);
 */
#define TC_REGISTER_UNARY_OP(Name, py_name, desc, func_body) \
    struct Name { \
        template <typename T> \
        TC_DEVICE_INLINE T operator()(T x) const func_body \
    }; \
    template <typename T> \
    void Name##_launch(const T* input, T* output, size_t n, cudaStream_t stream = nullptr) { \
        launch_elementwise(input, output, n, Name{}, stream); \
    }

/**
 * @brief Register a binary operation
 */
#define TC_REGISTER_BINARY_OP(Name, py_name, desc, func_body) \
    struct Name { \
        template <typename T> \
        TC_DEVICE_INLINE T operator()(T a, T b) const func_body \
    }; \
    template <typename T> \
    void Name##_launch(const T* a, const T* b, T* output, size_t n, cudaStream_t stream = nullptr) { \
        launch_elementwise_binary(a, b, output, n, Name{}, stream); \
    }

}  // namespace kernels
}  // namespace tensorcraft

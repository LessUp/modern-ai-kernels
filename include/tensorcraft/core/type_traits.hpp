#pragma once
/**
 * @file type_traits.hpp
 * @brief Type traits and concepts for numeric types
 * 
 * Provides type traits for half-precision types and numeric concepts
 * with C++20 Concepts when available, falling back to SFINAE for C++17.
 */

#include "features.hpp"
#include <type_traits>
#include <cstdint>

// CUDA half-precision types
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#if defined(TC_HAS_FP8)
#include <cuda_fp8.h>
#endif

namespace tensorcraft {

// ============================================================================
// Half Precision Type Traits
// ============================================================================

/**
 * @brief Type trait to check if T is a half-precision floating point type
 */
template<typename T>
struct is_half : std::false_type {};

template<>
struct is_half<__half> : std::true_type {};

template<>
struct is_half<__half2> : std::true_type {};

template<>
struct is_half<__nv_bfloat16> : std::true_type {};

template<>
struct is_half<__nv_bfloat162> : std::true_type {};

template<typename T>
inline constexpr bool is_half_v = is_half<T>::value;

// ============================================================================
// FP8 Type Traits (CUDA 12.0+)
// ============================================================================

#if defined(TC_HAS_FP8)
template<typename T>
struct is_fp8 : std::false_type {};

template<>
struct is_fp8<__nv_fp8_e4m3> : std::true_type {};

template<>
struct is_fp8<__nv_fp8_e5m2> : std::true_type {};

template<typename T>
inline constexpr bool is_fp8_v = is_fp8<T>::value;
#else
template<typename T>
struct is_fp8 : std::false_type {};

template<typename T>
inline constexpr bool is_fp8_v = false;
#endif

// ============================================================================
// Floating Point Type Traits
// ============================================================================

/**
 * @brief Type trait to check if T is any floating point type (including half)
 */
template<typename T>
struct is_floating : std::bool_constant<
    std::is_floating_point_v<T> || is_half_v<T> || is_fp8_v<T>
> {};

template<typename T>
inline constexpr bool is_floating_v = is_floating<T>::value;

// ============================================================================
// Numeric Type Traits
// ============================================================================

/**
 * @brief Type trait to check if T is a numeric type (arithmetic or half)
 */
template<typename T>
struct is_numeric : std::bool_constant<
    std::is_arithmetic_v<T> || is_half_v<T> || is_fp8_v<T>
> {};

template<typename T>
inline constexpr bool is_numeric_v = is_numeric<T>::value;

// ============================================================================
// C++20 Concepts (with C++17 SFINAE fallback)
// ============================================================================

#if defined(TC_CPP20)

/**
 * @brief Concept for numeric types
 */
template<typename T>
concept Numeric = is_numeric_v<T>;

/**
 * @brief Concept for floating point types (including half precision)
 */
template<typename T>
concept FloatingPoint = is_floating_v<T>;

/**
 * @brief Concept for half precision types
 */
template<typename T>
concept HalfPrecision = is_half_v<T>;

/**
 * @brief Concept for standard floating point types
 */
template<typename T>
concept StandardFloat = std::is_floating_point_v<T>;

/**
 * @brief Concept for integral types
 */
template<typename T>
concept Integral = std::is_integral_v<T>;

#else // C++17 SFINAE fallback

// Enable if numeric
template<typename T, typename = void>
struct enable_if_numeric {};

template<typename T>
struct enable_if_numeric<T, std::enable_if_t<is_numeric_v<T>>> {
    using type = T;
};

template<typename T>
using enable_if_numeric_t = typename enable_if_numeric<T>::type;

// Enable if floating
template<typename T, typename = void>
struct enable_if_floating {};

template<typename T>
struct enable_if_floating<T, std::enable_if_t<is_floating_v<T>>> {
    using type = T;
};

template<typename T>
using enable_if_floating_t = typename enable_if_floating<T>::type;

#endif // TC_CPP20

// ============================================================================
// Type Size Traits
// ============================================================================

/**
 * @brief Get the size in bits of a numeric type
 */
template<typename T>
struct type_bits : std::integral_constant<size_t, sizeof(T) * 8> {};

template<>
struct type_bits<__half> : std::integral_constant<size_t, 16> {};

template<>
struct type_bits<__nv_bfloat16> : std::integral_constant<size_t, 16> {};

#if defined(TC_HAS_FP8)
template<>
struct type_bits<__nv_fp8_e4m3> : std::integral_constant<size_t, 8> {};

template<>
struct type_bits<__nv_fp8_e5m2> : std::integral_constant<size_t, 8> {};
#endif

template<typename T>
inline constexpr size_t type_bits_v = type_bits<T>::value;

// ============================================================================
// Type Conversion Utilities
// ============================================================================

/**
 * @brief Convert value to float for computation
 */
template<typename T>
TC_HOST_DEVICE_INLINE float to_float(T val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __half2float(val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __bfloat162float(val);
    } else {
        return static_cast<float>(val);
    }
}

/**
 * @brief Convert float to target type
 */
template<typename T>
TC_HOST_DEVICE_INLINE T from_float(float val) {
    if constexpr (std::is_same_v<T, __half>) {
        return __float2half(val);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return __float2bfloat16(val);
    } else {
        return static_cast<T>(val);
    }
}

// ============================================================================
// Data Type Enumeration
// ============================================================================

/**
 * @brief Enumeration of supported data types
 */
enum class DataType {
    FP32,
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    INT8,
    INT32,
    INT64
};

/**
 * @brief Get DataType enum from C++ type
 */
template<typename T>
constexpr DataType get_dtype() {
    if constexpr (std::is_same_v<T, float>) {
        return DataType::FP32;
    } else if constexpr (std::is_same_v<T, __half>) {
        return DataType::FP16;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return DataType::BF16;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return DataType::INT8;
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return DataType::INT32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
        return DataType::INT64;
    } else {
        return DataType::FP32;  // Default
    }
}

/**
 * @brief Get size in bytes for DataType
 */
constexpr size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:     return 4;
        case DataType::FP16:     return 2;
        case DataType::BF16:     return 2;
        case DataType::FP8_E4M3: return 1;
        case DataType::FP8_E5M2: return 1;
        case DataType::INT8:     return 1;
        case DataType::INT32:    return 4;
        case DataType::INT64:    return 8;
        default:                 return 4;
    }
}

} // namespace tensorcraft

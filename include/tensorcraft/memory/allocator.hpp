#pragma once
/**
 * @file allocator.hpp
 * @brief Memory allocator abstractions for GPU memory management
 *
 * Provides a Seam for memory allocation strategies, enabling:
 * - Pool-based allocation (default, via MemoryPool)
 * - Direct allocation (cudaMalloc/cudaFree)
 * - Custom allocators for testing or special use cases
 *
 * This module creates a Seam between Tensor and memory allocation,
 * allowing different allocation strategies without modifying Tensor.
 */

#include "../core/cuda_check.hpp"

namespace tensorcraft {

// ============================================================================
// Allocator Concept (implicit, via duck typing)
// ============================================================================
//
// An Allocator must provide:
//   void* allocate(size_t bytes)    - Allocate `bytes` of GPU memory
//   void deallocate(void* ptr)      - Free previously allocated memory
//
// Allocators may be stateless (singleton-like) or stateful.
// Stateless allocators benefit from empty base optimization.

// ============================================================================
// Direct Allocator - uses cudaMalloc/cudaFree directly
// ============================================================================

/**
 * @brief Allocator that uses CUDA directly (no pooling)
 *
 * Use when:
 * - Memory pool overhead is undesirable
 * - Need precise control over allocation timing
 * - Testing memory behavior in isolation
 */
class DirectAllocator {
public:
    void* allocate(size_t bytes) {
        if (bytes == 0)
            return nullptr;
        void* ptr = nullptr;
        TC_CUDA_CHECK(cudaMalloc(&ptr, bytes));
        return ptr;
    }

    void deallocate(void* ptr) {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    /// Get singleton instance (for convenience)
    static DirectAllocator& instance() {
        static DirectAllocator alloc;
        return alloc;
    }
};

// ============================================================================
// Pool Allocator - uses MemoryPool singleton
// ============================================================================

/**
 * @brief Internal MemoryPool interface for PoolAllocator
 *
 * This is a minimal interface declaration. The full MemoryPool
 * implementation is in memory_pool.hpp.
 */
class MemoryPool;

namespace detail {

/// MemoryPool accessor functions (implemented in memory_pool.hpp)
void* memory_pool_allocate(size_t bytes);
void memory_pool_deallocate(void* ptr);

}  // namespace detail

/**
 * @brief Allocator that uses the global MemoryPool
 *
 * This is the default allocator for Tensor, providing:
 * - Reduced allocation overhead via caching
 * - Thread safety
 * - Automatic memory reuse
 */
class PoolAllocator {
public:
    void* allocate(size_t bytes) {
        return detail::memory_pool_allocate(bytes);
    }

    void deallocate(void* ptr) {
        detail::memory_pool_deallocate(ptr);
    }

    /// Get singleton instance (for convenience)
    static PoolAllocator& instance() {
        static PoolAllocator alloc;
        return alloc;
    }
};

// ============================================================================
// Allocator Traits
// ============================================================================

/**
 * @brief Traits for allocator behavior
 *
 * Specialize for custom allocators to control:
 * - Whether allocator is stateless (can use EBO)
 * - Thread safety guarantees
 */
template <typename Alloc>
struct AllocatorTraits {
    /// True if allocator has no state (can be optimized away)
    static constexpr bool is_stateless = false;

    /// True if allocator is thread-safe
    static constexpr bool is_thread_safe = false;
};

// DirectAllocator is stateless and thread-safe (CUDA calls are)
template <>
struct AllocatorTraits<DirectAllocator> {
    static constexpr bool is_stateless = true;
    static constexpr bool is_thread_safe = true;
};

// PoolAllocator is stateless (uses singleton) and thread-safe
template <>
struct AllocatorTraits<PoolAllocator> {
    static constexpr bool is_stateless = true;
    static constexpr bool is_thread_safe = true;
};

// ============================================================================
// Allocator-aware memory management helper
// ============================================================================

/**
 * @brief RAII wrapper for allocator-allocated memory
 *
 * @tparam T Element type
 * @tparam Allocator Allocator type
 *
 * Provides scoped ownership of memory allocated via the allocator.
 */
template <typename T, typename Allocator = PoolAllocator>
class AllocatorPtr {
public:
    AllocatorPtr() = default;

    explicit AllocatorPtr(size_t count, Allocator& alloc = Allocator::instance())
        : alloc_(&alloc), ptr_(alloc.allocate(count * sizeof(T))), count_(count) {}

    ~AllocatorPtr() {
        if (ptr_ && alloc_) {
            alloc_->deallocate(ptr_);
        }
    }

    // Move only
    AllocatorPtr(AllocatorPtr&& other) noexcept
        : alloc_(other.alloc_), ptr_(other.ptr_), count_(other.count_) {
        other.alloc_ = nullptr;
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    AllocatorPtr& operator=(AllocatorPtr&& other) noexcept {
        if (this != &other) {
            if (ptr_ && alloc_) {
                alloc_->deallocate(ptr_);
            }
            alloc_ = other.alloc_;
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.alloc_ = nullptr;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    AllocatorPtr(const AllocatorPtr&) = delete;
    AllocatorPtr& operator=(const AllocatorPtr&) = delete;

    T* get() { return static_cast<T*>(ptr_); }
    const T* get() const { return static_cast<const T*>(ptr_); }
    size_t count() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

    explicit operator bool() const { return ptr_ != nullptr; }

private:
    Allocator* alloc_ = nullptr;
    void* ptr_ = nullptr;
    size_t count_ = 0;
};

}  // namespace tensorcraft

# Memory Module API

The Memory module provides memory management utilities including aligned vectors for vectorized access, RAII tensor wrappers, and a thread-safe memory pool.

## Headers

| Header | Description |
|--------|-------------|
| `aligned_vector.hpp` | Aligned vector types for vectorized memory access |
| `tensor.hpp` | RAII-style GPU tensor wrapper |
| `memory_pool.hpp` | Thread-safe GPU memory pool |

---

## aligned_vector.hpp

Aligned vector types enabling efficient vectorized loads/stores on GPU.

### AlignedVector

```cpp
namespace tensorcraft;

template<typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector {
    T val[N];
    
    // Number of elements
    static constexpr int size = N;
    
    // Size in bytes
    static constexpr size_t byte_size = sizeof(T) * N;
    
    // Element access
    TC_HOST_DEVICE_INLINE T& operator[](int i);
    TC_HOST_DEVICE_INLINE const T& operator[](int i) const;
    
    // Pointer access
    TC_HOST_DEVICE_INLINE T* data();
    TC_HOST_DEVICE_INLINE const T* data() const;
    
    // Fill with value
    TC_HOST_DEVICE_INLINE void fill(T value);
    
    // Zero initialization
    TC_HOST_DEVICE_INLINE void zero();
};
```

### Type Aliases

```cpp
// Generic vector types
template<typename T> using Vec2 = AlignedVector<T, 2>;
template<typename T> using Vec4 = AlignedVector<T, 4>;
template<typename T> using Vec8 = AlignedVector<T, 8>;

// Float vectors
using float2_t = Vec2<float>;
using float4_t = Vec4<float>;

// Half vectors
using half2_t = Vec2<__half>;
using half4_t = Vec4<__half>;
using half8_t = Vec8<__half>;

// Integer vectors
using int2_t = Vec2<int>;
using int4_t = Vec4<int>;
using int8_v = Vec8<int8_t>;
```

### Utility Functions

```cpp
// Load aligned vector from memory
template<typename T, int N>
TC_DEVICE_INLINE AlignedVector<T, N> load_vector(const T* ptr);

// Store aligned vector to memory
template<typename T, int N>
TC_DEVICE_INLINE void store_vector(T* ptr, const AlignedVector<T, N>& vec);

// Check if pointer is aligned
template<typename T, int N>
TC_HOST_DEVICE_INLINE bool is_aligned(const T* ptr);

// Get optimal vector size for type
template<typename T>
constexpr int optimal_vec_size();
// Returns: 8 for 1-byte types, 8 for 2-byte, 4 for 4-byte, 2 for 8-byte
```

### Vector Operations

```cpp
// Element-wise addition
template<typename T, int N>
TC_DEVICE_INLINE AlignedVector<T, N> operator+(const AlignedVector<T, N>& a, const AlignedVector<T, N>& b);

// Element-wise subtraction
template<typename T, int N>
TC_DEVICE_INLINE AlignedVector<T, N> operator-(const AlignedVector<T, N>& a, const AlignedVector<T, N>& b);

// Element-wise multiplication
template<typename T, int N>
TC_DEVICE_INLINE AlignedVector<T, N> operator*(const AlignedVector<T, N>& a, const AlignedVector<T, N>& b);

// Scalar multiplication
template<typename T, int N>
TC_DEVICE_INLINE AlignedVector<T, N> operator*(const AlignedVector<T, N>& a, T scalar);

// Fused multiply-add
template<typename T, int N>
TC_DEVICE_INLINE AlignedVector<T, N> fma(const AlignedVector<T, N>& a, const AlignedVector<T, N>& b, const AlignedVector<T, N>& c);
```

---

## tensor.hpp

RAII-style GPU tensor with automatic memory management.

### Tensor Class

```cpp
namespace tensorcraft;

template<typename T>
class Tensor {
public:
    using value_type = T;
    using size_type = size_t;
    using shape_type = std::vector<size_t>;
    
    // Constructors
    Tensor() = default;
    explicit Tensor(const shape_type& shape);
    Tensor(std::initializer_list<size_t> shape);
    explicit Tensor(size_t n);                    // 1D
    Tensor(size_t rows, size_t cols);             // 2D
    Tensor(size_t d0, size_t d1, size_t d2);      // 3D
    Tensor(size_t d0, size_t d1, size_t d2, size_t d3);  // 4D
    
    // Destructor - automatically frees GPU memory
    ~Tensor();
    
    // Move semantics (no copy)
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    
    // Accessors
    T* data();
    const T* data() const;
    size_type size() const;
    size_type bytes() const;
    const shape_type& shape() const;
    const shape_type& strides() const;
    size_t ndim() const;
    size_t dim(size_t i) const;
    bool empty() const;
    bool is_contiguous() const;
    
    // Data transfer
    void copy_from_host(const T* host_data);
    void copy_from_host(const std::vector<T>& host_data);
    void copy_to_host(T* host_data) const;
    std::vector<T> to_host() const;
    
    // Async data transfer
    void copy_from_host_async(const T* host_data, cudaStream_t stream);
    void copy_to_host_async(T* host_data, cudaStream_t stream) const;
    
    // Memory operations
    void fill(T value);
    void zero();
    Tensor clone() const;
    Tensor& reshape(const shape_type& new_shape);
    
    // Factory methods
    static Tensor from_host(const T* data, const shape_type& shape);
    static Tensor from_host(const std::vector<T>& data, const shape_type& shape);
    static Tensor zeros(const shape_type& shape);
    static Tensor ones(const shape_type& shape);
};
```

### Type Aliases

```cpp
using FloatTensor = Tensor<float>;
using HalfTensor = Tensor<__half>;
using IntTensor = Tensor<int>;
using Int8Tensor = Tensor<int8_t>;
#if defined(TC_HAS_BF16)
using BFloat16Tensor = Tensor<__nv_bfloat16>;
#endif
```

### Example

```cpp
#include "tensorcraft/memory/tensor.hpp"

using namespace tensorcraft;

// Create tensors
FloatTensor A({128, 256});  // 2D tensor
FloatTensor B({256, 512});
FloatTensor C({128, 512});

// Initialize from host
std::vector<float> h_a(128 * 256, 1.0f);
A.copy_from_host(h_a);

// Or use factory method
auto B = FloatTensor::zeros({256, 512});

// Fill with value
C.zero();

// Get data back
auto result = C.to_host();
```

---

## memory_pool.hpp

Thread-safe GPU memory pool for reducing allocation overhead.

### MemoryPool Class

```cpp
namespace tensorcraft;

class MemoryPool {
public:
    // Alignment for all allocations
    static constexpr size_t ALIGNMENT = 256;
    
    // Get singleton instance
    static MemoryPool& instance();
    
    // Allocate memory from pool
    void* allocate(size_t bytes);
    
    // Return memory to pool
    void deallocate(void* ptr);
    
    // Free all cached (inactive) blocks
    void clear();
    
    // Trim cache to reduce memory usage
    void trim(size_t max_cached_bytes = 0);
    
    // Get statistics
    struct Stats {
        size_t allocations = 0;
        size_t deallocations = 0;
        size_t cache_hits = 0;
        size_t total_allocated = 0;
        size_t total_freed = 0;
    };
    Stats get_stats() const;
    
    // Get current cached memory size
    size_t cached_bytes() const;
};
```

### PoolPtr (RAII Wrapper)

```cpp
namespace tensorcraft;

template<typename T>
class PoolPtr {
public:
    PoolPtr() = default;
    explicit PoolPtr(size_t count);
    ~PoolPtr();
    
    // Move only
    PoolPtr(PoolPtr&& other) noexcept;
    PoolPtr& operator=(PoolPtr&& other) noexcept;
    PoolPtr(const PoolPtr&) = delete;
    PoolPtr& operator=(const PoolPtr&) = delete;
    
    T* get();
    const T* get() const;
    size_t count() const;
    size_t bytes() const;
    
    T* operator->();
    const T* operator->() const;
    
    explicit operator bool() const;
};
```

### Pinned Memory

```cpp
namespace tensorcraft;

class PinnedMemory {
public:
    static void* allocate(size_t size);
    static void deallocate(void* ptr);
};

template<typename T>
class PinnedPtr {
public:
    PinnedPtr() = default;
    explicit PinnedPtr(size_t count);
    ~PinnedPtr();
    
    // Move only, same interface as PoolPtr
    T* get();
    const T* get() const;
    T& operator[](size_t i);
    const T& operator[](size_t i) const;
    size_t count() const;
    
    explicit operator bool() const;
};
```

### Example

```cpp
#include "tensorcraft/memory/memory_pool.hpp"

using namespace tensorcraft;

// Using the pool directly
auto& pool = MemoryPool::instance();
void* ptr = pool.allocate(1024 * sizeof(float));
// ... use ptr ...
pool.deallocate(ptr);

// Using RAII wrapper
PoolPtr<float> buffer(1024);
// buffer automatically returned to pool on destruction

// Pinned memory for faster transfers
PinnedPtr<float> host_data(1024);
cudaMemcpy(device_ptr, host_data.get(), 1024 * sizeof(float), cudaMemcpyHostToDevice);
```

#pragma once
/**
 * @file sparse.hpp
 * @brief Sparse matrix operations (CSR/CSC formats, SpMV, SpMM)
 *
 * Provides sparse matrix formats with RAII memory management,
 * consistent with the Tensor class design philosophy.
 */

#include <stdexcept>
#include <string>

#include "../core/cuda_check.hpp"
#include "../core/features.hpp"
#include "../core/type_traits.hpp"
#include "../memory/memory_pool.hpp"

namespace tensorcraft {
namespace kernels {

// ============================================================================
// CSR (Compressed Sparse Row) Format - Non-Owning View
// ============================================================================

/**
 * @brief Non-owning view of CSR sparse matrix data
 *
 * Used for kernel launches where memory is managed externally.
 * Provides a safe interface for passing sparse matrix data to kernels.
 */
template <typename T>
struct CSRMatrixView {
    const T* values;         // Non-zero values [nnz]
    const int* col_indices;  // Column indices [nnz]
    const int* row_ptrs;     // Row pointers [rows + 1]
    int rows;
    int cols;
    int nnz;
};

// ============================================================================
// CSC (Compressed Sparse Column) Format - Non-Owning View
// ============================================================================

/**
 * @brief Non-owning view of CSC sparse matrix data
 */
template <typename T>
struct CSCMatrixView {
    const T* values;         // Non-zero values [nnz]
    const int* row_indices;  // Row indices [nnz]
    const int* col_ptrs;     // Column pointers [cols + 1]
    int rows;
    int cols;
    int nnz;
};

// ============================================================================
// CSR Sparse Matrix with RAII Memory Management
// ============================================================================

/**
 * @brief CSR sparse matrix with automatic memory management
 *
 * This class provides RAII-style memory management for sparse matrices,
 * consistent with the Tensor class design. Memory is allocated from
 * MemoryPool for reduced allocation overhead.
 *
 * @tparam T Element type (float, half, etc.)
 */
template <typename T>
class CSRMatrix {
public:
    using value_type = T;

    /// Default constructor (empty matrix)
    CSRMatrix() = default;

    /**
     * @brief Construct CSR matrix with given dimensions
     * @param rows Number of rows
     * @param cols Number of columns
     * @param nnz Number of non-zero elements
     * @param zero_init If true, initialize all arrays to zero (default: false)
     */
    CSRMatrix(int rows, int cols, int nnz, bool zero_init = false)
        : rows_(rows), cols_(cols), nnz_(nnz) {
        if (nnz > 0) {
            values_ = static_cast<T*>(MemoryPool::instance().allocate(nnz * sizeof(T)));
            col_indices_ = static_cast<int*>(MemoryPool::instance().allocate(nnz * sizeof(int)));
            if (zero_init) {
                TC_CUDA_CHECK(cudaMemset(values_, 0, nnz * sizeof(T)));
                TC_CUDA_CHECK(cudaMemset(col_indices_, 0, nnz * sizeof(int)));
            }
        }
        if (rows > 0) {
            row_ptrs_ = static_cast<int*>(MemoryPool::instance().allocate((rows + 1) * sizeof(int)));
            if (zero_init) {
                TC_CUDA_CHECK(cudaMemset(row_ptrs_, 0, (rows + 1) * sizeof(int)));
            }
        }
    }

    /// Destructor - returns memory to pool
    ~CSRMatrix() {
        if (values_) {
            MemoryPool::instance().deallocate(values_);
        }
        if (col_indices_) {
            MemoryPool::instance().deallocate(col_indices_);
        }
        if (row_ptrs_) {
            MemoryPool::instance().deallocate(row_ptrs_);
        }
    }

    // Move only
    CSRMatrix(CSRMatrix&& other) noexcept
        : values_(other.values_),
          col_indices_(other.col_indices_),
          row_ptrs_(other.row_ptrs_),
          rows_(other.rows_),
          cols_(other.cols_),
          nnz_(other.nnz_) {
        other.values_ = nullptr;
        other.col_indices_ = nullptr;
        other.row_ptrs_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
        other.nnz_ = 0;
    }

    CSRMatrix& operator=(CSRMatrix&& other) noexcept {
        if (this != &other) {
            if (values_) MemoryPool::instance().deallocate(values_);
            if (col_indices_) MemoryPool::instance().deallocate(col_indices_);
            if (row_ptrs_) MemoryPool::instance().deallocate(row_ptrs_);

            values_ = other.values_;
            col_indices_ = other.col_indices_;
            row_ptrs_ = other.row_ptrs_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            nnz_ = other.nnz_;

            other.values_ = nullptr;
            other.col_indices_ = nullptr;
            other.row_ptrs_ = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
            other.nnz_ = 0;
        }
        return *this;
    }

    CSRMatrix(const CSRMatrix&) = delete;
    CSRMatrix& operator=(const CSRMatrix&) = delete;

    // Accessors
    T* values() { return values_; }
    const T* values() const { return values_; }

    int* col_indices() { return col_indices_; }
    const int* col_indices() const { return col_indices_; }

    int* row_ptrs() { return row_ptrs_; }
    const int* row_ptrs() const { return row_ptrs_; }

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int nnz() const { return nnz_; }

    /// Get a view for kernel launches
    CSRMatrixView<T> view() const {
        return CSRMatrixView<T>{values_, col_indices_, row_ptrs_, rows_, cols_, nnz_};
    }

    /// Check if matrix is empty
    bool empty() const { return nnz_ == 0; }

private:
    T* values_ = nullptr;
    int* col_indices_ = nullptr;
    int* row_ptrs_ = nullptr;
    int rows_ = 0;
    int cols_ = 0;
    int nnz_ = 0;
};

// ============================================================================
// Dense to CSR Conversion
// ============================================================================

template <typename T>
__global__ void count_nnz_per_row_kernel(const T* TC_RESTRICT dense, int* TC_RESTRICT row_counts,
                                         int rows, int cols, T threshold) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;

    int count = 0;
    for (int col = 0; col < cols; ++col) {
        T val = dense[row * cols + col];
        if (fabsf(to_float(val)) > to_float(threshold)) {
            ++count;
        }
    }
    row_counts[row] = count;
}

template <typename T>
__global__ void dense_to_csr_kernel(const T* TC_RESTRICT dense, T* TC_RESTRICT values,
                                    int* TC_RESTRICT col_indices, const int* TC_RESTRICT row_ptrs,
                                    int rows, int cols, T threshold) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;

    int write_idx = row_ptrs[row];
    for (int col = 0; col < cols; ++col) {
        T val = dense[row * cols + col];
        if (fabsf(to_float(val)) > to_float(threshold)) {
            values[write_idx] = val;
            col_indices[write_idx] = col;
            ++write_idx;
        }
    }
}

// ============================================================================
// CSR to Dense Conversion
// ============================================================================

template <typename T>
__global__ void csr_to_dense_kernel(const T* TC_RESTRICT values, const int* TC_RESTRICT col_indices,
                                    const int* TC_RESTRICT row_ptrs, T* TC_RESTRICT dense, int rows,
                                    int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;

    // Initialize row to zero
    for (int col = 0; col < cols; ++col) {
        dense[row * cols + col] = T(0);
    }

    // Fill non-zero values
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];
    for (int i = start; i < end; ++i) {
        int col = col_indices[i];
        dense[row * cols + col] = values[i];
    }
}

// ============================================================================
// SpMV (Sparse Matrix-Vector Multiplication)
// ============================================================================

/**
 * @brief CSR SpMV kernel: y = A * x
 * @note Assumes col_indices are valid (within [0, cols)). Invalid indices may cause undefined
 * behavior.
 */
template <typename T>
__global__ void spmv_csr_kernel(const T* TC_RESTRICT values, const int* TC_RESTRICT col_indices,
                                const int* TC_RESTRICT row_ptrs, const T* TC_RESTRICT x,
                                T* TC_RESTRICT y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;

    float sum = 0.0f;
    int start = row_ptrs[row];
    int end = row_ptrs[row + 1];

    for (int i = start; i < end; ++i) {
        int col = col_indices[i];
        // Bounds check in debug builds
        assert(col >= 0 && col < cols && "CSR col_indices out of bounds");
        sum += to_float(values[i]) * to_float(x[col]);
    }

    y[row] = from_float<T>(sum);
}

/**
 * @brief Vectorized CSR SpMV using warp reduction
 */
template <typename T, int WARP_SIZE = 32>
__global__ void spmv_csr_vector_kernel(const T* TC_RESTRICT values,
                                       const int* TC_RESTRICT col_indices,
                                       const int* TC_RESTRICT row_ptrs, const T* TC_RESTRICT x,
                                       T* TC_RESTRICT y, int rows) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (warp_id >= rows)
        return;

    int start = row_ptrs[warp_id];
    int end = row_ptrs[warp_id + 1];

    float sum = 0.0f;
    for (int i = start + lane; i < end; i += WARP_SIZE) {
        int col = col_indices[i];
        sum += to_float(values[i]) * to_float(x[col]);
    }

// Warp reduction
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[warp_id] = from_float<T>(sum);
    }
}

// ============================================================================
// SpMM (Sparse Matrix-Matrix Multiplication)
// ============================================================================

/**
 * @brief CSR SpMM kernel: C = A * B
 * A: sparse [M, K], B: dense [K, N], C: dense [M, N]
 */
template <typename T>
__global__ void spmm_csr_kernel(const T* TC_RESTRICT A_values, const int* TC_RESTRICT A_col_indices,
                                const int* TC_RESTRICT A_row_ptrs, const T* TC_RESTRICT B,
                                T* TC_RESTRICT C, int M, int K, int N) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;

    float sum = 0.0f;
    int start = A_row_ptrs[row];
    int end = A_row_ptrs[row + 1];

    for (int i = start; i < end; ++i) {
        int k = A_col_indices[i];
        sum += to_float(A_values[i]) * to_float(B[k * N + col]);
    }

    C[row * N + col] = from_float<T>(sum);
}

/**
 * @brief Tiled CSR SpMM for better memory access
 */
template <typename T, int TILE_N = 32>
__global__ void spmm_csr_tiled_kernel(const T* TC_RESTRICT A_values,
                                      const int* TC_RESTRICT A_col_indices,
                                      const int* TC_RESTRICT A_row_ptrs, const T* TC_RESTRICT B,
                                      T* TC_RESTRICT C, int M, int K, int N) {
    __shared__ float B_shared[32][TILE_N + 1];  // +1 to avoid bank conflicts

    int row = blockIdx.y;
    int tile_col = blockIdx.x * TILE_N;
    int tx = threadIdx.x;

    if (row >= M)
        return;

    float sums[TILE_N / 32];
#pragma unroll
    for (int i = 0; i < TILE_N / 32; ++i) {
        sums[i] = 0.0f;
    }

    int start = A_row_ptrs[row];
    int end = A_row_ptrs[row + 1];

    // Process in chunks of 32 K values
    for (int k_base = 0; k_base < K; k_base += 32) {
        // Load B tile into shared memory
        for (int n = 0; n < TILE_N && tile_col + n < N; ++n) {
            int k = k_base + tx;
            B_shared[tx][n] = (k < K) ? to_float(B[k * N + tile_col + n]) : 0.0f;
        }
        __syncthreads();

        // Accumulate
        for (int i = start; i < end; ++i) {
            int k = A_col_indices[i];
            if (k >= k_base && k < k_base + 32) {
                float a_val = to_float(A_values[i]);
                int k_local = k - k_base;
                for (int n = 0; n < TILE_N / 32; ++n) {
                    int col = tx + n * 32;
                    if (tile_col + col < N) {
                        sums[n] += a_val * B_shared[k_local][col];
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write results
    for (int n = 0; n < TILE_N / 32; ++n) {
        int col = tile_col + tx + n * 32;
        if (col < N) {
            C[row * N + col] = from_float<T>(sums[n]);
        }
    }
}

// ============================================================================
// Launcher Functions
// ============================================================================

namespace detail {

template <typename T>
void validate_csr_view(CSRMatrixView<T> A, const char* op_name) {
    if (A.rows < 0) {
        throw std::invalid_argument(std::string(op_name) + " requires rows >= 0");
    }
    if (A.cols < 0) {
        throw std::invalid_argument(std::string(op_name) + " requires cols >= 0");
    }
    if (A.nnz < 0) {
        throw std::invalid_argument(std::string(op_name) + " requires nnz >= 0");
    }
    if (A.rows > 0 && A.row_ptrs == nullptr) {
        throw std::invalid_argument(std::string(op_name) + " requires row_ptrs when rows > 0");
    }
    if (A.nnz > 0 && (A.values == nullptr || A.col_indices == nullptr)) {
        throw std::invalid_argument(
            std::string(op_name) + " requires values and col_indices when nnz > 0");
    }
}

template <typename T>
void validate_spmv_launch(CSRMatrixView<T> A, const T* x, T* y) {
    validate_csr_view(A, "launch_spmv_csr");
    if (A.rows == 0) {
        return;
    }
    if (y == nullptr) {
        throw std::invalid_argument("launch_spmv_csr requires y when rows > 0");
    }
    if (A.nnz > 0 && x == nullptr) {
        throw std::invalid_argument("launch_spmv_csr requires x when nnz > 0");
    }
}

template <typename T>
void validate_spmm_launch(CSRMatrixView<T> A, const T* B, T* C, int N) {
    if (N < 0) {
        throw std::invalid_argument("launch_spmm_csr requires N >= 0");
    }
    validate_csr_view(A, "launch_spmm_csr");
    if (A.rows == 0 || N == 0) {
        return;
    }
    if (C == nullptr) {
        throw std::invalid_argument("launch_spmm_csr requires C when rows > 0 and N > 0");
    }
    if (A.nnz > 0 && B == nullptr) {
        throw std::invalid_argument("launch_spmm_csr requires B when nnz > 0 and N > 0");
    }
}

}  // namespace detail

template <typename T>
void launch_spmv_csr(CSRMatrixView<T> A, const T* x, T* y, bool use_vector = true,
                     cudaStream_t stream = nullptr) {
    detail::validate_spmv_launch(A, x, y);
    if (A.rows == 0)
        return;

    if (use_vector) {
        int warps_per_block = 4;
        int threads = warps_per_block * 32;
        int blocks = (A.rows + warps_per_block - 1) / warps_per_block;
        spmv_csr_vector_kernel<T>
            <<<blocks, threads, 0, stream>>>(A.values, A.col_indices, A.row_ptrs, x, y, A.rows);
    } else {
        int block_size = 256;
        int grid_size = (A.rows + block_size - 1) / block_size;
        spmv_csr_kernel<T>
            <<<grid_size, block_size, 0, stream>>>(A.values, A.col_indices, A.row_ptrs, x, y, A.rows,
                                                   A.cols);
    }
    TC_CUDA_CHECK_LAST();
}

template <typename T>
void launch_spmm_csr(CSRMatrixView<T> A, const T* B, T* C, int N, cudaStream_t stream = nullptr) {
    detail::validate_spmm_launch(A, B, C, N);
    if (A.rows == 0 || N == 0)
        return;

    constexpr int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, A.rows);

    spmm_csr_kernel<T>
        <<<grid, block, 0, stream>>>(A.values, A.col_indices, A.row_ptrs, B, C, A.rows, A.cols, N);
    TC_CUDA_CHECK_LAST();
}

template <typename T>
void launch_csr_to_dense(const T* values, const int* col_indices, const int* row_ptrs, T* dense,
                         int rows, int cols, cudaStream_t stream = nullptr) {
    if (rows == 0)
        return;

    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;
    csr_to_dense_kernel<T>
        <<<grid_size, block_size, 0, stream>>>(values, col_indices, row_ptrs, dense, rows, cols);
    TC_CUDA_CHECK_LAST();
}

// Convenience functions for RAII CSRMatrix class
template <typename T>
void spmv(const CSRMatrix<T>& A, const T* x, T* y, cudaStream_t stream = nullptr) {
    launch_spmv_csr(A.view(), x, y, true, stream);
}

template <typename T>
void spmm(const CSRMatrix<T>& A, const T* B, T* C, int N, cudaStream_t stream = nullptr) {
    launch_spmm_csr(A.view(), B, C, N, stream);
}

// Convenience functions for CSRMatrixView (non-owning)
template <typename T>
void spmv(CSRMatrixView<T> A, const T* x, T* y, cudaStream_t stream = nullptr) {
    launch_spmv_csr(A, x, y, true, stream);
}

template <typename T>
void spmm(CSRMatrixView<T> A, const T* B, T* C, int N, cudaStream_t stream = nullptr) {
    launch_spmm_csr(A, B, C, N, stream);
}

}  // namespace kernels
}  // namespace tensorcraft

/**
 * @file test_sparse.cpp
 * @brief Unit tests for sparse matrix operations (CSR, SpMV, SpMM)
 */

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <tensorcraft/core/cuda_check.hpp>
#include <tensorcraft/kernels/sparse.hpp>
#include <vector>

using namespace tensorcraft::kernels;

class SparseTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA device
        int device_count;
        cudaGetDeviceCount(&device_count);
        ASSERT_GT(device_count, 0) << "No CUDA devices found";
    }

    // Helper to create a sparse matrix from dense
    void dense_to_csr_host(const std::vector<float>& dense, std::vector<float>& values,
                           std::vector<int>& col_indices, std::vector<int>& row_ptrs, int rows,
                           int cols, float threshold = 1e-6f) {
        row_ptrs.push_back(0);

        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                float val = dense[r * cols + c];
                if (std::fabs(val) > threshold) {
                    values.push_back(val);
                    col_indices.push_back(c);
                }
            }
            row_ptrs.push_back(static_cast<int>(values.size()));
        }
    }

    // Reference SpMV on CPU
    void spmv_csr_host(const std::vector<float>& values, const std::vector<int>& col_indices,
                       const std::vector<int>& row_ptrs, const std::vector<float>& x,
                       std::vector<float>& y, int rows) {
        for (int r = 0; r < rows; ++r) {
            float sum = 0.0f;
            for (int i = row_ptrs[r]; i < row_ptrs[r + 1]; ++i) {
                sum += values[i] * x[col_indices[i]];
            }
            y[r] = sum;
        }
    }

    // Reference SpMM on CPU
    void spmm_csr_host(const std::vector<float>& A_values, const std::vector<int>& A_col_indices,
                       const std::vector<int>& A_row_ptrs, const std::vector<float>& B,
                       std::vector<float>& C, int M, int K, int N) {
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < N; ++c) {
                float sum = 0.0f;
                for (int i = A_row_ptrs[r]; i < A_row_ptrs[r + 1]; ++i) {
                    int k = A_col_indices[i];
                    sum += A_values[i] * B[k * N + c];
                }
                C[r * N + c] = sum;
            }
        }
    }

    bool compare_results(const std::vector<float>& a, const std::vector<float>& b,
                         float tolerance = 1e-4f) {
        if (a.size() != b.size())
            return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::fabs(a[i] - b[i]) > tolerance * std::max(1.0f, std::fabs(a[i]))) {
                return false;
            }
        }
        return true;
    }
};

// Test SpMV with a simple 3x3 matrix
TEST_F(SparseTest, SpMV_Simple3x3) {
    // Dense matrix: [1, 0, 2]
    //               [0, 3, 0]
    //               [4, 0, 5]
    std::vector<float> dense = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f, 5.0f};

    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptrs;
    dense_to_csr_host(dense, values, col_indices, row_ptrs, 3, 3);

    int rows = 3;
    int nnz = static_cast<int>(values.size());

    // Input vector
    std::vector<float> h_x = {1.0f, 2.0f, 3.0f};
    std::vector<float> h_y(rows);
    std::vector<float> h_y_ref(rows);

    // Compute reference
    spmv_csr_host(values, col_indices, row_ptrs, h_x, h_y_ref, rows);

    // Allocate device memory
    float *d_values, *d_x, *d_y;
    int *d_col_indices, *d_row_ptrs;

    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_ptrs, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));

    // Copy data
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_col_indices, col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_row_ptrs, row_ptrs.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), rows * sizeof(float), cudaMemcpyHostToDevice));

    // Launch SpMV
    launch_spmv_csr(d_values, d_col_indices, d_row_ptrs, d_x, d_y, rows);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    EXPECT_TRUE(compare_results(h_y, h_y_ref)) << "SpMV result mismatch";

    // Cleanup
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_row_ptrs));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

// Test SpMV with a larger random sparse matrix
TEST_F(SparseTest, SpMV_RandomSparse) {
    const int rows = 256;
    const int cols = 256;
    const float sparsity = 0.3f;  // 30% non-zero elements

    // Generate random sparse matrix
    std::vector<float> dense(rows * cols, 0.0f);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    for (int i = 0; i < rows * cols; ++i) {
        if (prob_dist(gen) < sparsity) {
            dense[i] = val_dist(gen);
        }
    }

    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptrs;
    dense_to_csr_host(dense, values, col_indices, row_ptrs, rows, cols);

    int nnz = static_cast<int>(values.size());

    // Random input vector
    std::vector<float> h_x(cols);
    for (auto& v : h_x)
        v = val_dist(gen);

    std::vector<float> h_y(rows);
    std::vector<float> h_y_ref(rows);

    // Compute reference
    spmv_csr_host(values, col_indices, row_ptrs, h_x, h_y_ref, rows);

    // Allocate device memory
    float *d_values, *d_x, *d_y;
    int *d_col_indices, *d_row_ptrs;

    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_ptrs, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));

    // Copy data
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_col_indices, col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_row_ptrs, row_ptrs.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    // Launch SpMV (both scalar and vector versions)
    launch_spmv_csr(d_values, d_col_indices, d_row_ptrs, d_x, d_y, rows, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    std::vector<float> h_y_scalar(rows);
    CUDA_CHECK(cudaMemcpy(h_y_scalar.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Vector version
    launch_spmv_csr(d_values, d_col_indices, d_row_ptrs, d_x, d_y, rows, true);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_y_vector(rows);
    CUDA_CHECK(cudaMemcpy(h_y_vector.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify both versions
    EXPECT_TRUE(compare_results(h_y_scalar, h_y_ref)) << "Scalar SpMV result mismatch";
    EXPECT_TRUE(compare_results(h_y_vector, h_y_ref)) << "Vector SpMV result mismatch";

    // Cleanup
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_row_ptrs));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

// Test SpMM with a small matrix
TEST_F(SparseTest, SpMM_Small) {
    // Sparse matrix A (3x4):
    // [1, 0, 2, 0]
    // [0, 3, 0, 4]
    // [5, 0, 6, 0]
    std::vector<float> dense_A = {1.0f, 0.0f, 2.0f, 0.0f, 0.0f, 3.0f,
                                  0.0f, 4.0f, 5.0f, 0.0f, 6.0f, 0.0f};

    int M = 3, K = 4, N = 3;

    std::vector<float> A_values;
    std::vector<int> A_col_indices;
    std::vector<int> A_row_ptrs;
    dense_to_csr_host(dense_A, A_values, A_col_indices, A_row_ptrs, M, K);

    int nnz = static_cast<int>(A_values.size());

    // Dense matrix B (4x3)
    std::vector<float> h_B = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                              7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    std::vector<float> h_C(M * N);
    std::vector<float> h_C_ref(M * N);

    // Compute reference
    spmm_csr_host(A_values, A_col_indices, A_row_ptrs, h_B, h_C_ref, M, K, N);

    // Allocate device memory
    float *d_A_values, *d_B, *d_C;
    int *d_A_col_indices, *d_A_row_ptrs;

    CUDA_CHECK(cudaMalloc(&d_A_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_col_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_A_row_ptrs, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy data
    CUDA_CHECK(
        cudaMemcpy(d_A_values, A_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_col_indices, A_col_indices.data(), nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_A_row_ptrs, A_row_ptrs.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch SpMM
    launch_spmm_csr(d_A_values, d_A_col_indices, d_A_row_ptrs, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    EXPECT_TRUE(compare_results(h_C, h_C_ref)) << "SpMM result mismatch";

    // Cleanup
    CUDA_CHECK(cudaFree(d_A_values));
    CUDA_CHECK(cudaFree(d_A_col_indices));
    CUDA_CHECK(cudaFree(d_A_row_ptrs));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// Test CSR to dense conversion
TEST_F(SparseTest, CSRToDense) {
    // Create a sparse matrix
    std::vector<float> dense_original = {1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f, 5.0f};
    int rows = 3, cols = 3;

    std::vector<float> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptrs;
    dense_to_csr_host(dense_original, values, col_indices, row_ptrs, rows, cols);

    int nnz = static_cast<int>(values.size());

    // Allocate device memory
    float *d_values, *d_dense;
    int *d_col_indices, *d_row_ptrs;

    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col_indices, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_ptrs, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dense, rows * cols * sizeof(float)));

    // Copy CSR data
    CUDA_CHECK(cudaMemcpy(d_values, values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_col_indices, col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_row_ptrs, row_ptrs.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Convert CSR to dense
    launch_csr_to_dense(d_values, d_col_indices, d_row_ptrs, d_dense, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    std::vector<float> dense_result(rows * cols);
    CUDA_CHECK(cudaMemcpy(dense_result.data(), d_dense, rows * cols * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Verify
    EXPECT_TRUE(compare_results(dense_result, dense_original))
        << "CSR to dense conversion mismatch";

    // Cleanup
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_row_ptrs));
    CUDA_CHECK(cudaFree(d_dense));
}

// Test empty matrix edge case
TEST_F(SparseTest, EmptyMatrix) {
    std::vector<float> values;              // Empty
    std::vector<int> col_indices;           // Empty
    std::vector<int> row_ptrs = {0, 0, 0};  // 2 rows, 0 non-zeros

    int rows = 2;

    std::vector<float> h_x = {1.0f, 2.0f};
    std::vector<float> h_y(rows, -1.0f);     // Initialize to -1
    std::vector<float> h_y_ref(rows, 0.0f);  // Expected: all zeros

    // Allocate device memory
    float *d_values, *d_x, *d_y;
    int *d_col_indices, *d_row_ptrs;

    CUDA_CHECK(cudaMalloc(&d_values, 1 * sizeof(float)));  // Allocate minimum
    CUDA_CHECK(cudaMalloc(&d_col_indices, 1 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_row_ptrs, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(float)));

    CUDA_CHECK(
        cudaMemcpy(d_row_ptrs, row_ptrs.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), rows * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0xFF, rows * sizeof(float)));  // Set to NaN pattern

    // This should not crash
    launch_spmv_csr(d_values, d_col_indices, d_row_ptrs, d_x, d_y, rows);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Result should be zeros
    EXPECT_TRUE(compare_results(h_y, h_y_ref)) << "Empty matrix SpMV should produce zeros";

    // Cleanup
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_col_indices));
    CUDA_CHECK(cudaFree(d_row_ptrs));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}

/**
 * @file basic_gemm.cu
 * @brief Example demonstrating GEMM kernel usage at various optimization levels
 *
 * This example shows how to use TensorCraft-HPC's GEMM implementations:
 * - Naive GEMM (baseline)
 * - Tiled GEMM (shared memory optimization)
 * - Double-buffered GEMM (latency hiding)
 * - Tensor Core GEMM (hardware acceleration)
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

#include <tensorcraft/kernels/gemm.hpp>
#include <tensorcraft/core/cuda_check.hpp>

// Matrix dimensions
constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

/**
 * @brief Initialize matrix with random values
 */
void initialize_matrix(std::vector<float>& matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dist(gen);
    }
}

/**
 * @brief Verify GEMM result against reference
 */
bool verify_result(const std::vector<float>& C,
                   const std::vector<float>& C_ref,
                   float tolerance = 1e-3f) {
    for (size_t i = 0; i < C.size(); ++i) {
        float diff = std::abs(C[i] - C_ref[i]);
        float max_val = std::max(std::abs(C[i]), std::abs(C_ref[i]));
        if (diff > tolerance * max_val && diff > tolerance) {
            std::cerr << "Mismatch at index " << i
                      << ": got " << C[i]
                      << ", expected " << C_ref[i]
                      << ", diff = " << diff << std::endl;
            return false;
        }
    }
    return true;
}

/**
 * @brief CPU reference GEMM implementation
 */
void gemm_cpu_reference(const std::vector<float>& A,
                        const std::vector<float>& B,
                        std::vector<float>& C,
                        int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * @brief Benchmark a GEMM kernel
 */
template<typename GemmFunc>
float benchmark_gemm(GemmFunc gemm_func,
                     float* d_A, float* d_B, float* d_C,
                     int M, int N, int K,
                     int warmup_iters = 5,
                     int benchmark_iters = 20) {
    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        gemm_func(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_iters; ++i) {
        gemm_func(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return static_cast<float>(duration.count()) / benchmark_iters / 1000.0f; // ms
}

int main() {
    std::cout << "=== TensorCraft-HPC GEMM Example ===" << std::endl;
    std::cout << "Matrix dimensions: " << M << " x " << K << " x " << N << std::endl;
    std::cout << std::endl;

    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;

    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_ref(M * N);

    // Initialize matrices
    std::cout << "Initializing matrices..." << std::endl;
    initialize_matrix(h_A, M, K);
    initialize_matrix(h_B, K, N);

    // Compute CPU reference (for small matrices only)
    if (M <= 512 && N <= 512 && K <= 512) {
        std::cout << "Computing CPU reference..." << std::endl;
        gemm_cpu_reference(h_A, h_B, h_C_ref, M, N, K);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate FLOPS
    double flops = 2.0 * M * N * K;

    std::cout << std::endl;
    std::cout << "=== Benchmarking GEMM Implementations ===" << std::endl;
    std::cout << std::endl;

    // 1. Naive GEMM
    {
        std::cout << "1. Naive GEMM:" << std::endl;
        auto naive_gemm = [](float* A, float* B, float* C, int M, int N, int K) {
            tensorcraft::kernels::gemm_naive(A, B, C, M, N, K);
        };

        float time_ms = benchmark_gemm(naive_gemm, d_A, d_B, d_C, M, N, K);
        double gflops = flops / (time_ms * 1e6);

        std::cout << "   Time: " << time_ms << " ms" << std::endl;
        std::cout << "   Performance: " << gflops << " GFLOPS" << std::endl;

        // Verify
        if (M <= 512) {
            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
            bool correct = verify_result(h_C, h_C_ref);
            std::cout << "   Verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
        }
        std::cout << std::endl;
    }

    // 2. Tiled GEMM
    {
        std::cout << "2. Tiled GEMM (Shared Memory):" << std::endl;
        auto tiled_gemm = [](float* A, float* B, float* C, int M, int N, int K) {
            tensorcraft::kernels::gemm_tiled(A, B, C, M, N, K);
        };

        float time_ms = benchmark_gemm(tiled_gemm, d_A, d_B, d_C, M, N, K);
        double gflops = flops / (time_ms * 1e6);

        std::cout << "   Time: " << time_ms << " ms" << std::endl;
        std::cout << "   Performance: " << gflops << " GFLOPS" << std::endl;

        if (M <= 512) {
            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
            bool correct = verify_result(h_C, h_C_ref);
            std::cout << "   Verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
        }
        std::cout << std::endl;
    }

    // 3. Double-Buffered GEMM
    {
        std::cout << "3. Double-Buffered GEMM:" << std::endl;
        auto db_gemm = [](float* A, float* B, float* C, int M, int N, int K) {
            tensorcraft::kernels::gemm_double_buffer(A, B, C, M, N, K);
        };

        float time_ms = benchmark_gemm(db_gemm, d_A, d_B, d_C, M, N, K);
        double gflops = flops / (time_ms * 1e6);

        std::cout << "   Time: " << time_ms << " ms" << std::endl;
        std::cout << "   Performance: " << gflops << " GFLOPS" << std::endl;

        if (M <= 512) {
            CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
            bool correct = verify_result(h_C, h_C_ref);
            std::cout << "   Verification: " << (correct ? "PASSED" : "FAILED") << std::endl;
        }
        std::cout << std::endl;
    }

    // 4. Tensor Core GEMM (if supported)
    if (prop.major >= 7) {
        std::cout << "4. Tensor Core GEMM (WMMA):" << std::endl;
        auto tc_gemm = [](float* A, float* B, float* C, int M, int N, int K) {
            tensorcraft::kernels::gemm_tensor_core(A, B, C, M, N, K);
        };

        float time_ms = benchmark_gemm(tc_gemm, d_A, d_B, d_C, M, N, K);
        double gflops = flops / (time_ms * 1e6);

        std::cout << "   Time: " << time_ms << " ms" << std::endl;
        std::cout << "   Performance: " << gflops << " GFLOPS" << std::endl;
        std::cout << std::endl;
    } else {
        std::cout << "4. Tensor Core GEMM: Skipped (requires SM 7.0+)" << std::endl;
        std::cout << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    std::cout << "=== Example Complete ===" << std::endl;
    return 0;
}

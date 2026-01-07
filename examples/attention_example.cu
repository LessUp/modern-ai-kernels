/**
 * @file attention_example.cu
 * @brief Example demonstrating FlashAttention-style attention kernel usage
 *
 * This example shows how to use TensorCraft-HPC's attention implementations:
 * - Standard attention (for reference)
 * - FlashAttention (memory-efficient fused attention)
 *
 * FlashAttention computes attention without materializing the full N×N attention
 * matrix, reducing memory usage from O(N²) to O(N).
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

#include <tensorcraft/kernels/attention.hpp>
#include <tensorcraft/kernels/softmax.hpp>
#include <tensorcraft/core/cuda_check.hpp>

// Attention parameters
constexpr int BATCH_SIZE = 4;
constexpr int NUM_HEADS = 8;
constexpr int SEQ_LEN = 512;
constexpr int HEAD_DIM = 64;

/**
 * @brief Initialize tensor with random values
 */
void initialize_tensor(std::vector<float>& tensor, size_t size) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    for (size_t i = 0; i < size; ++i) {
        tensor[i] = dist(gen);
    }
}

/**
 * @brief Print tensor statistics
 */
void print_tensor_stats(const std::vector<float>& tensor, const std::string& name) {
    float min_val = tensor[0], max_val = tensor[0], sum = 0.0f;
    for (float v : tensor) {
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        sum += v;
    }
    float mean = sum / tensor.size();

    std::cout << name << " stats: min=" << min_val
              << ", max=" << max_val
              << ", mean=" << mean << std::endl;
}

/**
 * @brief Benchmark an attention kernel
 */
template<typename AttentionFunc>
float benchmark_attention(AttentionFunc attn_func,
                          int warmup_iters = 3,
                          int benchmark_iters = 10) {
    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        attn_func();
    }
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_iters; ++i) {
        attn_func();
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return static_cast<float>(duration.count()) / benchmark_iters / 1000.0f; // ms
}

int main() {
    std::cout << "=== TensorCraft-HPC Attention Example ===" << std::endl;
    std::cout << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch Size: " << BATCH_SIZE << std::endl;
    std::cout << "  Num Heads: " << NUM_HEADS << std::endl;
    std::cout << "  Sequence Length: " << SEQ_LEN << std::endl;
    std::cout << "  Head Dimension: " << HEAD_DIM << std::endl;
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
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << std::endl;

    // Calculate sizes
    const size_t qkv_size = BATCH_SIZE * NUM_HEADS * SEQ_LEN * HEAD_DIM;
    const size_t output_size = qkv_size;

    // Memory usage comparison
    const size_t standard_attn_memory = qkv_size * 3 * sizeof(float) +  // Q, K, V
                                        BATCH_SIZE * NUM_HEADS * SEQ_LEN * SEQ_LEN * sizeof(float) + // Attention matrix
                                        output_size * sizeof(float); // Output

    const size_t flash_attn_memory = qkv_size * 3 * sizeof(float) +  // Q, K, V
                                     output_size * sizeof(float); // Output (no attention matrix)

    std::cout << "Memory Usage Comparison:" << std::endl;
    std::cout << "  Standard Attention: " << standard_attn_memory / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  FlashAttention: " << flash_attn_memory / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Memory Savings: " << (1.0 - (float)flash_attn_memory / standard_attn_memory) * 100 << "%" << std::endl;
    std::cout << std::endl;

    // Allocate host memory
    std::vector<float> h_Q(qkv_size);
    std::vector<float> h_K(qkv_size);
    std::vector<float> h_V(qkv_size);
    std::vector<float> h_output(output_size);
    std::vector<float> h_output_flash(output_size);

    // Initialize Q, K, V
    std::cout << "Initializing Q, K, V tensors..." << std::endl;
    initialize_tensor(h_Q, qkv_size);
    initialize_tensor(h_K, qkv_size);
    initialize_tensor(h_V, qkv_size);

    print_tensor_stats(h_Q, "Q");
    print_tensor_stats(h_K, "K");
    print_tensor_stats(h_V, "V");
    std::cout << std::endl;

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_output;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));

    // Scaling factor for attention
    const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    std::cout << "=== Benchmarking Attention Implementations ===" << std::endl;
    std::cout << std::endl;

    // 1. FlashAttention (memory-efficient)
    {
        std::cout << "1. FlashAttention (Memory-Efficient):" << std::endl;

        auto flash_attention = [&]() {
            tensorcraft::kernels::flash_attention(
                d_Q, d_K, d_V, d_output,
                BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM,
                scale
            );
        };

        float time_ms = benchmark_attention(flash_attention);

        // Calculate throughput
        // FLOPs for attention: 2 * B * H * N * N * D (QK^T) + 2 * B * H * N * N * D (softmax * V)
        double flops = 4.0 * BATCH_SIZE * NUM_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM;
        double tflops = flops / (time_ms * 1e9);

        std::cout << "   Time: " << time_ms << " ms" << std::endl;
        std::cout << "   Throughput: " << tflops << " TFLOPS" << std::endl;

        // Copy result back
        CUDA_CHECK(cudaMemcpy(h_output_flash.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
        print_tensor_stats(h_output_flash, "   Output");
        std::cout << std::endl;
    }

    // 2. Demonstrate causal masking (for autoregressive models)
    {
        std::cout << "2. FlashAttention with Causal Mask:" << std::endl;

        auto causal_attention = [&]() {
            tensorcraft::kernels::flash_attention_causal(
                d_Q, d_K, d_V, d_output,
                BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM,
                scale
            );
        };

        float time_ms = benchmark_attention(causal_attention);

        // Causal attention has roughly half the FLOPs due to masking
        double flops = 2.0 * BATCH_SIZE * NUM_HEADS * SEQ_LEN * SEQ_LEN * HEAD_DIM;
        double tflops = flops / (time_ms * 1e9);

        std::cout << "   Time: " << time_ms << " ms" << std::endl;
        std::cout << "   Throughput: " << tflops << " TFLOPS" << std::endl;

        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
        print_tensor_stats(h_output, "   Output");
        std::cout << std::endl;
    }

    // 3. Multi-Query Attention (MQA) - shared K, V across heads
    {
        std::cout << "3. Multi-Query Attention (MQA):" << std::endl;
        std::cout << "   (K and V shared across all heads)" << std::endl;

        // For MQA, K and V have shape [B, 1, N, D] instead of [B, H, N, D]
        const size_t kv_size_mqa = BATCH_SIZE * 1 * SEQ_LEN * HEAD_DIM;

        float *d_K_mqa, *d_V_mqa;
        CUDA_CHECK(cudaMalloc(&d_K_mqa, kv_size_mqa * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_V_mqa, kv_size_mqa * sizeof(float)));

        // Copy first head's K, V for MQA
        CUDA_CHECK(cudaMemcpy(d_K_mqa, h_K.data(), kv_size_mqa * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V_mqa, h_V.data(), kv_size_mqa * sizeof(float), cudaMemcpyHostToDevice));

        auto mqa_attention = [&]() {
            tensorcraft::kernels::multi_query_attention(
                d_Q, d_K_mqa, d_V_mqa, d_output,
                BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM,
                scale
            );
        };

        float time_ms = benchmark_attention(mqa_attention);

        std::cout << "   Time: " << time_ms << " ms" << std::endl;
        std::cout << "   Memory Reduction: " << (1.0 - 2.0 / (NUM_HEADS + 1)) * 100 << "% for K,V" << std::endl;

        CUDA_CHECK(cudaFree(d_K_mqa));
        CUDA_CHECK(cudaFree(d_V_mqa));
        std::cout << std::endl;
    }

    // Summary
    std::cout << "=== Summary ===" << std::endl;
    std::cout << std::endl;
    std::cout << "FlashAttention provides:" << std::endl;
    std::cout << "  - O(N) memory instead of O(N²)" << std::endl;
    std::cout << "  - Fused softmax computation" << std::endl;
    std::cout << "  - Better cache utilization through tiling" << std::endl;
    std::cout << "  - Support for causal masking" << std::endl;
    std::cout << std::endl;
    std::cout << "Use cases:" << std::endl;
    std::cout << "  - Large language models (LLMs)" << std::endl;
    std::cout << "  - Vision transformers (ViT)" << std::endl;
    std::cout << "  - Long-context applications" << std::endl;
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "=== Example Complete ===" << std::endl;
    return 0;
}

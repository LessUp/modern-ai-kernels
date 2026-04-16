#include <gtest/gtest.h>
#include <stdexcept>

#include "cuda_test_ops.hpp"

using namespace tensorcraft::tests;

TEST(GemmContractTest, TensorCoreVersionThrowsInLaunchGemm) {
    float* null_ptr = nullptr;
    EXPECT_THROW(
        launch_gemm(null_ptr, null_ptr, null_ptr, 1, 1, 1, 1.0f, 0.0f, GemmVersion::TensorCore),
        std::invalid_argument);
}

TEST(AttentionContractTest, FlashAttentionRejectsUnsupportedHeadDim) {
    float* null_ptr = nullptr;
    EXPECT_THROW(launch_flash_attention(null_ptr, null_ptr, null_ptr, null_ptr, 1, 1, 1, 32, 1.0f),
                 std::invalid_argument);
}

TEST(AttentionContractTest, RopeRejectsOddHeadDim) {
    float* null_ptr = nullptr;
    EXPECT_THROW(launch_rope(null_ptr, null_ptr, null_ptr, 1, 1, 1, 63, 0), std::invalid_argument);
}

TEST(AttentionContractTest, RopeRejectsNegativeStartPos) {
    float* null_ptr = nullptr;
    EXPECT_THROW(launch_rope(null_ptr, null_ptr, null_ptr, 1, 1, 1, 64, -1), std::invalid_argument);
}

TEST(AttentionContractTest, MoeRouterRejectsInvalidExpertBounds) {
    float* null_logits = nullptr;
    int* null_indices = nullptr;
    float* null_weights = nullptr;

    EXPECT_THROW(launch_moe_router(null_logits, null_indices, null_weights, 1, 9, 1),
                 std::invalid_argument);
    EXPECT_THROW(launch_moe_router(null_logits, null_indices, null_weights, 1, 4, 5),
                 std::invalid_argument);
}

TEST(AttentionContractTest, PrecomputeRopeCacheRejectsOddHeadDim) {
    float* null_ptr = nullptr;
    EXPECT_THROW(precompute_rope_cache(null_ptr, null_ptr, 16, 63), std::invalid_argument);
}

TEST(AttentionContractTest, PrecomputeRopeCacheRejectsNegativeSeqLen) {
    float* null_ptr = nullptr;
    EXPECT_THROW(precompute_rope_cache(null_ptr, null_ptr, -1, 64), std::invalid_argument);
}

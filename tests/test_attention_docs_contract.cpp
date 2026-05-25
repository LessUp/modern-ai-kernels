#include <gtest/gtest.h>
#include <stdexcept>

#include "tensorcraft/kernels/attention.hpp"
#include "tensorcraft/kernels/sparse.hpp"
using namespace tensorcraft::kernels;

TEST(SparseContractTest, SpmvRejectsNegativeMetadata) {
    float* null_ptr = nullptr;
    CSRMatrixView<float> negative_rows{nullptr, nullptr, nullptr, -1, 4, 0};
    CSRMatrixView<float> negative_cols{nullptr, nullptr, nullptr, 1, -1, 0};
    CSRMatrixView<float> negative_nnz{nullptr, nullptr, nullptr, 1, 4, -1};

    EXPECT_THROW(launch_spmv_csr(negative_rows, null_ptr, null_ptr), std::invalid_argument);
    EXPECT_THROW(launch_spmv_csr(negative_cols, null_ptr, null_ptr), std::invalid_argument);
    EXPECT_THROW(launch_spmv_csr(negative_nnz, null_ptr, null_ptr), std::invalid_argument);
}

TEST(SparseContractTest, SpmvRejectsMissingRequiredPointers) {
    float output = 0.0f;
    int row_ptrs[2] = {0, 0};
    CSRMatrixView<float> missing_row_ptrs{nullptr, nullptr, nullptr, 1, 1, 0};
    CSRMatrixView<float> missing_values{nullptr, nullptr, row_ptrs, 1, 1, 1};

    EXPECT_THROW(launch_spmv_csr(missing_row_ptrs, nullptr, &output), std::invalid_argument);
    EXPECT_THROW(launch_spmv_csr(missing_values, nullptr, &output), std::invalid_argument);
}

TEST(SparseContractTest, SpmmRejectsInvalidLaunchContract) {
    float* null_ptr = nullptr;
    int row_ptrs[2] = {0, 0};
    CSRMatrixView<float> negative_cols{nullptr, nullptr, row_ptrs, 1, -1, 0};
    CSRMatrixView<float> missing_row_ptrs{nullptr, nullptr, nullptr, 1, 1, 0};
    CSRMatrixView<float> missing_values{nullptr, nullptr, row_ptrs, 1, 1, 1};

    EXPECT_THROW(launch_spmm_csr(negative_cols, null_ptr, null_ptr, 1), std::invalid_argument);
    EXPECT_THROW(launch_spmm_csr(missing_row_ptrs, null_ptr, null_ptr, 1), std::invalid_argument);
    EXPECT_THROW(launch_spmm_csr(missing_values, null_ptr, null_ptr, 1), std::invalid_argument);
    EXPECT_THROW(launch_spmm_csr(CSRMatrixView<float>{nullptr, nullptr, row_ptrs, 1, 1, 0}, null_ptr,
                                 null_ptr, -1),
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

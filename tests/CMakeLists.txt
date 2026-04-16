# TensorCraft-HPC Tests
add_executable(tensorcraft_tests
    test_main.cpp
    test_elementwise.cpp
    test_softmax.cpp
    test_normalization.cpp
    test_gemm.cpp
    test_conv2d.cpp
    test_fusion.cpp
    test_attention_docs_contract.cpp
    test_sparse.cpp
    cuda_test_ops.cu
)

target_link_libraries(tensorcraft_tests PRIVATE
    tensorcraft
    GTest::gtest
    CUDA::cudart
)

target_include_directories(tensorcraft_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

set_target_properties(tensorcraft_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
)

set_source_files_properties(cuda_test_ops.cu PROPERTIES LANGUAGE CUDA)

set_source_files_properties(
    test_main.cpp
    test_elementwise.cpp
    test_softmax.cpp
    test_normalization.cpp
    test_gemm.cpp
    test_conv2d.cpp
    test_fusion.cpp
    test_attention_docs_contract.cpp
    test_sparse.cpp
    PROPERTIES LANGUAGE CXX
)

include(GoogleTest)
gtest_discover_tests(tensorcraft_tests)

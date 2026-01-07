# TensorCraft-HPC Examples
cmake_minimum_required(VERSION 3.18)

# Check if being built as part of main project or standalone
if(NOT TARGET tensorcraft)
    project(tensorcraft_examples LANGUAGES CXX CUDA)

    # Find CUDA
    find_package(CUDAToolkit REQUIRED)

    # Set C++ standard
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Include directories
    set(TENSORCRAFT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
endif()

# Helper function to add example
function(add_tensorcraft_example name)
    add_executable(${name} ${name}.cu)

    target_include_directories(${name} PRIVATE
        ${TENSORCRAFT_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )

    target_link_libraries(${name} PRIVATE
        CUDA::cudart
    )

    # Set CUDA architectures
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(${name} PROPERTIES
            CUDA_ARCHITECTURES "70;75;80;86"
        )
    endif()

    # Enable separable compilation for device code
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endfunction()

# Add examples
add_tensorcraft_example(basic_gemm)
add_tensorcraft_example(attention_example)

# Custom target to build all examples
add_custom_target(examples DEPENDS
    basic_gemm
    attention_example
)

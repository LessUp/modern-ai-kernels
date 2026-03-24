cmake_minimum_required(VERSION 3.20)
project(TensorCraft-HPC
    VERSION 2.0.0
    DESCRIPTION "High-Performance AI Kernels with Modern C++ & CUDA"
    LANGUAGES CXX
)

# ============================================================================
# C++ Standard Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detect C++20/23 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++20" COMPILER_SUPPORTS_CXX20)
check_cxx_compiler_flag("-std=c++23" COMPILER_SUPPORTS_CXX23)

if(COMPILER_SUPPORTS_CXX23)
    set(CMAKE_CXX_STANDARD 23)
    message(STATUS "C++23 support enabled")
elseif(COMPILER_SUPPORTS_CXX20)
    set(CMAKE_CXX_STANDARD 20)
    message(STATUS "C++20 support enabled")
else()
    message(STATUS "C++17 support enabled")
endif()

# ============================================================================
# CUDA Configuration
# ============================================================================
option(TC_ENABLE_CUDA "Enable CUDA support" ON)
set(TC_CUDA_ENABLED OFF)

if(TC_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(TC_CUDA_ENABLED ON)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)

        # Check CUDA version
        if(CUDAToolkit_VERSION VERSION_LESS "10.1")
            message(FATAL_ERROR "CUDA 10.1 or higher is required. Found: ${CUDAToolkit_VERSION}")
        endif()

        message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")

        # Set a CUDA dialect compatible with the detected toolkit.
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
            set(CMAKE_CUDA_STANDARD 17)
        else()
            set(CMAKE_CUDA_STANDARD 14)
        endif()
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        # CUDA architectures - keep defaults compatible with the detected toolkit.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
            if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
                set(TC_DEFAULT_CUDA_ARCHITECTURES "75;80;86;89;90")
            elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
                set(TC_DEFAULT_CUDA_ARCHITECTURES "70;75;80;86")
            else()
                set(TC_DEFAULT_CUDA_ARCHITECTURES "70;75")
            endif()
            set(CMAKE_CUDA_ARCHITECTURES "${TC_DEFAULT_CUDA_ARCHITECTURES}" CACHE STRING "CUDA architectures" FORCE)
        endif()

        # Feature detection based on CUDA version
        if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
            add_compile_definitions(TC_CUDA_13=1 TC_CUDA_12=1 TC_CUDA_11=1 TC_CUDA_10=1)
            add_compile_definitions(TC_HAS_TMA=1 TC_HAS_WGMMA=1 TC_HAS_FP8=1 TC_HAS_BF16=1 TC_HAS_WMMA=1)
            message(STATUS "CUDA 13.x features enabled (TMA, WGMMA, FP8, BF16, WMMA)")
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
            add_compile_definitions(TC_CUDA_12=1 TC_CUDA_11=1 TC_CUDA_10=1)
            add_compile_definitions(TC_HAS_TMA=1 TC_HAS_WGMMA=1 TC_HAS_FP8=1 TC_HAS_BF16=1 TC_HAS_WMMA=1)
            message(STATUS "CUDA 12.x features enabled (TMA, WGMMA, FP8, BF16, WMMA)")
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
            add_compile_definitions(TC_CUDA_11=1 TC_CUDA_10=1)
            add_compile_definitions(TC_HAS_BF16=1 TC_HAS_WMMA=1)
            message(STATUS "CUDA 11.x features enabled (BF16, WMMA)")
        else()
            add_compile_definitions(TC_CUDA_10=1)
            message(STATUS "CUDA 10.x compatibility mode enabled")
        endif()
    else()
        message(WARNING "Failed to find nvcc (CUDA compiler). Configure will continue without CUDA. To enable CUDA, set CUDAToolkit_ROOT or ensure nvcc is on PATH.")
    endif()
endif()

# ============================================================================
# Build Options
# ============================================================================
option(TC_BUILD_TESTS "Build unit tests" ON)
option(TC_BUILD_BENCHMARKS "Build benchmarks" ON)
option(TC_BUILD_PYTHON "Build Python bindings" ON)
option(TC_BUILD_DOCS "Build documentation" OFF)

if(NOT TC_CUDA_ENABLED)
    if(TC_BUILD_TESTS OR TC_BUILD_BENCHMARKS OR TC_BUILD_PYTHON)
        message(WARNING "CUDA is unavailable; disabling tests, benchmarks, and Python bindings.")
    endif()
    set(TC_BUILD_TESTS OFF CACHE BOOL "Build unit tests" FORCE)
    set(TC_BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmarks" FORCE)
    set(TC_BUILD_PYTHON OFF CACHE BOOL "Build Python bindings" FORCE)
endif()

# ============================================================================
# Dependencies via FetchContent
# ============================================================================
include(FetchContent)

if(TC_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

if(TC_BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()

if(TC_BUILD_PYTHON)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# ============================================================================
# Main Library Target (Header-Only)
# ============================================================================
add_library(tensorcraft INTERFACE)
target_include_directories(tensorcraft INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
if(TC_CUDA_ENABLED)
    target_link_libraries(tensorcraft INTERFACE CUDA::cudart)
endif()

# Add alias for consistent naming
add_library(TensorCraft::tensorcraft ALIAS tensorcraft)

# ============================================================================
# Subdirectories
# ============================================================================
if(TC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TC_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(TC_BUILD_PYTHON)
    add_subdirectory(src/python_ops)
endif()

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(TARGETS tensorcraft
    EXPORT TensorCraftTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT TensorCraftTargets
    FILE TensorCraftTargets.cmake
    NAMESPACE TensorCraft::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TensorCraft
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "TensorCraft-HPC Configuration Summary:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
if(TC_CUDA_ENABLED)
    set(TC_CUDA_VERSION_STR "${CUDAToolkit_VERSION}")
    set(TC_CUDA_ARCHS_STR "${CMAKE_CUDA_ARCHITECTURES}")
else()
    set(TC_CUDA_VERSION_STR "DISABLED")
    set(TC_CUDA_ARCHS_STR "N/A")
endif()
message(STATUS "  CUDA Enabled:   ${TC_CUDA_ENABLED}")
message(STATUS "  CUDA Version:   ${TC_CUDA_VERSION_STR}")
message(STATUS "  CUDA Archs:     ${TC_CUDA_ARCHS_STR}")
message(STATUS "  Build Tests:    ${TC_BUILD_TESTS}")
message(STATUS "  Build Bench:    ${TC_BUILD_BENCHMARKS}")
message(STATUS "  Build Python:   ${TC_BUILD_PYTHON}")
message(STATUS "")

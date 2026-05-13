/**
 * @file bindings.cpp
 * @brief Python bindings for TensorCraft-HPC using pybind11
 *
 * Direct kernel bindings without shallow intermediate layer.
 * Memory management uses MemoryPool for efficient allocation.
 */

#include <initializer_list>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensorcraft/core/cuda_check.hpp"
#include "tensorcraft/kernels/elementwise.hpp"
#include "tensorcraft/kernels/gemm.hpp"
#include "tensorcraft/kernels/normalization.hpp"
#include "tensorcraft/kernels/softmax.hpp"
#include "tensorcraft/memory/memory_pool.hpp"

namespace py = pybind11;
using namespace tensorcraft;

// ============================================================================
// Memory Management (using MemoryPool)
// ============================================================================

/**
 * @brief RAII wrapper for device memory using MemoryPool
 *
 * Replaces the previous DeviceArray/DeviceBuffer with unified
 * PoolPtr-based implementation for reduced allocation overhead.
 */
template <typename T>
class PooledDeviceMemory {
public:
    PooledDeviceMemory() = default;

    /// Allocate uninitialized device memory
    explicit PooledDeviceMemory(size_t count)
        : ptr_(static_cast<T*>(MemoryPool::instance().allocate(count * sizeof(T)))),
          count_(count) {}

    /// Allocate and copy from host
    static PooledDeviceMemory from_host(const T* host_data, size_t count) {
        PooledDeviceMemory mem(count);
        TC_CUDA_CHECK(cudaMemcpy(mem.ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
        return mem;
    }

    /// Allocate and copy from numpy array
    static PooledDeviceMemory from_numpy(py::array_t<T> arr) {
        py::buffer_info buf = arr.request();
        return from_host(static_cast<const T*>(buf.ptr), static_cast<size_t>(buf.size));
    }

    ~PooledDeviceMemory() {
        if (ptr_) {
            MemoryPool::instance().deallocate(ptr_);
        }
    }

    // Move only
    PooledDeviceMemory(PooledDeviceMemory&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    PooledDeviceMemory& operator=(PooledDeviceMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                MemoryPool::instance().deallocate(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    PooledDeviceMemory(const PooledDeviceMemory&) = delete;
    PooledDeviceMemory& operator=(const PooledDeviceMemory&) = delete;

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }

    /// Copy to host
    void to_host(T* host_data) const {
        TC_CUDA_CHECK(cudaMemcpy(host_data, ptr_, bytes(), cudaMemcpyDeviceToHost));
    }

    /// Copy to numpy array with given shape
    py::array_t<T> to_numpy(const std::vector<ssize_t>& shape) const {
        py::array_t<T> result(shape);
        py::buffer_info buf = result.request();
        to_host(static_cast<T*>(buf.ptr));
        return result;
    }

    explicit operator bool() const { return ptr_ != nullptr; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

// Type alias for common float memory
using FloatDeviceMemory = PooledDeviceMemory<float>;

// ============================================================================
// Utility Functions
// ============================================================================

void sync_cuda() {
    TC_CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<ssize_t> array_shape(const py::buffer_info& buf) {
    return std::vector<ssize_t>(buf.shape.begin(), buf.shape.end());
}

size_t array_size(const py::buffer_info& buf) {
    return static_cast<size_t>(buf.size);
}

void require_same_shape(const py::buffer_info& lhs, const py::buffer_info& rhs,
                        const char* message) {
    if (lhs.ndim != rhs.ndim || lhs.shape != rhs.shape) {
        throw std::invalid_argument(message);
    }
}

void require_non_empty_last_dim(const py::buffer_info& buf, const char* message) {
    if (buf.ndim == 0 || buf.shape[buf.ndim - 1] <= 0) {
        throw std::invalid_argument(message);
    }
}

void require_vector_size(const py::buffer_info& buf, ssize_t expected, const char* name) {
    if (buf.size != expected) {
        throw std::invalid_argument(std::string(name) + " must contain exactly " +
                                    std::to_string(expected) + " elements");
    }
}

size_t matrix_output_size(int rows, int cols) {
    return static_cast<size_t>(rows) * static_cast<size_t>(cols);
}

// Parse GEMM version string to enum
kernels::GemmVersion parse_gemm_version(const std::string& name) {
    if (name == "naive")
        return kernels::GemmVersion::Naive;
    if (name == "tiled")
        return kernels::GemmVersion::Tiled;
    if (name == "double_buffer")
        return kernels::GemmVersion::DoubleBuffer;
    throw std::invalid_argument("Unsupported GEMM version: " + name);
}

// ============================================================================
// Operation Templates
// ============================================================================

template <typename Launcher>
py::array_t<float> run_shape_preserving_float_op(py::array_t<float> input, Launcher&& launch) {
    py::buffer_info buf = input.request();
    FloatDeviceMemory d_input = FloatDeviceMemory::from_numpy(input);
    FloatDeviceMemory d_output(array_size(buf));

    launch(d_input.get(), d_output.get(), array_size(buf));
    sync_cuda();
    return d_output.to_numpy(array_shape(buf));
}

template <typename Launcher>
py::array_t<float> run_binary_float_op(py::array_t<float> a, py::array_t<float> b,
                                       Launcher&& launch) {
    py::buffer_info buf_a = a.request();
    py::buffer_info buf_b = b.request();

    require_same_shape(buf_a, buf_b, "Input arrays must have the same shape");

    FloatDeviceMemory d_a = FloatDeviceMemory::from_numpy(a);
    FloatDeviceMemory d_b = FloatDeviceMemory::from_numpy(b);
    FloatDeviceMemory d_output(array_size(buf_a));

    launch(d_a.get(), d_b.get(), d_output.get(), array_size(buf_a));
    sync_cuda();
    return d_output.to_numpy(array_shape(buf_a));
}

template <typename Launcher>
py::array_t<float> run_matrix_float_op(py::array_t<float> input, Launcher&& launch) {
    py::buffer_info buf = input.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D matrix");
    }

    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);
    FloatDeviceMemory d_input = FloatDeviceMemory::from_numpy(input);
    FloatDeviceMemory d_output(matrix_output_size(rows, cols));

    launch(d_input.get(), d_output.get(), rows, cols);
    sync_cuda();
    return d_output.to_numpy({cols, rows});
}

template <typename Launcher>
py::array_t<float> run_last_dim_float_op(py::array_t<float> input, Launcher&& launch) {
    py::buffer_info buf = input.request();

    require_non_empty_last_dim(buf,
                               "Input must have at least 1 dimension with a non-empty last axis");

    int cols = static_cast<int>(buf.shape[buf.ndim - 1]);
    int rows = static_cast<int>(buf.size / cols);
    FloatDeviceMemory d_input = FloatDeviceMemory::from_numpy(input);
    FloatDeviceMemory d_output(array_size(buf));

    launch(d_input.get(), d_output.get(), rows, cols);
    sync_cuda();
    return d_output.to_numpy(array_shape(buf));
}

template <typename Launcher>
py::array_t<float> run_norm_float_op(py::array_t<float> input, Launcher&& launch) {
    py::buffer_info buf = input.request();

    if (buf.ndim < 2) {
        throw std::invalid_argument("Input must have at least 2 dimensions");
    }
    require_non_empty_last_dim(buf, "Input must have a non-empty hidden dimension");

    int hidden_size = static_cast<int>(buf.shape[buf.ndim - 1]);
    int batch_size = static_cast<int>(buf.size / hidden_size);
    FloatDeviceMemory d_input = FloatDeviceMemory::from_numpy(input);
    FloatDeviceMemory d_output(array_size(buf));

    launch(d_input.get(), d_output.get(), batch_size, hidden_size);
    sync_cuda();
    return d_output.to_numpy(array_shape(buf));
}

template <typename Launcher>
py::array_t<float> run_gemm_float_op(py::array_t<float> A, py::array_t<float> B,
                                     Launcher&& launch) {
    py::buffer_info buf_a = A.request();
    py::buffer_info buf_b = B.request();

    if (buf_a.ndim != 2 || buf_b.ndim != 2) {
        throw std::runtime_error("A and B must be 2D matrices");
    }

    int M = static_cast<int>(buf_a.shape[0]);
    int K = static_cast<int>(buf_a.shape[1]);
    int N = static_cast<int>(buf_b.shape[1]);

    if (static_cast<int>(buf_b.shape[0]) != K) {
        throw std::runtime_error("Matrix dimensions don't match for multiplication");
    }

    FloatDeviceMemory d_A = FloatDeviceMemory::from_numpy(A);
    FloatDeviceMemory d_B = FloatDeviceMemory::from_numpy(B);
    FloatDeviceMemory d_C(matrix_output_size(M, N));
    TC_CUDA_CHECK(cudaMemset(d_C.get(), 0, matrix_output_size(M, N) * sizeof(float)));

    launch(d_A.get(), d_B.get(), d_C.get(), M, N, K);
    sync_cuda();
    return d_C.to_numpy({M, N});
}

// ============================================================================
// Python Functions
// ============================================================================

py::array_t<float> py_relu(py::array_t<float> input) {
    return run_shape_preserving_float_op(
        input, [](const float* in, float* out, size_t n) { kernels::relu(in, out, n); });
}

py::array_t<float> py_silu(py::array_t<float> input) {
    return run_shape_preserving_float_op(
        input, [](const float* in, float* out, size_t n) { kernels::silu(in, out, n); });
}

py::array_t<float> py_gelu(py::array_t<float> input) {
    return run_shape_preserving_float_op(
        input, [](const float* in, float* out, size_t n) { kernels::gelu(in, out, n); });
}

py::array_t<float> py_sigmoid(py::array_t<float> input) {
    return run_shape_preserving_float_op(
        input, [](const float* in, float* out, size_t n) { kernels::sigmoid(in, out, n); });
}

py::array_t<float> py_vector_add(py::array_t<float> a, py::array_t<float> b) {
    return run_binary_float_op(a, b, [](const float* lhs, const float* rhs, float* out, size_t n) {
        kernels::vector_add(lhs, rhs, out, n);
    });
}

py::array_t<float> py_softmax(py::array_t<float> input) {
    return run_last_dim_float_op(input, [](const float* in, float* out, int rows, int cols) {
        kernels::softmax(in, out, rows, cols);
    });
}

py::array_t<float> py_layernorm(py::array_t<float> input, py::array_t<float> gamma,
                                py::array_t<float> beta, float eps = 1e-5f) {
    py::buffer_info input_buf = input.request();
    require_non_empty_last_dim(input_buf, "Input must have a non-empty hidden dimension");
    ssize_t hidden_size = input_buf.shape[input_buf.ndim - 1];
    require_vector_size(gamma.request(), hidden_size, "gamma");
    require_vector_size(beta.request(), hidden_size, "beta");

    FloatDeviceMemory d_gamma = FloatDeviceMemory::from_numpy(gamma);
    FloatDeviceMemory d_beta = FloatDeviceMemory::from_numpy(beta);
    return run_norm_float_op(input, [&](const float* in, float* out, int batch_size,
                                        int hidden_size_int) {
        kernels::layernorm(in, d_gamma.get(), d_beta.get(), out, batch_size, hidden_size_int, eps);
    });
}

py::array_t<float> py_rmsnorm(py::array_t<float> input, py::array_t<float> weight,
                              float eps = 1e-6f) {
    py::buffer_info input_buf = input.request();
    require_non_empty_last_dim(input_buf, "Input must have a non-empty hidden dimension");
    ssize_t hidden_size = input_buf.shape[input_buf.ndim - 1];
    require_vector_size(weight.request(), hidden_size, "weight");

    FloatDeviceMemory d_weight = FloatDeviceMemory::from_numpy(weight);
    return run_norm_float_op(
        input, [&](const float* in, float* out, int batch_size, int hidden_size_int) {
            kernels::rmsnorm(in, d_weight.get(), out, batch_size, hidden_size_int, eps);
        });
}

py::array_t<float> py_gemm(py::array_t<float> A, py::array_t<float> B, float alpha = 1.0f,
                           float beta = 0.0f, const std::string& version = "tiled") {
    return run_gemm_float_op(
        A, B, [&](const float* lhs, const float* rhs, float* out, int M, int N, int K) {
            kernels::launch_gemm(lhs, rhs, out, M, N, K, alpha, beta, parse_gemm_version(version));
        });
}

py::array_t<float> py_transpose(py::array_t<float> input) {
    return run_matrix_float_op(input, [](const float* in, float* out, int rows, int cols) {
        kernels::transpose(in, out, rows, cols);
    });
}

// ============================================================================
// Module Definition
// ============================================================================

PYBIND11_MODULE(tensorcraft_ops, m) {
    m.doc() = "TensorCraft-HPC: High-Performance AI Kernels";

    // Register exception translator for CUDA errors
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const tensorcraft::CudaException& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    m.def("relu", &py_relu, "ReLU activation", py::arg("input"));
    m.def("silu", &py_silu, "SiLU (Swish) activation", py::arg("input"));
    m.def("gelu", &py_gelu, "GeLU activation", py::arg("input"));
    m.def("sigmoid", &py_sigmoid, "Sigmoid activation", py::arg("input"));
    m.def("vector_add", &py_vector_add, "Element-wise vector addition", py::arg("a"), py::arg("b"));
    m.def("softmax", &py_softmax, "Softmax along last dimension", py::arg("input"));
    m.def("layernorm", &py_layernorm, "Layer normalization", py::arg("input"), py::arg("gamma"),
          py::arg("beta"), py::arg("eps") = 1e-5f);
    m.def("rmsnorm", &py_rmsnorm, "RMS normalization", py::arg("input"), py::arg("weight"),
          py::arg("eps") = 1e-6f);
    m.def("gemm", &py_gemm, "General matrix multiplication", py::arg("A"), py::arg("B"),
          py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f, py::arg("version") = "tiled");
    m.def("transpose", &py_transpose, "Matrix transpose", py::arg("input"));

    m.attr("__version__") = "2.0.0";
}

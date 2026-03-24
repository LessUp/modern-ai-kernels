/**
 * @file bindings.cpp
 * @brief Python bindings for TensorCraft-HPC using pybind11
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_ops.hpp"
#include "tensorcraft/core/cuda_check.hpp"

namespace py = pybind11;
using namespace tensorcraft;

// Helper to convert numpy array to device pointer
template<typename T>
T* numpy_to_device(py::array_t<T> arr, size_t& size) {
    py::buffer_info buf = arr.request();
    size = static_cast<size_t>(buf.size);

    T* d_ptr = nullptr;
    TC_CUDA_CHECK(cudaMalloc(&d_ptr, size * sizeof(T)));
    TC_CUDA_CHECK(cudaMemcpy(d_ptr, buf.ptr, size * sizeof(T), cudaMemcpyHostToDevice));
    return d_ptr;
}

// Helper to copy device data to numpy array
template<typename T>
py::array_t<T> device_to_numpy(T* d_ptr, const std::vector<ssize_t>& shape) {
    size_t size = 1;
    for (auto s : shape) size *= static_cast<size_t>(s);

    py::array_t<T> result(shape);
    py::buffer_info buf = result.request();
    TC_CUDA_CHECK(cudaMemcpy(buf.ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}

// RAII wrapper for device memory
template<typename T>
struct DeviceArray {
    T* ptr = nullptr;
    size_t size = 0;

    explicit DeviceArray(py::array_t<T> arr) {
        ptr = numpy_to_device(arr, size);
    }

    ~DeviceArray() {
        if (ptr) cudaFree(ptr);
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
};

template<typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t size = 0;

    explicit DeviceBuffer(size_t count) : size(count) {
        TC_CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (ptr) cudaFree(ptr);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

py::array_t<float> copy_float_output(const DeviceBuffer<float>& output,
                                     const std::vector<ssize_t>& shape) {
    return device_to_numpy(output.ptr, shape);
}

py::array_t<float> copy_float_output(const DeviceBuffer<float>& output,
                                     std::initializer_list<ssize_t> shape) {
    return device_to_numpy(output.ptr, std::vector<ssize_t>(shape));
}

void sync_cuda() {
    TC_CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<ssize_t> array_shape(const py::buffer_info& buf) {
    return std::vector<ssize_t>(buf.shape.begin(), buf.shape.end());
}

size_t array_size(const py::buffer_info& buf) {
    return static_cast<size_t>(buf.size);
}

void require_same_shape(const py::buffer_info& lhs, const py::buffer_info& rhs, const char* message) {
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
        throw std::invalid_argument(std::string(name) + " must contain exactly " + std::to_string(expected) + " elements");
    }
}

size_t matrix_output_size(int rows, int cols) {
    return static_cast<size_t>(rows) * static_cast<size_t>(cols);
}

template<typename Launcher>
py::array_t<float> run_shape_preserving_float_op(py::array_t<float> input, Launcher&& launch) {
    py::buffer_info buf = input.request();
    DeviceArray<float> d_input(input);
    DeviceBuffer<float> d_output(array_size(buf));

    launch(d_input.ptr, d_output.ptr, array_size(buf));
    sync_cuda();
    return copy_float_output(d_output, array_shape(buf));
}

template<typename Launcher>
py::array_t<float> run_binary_float_op(py::array_t<float> a,
                                       py::array_t<float> b,
                                       Launcher&& launch) {
    py::buffer_info buf_a = a.request();
    py::buffer_info buf_b = b.request();

    require_same_shape(buf_a, buf_b, "Input arrays must have the same shape");

    DeviceArray<float> d_a(a);
    DeviceArray<float> d_b(b);
    DeviceBuffer<float> d_output(array_size(buf_a));

    launch(d_a.ptr, d_b.ptr, d_output.ptr, array_size(buf_a));
    sync_cuda();
    return copy_float_output(d_output, array_shape(buf_a));
}

template<typename Launcher>
py::array_t<float> run_matrix_float_op(py::array_t<float> input, Launcher&& launch) {
    py::buffer_info buf = input.request();

    if (buf.ndim != 2) {
        throw std::runtime_error("Input must be a 2D matrix");
    }

    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);
    DeviceArray<float> d_input(input);
    DeviceBuffer<float> d_output(matrix_output_size(rows, cols));

    launch(d_input.ptr, d_output.ptr, rows, cols);
    sync_cuda();
    return copy_float_output(d_output, {cols, rows});
}

template<typename Launcher>
py::array_t<float> run_last_dim_float_op(py::array_t<float> input, Launcher&& launch) {
    py::buffer_info buf = input.request();

    require_non_empty_last_dim(buf, "Input must have at least 1 dimension with a non-empty last axis");

    int cols = static_cast<int>(buf.shape[buf.ndim - 1]);
    int rows = static_cast<int>(buf.size / cols);
    DeviceArray<float> d_input(input);
    DeviceBuffer<float> d_output(array_size(buf));

    launch(d_input.ptr, d_output.ptr, rows, cols);
    sync_cuda();
    return copy_float_output(d_output, array_shape(buf));
}

template<typename Launcher>
py::array_t<float> run_norm_float_op(py::array_t<float> input, Launcher&& launch) {
    py::buffer_info buf = input.request();

    if (buf.ndim < 2) {
        throw std::invalid_argument("Input must have at least 2 dimensions");
    }
    require_non_empty_last_dim(buf, "Input must have a non-empty hidden dimension");

    int hidden_size = static_cast<int>(buf.shape[buf.ndim - 1]);
    int batch_size = static_cast<int>(buf.size / hidden_size);
    DeviceArray<float> d_input(input);
    DeviceBuffer<float> d_output(array_size(buf));

    launch(d_input.ptr, d_output.ptr, batch_size, hidden_size);
    sync_cuda();
    return copy_float_output(d_output, array_shape(buf));
}

template<typename Launcher>
py::array_t<float> run_gemm_float_op(py::array_t<float> A,
                                     py::array_t<float> B,
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

    DeviceArray<float> d_A(A);
    DeviceArray<float> d_B(B);
    DeviceBuffer<float> d_C(matrix_output_size(M, N));
    TC_CUDA_CHECK(cudaMemset(d_C.ptr, 0, matrix_output_size(M, N) * sizeof(float)));

    launch(d_A.ptr, d_B.ptr, d_C.ptr, M, N, K);
    sync_cuda();
    return copy_float_output(d_C, {M, N});
}

py::array_t<float> py_relu(py::array_t<float> input) {
    return run_shape_preserving_float_op(input, [](const float* in, float* out, size_t n) {
        python_ops::relu(in, out, n);
    });
}

py::array_t<float> py_silu(py::array_t<float> input) {
    return run_shape_preserving_float_op(input, [](const float* in, float* out, size_t n) {
        python_ops::silu(in, out, n);
    });
}

py::array_t<float> py_gelu(py::array_t<float> input) {
    return run_shape_preserving_float_op(input, [](const float* in, float* out, size_t n) {
        python_ops::gelu(in, out, n);
    });
}

py::array_t<float> py_sigmoid(py::array_t<float> input) {
    return run_shape_preserving_float_op(input, [](const float* in, float* out, size_t n) {
        python_ops::sigmoid(in, out, n);
    });
}

py::array_t<float> py_vector_add(py::array_t<float> a, py::array_t<float> b) {
    return run_binary_float_op(a, b, [](const float* lhs, const float* rhs, float* out, size_t n) {
        python_ops::vector_add(lhs, rhs, out, n);
    });
}

py::array_t<float> py_softmax(py::array_t<float> input) {
    return run_last_dim_float_op(input, [](const float* in, float* out, int rows, int cols) {
        python_ops::softmax(in, out, rows, cols);
    });
}

py::array_t<float> py_layernorm(
    py::array_t<float> input,
    py::array_t<float> gamma,
    py::array_t<float> beta,
    float eps = 1e-5f) {
    py::buffer_info input_buf = input.request();
    require_non_empty_last_dim(input_buf, "Input must have a non-empty hidden dimension");
    ssize_t hidden_size = input_buf.shape[input_buf.ndim - 1];
    require_vector_size(gamma.request(), hidden_size, "gamma");
    require_vector_size(beta.request(), hidden_size, "beta");

    DeviceArray<float> d_gamma(gamma);
    DeviceArray<float> d_beta(beta);
    return run_norm_float_op(input, [&](const float* in, float* out, int batch_size, int hidden_size_int) {
        python_ops::layernorm(in, d_gamma.ptr, d_beta.ptr, out, batch_size, hidden_size_int, eps);
    });
}

py::array_t<float> py_rmsnorm(
    py::array_t<float> input,
    py::array_t<float> weight,
    float eps = 1e-6f) {
    py::buffer_info input_buf = input.request();
    require_non_empty_last_dim(input_buf, "Input must have a non-empty hidden dimension");
    ssize_t hidden_size = input_buf.shape[input_buf.ndim - 1];
    require_vector_size(weight.request(), hidden_size, "weight");

    DeviceArray<float> d_weight(weight);
    return run_norm_float_op(input, [&](const float* in, float* out, int batch_size, int hidden_size_int) {
        python_ops::rmsnorm(in, d_weight.ptr, out, batch_size, hidden_size_int, eps);
    });
}

py::array_t<float> py_gemm(
    py::array_t<float> A,
    py::array_t<float> B,
    float alpha = 1.0f,
    float beta = 0.0f,
    const std::string& version = "tiled") {
    return run_gemm_float_op(A, B, [&](const float* lhs, const float* rhs, float* out, int M, int N, int K) {
        python_ops::gemm(lhs, rhs, out, M, N, K, alpha, beta, version.c_str());
    });
}

py::array_t<float> py_transpose(py::array_t<float> input) {
    return run_matrix_float_op(input, [](const float* in, float* out, int rows, int cols) {
        python_ops::transpose(in, out, rows, cols);
    });
}

PYBIND11_MODULE(tensorcraft_ops, m) {
    m.doc() = "TensorCraft-HPC: High-Performance AI Kernels";

    m.def("relu", &py_relu, "ReLU activation", py::arg("input"));
    m.def("silu", &py_silu, "SiLU (Swish) activation", py::arg("input"));
    m.def("gelu", &py_gelu, "GeLU activation", py::arg("input"));
    m.def("sigmoid", &py_sigmoid, "Sigmoid activation", py::arg("input"));
    m.def("vector_add", &py_vector_add, "Element-wise vector addition", py::arg("a"), py::arg("b"));
    m.def("softmax", &py_softmax, "Softmax along last dimension", py::arg("input"));
    m.def("layernorm", &py_layernorm, "Layer normalization",
          py::arg("input"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f);
    m.def("rmsnorm", &py_rmsnorm, "RMS normalization",
          py::arg("input"), py::arg("weight"), py::arg("eps") = 1e-6f);
    m.def("gemm", &py_gemm, "General matrix multiplication",
          py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
          py::arg("version") = "tiled");
    m.def("transpose", &py_transpose, "Matrix transpose", py::arg("input"));

    m.attr("__version__") = "0.1.0";
}

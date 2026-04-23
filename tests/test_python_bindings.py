import numpy as np
import tensorcraft_ops as tc

EXPECTED_EXPORTS = {
    "relu",
    "silu",
    "gelu",
    "sigmoid",
    "vector_add",
    "softmax",
    "layernorm",
    "rmsnorm",
    "gemm",
    "transpose",
    "__version__",
}


for name in EXPECTED_EXPORTS:
    if not hasattr(tc, name):
        raise AssertionError(f"missing export: {name}")

x = np.array([[-1.0, 0.0], [1.0, 2.0]], dtype=np.float32)
sigmoid_expected = 1.0 / (1.0 + np.exp(-x))
np.testing.assert_allclose(tc.sigmoid(x), sigmoid_expected, rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(tc.transpose(x), x.T, rtol=1e-6, atol=1e-6)

a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
np.testing.assert_allclose(tc.gemm(a, b), a @ b, rtol=1e-5, atol=1e-5)

try:
    tc.gemm(a, b, version="bogus")
    raise AssertionError("expected tc.gemm(..., version='bogus') to fail")
except ValueError:
    pass

print(f"tensorcraft_ops smoke test passed: version={tc.__version__}")

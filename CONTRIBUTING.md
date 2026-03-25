# 贡献指南

感谢您对 TensorCraft-HPC 的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告问题

如果您发现 bug 或有功能建议，请在 GitHub Issues 中提交：

1. 搜索现有 issues，避免重复
2. 使用清晰的标题描述问题
3. 提供复现步骤（如果是 bug）
4. 包含环境信息：
   - 操作系统
   - CUDA 版本
   - GPU 型号
   - 编译器版本

### 提交代码

1. Fork 仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'Add some feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request

## 代码规范

### C++ 风格

- 使用 C++17 作为基础标准
- 遵循 Google C++ Style Guide（有以下例外）
- 缩进：4 空格
- 命名约定：
  - 类名：`PascalCase`
  - 函数名：`snake_case`
  - 变量名：`snake_case`
  - 常量：`kConstantName` 或 `CONSTANT_NAME`
  - 模板参数：`PascalCase`

```cpp
// 示例
template<typename T, int TileSize = 32>
class GemmKernel {
public:
    void launch(const T* A, const T* B, T* C, int M, int N, int K);
    
private:
    static constexpr int kBlockSize = 256;
    int tile_size_ = TileSize;
};
```

### CUDA 风格

- Kernel 函数使用 `__global__` 前缀
- Device 函数使用 `__device__ __forceinline__`
- 使用 `__restrict__` 提示编译器
- 显式指定 `__launch_bounds__`

```cpp
template<typename T>
__global__ void __launch_bounds__(256, 4)
my_kernel(const T* __restrict__ input, T* __restrict__ output, size_t n) {
    // ...
}
```

### 文档

- 所有公共 API 需要文档注释
- 使用 Doxygen 风格注释
- 复杂算法需要解释原理

```cpp
/**
 * @brief 执行矩阵乘法 C = alpha * A * B + beta * C
 * 
 * @tparam T 数据类型 (float, half, etc.)
 * @param A 输入矩阵 A [M x K]
 * @param B 输入矩阵 B [K x N]
 * @param C 输出矩阵 C [M x N]
 * @param M 矩阵 A 的行数
 * @param N 矩阵 B 的列数
 * @param K 矩阵 A 的列数 / B 的行数
 * @param alpha 缩放因子
 * @param beta 累加因子
 * @param stream CUDA stream
 */
template<typename T>
void gemm(const T* A, const T* B, T* C, int M, int N, int K,
          float alpha = 1.0f, float beta = 0.0f,
          cudaStream_t stream = 0);
```

## 测试要求

### 单元测试

- 所有新功能需要测试
- 使用 GoogleTest 框架
- 测试文件放在 `tests/` 目录

```cpp
TEST(GemmTest, BasicMultiplication) {
    // 准备数据
    std::vector<float> A = {...};
    std::vector<float> B = {...};
    std::vector<float> C(M * N);
    
    // 执行
    tensorcraft::kernels::gemm(d_A, d_B, d_C, M, N, K);
    
    // 验证
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(C[i], expected[i], 1e-5f);
    }
}
```

### 运行测试

```bash
# 推荐的日常 CUDA 开发验证路径
cmake --preset dev
cmake --build --preset dev --parallel 2
ctest --preset dev --output-on-failure

# 调试配置下运行测试
cmake --preset debug
cmake --build --preset debug --parallel 2
ctest --preset debug --output-on-failure
```

当前 GitHub CI 主要覆盖 CPU configure/install smoke 和 Python packaging smoke；真实 CUDA 构建与测试仍需要在具备 CUDA 的机器上验证。

## 性能基准

### 添加基准测试

新算子应该包含性能基准：

```cpp
static void BM_Gemm_Tiled(benchmark::State& state) {
    int M = state.range(0);
    int N = state.range(0);
    int K = state.range(0);
    
    // 准备数据...
    
    for (auto _ : state) {
        tensorcraft::kernels::launch_gemm(d_A, d_B, d_C, M, N, K,
                                          1.0f, 0.0f, GemmVersion::Tiled);
        cudaDeviceSynchronize();
    }
    
    // 报告 GFLOPS
    double flops = 2.0 * M * N * K;
    state.SetItemsProcessed(state.iterations() * flops);
}

BENCHMARK(BM_Gemm_Tiled)->RangeMultiplier(2)->Range(256, 4096);
```

### 运行基准测试

```bash
cmake --preset release
cmake --build --preset release --parallel 2
./build/release/benchmarks/gemm_benchmark --benchmark_format=console
```

## Pull Request 检查清单

提交 PR 前请确认：

- [ ] 代码遵循项目风格规范
- [ ] 所有测试通过
- [ ] 新功能有对应测试
- [ ] 文档已更新
- [ ] 没有编译警告
- [ ] 性能没有明显退化

## 开发环境设置

### 推荐工具

- **IDE**: VS Code + CUDA 扩展 / CLion
- **格式化**: clang-format
- **静态分析**: clang-tidy
- **性能分析**: Nsight Compute, Nsight Systems

### 构建调试版本

```bash
cmake --preset debug
cmake --build --preset debug --parallel 2
```

### 使用 Sanitizers

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"
```

## 社区准则

- 尊重所有贡献者
- 建设性地提供反馈
- 保持专业和友好
- 遵循 [Contributor Covenant](https://www.contributor-covenant.org/)

## 联系方式

- GitHub Issues: 技术问题和功能请求
- Discussions: 一般讨论和问答

感谢您的贡献！🎉

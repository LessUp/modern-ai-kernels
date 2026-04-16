# Contributing to TensorCraft-HPC

Thank you for your interest in contributing to TensorCraft-HPC! We welcome all forms of contributions.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature suggestion, please submit it via GitHub Issues:

1. Search existing issues to avoid duplicates
2. Use a clear title to describe the problem
3. Provide reproduction steps (if it's a bug)
4. Include environment information:
   - Operating system
   - CUDA version
   - GPU model
   - Compiler version

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes: `git commit -m 'Add some feature'`
4. Push the branch: `git push origin feature/your-feature`
5. Create a Pull Request

## Code Style

### C++ Style

- Use C++17 as the base standard
- Follow Google C++ Style Guide (with exceptions below)
- Indentation: 4 spaces
- Naming conventions:
  - Class names: `PascalCase`
  - Function names: `snake_case`
  - Variable names: `snake_case`
  - Constants: `kConstantName` or `CONSTANT_NAME`
  - Template parameters: `PascalCase`

### CUDA Style

- Kernel functions use `__global__` prefix
- Device functions use `__device__ __forceinline__`
- Use `__restrict__` to hint the compiler
- Explicitly specify `__launch_bounds__`

### Documentation

- All public APIs need documentation comments
- Use Doxygen style comments
- Complex algorithms need explanation

## Testing

### Unit Tests

- All new features need tests
- Use GoogleTest framework
- Test files go in `tests/` directory

### Running Tests

```bash
cmake --preset dev
cmake --build --preset dev --parallel 2
ctest --preset dev --output-on-failure
```

## Pull Request Checklist

Before submitting a PR, please confirm:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New features have corresponding tests
- [ ] Documentation is updated
- [ ] No compiler warnings
- [ ] No significant performance regression

## Contact

- GitHub Issues: Technical issues and feature requests
- Discussions: General discussion and Q&A

Thank you for your contribution! 🎉

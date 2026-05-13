# TensorCraft-HPC Python Bindings
# Direct kernel bindings - no shallow intermediate layer
pybind11_add_module(tensorcraft_ops bindings.cpp)

target_link_libraries(tensorcraft_ops PRIVATE tensorcraft CUDA::cudart)

# Bindings need CUDA compilation for kernel launches
set_source_files_properties(bindings.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(
  tensorcraft_ops
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
             CUDA_STANDARD_REQUIRED ON
             CUDA_SEPARABLE_COMPILATION OFF
             CUDA_RESOLVE_DEVICE_SYMBOLS OFF
             POSITION_INDEPENDENT_CODE ON
             CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
             CUDA_RUNTIME_LIBRARY Shared)

target_compile_features(tensorcraft_ops PRIVATE cxx_std_17)

install(
  TARGETS tensorcraft_ops
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .)

if(TC_BUILD_TESTS)
  find_package(
    Python3
    COMPONENTS Interpreter
    REQUIRED)

  add_test(
    NAME tensorcraft_python_smoke
    COMMAND
      ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:tensorcraft_ops>
      ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/test_python_bindings.py)
endif()

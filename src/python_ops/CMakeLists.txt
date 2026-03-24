# TensorCraft-HPC Python Bindings
pybind11_add_module(tensorcraft_ops
    bindings.cpp
)

set_source_files_properties(bindings.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(tensorcraft_ops PRIVATE
    tensorcraft
    CUDA::cudart
)

set_target_properties(tensorcraft_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
)

# Install Python module
if(DEFINED SKBUILD_PLATLIB_DIR)
    set(TC_PYTHON_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}")
else()
    set(TC_PYTHON_INSTALL_DIR ".")
endif()

install(TARGETS tensorcraft_ops
    LIBRARY DESTINATION ${TC_PYTHON_INSTALL_DIR}
    RUNTIME DESTINATION ${TC_PYTHON_INSTALL_DIR}
)

if(TC_BUILD_TESTS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    add_test(
        NAME tensorcraft_python_smoke
        COMMAND ${CMAKE_COMMAND} -E env
            PYTHONPATH=$<TARGET_FILE_DIR:tensorcraft_ops>
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tests/test_python_bindings.py
    )
endif()

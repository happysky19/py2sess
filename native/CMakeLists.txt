find_package(Torch REQUIRED)

find_library(TORCH_PYTHON_LIBRARY
  NAMES torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib"
  NO_DEFAULT_PATH
)
if(NOT TORCH_PYTHON_LIBRARY)
  message(FATAL_ERROR "Could not find libtorch_python in ${TORCH_INSTALL_PREFIX}/lib")
endif()

find_library(PY2SESS_TORCH_LIBRARY
  NAMES torch
  PATHS "${TORCH_INSTALL_PREFIX}/lib"
  NO_DEFAULT_PATH
)
find_library(PY2SESS_TORCH_CPU_LIBRARY
  NAMES torch_cpu
  PATHS "${TORCH_INSTALL_PREFIX}/lib"
  NO_DEFAULT_PATH
)
find_library(PY2SESS_C10_LIBRARY
  NAMES c10
  PATHS "${TORCH_INSTALL_PREFIX}/lib"
  NO_DEFAULT_PATH
)
foreach(_py2sess_torch_cpu_lib
    PY2SESS_TORCH_LIBRARY
    PY2SESS_TORCH_CPU_LIBRARY
    PY2SESS_C10_LIBRARY)
  if(NOT ${_py2sess_torch_cpu_lib})
    message(FATAL_ERROR
      "Could not find ${_py2sess_torch_cpu_lib} in ${TORCH_INSTALL_PREFIX}/lib")
  endif()
endforeach()
set(PY2SESS_TORCH_CPU_LIBRARIES
  ${PY2SESS_TORCH_LIBRARY}
  ${PY2SESS_TORCH_CPU_LIBRARY}
  ${PY2SESS_C10_LIBRARY}
)

if(APPLE)
  set(PY2SESS_NATIVE_INSTALL_RPATH "@loader_path")
elseif(UNIX)
  set(PY2SESS_NATIVE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../torch/lib;$ORIGIN/../nvidia/cuda_runtime/lib;$ORIGIN/../nvidia/cuda_nvrtc/lib")
else()
  set(PY2SESS_NATIVE_INSTALL_RPATH "")
endif()

function(py2sess_configure_native_core target_name)
  target_include_directories(${target_name}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc
      ${TORCH_INCLUDE_DIRS}
  )
  target_link_libraries(${target_name}
    PRIVATE
      ${PY2SESS_TORCH_CPU_LIBRARIES}
  )
  set_target_properties(${target_name}
    PROPERTIES
      POSITION_INDEPENDENT_CODE ON
  )
endfunction()

function(py2sess_configure_pybind_module target_name output_name)
  target_include_directories(${target_name}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc
      ${TORCH_INCLUDE_DIRS}
  )
  target_link_libraries(${target_name}
    PRIVATE
      Python3::Module
      ${TORCH_PYTHON_LIBRARY}
      ${PY2SESS_TORCH_CPU_LIBRARIES}
  )
  set_target_properties(${target_name}
    PROPERTIES
      PREFIX ""
      OUTPUT_NAME "${output_name}"
      POSITION_INDEPENDENT_CODE ON
      INSTALL_RPATH "${PY2SESS_NATIVE_INSTALL_RPATH}"
  )
endfunction()

add_library(py2sess_native_core SHARED
  csrc/native_dispatch.cpp
  csrc/native_module.cpp
)

py2sess_configure_native_core(py2sess_native_core)
set_target_properties(py2sess_native_core
  PROPERTIES
    INSTALL_RPATH "${PY2SESS_NATIVE_INSTALL_RPATH}"
)

if(PY2SESS_NATIVE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(_py2sess_cuda_architectures 70 75 80 86 89)
    if(DEFINED CUDAToolkit_VERSION AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
      list(APPEND _py2sess_cuda_architectures 90)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${_py2sess_cuda_architectures})
  endif()
  message(STATUS "py2sess CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

  add_library(py2sess_native_cuda SHARED
    csrc/native_dispatch_cuda.cu
  )
  target_include_directories(py2sess_native_cuda
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc
      ${TORCH_INCLUDE_DIRS}
  )
  target_link_libraries(py2sess_native_cuda
    PRIVATE
      CUDA::cudart
      ${TORCH_LIBRARIES}
      ${TORCH_CUDA_LIBRARY}
      ${C10_CUDA_LIBRARY}
  )
  target_compile_definitions(py2sess_native_cuda
    PRIVATE
      PY2SESS_WITH_CUDA=1
  )
  set_target_properties(py2sess_native_cuda
    PROPERTIES
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      CUDA_STANDARD 17
      CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON
      INSTALL_RPATH "${PY2SESS_NATIVE_INSTALL_RPATH}"
  )

  add_library(py2sess_native_core_cuda OBJECT
    csrc/native_dispatch.cpp
  )
  py2sess_configure_native_core(py2sess_native_core_cuda)
  target_compile_definitions(py2sess_native_core_cuda
    PRIVATE
      PY2SESS_WITH_CUDA=1
  )
endif()

add_library(py2sess_native_pybind MODULE
  csrc/native_bindings.cpp
)

target_compile_definitions(py2sess_native_pybind
  PRIVATE
    TORCH_EXTENSION_NAME=_native
    PY2SESS_BIND_NATIVE_MODULE_CLASS=1
)

py2sess_configure_pybind_module(py2sess_native_pybind "_native")
target_link_libraries(py2sess_native_pybind
  PRIVATE
    py2sess_native_core
)

if(PY2SESS_NATIVE_CUDA)
  add_library(py2sess_native_cuda_pybind MODULE
    csrc/native_bindings.cpp
    $<TARGET_OBJECTS:py2sess_native_core_cuda>
  )
  target_compile_definitions(py2sess_native_cuda_pybind
    PRIVATE
      TORCH_EXTENSION_NAME=_native_cuda
      PY2SESS_WITH_CUDA=1
      PY2SESS_BIND_NATIVE_MODULE_CLASS=0
  )
  py2sess_configure_pybind_module(py2sess_native_cuda_pybind "_native_cuda")
  target_link_libraries(py2sess_native_cuda_pybind
    PRIVATE
      py2sess_native_cuda
  )
endif()

install(TARGETS py2sess_native_core py2sess_native_pybind
  LIBRARY DESTINATION py2sess
  RUNTIME DESTINATION py2sess
  ARCHIVE DESTINATION py2sess
)

if(PY2SESS_NATIVE_CUDA)
  install(TARGETS py2sess_native_cuda py2sess_native_cuda_pybind
    LIBRARY DESTINATION py2sess
    RUNTIME DESTINATION py2sess
    ARCHIVE DESTINATION py2sess
  )
endif()

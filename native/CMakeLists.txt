find_package(Torch REQUIRED)

find_library(TORCH_PYTHON_LIBRARY
  NAMES torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib"
  NO_DEFAULT_PATH
)
if(NOT TORCH_PYTHON_LIBRARY)
  message(FATAL_ERROR "Could not find libtorch_python in ${TORCH_INSTALL_PREFIX}/lib")
endif()

if(APPLE)
  set(PY2SESS_NATIVE_INSTALL_RPATH "@loader_path")
elseif(UNIX)
  set(PY2SESS_NATIVE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../torch/lib")
else()
  set(PY2SESS_NATIVE_INSTALL_RPATH "")
endif()

add_library(py2sess_native_core SHARED
  csrc/native_dispatch.cpp
  csrc/native_module.cpp
)

target_include_directories(py2sess_native_core
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

target_link_libraries(py2sess_native_core
  PRIVATE
    ${TORCH_LIBRARIES}
)

if(PY2SESS_NATIVE_CUDA)
  target_compile_definitions(py2sess_native_core
    PRIVATE
      PY2SESS_WITH_CUDA=1
  )
endif()

set_target_properties(py2sess_native_core
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    INSTALL_RPATH "${PY2SESS_NATIVE_INSTALL_RPATH}"
)

if(PY2SESS_NATIVE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_library(py2sess_native_cuda SHARED
    csrc/native_dispatch_cuda.cu
  )
  target_include_directories(py2sess_native_cuda
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/csrc
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
      CUDA_STANDARD 17
      CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON
      INSTALL_RPATH "${PY2SESS_NATIVE_INSTALL_RPATH}"
  )
endif()

add_library(py2sess_native_pybind MODULE
  csrc/native_bindings.cpp
)

target_include_directories(py2sess_native_pybind
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

target_compile_definitions(py2sess_native_pybind
  PRIVATE
    TORCH_EXTENSION_NAME=_native
)

target_link_libraries(py2sess_native_pybind
  PRIVATE
    py2sess_native_core
    Python3::Module
    ${TORCH_PYTHON_LIBRARY}
    ${TORCH_LIBRARIES}
)

if(PY2SESS_NATIVE_CUDA)
  target_link_libraries(py2sess_native_pybind
    PRIVATE
      py2sess_native_cuda
  )
  target_compile_definitions(py2sess_native_pybind
    PRIVATE
      PY2SESS_WITH_CUDA=1
  )
endif()

set_target_properties(py2sess_native_pybind
  PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_native"
    POSITION_INDEPENDENT_CODE ON
    INSTALL_RPATH "${PY2SESS_NATIVE_INSTALL_RPATH}"
)

install(TARGETS py2sess_native_core py2sess_native_pybind
  LIBRARY DESTINATION py2sess
  RUNTIME DESTINATION py2sess
  ARCHIVE DESTINATION py2sess
)

if(PY2SESS_NATIVE_CUDA)
  install(TARGETS py2sess_native_cuda
    LIBRARY DESTINATION py2sess
    RUNTIME DESTINATION py2sess
    ARCHIVE DESTINATION py2sess
  )
endif()

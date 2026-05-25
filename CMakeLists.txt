cmake_minimum_required(VERSION 3.18)

if(DEFINED SKBUILD_PROJECT_VERSION)
  set(PY2SESS_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
  set(PY2SESS_VERSION "0.4.1")
endif()

project(py2sess_native
  LANGUAGES CXX
  DESCRIPTION "Native C++/CUDA extension scaffold for py2sess")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PY2SESS_BUILD_NATIVE "Build the optional py2sess native extension" OFF)
option(PY2SESS_NATIVE_CUDA "Build CUDA native dispatch stubs" OFF)

if(PY2SESS_BUILD_NATIVE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch, pathlib; print(pathlib.Path(torch.utils.cmake_prefix_path).as_posix())"
    RESULT_VARIABLE _TORCH_PREFIX_RESULT
    OUTPUT_VARIABLE _TORCH_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(_TORCH_PREFIX_RESULT EQUAL 0 AND NOT _TORCH_PREFIX STREQUAL "")
    list(APPEND CMAKE_PREFIX_PATH "${_TORCH_PREFIX}")
  endif()

  if(PY2SESS_NATIVE_CUDA)
    enable_language(CUDA)
  endif()

  add_subdirectory(native)
else()
  find_package(Python REQUIRED COMPONENTS Interpreter)
  add_custom_target(
    py2sess_python_sources
    ALL
    COMMAND ${CMAKE_COMMAND} -E echo
            "Preparing py2sess ${PY2SESS_VERSION} for Python ${Python_VERSION}"
    VERBATIM
  )
  install(CODE "message(STATUS \"py2sess ${PY2SESS_VERSION} has no native CMake artifacts in this build\")")
endif()

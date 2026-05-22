cmake_minimum_required(VERSION 3.18)

project(py2sess_native
  LANGUAGES CXX
  DESCRIPTION "Native C++/CUDA extension scaffold for py2sess")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(PY2SESS_NATIVE_CUDA "Build CUDA native dispatch stubs" OFF)

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

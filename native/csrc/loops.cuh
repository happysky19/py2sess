#pragma once

#include <ATen/ATen.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#ifndef PY2SESS_NATIVE_BLOCK_SIZE
#define PY2SESS_NATIVE_BLOCK_SIZE 64
#endif

namespace py2sess_native {

static_assert(PY2SESS_NATIVE_BLOCK_SIZE > 0, "PY2SESS_NATIVE_BLOCK_SIZE must be positive");

template <typename Func>
__global__ void element_kernel(int64_t numel, Func f, char* work) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    f(idx, work);
  }
}

template <int Arity, typename Func>
void gpu_kernel(at::TensorIterator& iter, const Func& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; ++i) {
    data[i] = reinterpret_cast<char*>(iter.data_ptr(i));
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  const int64_t numel = iter.numel();

  at::native::launch_legacy_kernel<128, 1>(numel, [=] __device__(int idx) {
    auto offsets = offset_calc.get(idx);
    f(data.data(), offsets.data());
  });
}

template <int Chunks, int Arity, typename Func>
void gpu_chunk_kernel(at::TensorIterator& iter, int work_size, const Func& f) {
  TORCH_CHECK(iter.ninputs() + iter.noutputs() == Arity);

  std::array<char*, Arity> data;
  for (int i = 0; i < Arity; ++i) {
    data[i] = reinterpret_cast<char*>(iter.data_ptr(i));
  }

  auto offset_calc = ::make_offset_calculator<Arity>(iter);
  const int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  const int64_t chunks = Chunks > numel ? numel : Chunks;
  const int64_t base = numel / chunks;
  const int64_t rem = numel % chunks;
  const size_t workspace_bytes = static_cast<size_t>(work_size) * (base + (rem > 0 ? 1 : 0));
  auto workspace =
      at::empty({static_cast<int64_t>(workspace_bytes)}, iter.input(0).options().dtype(at::kByte));
  auto* d_workspace = static_cast<char*>(workspace.data_ptr());
  const auto stream = c10::cuda::getCurrentCUDAStream();

  int64_t chunk_start = 0;
  for (int64_t chunk = 0; chunk < chunks; ++chunk) {
    const int64_t chunk_numel = base + (chunk < rem ? 1 : 0);

    dim3 block(PY2SESS_NATIVE_BLOCK_SIZE);
    dim3 grid((chunk_numel + block.x - 1) / block.x);
    auto device_lambda = [=] __device__(int idx, char* work) {
      auto offsets = offset_calc.get(idx + chunk_start);
      f(data.data(), offsets.data(), work + idx * work_size);
    };

    element_kernel<<<grid, block, 0, stream.stream()>>>(
        chunk_numel, device_lambda, d_workspace);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    chunk_start += chunk_numel;
  }
}

}  // namespace py2sess_native

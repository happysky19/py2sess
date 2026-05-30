#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "loops.cuh"
#include "native_dispatch.hpp"
#include "thermal_2s_impl.hpp"

namespace py2sess_native {

namespace {

int last_dim_size(const at::TensorBase& tensor) {
  TORCH_CHECK(tensor.dim() > 0, "py2sess native kernels expect non-scalar row tensors");
  return static_cast<int>(tensor.size(tensor.dim() - 1));
}

constexpr int kBlock = 128;

dim3 grid_1d(int64_t count) {
  return dim3(static_cast<unsigned int>((count + kBlock - 1) / kBlock));
}

template <typename T>
__global__ void solar_fo_extinction_kernel(
    int64_t rows,
    int nlay,
    const T* tau,
    const T* omega,
    const T* scaling,
    const T* inv_layer_thickness,
    T* extinction) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t count = rows * static_cast<int64_t>(nlay);
  if (idx >= count) {
    return;
  }
  const int layer = static_cast<int>(idx % nlay);
  const T tau_scaled = tau[idx] * (T(1) - omega[idx] * scaling[idx]);
  extinction[idx] = tau_scaled * inv_layer_thickness[layer];
}

template <typename T>
__global__ void solar_fo_fine_attenuation_kernel(
    int64_t rows,
    int nlay,
    int nfine,
    const T* extinction,
    const T* sunpathsfine,
    const std::int64_t* nfinedivs,
    const std::int64_t* ntraversefine,
    T* fine_attenuation) {
  const int fine_slots = nfine * nlay;
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t count = rows * static_cast<int64_t>(fine_slots);
  if (idx >= count) {
    return;
  }
  const int slot = static_cast<int>(idx % fine_slots);
  const int64_t row = idx / fine_slots;
  const int j = slot / nlay;
  const int layer = slot - j * nlay;
  if (j >= static_cast<int>(nfinedivs[layer])) {
    return;
  }
  const int ntrav = static_cast<int>(ntraversefine[j * nlay + layer]);
  const T* row_extinction = extinction + row * nlay;
  T fine_tau = T(0);
  for (int k = 0; k < ntrav; ++k) {
    fine_tau += row_extinction[k] * sunpathsfine[(k * nfine + j) * nlay + layer];
  }
  fine_attenuation[idx] = exp_cutoff(fine_tau, T(88));
}

template <typename T>
__global__ void solar_fo_nonnadir_endpoint_kernel(
    int64_t rows,
    int nlay,
    int nfine,
    int ntrav_nl,
    bool return_components,
    T mu0,
    T rayconv,
    const T* albedo,
    const T* flux_factor,
    const T* exact_scatter,
    const T* sunpathsnl,
    const T* cota,
    const T* cotfine,
    const T* csqfine,
    const T* wfine,
    const std::int64_t* nfinedivs,
    const T* extinction,
    const T* fine_attenuation,
    T* out) {
  const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const int fine_slots = nfine * nlay;
  const T* row_extinction = extinction + row * nlay;
  const T* row_fine_attenuation = fine_attenuation + row * fine_slots;
  const T* row_scatter = exact_scatter + row * nlay;

  T total_tau = T(0);
  for (int k = 0; k < ntrav_nl; ++k) {
    total_tau += row_extinction[k] * sunpathsnl[k];
  }
  const T attenuation_nl = exp_cutoff(total_tau, T(88));
  T cumsource_up = T(0);
  T cumsource_db = T(4) * mu0 * albedo[row] * attenuation_nl;

  for (int n = nlay; n > 0; --n) {
    const int layer = n - 1;
    const T cot_2 = cota[layer];
    const T cot_1 = cota[layer + 1];
    const T ke = rayconv * row_extinction[layer];
    const T lostrans = exp(-ke * (cot_2 - cot_1));
    T layer_sum = T(0);
    const int nfine_layer = static_cast<int>(nfinedivs[layer]);
    for (int j = 0; j < nfine_layer; ++j) {
      const int index = j * nlay + layer;
      const T tran = exp(-ke * (cot_2 - cotfine[index]));
      layer_sum += row_scatter[layer] * row_fine_attenuation[index] * csqfine[index] * tran *
                   wfine[index];
    }
    const T source = layer_sum * ke;
    cumsource_db = lostrans * cumsource_db;
    cumsource_up = lostrans * cumsource_up + source;
  }

  const T pi = T(3.141592653589793238462643383279502884);
  const T scale = T(0.25) * flux_factor[row] / pi;
  const T single_scatter = scale * cumsource_up;
  const T direct_beam = scale * cumsource_db;
  const int out_cols = return_components ? 3 : 1;
  T* row_out = out + row * out_cols;
  row_out[0] = single_scatter + direct_beam;
  if (return_components) {
    row_out[1] = single_scatter;
    row_out[2] = direct_beam;
  }
}

}  // namespace

void thermal_2s_cuda(at::TensorIterator& iter, const Thermal2sParams& params) {
  at::cuda::CUDAGuard device_guard(iter.device());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_thermal_2s_cuda", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const int workspace_size = static_cast<int>(thermal_2s_workspace_bytes<scalar_t>(nlay));
    gpu_chunk_kernel<8, 11>(
        iter,
        workspace_size,
        [=] __device__(char* const data[11], unsigned int offsets[11], char* work) {
          auto* out = reinterpret_cast<scalar_t*>(data[0] + offsets[0]);
          auto* tau = reinterpret_cast<const scalar_t*>(data[1] + offsets[1]);
          auto* omega = reinterpret_cast<const scalar_t*>(data[2] + offsets[2]);
          auto* asymm = reinterpret_cast<const scalar_t*>(data[3] + offsets[3]);
          auto* scaling = reinterpret_cast<const scalar_t*>(data[4] + offsets[4]);
          auto* planck = reinterpret_cast<const scalar_t*>(data[5] + offsets[5]);
          auto* surfbb = reinterpret_cast<const scalar_t*>(data[6] + offsets[6]);
          auto* emissivity = reinterpret_cast<const scalar_t*>(data[7] + offsets[7]);
          auto* albedo = reinterpret_cast<const scalar_t*>(data[8] + offsets[8]);
          auto* brdf_f = reinterpret_cast<const scalar_t*>(data[9] + offsets[9]);
          auto* ubrdf_f = reinterpret_cast<const scalar_t*>(data[10] + offsets[10]);
          thermal_2s_row<scalar_t>(
              nlay,
              static_cast<scalar_t>(params.stream_value),
              static_cast<scalar_t>(params.user_stream),
              static_cast<scalar_t>(params.thermal_tcutoff),
              tau,
              omega,
              asymm,
              scaling,
              planck,
              surfbb,
              emissivity,
              albedo,
              brdf_f,
              ubrdf_f,
              out,
              params.return_profile,
              params.return_fluxes,
              params.do_upwelling,
              params.do_dnwelling,
              params.use_brdf,
              work);
        });
  });
}

void thermal_2s_flux_cuda(at::TensorIterator& iter, const Thermal2sParams& params) {
  at::cuda::CUDAGuard device_guard(iter.device());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_thermal_2s_flux_cuda", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const int nlev = nlay + 1;
    const int packed_cols = 1 + 4 * nlev;
    const int packed_bytes = static_cast<int>(packed_cols * sizeof(scalar_t));
    const int workspace_size =
        packed_bytes + static_cast<int>(thermal_2s_workspace_bytes<scalar_t>(nlay));
    gpu_chunk_kernel<8, 11>(
        iter,
        workspace_size,
        [=] __device__(char* const data[11], unsigned int offsets[11], char* work) {
          auto* out = reinterpret_cast<scalar_t*>(data[0] + offsets[0]);
          auto* tau = reinterpret_cast<const scalar_t*>(data[1] + offsets[1]);
          auto* omega = reinterpret_cast<const scalar_t*>(data[2] + offsets[2]);
          auto* asymm = reinterpret_cast<const scalar_t*>(data[3] + offsets[3]);
          auto* scaling = reinterpret_cast<const scalar_t*>(data[4] + offsets[4]);
          auto* planck = reinterpret_cast<const scalar_t*>(data[5] + offsets[5]);
          auto* surfbb = reinterpret_cast<const scalar_t*>(data[6] + offsets[6]);
          auto* emissivity = reinterpret_cast<const scalar_t*>(data[7] + offsets[7]);
          auto* albedo = reinterpret_cast<const scalar_t*>(data[8] + offsets[8]);
          auto* brdf_f = reinterpret_cast<const scalar_t*>(data[9] + offsets[9]);
          auto* ubrdf_f = reinterpret_cast<const scalar_t*>(data[10] + offsets[10]);
          auto* packed = reinterpret_cast<scalar_t*>(work);
          char* row_work = work + packed_bytes;
          thermal_2s_row<scalar_t>(
              nlay,
              static_cast<scalar_t>(params.stream_value),
              static_cast<scalar_t>(params.user_stream),
              static_cast<scalar_t>(params.thermal_tcutoff),
              tau,
              omega,
              asymm,
              scaling,
              planck,
              surfbb,
              emissivity,
              albedo,
              brdf_f,
              ubrdf_f,
              packed,
              false,
              true,
              params.do_upwelling,
              params.do_dnwelling,
              params.use_brdf,
              row_work);
          const scalar_t* flux_up = packed + 1;
          const scalar_t* flux_down = flux_up + nlev;
          for (int level = 0; level < nlev; ++level) {
            out[2 * level] = flux_up[level];
            out[2 * level + 1] = flux_down[level];
          }
        });
  });
}

void thermal_2s_prop_flux_cuda(at::TensorIterator& iter, const Thermal2sPropParams& params) {
  at::cuda::CUDAGuard device_guard(iter.device());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_thermal_2s_prop_flux_cuda", [&] {
    const int nlay = static_cast<int>(params.nlay);
    const int nprop = static_cast<int>(params.nprop);
    const int staging_size = static_cast<int>((4 * nlay + (nlay + 1)) * sizeof(scalar_t));
    const int workspace_size =
        staging_size +
        static_cast<int>(two_stream_flux_pair_packed_cols<scalar_t>(nlay) * sizeof(scalar_t)) +
        static_cast<int>(thermal_2s_workspace_bytes<scalar_t>(nlay));
    gpu_chunk_kernel<8, 8>(
        iter,
        workspace_size,
        [=] __device__(char* const data[8], unsigned int offsets[8], char* work) {
          auto* out = reinterpret_cast<scalar_t*>(data[0] + offsets[0]);
          auto* prop = reinterpret_cast<const scalar_t*>(data[1] + offsets[1]);
          auto* planck = reinterpret_cast<const scalar_t*>(data[2] + offsets[2]);
          auto* surfbb = reinterpret_cast<const scalar_t*>(data[3] + offsets[3]);
          auto* emissivity = reinterpret_cast<const scalar_t*>(data[4] + offsets[4]);
          auto* albedo = reinterpret_cast<const scalar_t*>(data[5] + offsets[5]);
          auto* brdf_f = reinterpret_cast<const scalar_t*>(data[6] + offsets[6]);
          auto* ubrdf_f = reinterpret_cast<const scalar_t*>(data[7] + offsets[7]);
          thermal_2s_prop_flux_pair_row<scalar_t>(
              nlay,
              nprop,
              params.flip_layers,
              static_cast<scalar_t>(params.stream_value),
              static_cast<scalar_t>(params.user_stream),
              static_cast<scalar_t>(params.thermal_tcutoff),
              prop,
              planck,
              surfbb,
              emissivity,
              albedo,
              brdf_f,
              ubrdf_f,
              out,
              params.do_upwelling,
              params.do_dnwelling,
              params.use_brdf,
              work);
        });
  });
}

void solar_2s_cuda(at::TensorIterator& iter, const Solar2sParams& params) {
  at::cuda::CUDAGuard device_guard(iter.device());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_solar_2s_cuda", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const int workspace_size = static_cast<int>(solar_2s_workspace_bytes<scalar_t>(nlay));
    const auto* chapman = reinterpret_cast<const scalar_t*>(params.chapman);
    const auto* pxsq = reinterpret_cast<const scalar_t*>(params.pxsq);
    const auto* px0x = reinterpret_cast<const scalar_t*>(params.px0x);
    gpu_chunk_kernel<8, 12>(
        iter,
        workspace_size,
        [=] __device__(char* const data[12], unsigned int offsets[12], char* work) {
          auto* out = reinterpret_cast<scalar_t*>(data[0] + offsets[0]);
          auto* tau = reinterpret_cast<const scalar_t*>(data[1] + offsets[1]);
          auto* omega = reinterpret_cast<const scalar_t*>(data[2] + offsets[2]);
          auto* asymm = reinterpret_cast<const scalar_t*>(data[3] + offsets[3]);
          auto* scaling = reinterpret_cast<const scalar_t*>(data[4] + offsets[4]);
          auto* albedo = reinterpret_cast<const scalar_t*>(data[5] + offsets[5]);
          auto* flux_factor = reinterpret_cast<const scalar_t*>(data[6] + offsets[6]);
          auto* brdf_f0 = reinterpret_cast<const scalar_t*>(data[7] + offsets[7]);
          auto* brdf_f = reinterpret_cast<const scalar_t*>(data[8] + offsets[8]);
          auto* ubrdf_f = reinterpret_cast<const scalar_t*>(data[9] + offsets[9]);
          auto* slterm_isotropic = reinterpret_cast<const scalar_t*>(data[10] + offsets[10]);
          auto* slterm_f0 = reinterpret_cast<const scalar_t*>(data[11] + offsets[11]);
          solar_2s_row<scalar_t>(
              nlay,
              static_cast<scalar_t>(params.stream_value),
              static_cast<scalar_t>(params.x0),
              static_cast<scalar_t>(params.user_stream),
              static_cast<scalar_t>(params.user_secant),
              static_cast<scalar_t>(params.azmfac),
              static_cast<scalar_t>(params.px11),
              static_cast<scalar_t>(params.ulp),
              chapman,
              pxsq,
              px0x,
              tau,
              omega,
              asymm,
              scaling,
              albedo,
              flux_factor,
              brdf_f0,
              brdf_f,
              ubrdf_f,
              slterm_isotropic,
              slterm_f0,
              out,
              params.return_profile,
              params.return_fluxes,
              params.do_upwelling,
              params.do_dnwelling,
              params.use_brdf,
              params.use_surface_leaving,
              params.sl_isotropic,
              work);
        });
  });
}

void solar_2s_flux_cuda(at::TensorIterator& iter, const Solar2sParams& params) {
  at::cuda::CUDAGuard device_guard(iter.device());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_solar_2s_flux_cuda", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const int workspace_size = static_cast<int>(solar_2s_flux_workspace_bytes<scalar_t>(nlay));
    const auto* chapman = reinterpret_cast<const scalar_t*>(params.chapman);
    const auto* pxsq = reinterpret_cast<const scalar_t*>(params.pxsq);
    const auto* px0x = reinterpret_cast<const scalar_t*>(params.px0x);
    gpu_chunk_kernel<8, 12>(
        iter,
        workspace_size,
        [=] __device__(char* const data[12], unsigned int offsets[12], char* work) {
          auto* out = reinterpret_cast<scalar_t*>(data[0] + offsets[0]);
          auto* tau = reinterpret_cast<const scalar_t*>(data[1] + offsets[1]);
          auto* omega = reinterpret_cast<const scalar_t*>(data[2] + offsets[2]);
          auto* asymm = reinterpret_cast<const scalar_t*>(data[3] + offsets[3]);
          auto* scaling = reinterpret_cast<const scalar_t*>(data[4] + offsets[4]);
          auto* albedo = reinterpret_cast<const scalar_t*>(data[5] + offsets[5]);
          auto* flux_factor = reinterpret_cast<const scalar_t*>(data[6] + offsets[6]);
          auto* brdf_f0 = reinterpret_cast<const scalar_t*>(data[7] + offsets[7]);
          auto* brdf_f = reinterpret_cast<const scalar_t*>(data[8] + offsets[8]);
          auto* ubrdf_f = reinterpret_cast<const scalar_t*>(data[9] + offsets[9]);
          auto* slterm_isotropic = reinterpret_cast<const scalar_t*>(data[10] + offsets[10]);
          auto* slterm_f0 = reinterpret_cast<const scalar_t*>(data[11] + offsets[11]);
          solar_2s_flux_pair_row<scalar_t>(
              nlay,
              static_cast<scalar_t>(params.stream_value),
              static_cast<scalar_t>(params.x0),
              static_cast<scalar_t>(params.user_stream),
              static_cast<scalar_t>(params.user_secant),
              static_cast<scalar_t>(params.azmfac),
              static_cast<scalar_t>(params.px11),
              static_cast<scalar_t>(params.ulp),
              chapman,
              pxsq,
              px0x,
              tau,
              omega,
              asymm,
              scaling,
              albedo,
              flux_factor,
              brdf_f0,
              brdf_f,
              ubrdf_f,
              slterm_isotropic,
              slterm_f0,
              out,
              params.do_upwelling,
              params.do_dnwelling,
              params.use_brdf,
              params.use_surface_leaving,
              params.sl_isotropic,
              params.plane_parallel_chapman,
              work);
        });
  });
}

void solar_2s_prop_flux_cuda(at::TensorIterator& iter, const Solar2sPropParams& params) {
  at::cuda::CUDAGuard device_guard(iter.device());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_solar_2s_prop_flux_cuda", [&] {
    const int nlay = static_cast<int>(params.nlay);
    const int nprop = static_cast<int>(params.nprop);
    const int staging_size = static_cast<int>(4 * nlay * sizeof(scalar_t));
    const int workspace_size =
        staging_size + static_cast<int>(solar_2s_flux_workspace_bytes<scalar_t>(nlay));
    const auto* chapman = reinterpret_cast<const scalar_t*>(params.chapman);
    const auto* pxsq = reinterpret_cast<const scalar_t*>(params.pxsq);
    const auto* px0x = reinterpret_cast<const scalar_t*>(params.px0x);
    gpu_chunk_kernel<8, 9>(
        iter,
        workspace_size,
        [=] __device__(char* const data[9], unsigned int offsets[9], char* work) {
          auto* out = reinterpret_cast<scalar_t*>(data[0] + offsets[0]);
          auto* prop = reinterpret_cast<const scalar_t*>(data[1] + offsets[1]);
          auto* albedo = reinterpret_cast<const scalar_t*>(data[2] + offsets[2]);
          auto* flux_factor = reinterpret_cast<const scalar_t*>(data[3] + offsets[3]);
          auto* brdf_f0 = reinterpret_cast<const scalar_t*>(data[4] + offsets[4]);
          auto* brdf_f = reinterpret_cast<const scalar_t*>(data[5] + offsets[5]);
          auto* ubrdf_f = reinterpret_cast<const scalar_t*>(data[6] + offsets[6]);
          auto* slterm_isotropic = reinterpret_cast<const scalar_t*>(data[7] + offsets[7]);
          auto* slterm_f0 = reinterpret_cast<const scalar_t*>(data[8] + offsets[8]);
          solar_2s_prop_flux_pair_row<scalar_t>(
              nlay,
              nprop,
              params.flip_layers,
              static_cast<scalar_t>(params.stream_value),
              static_cast<scalar_t>(params.x0),
              static_cast<scalar_t>(params.user_stream),
              static_cast<scalar_t>(params.user_secant),
              static_cast<scalar_t>(params.azmfac),
              static_cast<scalar_t>(params.px11),
              static_cast<scalar_t>(params.ulp),
              chapman,
              pxsq,
              px0x,
              prop,
              albedo,
              flux_factor,
              brdf_f0,
              brdf_f,
              ubrdf_f,
              slterm_isotropic,
              slterm_f0,
              out,
              params.do_upwelling,
              params.do_dnwelling,
              params.use_brdf,
              params.use_surface_leaving,
              params.sl_isotropic,
              params.plane_parallel_chapman,
              work);
        });
  });
}

void thermal_fo_cuda(at::TensorIterator& iter, const ThermalFoParams& params) {
  at::cuda::CUDAGuard device_guard(iter.device());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_thermal_fo_cuda", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const int workspace_size = static_cast<int>(thermal_fo_workspace_bytes<scalar_t>(nlay));
    const auto* height_delta = reinterpret_cast<const scalar_t*>(params.height_delta);
    const auto* xfine = reinterpret_cast<const scalar_t*>(params.xfine);
    const auto* wfine = reinterpret_cast<const scalar_t*>(params.wfine);
    const auto* cota = reinterpret_cast<const scalar_t*>(params.cota);
    const auto* cotfine = reinterpret_cast<const scalar_t*>(params.cotfine);
    const auto* csqfine = reinterpret_cast<const scalar_t*>(params.csqfine);
    gpu_chunk_kernel<8, 7>(
        iter,
        workspace_size,
        [=] __device__(char* const data[7], unsigned int offsets[7], char* work) {
          auto* out = reinterpret_cast<scalar_t*>(data[0] + offsets[0]);
          auto* tau = reinterpret_cast<const scalar_t*>(data[1] + offsets[1]);
          auto* omega = reinterpret_cast<const scalar_t*>(data[2] + offsets[2]);
          auto* scaling = reinterpret_cast<const scalar_t*>(data[3] + offsets[3]);
          auto* planck = reinterpret_cast<const scalar_t*>(data[4] + offsets[4]);
          auto* surfbb = reinterpret_cast<const scalar_t*>(data[5] + offsets[5]);
          auto* emissivity = reinterpret_cast<const scalar_t*>(data[6] + offsets[6]);
          thermal_fo_row<scalar_t>(
              nlay,
              params.nfine,
              params.do_nadir,
              params.do_optical_deltam_scaling,
              params.do_source_deltam_scaling,
              tau,
              omega,
              scaling,
              planck,
              surfbb,
              emissivity,
              height_delta,
              xfine,
              wfine,
              cota,
              cotfine,
              csqfine,
              static_cast<scalar_t>(params.rayconv),
              params.return_components,
              params.return_profile,
              out,
              work);
        });
  });
}

void thermal_fo_flux_correction_cuda(
    at::TensorIterator& iter,
    const ThermalFoFluxParams& params) {
  at::cuda::CUDAGuard device_guard(iter.device());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_thermal_fo_flux_correction_cuda", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const int workspace_size =
        static_cast<int>(thermal_fo_flux_workspace_bytes<scalar_t>(nlay));
    gpu_chunk_kernel<8, 7>(
        iter,
        workspace_size,
        [=] __device__(char* const data[7], unsigned int offsets[7], char* work) {
          auto* out = reinterpret_cast<scalar_t*>(data[0] + offsets[0]);
          auto* tau = reinterpret_cast<const scalar_t*>(data[1] + offsets[1]);
          auto* omega = reinterpret_cast<const scalar_t*>(data[2] + offsets[2]);
          auto* scaling = reinterpret_cast<const scalar_t*>(data[3] + offsets[3]);
          auto* planck = reinterpret_cast<const scalar_t*>(data[4] + offsets[4]);
          auto* surfbb = reinterpret_cast<const scalar_t*>(data[5] + offsets[5]);
          auto* emissivity = reinterpret_cast<const scalar_t*>(data[6] + offsets[6]);
          thermal_fo_flux_correction_row<scalar_t>(
              nlay,
              static_cast<scalar_t>(params.stream_value),
              params.do_optical_deltam_scaling,
              params.do_source_deltam_scaling,
              params.n_mu,
              static_cast<const scalar_t*>(params.mu_nodes),
              static_cast<const scalar_t*>(params.mu_weights),
              tau,
              omega,
              scaling,
              planck,
              surfbb,
              emissivity,
              out,
              work);
        });
  });
}

void solar_fo_cuda(at::TensorIterator& iter, const SolarFoParams& params) {
  at::cuda::CUDAGuard device_guard(iter.device());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_solar_fo_cuda", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const auto* inv_layer_thickness =
        reinterpret_cast<const scalar_t*>(params.inv_layer_thickness);
    const auto* sunpathsnl = reinterpret_cast<const scalar_t*>(params.sunpathsnl);
    const auto* cota = reinterpret_cast<const scalar_t*>(params.cota);
    const auto* cotfine = reinterpret_cast<const scalar_t*>(params.cotfine);
    const auto* csqfine = reinterpret_cast<const scalar_t*>(params.csqfine);
    const auto* wfine = reinterpret_cast<const scalar_t*>(params.wfine);
    const auto* xfine = reinterpret_cast<const scalar_t*>(params.xfine);
    const auto* sunpathsfine = reinterpret_cast<const scalar_t*>(params.sunpathsfine);
    const auto* nfinedivs = reinterpret_cast<const std::int64_t*>(params.nfinedivs);
    const auto* ntraversefine = reinterpret_cast<const std::int64_t*>(params.ntraversefine);
    if (!params.do_nadir && !params.return_profile) {
      const int64_t rows = iter.numel();
      if (rows == 0) {
        return;
      }
      const int fine_slots = params.nfine * nlay;
      auto options = iter.input(0).options();
      auto extinction = at::empty({rows, nlay}, options);
      auto fine_attenuation = at::empty({rows, fine_slots}, options);
      auto* d_extinction = extinction.data_ptr<scalar_t>();
      auto* d_fine_attenuation = fine_attenuation.data_ptr<scalar_t>();
      auto* out = reinterpret_cast<scalar_t*>(iter.data_ptr(0));
      const auto* tau = reinterpret_cast<const scalar_t*>(iter.data_ptr(1));
      const auto* omega = reinterpret_cast<const scalar_t*>(iter.data_ptr(2));
      const auto* scaling = reinterpret_cast<const scalar_t*>(iter.data_ptr(3));
      const auto* albedo = reinterpret_cast<const scalar_t*>(iter.data_ptr(4));
      const auto* flux_factor = reinterpret_cast<const scalar_t*>(iter.data_ptr(5));
      const auto* exact_scatter = reinterpret_cast<const scalar_t*>(iter.data_ptr(6));
      const int64_t extinction_count = rows * static_cast<int64_t>(nlay);
      const int64_t fine_count = rows * static_cast<int64_t>(fine_slots);
      const auto stream = c10::cuda::getCurrentCUDAStream();
      solar_fo_extinction_kernel<scalar_t>
          <<<grid_1d(extinction_count), kBlock, 0, stream.stream()>>>(
              rows,
              nlay,
              tau,
              omega,
              scaling,
              inv_layer_thickness,
              d_extinction);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      solar_fo_fine_attenuation_kernel<scalar_t>
          <<<grid_1d(fine_count), kBlock, 0, stream.stream()>>>(
              rows,
              nlay,
              params.nfine,
              d_extinction,
              sunpathsfine,
              nfinedivs,
              ntraversefine,
              d_fine_attenuation);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      solar_fo_nonnadir_endpoint_kernel<scalar_t>
          <<<grid_1d(rows), kBlock, 0, stream.stream()>>>(
              rows,
              nlay,
              params.nfine,
              params.ntrav_nl,
              params.return_components,
              static_cast<scalar_t>(params.mu0),
              static_cast<scalar_t>(params.rayconv),
              albedo,
              flux_factor,
              exact_scatter,
              sunpathsnl,
              cota,
              cotfine,
              csqfine,
              wfine,
              nfinedivs,
              d_extinction,
              d_fine_attenuation,
              out);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }

    const int workspace_size =
        static_cast<int>(solar_fo_workspace_bytes<scalar_t>(nlay, params.return_profile));
    gpu_chunk_kernel<8, 7>(
        iter,
        workspace_size,
        [=] __device__(char* const data[7], unsigned int offsets[7], char* work) {
          auto* out = reinterpret_cast<scalar_t*>(data[0] + offsets[0]);
          auto* tau = reinterpret_cast<const scalar_t*>(data[1] + offsets[1]);
          auto* omega = reinterpret_cast<const scalar_t*>(data[2] + offsets[2]);
          auto* scaling = reinterpret_cast<const scalar_t*>(data[3] + offsets[3]);
          auto* albedo = reinterpret_cast<const scalar_t*>(data[4] + offsets[4]);
          auto* flux_factor = reinterpret_cast<const scalar_t*>(data[5] + offsets[5]);
          auto* exact_scatter = reinterpret_cast<const scalar_t*>(data[6] + offsets[6]);
          solar_fo_row<scalar_t>(
              nlay,
              params.nfine,
              params.ntrav_nl,
              params.ntrav_max,
              params.do_nadir,
              static_cast<scalar_t>(params.mu0),
              static_cast<scalar_t>(params.rayconv),
              tau,
              omega,
              scaling,
              albedo,
              flux_factor,
              exact_scatter,
              inv_layer_thickness,
              sunpathsnl,
              cota,
              cotfine,
              csqfine,
              wfine,
              xfine,
              sunpathsfine,
              nfinedivs,
              ntraversefine,
              out,
              params.return_components,
              params.return_profile,
              work);
        });
  });
}

void solar_fo_plane_parallel_cuda(at::TensorIterator& iter, const SolarFoPpParams& params) {
  at::cuda::CUDAGuard device_guard(iter.device());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_solar_fo_plane_parallel_cuda", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const int workspace_size = static_cast<int>(solar_fo_pp_workspace_bytes<scalar_t>(nlay));
    gpu_chunk_kernel<8, 7>(
        iter,
        workspace_size,
        [=] __device__(char* const data[7], unsigned int offsets[7], char* work) {
          auto* out = reinterpret_cast<scalar_t*>(data[0] + offsets[0]);
          auto* tau = reinterpret_cast<const scalar_t*>(data[1] + offsets[1]);
          auto* omega = reinterpret_cast<const scalar_t*>(data[2] + offsets[2]);
          auto* scaling = reinterpret_cast<const scalar_t*>(data[3] + offsets[3]);
          auto* surface_reflectance = reinterpret_cast<const scalar_t*>(data[4] + offsets[4]);
          auto* flux_factor = reinterpret_cast<const scalar_t*>(data[5] + offsets[5]);
          auto* exact_scatter = reinterpret_cast<const scalar_t*>(data[6] + offsets[6]);
          solar_fo_pp_row<scalar_t>(
              nlay,
              static_cast<scalar_t>(params.mu0),
              static_cast<scalar_t>(params.user_stream),
              tau,
              omega,
              scaling,
              surface_reflectance,
              flux_factor,
              exact_scatter,
              out,
              params.return_profile,
              work);
        });
  });
}

void solar_fo_flux_correction_cuda(at::TensorIterator& iter, const SolarFoFluxParams& params) {
  at::cuda::CUDAGuard device_guard(iter.device());
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_solar_fo_flux_correction_cuda", [&] {
    const int nlay = last_dim_size(iter.input(0));
    gpu_kernel<6>(
        iter,
        [=] __device__(char* const data[6], unsigned int offsets[6]) {
          auto* out = reinterpret_cast<scalar_t*>(data[0] + offsets[0]);
          auto* tau = reinterpret_cast<const scalar_t*>(data[1] + offsets[1]);
          auto* omega = reinterpret_cast<const scalar_t*>(data[2] + offsets[2]);
          auto* scaling = reinterpret_cast<const scalar_t*>(data[3] + offsets[3]);
          auto* surface_reflectance = reinterpret_cast<const scalar_t*>(data[4] + offsets[4]);
          auto* flux_factor = reinterpret_cast<const scalar_t*>(data[5] + offsets[5]);
          solar_fo_flux_correction_row<scalar_t>(
              nlay,
              static_cast<scalar_t>(params.stream_value),
              static_cast<scalar_t>(params.mu0),
              params.do_optical_deltam_scaling,
              tau,
              omega,
              scaling,
              surface_reflectance,
              flux_factor,
              out);
        });
  });
}

}  // namespace py2sess_native

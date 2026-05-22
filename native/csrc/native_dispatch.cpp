#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Parallel.h>
#include <ATen/Tensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/ops/sub.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <memory>
#include <string>

#include "native_dispatch.hpp"
#include "native_workspace.hpp"
#include "thermal_2s_impl.hpp"

namespace py2sess_native {

namespace {

void check_same_dtype_device(const at::Tensor& reference, const at::Tensor& tensor, const char* label) {
  TORCH_CHECK(tensor.scalar_type() == reference.scalar_type(), label, " tensors must share dtype");
  TORCH_CHECK(tensor.device() == reference.device(), label, " tensors must share device");
}

template <typename... Tensors>
void check_all_same_dtype_device(const at::Tensor& reference, const char* label, const Tensors&... tensors) {
  (check_same_dtype_device(reference, tensors, label), ...);
}

void check_same_shape(const at::Tensor& tensor, const at::Tensor& reference, const char* message) {
  TORCH_CHECK(tensor.sizes() == reference.sizes(), message);
}

void check_vector_size(const at::Tensor& tensor, int64_t size, const char* message) {
  TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == size, message);
}

void check_matrix_shape(const at::Tensor& tensor, int64_t rows, int64_t cols, const char* message) {
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == cols, message);
}

void check_matrix_min_cols(
    const at::Tensor& tensor,
    int64_t rows,
    int64_t min_cols,
    const char* message) {
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) >= min_cols, message);
}

void check_thermal_inputs(
    const at::Tensor& tau,
    const at::Tensor& omega,
    const at::Tensor& asymm,
    const at::Tensor& scaling,
    const at::Tensor& planck,
    const at::Tensor& surfbb,
    const at::Tensor& emissivity,
    const at::Tensor& albedo,
    const at::Tensor& brdf_f,
    const at::Tensor& ubrdf_f) {
  TORCH_CHECK(tau.is_floating_point(), "thermal_2s expects floating tensors");
  TORCH_CHECK(tau.dim() == 2, "thermal_2s tau must have shape (rows, nlay)");
  check_same_shape(omega, tau, "thermal_2s omega must match tau shape");
  check_same_shape(asymm, tau, "thermal_2s asymm must match tau shape");
  check_same_shape(scaling, tau, "thermal_2s scaling must match tau shape");
  check_matrix_shape(planck, tau.size(0), tau.size(1) + 1,
                     "thermal_2s planck must have shape (rows, nlay + 1)");
  check_vector_size(surfbb, tau.size(0), "thermal_2s surfbb must have shape (rows,)");
  check_vector_size(emissivity, tau.size(0), "thermal_2s emissivity must have shape (rows,)");
  check_vector_size(albedo, tau.size(0), "thermal_2s albedo must have shape (rows,)");
  check_vector_size(brdf_f, tau.size(0), "thermal_2s brdf_f must have shape (rows,)");
  check_vector_size(ubrdf_f, tau.size(0), "thermal_2s ubrdf_f must have shape (rows,)");
  check_all_same_dtype_device(
      tau,
      "thermal_2s",
      omega,
      asymm,
      scaling,
      planck,
      surfbb,
      emissivity,
      albedo,
      brdf_f,
      ubrdf_f);
}

void check_solar_required_inputs(
    const at::Tensor& tau,
    const at::Tensor& omega,
    const at::Tensor& asymm,
    const at::Tensor& scaling,
    const at::Tensor& albedo,
    const at::Tensor& flux_factor,
    const at::Tensor& chapman,
    const at::Tensor& pxsq,
    const at::Tensor& px0x) {
  TORCH_CHECK(tau.is_floating_point(), "solar_2s expects floating tensors");
  TORCH_CHECK(tau.dim() == 2, "solar_2s tau must have shape (rows, nlay)");
  check_same_shape(omega, tau, "solar_2s omega must match tau shape");
  check_same_shape(asymm, tau, "solar_2s asymm must match tau shape");
  check_same_shape(scaling, tau, "solar_2s scaling must match tau shape");
  check_vector_size(albedo, tau.size(0), "solar_2s albedo must have shape (rows,)");
  check_vector_size(flux_factor, tau.size(0), "solar_2s flux_factor must have shape (rows,)");
  check_matrix_shape(chapman, tau.size(1), tau.size(1),
                     "solar_2s chapman must have shape (nlay, nlay)");
  TORCH_CHECK(pxsq.numel() >= 2, "solar_2s pxsq must contain two Fourier values");
  TORCH_CHECK(px0x.numel() >= 2, "solar_2s px0x must contain two Fourier values");
  check_all_same_dtype_device(
      tau,
      "solar_2s",
      omega,
      asymm,
      scaling,
      albedo,
      flux_factor,
      chapman,
      pxsq,
      px0x);
}

void check_prop_inputs(const at::Tensor& prop, const char* label) {
  TORCH_CHECK(prop.is_floating_point(), label, " expects floating prop tensor");
  TORCH_CHECK(prop.dim() == 4, label, " prop must have shape (nwave, ncol, nlay, nprop)");
  TORCH_CHECK(prop.size(2) > 0, label, " prop must contain at least one layer");
  TORCH_CHECK(prop.size(3) >= 3, label, " prop must contain tau, omega, and asymm channels");
}

int64_t prop_rows(const at::Tensor& prop) {
  return prop.size(0) * prop.size(1);
}

void check_thermal_fo_inputs(
    const at::Tensor& tau,
    const at::Tensor& omega,
    const at::Tensor& scaling,
    const at::Tensor& planck,
    const at::Tensor& surfbb,
    const at::Tensor& emissivity,
    const at::Tensor& heights,
    const at::Tensor& xfine,
    const at::Tensor& wfine,
    const at::Tensor& cota,
    const at::Tensor& cotfine,
    const at::Tensor& csqfine) {
  TORCH_CHECK(tau.is_floating_point(), "thermal_fo expects floating tensors");
  TORCH_CHECK(tau.dim() == 2, "thermal_fo tau must have shape (rows, nlay)");
  check_same_shape(omega, tau, "thermal_fo omega must match tau shape");
  check_same_shape(scaling, tau, "thermal_fo scaling must match tau shape");
  check_matrix_shape(planck, tau.size(0), tau.size(1) + 1,
                     "thermal_fo planck must have shape (rows, nlay + 1)");
  check_vector_size(surfbb, tau.size(0), "thermal_fo surfbb must have shape (rows,)");
  check_vector_size(emissivity, tau.size(0), "thermal_fo emissivity must have shape (rows,)");
  check_vector_size(heights, tau.size(1) + 1, "thermal_fo heights must have shape (nlay + 1,)");
  TORCH_CHECK(xfine.dim() == 2 && xfine.size(1) == tau.size(1),
              "thermal_fo xfine must have shape (nfine, nlay)");
  check_same_shape(wfine, xfine, "thermal_fo wfine must match xfine shape");
  check_vector_size(cota, tau.size(1) + 1, "thermal_fo cota must have shape (nlay + 1,)");
  check_same_shape(cotfine, xfine, "thermal_fo cotfine must match xfine shape");
  check_same_shape(csqfine, xfine, "thermal_fo csqfine must match xfine shape");
  check_all_same_dtype_device(
      tau,
      "thermal_fo",
      omega,
      scaling,
      planck,
      surfbb,
      emissivity,
      heights,
      xfine,
      wfine,
      cota,
      cotfine,
      csqfine);
}

void check_solar_fo_inputs(
    const at::Tensor& tau,
    const at::Tensor& omega,
    const at::Tensor& scaling,
    const at::Tensor& albedo,
    const at::Tensor& flux_factor,
    const at::Tensor& exact_scatter,
    const at::Tensor& inv_layer_thickness,
    const at::Tensor& sunpathsnl,
    const at::Tensor& cota,
    const at::Tensor& cotfine,
    const at::Tensor& csqfine,
    const at::Tensor& wfine,
    const at::Tensor& xfine,
    const at::Tensor& sunpathsfine,
    const at::Tensor& nfinedivs,
    const at::Tensor& ntraversefine) {
  TORCH_CHECK(tau.is_floating_point(), "solar_fo expects floating tensors");
  TORCH_CHECK(tau.dim() == 2, "solar_fo tau must have shape (rows, nlay)");
  check_same_shape(omega, tau, "solar_fo omega must match tau shape");
  check_same_shape(scaling, tau, "solar_fo scaling must match tau shape");
  check_same_shape(exact_scatter, tau, "solar_fo exact_scatter must match tau shape");
  check_vector_size(albedo, tau.size(0), "solar_fo albedo must have shape (rows,)");
  check_vector_size(flux_factor, tau.size(0), "solar_fo flux_factor must have shape (rows,)");
  check_vector_size(
      inv_layer_thickness, tau.size(1), "solar_fo inv_layer_thickness must have shape (nlay,)");
  TORCH_CHECK(sunpathsnl.dim() == 1, "solar_fo sunpathsnl must be 1D");
  check_vector_size(cota, tau.size(1) + 1, "solar_fo cota must have shape (nlay + 1,)");
  TORCH_CHECK(cotfine.dim() == 2 && cotfine.size(1) == tau.size(1),
              "solar_fo cotfine must have shape (nfine, nlay)");
  check_same_shape(csqfine, cotfine, "solar_fo csqfine must match cotfine");
  check_same_shape(wfine, cotfine, "solar_fo wfine must match cotfine");
  check_same_shape(xfine, cotfine, "solar_fo xfine must match cotfine");
  TORCH_CHECK(sunpathsfine.dim() == 3 && sunpathsfine.size(1) == cotfine.size(0) &&
                  sunpathsfine.size(2) == tau.size(1),
              "solar_fo sunpathsfine must have shape (ntrav_max, nfine, nlay)");
  check_vector_size(nfinedivs, tau.size(1), "solar_fo nfinedivs must have shape (nlay,)");
  check_same_shape(ntraversefine, cotfine, "solar_fo ntraversefine must have shape (nfine, nlay)");
  TORCH_CHECK(nfinedivs.scalar_type() == at::kLong, "solar_fo nfinedivs must be int64");
  TORCH_CHECK(ntraversefine.scalar_type() == at::kLong,
              "solar_fo ntraversefine must be int64");
  check_all_same_dtype_device(
      tau,
      "solar_fo",
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
      sunpathsfine);
  TORCH_CHECK(nfinedivs.device() == tau.device(), "solar_fo tensors must share device");
  TORCH_CHECK(ntraversefine.device() == tau.device(), "solar_fo tensors must share device");
}

int64_t two_stream_output_cols(bool return_profile, bool return_fluxes, int64_t nlay) {
  const auto nlev = nlay + 1;
  const auto radiance_cols = return_profile ? nlev : int64_t{1};
  return radiance_cols + (return_fluxes ? 4 * nlev : int64_t{0});
}

int64_t two_stream_flux_output_cols(int64_t nlay) {
  return 2 * (nlay + 1);
}

int last_dim_size(const at::TensorBase& tensor) {
  TORCH_CHECK(tensor.dim() > 0, "py2sess native kernels expect non-scalar row tensors");
  return static_cast<int>(tensor.size(tensor.dim() - 1));
}

#ifdef PY2SESS_WITH_CUDA
#define PY2SESS_DISPATCH_KERNEL(output, cpu_fn, cuda_fn, iter, params) \
  do { \
    if ((output).is_cuda()) { \
      cuda_fn((iter), (params)); \
    } else { \
      TORCH_CHECK((output).is_cpu(), "py2sess native backend supports CPU and CUDA tensors only"); \
      cpu_fn((iter), (params)); \
    } \
  } while (false)
#else
#define PY2SESS_DISPATCH_KERNEL(output, cpu_fn, cuda_fn, iter, params) \
  do { \
    TORCH_CHECK(!(output).is_cuda(), "py2sess native extension was built without CUDA support"); \
    TORCH_CHECK((output).is_cpu(), "py2sess native backend supports CPU tensors only in this build"); \
    cpu_fn((iter), (params)); \
  } while (false)
#endif

at::Tensor run_thermal_2s(
    const at::Tensor& tau,
    const at::Tensor& omega,
    const at::Tensor& asymm,
    const at::Tensor& scaling,
    const at::Tensor& planck,
    const at::Tensor& surfbb,
    const at::Tensor& emissivity,
    const at::Tensor& albedo,
    const at::Tensor& brdf_f,
    const at::Tensor& ubrdf_f,
    const Thermal2sParams& params) {
  const auto rows = tau.size(0);
  const auto nlay = tau.size(1);
  const auto output_cols = two_stream_output_cols(params.return_profile, params.return_fluxes, nlay);
  auto output = at::empty({rows, output_cols}, tau.options());

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape({rows, output_cols}, /*squash_dims=*/{1})
                  .add_output(output)
                  .add_input(tau)
                  .add_input(omega)
                  .add_input(asymm)
                  .add_input(scaling)
                  .add_input(planck)
                  .add_owned_input(surfbb.view({rows, 1}))
                  .add_owned_input(emissivity.view({rows, 1}))
                  .add_owned_input(albedo.view({rows, 1}))
                  .add_owned_input(brdf_f.view({rows, 1}))
                  .add_owned_input(ubrdf_f.view({rows, 1}))
                  .build();

  PY2SESS_DISPATCH_KERNEL(output, thermal_2s_cpu, thermal_2s_cuda, iter, params);
  return output;
}

at::Tensor run_solar_2s(
    const at::Tensor& tau,
    const at::Tensor& omega,
    const at::Tensor& asymm,
    const at::Tensor& scaling,
    const at::Tensor& albedo,
    const at::Tensor& flux_factor,
    const at::Tensor& brdf_f0,
    const at::Tensor& brdf_f,
    const at::Tensor& ubrdf_f,
    const at::Tensor& slterm_isotropic,
    const at::Tensor& slterm_f0,
    const Solar2sParams& params) {
  const auto rows = tau.size(0);
  const auto nlay = tau.size(1);
  const auto output_cols = two_stream_output_cols(params.return_profile, params.return_fluxes, nlay);
  auto output = at::empty({rows, output_cols}, tau.options());

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape({rows, output_cols}, /*squash_dims=*/{1})
                  .add_output(output)
                  .add_input(tau)
                  .add_input(omega)
                  .add_input(asymm)
                  .add_input(scaling)
                  .add_owned_input(albedo.view({rows, 1}))
                  .add_owned_input(flux_factor.view({rows, 1}))
                  .add_input(brdf_f0)
                  .add_input(brdf_f)
                  .add_input(ubrdf_f)
                  .add_owned_input(slterm_isotropic.view({rows, 1}))
                  .add_input(slterm_f0)
                  .build();

  PY2SESS_DISPATCH_KERNEL(output, solar_2s_cpu, solar_2s_cuda, iter, params);
  return output;
}

at::Tensor run_thermal_2s_flux(
    const at::Tensor& tau,
    const at::Tensor& omega,
    const at::Tensor& asymm,
    const at::Tensor& scaling,
    const at::Tensor& planck,
    const at::Tensor& surfbb,
    const at::Tensor& emissivity,
    const at::Tensor& albedo,
    const at::Tensor& brdf_f,
    const at::Tensor& ubrdf_f,
    const Thermal2sParams& params) {
  const auto rows = tau.size(0);
  const auto nlay = tau.size(1);
  const auto output_cols = two_stream_flux_output_cols(nlay);
  auto output = at::empty({rows, output_cols}, tau.options());

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape({rows, output_cols}, /*squash_dims=*/{1})
                  .add_output(output)
                  .add_input(tau)
                  .add_input(omega)
                  .add_input(asymm)
                  .add_input(scaling)
                  .add_input(planck)
                  .add_owned_input(surfbb.view({rows, 1}))
                  .add_owned_input(emissivity.view({rows, 1}))
                  .add_owned_input(albedo.view({rows, 1}))
                  .add_owned_input(brdf_f.view({rows, 1}))
                  .add_owned_input(ubrdf_f.view({rows, 1}))
                  .build();

  PY2SESS_DISPATCH_KERNEL(output, thermal_2s_flux_cpu, thermal_2s_flux_cuda, iter, params);
  return output.view({rows, nlay + 1, 2});
}

at::Tensor run_solar_2s_flux(
    const at::Tensor& tau,
    const at::Tensor& omega,
    const at::Tensor& asymm,
    const at::Tensor& scaling,
    const at::Tensor& albedo,
    const at::Tensor& flux_factor,
    const at::Tensor& brdf_f0,
    const at::Tensor& brdf_f,
    const at::Tensor& ubrdf_f,
    const at::Tensor& slterm_isotropic,
    const at::Tensor& slterm_f0,
    const Solar2sParams& params) {
  const auto rows = tau.size(0);
  const auto nlay = tau.size(1);
  const auto output_cols = two_stream_flux_output_cols(nlay);
  auto output = at::empty({rows, output_cols}, tau.options());

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape({rows, output_cols}, /*squash_dims=*/{1})
                  .add_output(output)
                  .add_input(tau)
                  .add_input(omega)
                  .add_input(asymm)
                  .add_input(scaling)
                  .add_owned_input(albedo.view({rows, 1}))
                  .add_owned_input(flux_factor.view({rows, 1}))
                  .add_input(brdf_f0)
                  .add_input(brdf_f)
                  .add_input(ubrdf_f)
                  .add_owned_input(slterm_isotropic.view({rows, 1}))
                  .add_input(slterm_f0)
                  .build();

  PY2SESS_DISPATCH_KERNEL(output, solar_2s_flux_cpu, solar_2s_flux_cuda, iter, params);
  return output.view({rows, nlay + 1, 2});
}

at::Tensor run_thermal_2s_prop_flux(
    const at::Tensor& prop,
    const at::Tensor& planck,
    const at::Tensor& surfbb,
    const at::Tensor& emissivity,
    const at::Tensor& albedo,
    const at::Tensor& brdf_f,
    const at::Tensor& ubrdf_f,
    const Thermal2sPropParams& params) {
  const auto rows = prop.size(0) * prop.size(1);
  const auto nlay = prop.size(2);
  const auto nprop = prop.size(3);
  const auto nlev = nlay + 1;
  auto output = at::empty({rows, two_stream_flux_output_cols(nlay)}, prop.options());
  auto prop_flat = prop.view({rows, nlay * nprop});

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(output.sizes(), /*squash_dims=*/{1})
                  .add_output(output)
                  .add_input(prop_flat)
                  .add_input(planck)
                  .add_owned_input(surfbb.view({rows, 1}))
                  .add_owned_input(emissivity.view({rows, 1}))
                  .add_owned_input(albedo.view({rows, 1}))
                  .add_owned_input(brdf_f.view({rows, 1}))
                  .add_owned_input(ubrdf_f.view({rows, 1}))
                  .build();

  PY2SESS_DISPATCH_KERNEL(output, thermal_2s_prop_flux_cpu, thermal_2s_prop_flux_cuda, iter, params);
  return output.view({prop.size(0), prop.size(1), nlev, 2});
}

at::Tensor run_solar_2s_prop_flux(
    const at::Tensor& prop,
    const at::Tensor& albedo,
    const at::Tensor& flux_factor,
    const at::Tensor& brdf_f0,
    const at::Tensor& brdf_f,
    const at::Tensor& ubrdf_f,
    const at::Tensor& slterm_isotropic,
    const at::Tensor& slterm_f0,
    const Solar2sPropParams& params) {
  const auto rows = prop.size(0) * prop.size(1);
  const auto nlay = prop.size(2);
  const auto nprop = prop.size(3);
  const auto nlev = nlay + 1;
  auto output = at::empty({rows, two_stream_flux_output_cols(nlay)}, prop.options());
  auto prop_flat = prop.view({rows, nlay * nprop});

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape(output.sizes(), /*squash_dims=*/{1})
                  .add_output(output)
                  .add_input(prop_flat)
                  .add_owned_input(albedo.view({rows, 1}))
                  .add_owned_input(flux_factor.view({rows, 1}))
                  .add_input(brdf_f0)
                  .add_input(brdf_f)
                  .add_input(ubrdf_f)
                  .add_owned_input(slterm_isotropic.view({rows, 1}))
                  .add_input(slterm_f0)
                  .build();

  PY2SESS_DISPATCH_KERNEL(output, solar_2s_prop_flux_cpu, solar_2s_prop_flux_cuda, iter, params);
  return output.view({prop.size(0), prop.size(1), nlev, 2});
}

}  // namespace

void thermal_2s_cpu(at::TensorIterator& iter, const Thermal2sParams& params) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_thermal_2s_cpu", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const auto grain_size =
        std::max<int64_t>(1, iter.numel() / std::max<int>(1, at::get_num_threads()));
    const auto workspace_size = thermal_2s_workspace_bytes<scalar_t>(nlay);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          std::unique_ptr<char[]> work = std::make_unique<char[]>(workspace_size);
          for (int64_t i = 0; i < n; ++i) {
            auto* out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto* tau = reinterpret_cast<const scalar_t*>(data[1] + i * strides[1]);
            auto* omega = reinterpret_cast<const scalar_t*>(data[2] + i * strides[2]);
            auto* asymm = reinterpret_cast<const scalar_t*>(data[3] + i * strides[3]);
            auto* scaling = reinterpret_cast<const scalar_t*>(data[4] + i * strides[4]);
            auto* planck = reinterpret_cast<const scalar_t*>(data[5] + i * strides[5]);
            auto* surfbb = reinterpret_cast<const scalar_t*>(data[6] + i * strides[6]);
            auto* emissivity = reinterpret_cast<const scalar_t*>(data[7] + i * strides[7]);
            auto* albedo = reinterpret_cast<const scalar_t*>(data[8] + i * strides[8]);
            auto* brdf_f = reinterpret_cast<const scalar_t*>(data[9] + i * strides[9]);
            auto* ubrdf_f = reinterpret_cast<const scalar_t*>(data[10] + i * strides[10]);
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
                work.get());
          }
        },
        grain_size);
  });
}

void thermal_2s_flux_cpu(at::TensorIterator& iter, const Thermal2sParams& params) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_thermal_2s_flux_cpu", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const auto grain_size =
        std::max<int64_t>(1, iter.numel() / std::max<int>(1, at::get_num_threads()));
    const auto workspace_size =
        two_stream_flux_pair_packed_cols<scalar_t>(nlay) * sizeof(scalar_t) +
        thermal_2s_workspace_bytes<scalar_t>(nlay);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          std::unique_ptr<char[]> work = std::make_unique<char[]>(workspace_size);
          for (int64_t i = 0; i < n; ++i) {
            auto* out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto* tau = reinterpret_cast<const scalar_t*>(data[1] + i * strides[1]);
            auto* omega = reinterpret_cast<const scalar_t*>(data[2] + i * strides[2]);
            auto* asymm = reinterpret_cast<const scalar_t*>(data[3] + i * strides[3]);
            auto* scaling = reinterpret_cast<const scalar_t*>(data[4] + i * strides[4]);
            auto* planck = reinterpret_cast<const scalar_t*>(data[5] + i * strides[5]);
            auto* surfbb = reinterpret_cast<const scalar_t*>(data[6] + i * strides[6]);
            auto* emissivity = reinterpret_cast<const scalar_t*>(data[7] + i * strides[7]);
            auto* albedo = reinterpret_cast<const scalar_t*>(data[8] + i * strides[8]);
            auto* brdf_f = reinterpret_cast<const scalar_t*>(data[9] + i * strides[9]);
            auto* ubrdf_f = reinterpret_cast<const scalar_t*>(data[10] + i * strides[10]);
            thermal_2s_flux_pair_row<scalar_t>(
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
                params.do_upwelling,
                params.do_dnwelling,
                params.use_brdf,
                work.get());
          }
        },
        grain_size);
  });
}

void thermal_2s_prop_flux_cpu(at::TensorIterator& iter, const Thermal2sPropParams& params) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_thermal_2s_prop_flux_cpu", [&] {
    const int nlay = static_cast<int>(params.nlay);
    const int nprop = static_cast<int>(params.nprop);
    const auto grain_size =
        std::max<int64_t>(1, iter.numel() / std::max<int>(1, at::get_num_threads()));
    const auto staging_size = (4 * nlay + (nlay + 1)) * sizeof(scalar_t);
    const auto workspace_size = staging_size +
                                two_stream_flux_pair_packed_cols<scalar_t>(nlay) *
                                    sizeof(scalar_t) +
                                thermal_2s_workspace_bytes<scalar_t>(nlay);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          std::unique_ptr<char[]> work = std::make_unique<char[]>(workspace_size);
          for (int64_t i = 0; i < n; ++i) {
            auto* out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto* prop = reinterpret_cast<const scalar_t*>(data[1] + i * strides[1]);
            auto* planck = reinterpret_cast<const scalar_t*>(data[2] + i * strides[2]);
            auto* surfbb = reinterpret_cast<const scalar_t*>(data[3] + i * strides[3]);
            auto* emissivity = reinterpret_cast<const scalar_t*>(data[4] + i * strides[4]);
            auto* albedo = reinterpret_cast<const scalar_t*>(data[5] + i * strides[5]);
            auto* brdf_f = reinterpret_cast<const scalar_t*>(data[6] + i * strides[6]);
            auto* ubrdf_f = reinterpret_cast<const scalar_t*>(data[7] + i * strides[7]);
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
                work.get());
          }
        },
        grain_size);
  });
}

void thermal_fo_cpu(at::TensorIterator& iter, const ThermalFoParams& params) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_thermal_fo_cpu", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const auto grain_size =
        std::max<int64_t>(1, iter.numel() / std::max<int>(1, at::get_num_threads()));
    const auto workspace_size = thermal_fo_workspace_bytes<scalar_t>(nlay);
    const auto* height_delta = reinterpret_cast<const scalar_t*>(params.height_delta);
    const auto* xfine = reinterpret_cast<const scalar_t*>(params.xfine);
    const auto* wfine = reinterpret_cast<const scalar_t*>(params.wfine);
    const auto* cota = reinterpret_cast<const scalar_t*>(params.cota);
    const auto* cotfine = reinterpret_cast<const scalar_t*>(params.cotfine);
    const auto* csqfine = reinterpret_cast<const scalar_t*>(params.csqfine);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          std::unique_ptr<char[]> work = std::make_unique<char[]>(workspace_size);
          for (int64_t i = 0; i < n; ++i) {
            auto* out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto* tau = reinterpret_cast<const scalar_t*>(data[1] + i * strides[1]);
            auto* omega = reinterpret_cast<const scalar_t*>(data[2] + i * strides[2]);
            auto* scaling = reinterpret_cast<const scalar_t*>(data[3] + i * strides[3]);
            auto* planck = reinterpret_cast<const scalar_t*>(data[4] + i * strides[4]);
            auto* surfbb = reinterpret_cast<const scalar_t*>(data[5] + i * strides[5]);
            auto* emissivity = reinterpret_cast<const scalar_t*>(data[6] + i * strides[6]);
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
                work.get());
          }
        },
        grain_size);
  });
}

void solar_fo_cpu(at::TensorIterator& iter, const SolarFoParams& params) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_solar_fo_cpu", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const auto grain_size =
        std::max<int64_t>(1, iter.numel() / std::max<int>(1, at::get_num_threads()));
    const auto workspace_size = solar_fo_workspace_bytes<scalar_t>(nlay, params.return_profile);
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
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          std::unique_ptr<char[]> work = std::make_unique<char[]>(workspace_size);
          for (int64_t i = 0; i < n; ++i) {
            auto* out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto* tau = reinterpret_cast<const scalar_t*>(data[1] + i * strides[1]);
            auto* omega = reinterpret_cast<const scalar_t*>(data[2] + i * strides[2]);
            auto* scaling = reinterpret_cast<const scalar_t*>(data[3] + i * strides[3]);
            auto* albedo = reinterpret_cast<const scalar_t*>(data[4] + i * strides[4]);
            auto* flux_factor = reinterpret_cast<const scalar_t*>(data[5] + i * strides[5]);
            auto* exact_scatter = reinterpret_cast<const scalar_t*>(data[6] + i * strides[6]);
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
                work.get());
          }
        },
        grain_size);
  });
}

void solar_2s_cpu(at::TensorIterator& iter, const Solar2sParams& params) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_solar_2s_cpu", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const auto grain_size =
        std::max<int64_t>(1, iter.numel() / std::max<int>(1, at::get_num_threads()));
    const auto workspace_size = solar_2s_workspace_bytes<scalar_t>(nlay);
    const auto* chapman = reinterpret_cast<const scalar_t*>(params.chapman);
    const auto* pxsq = reinterpret_cast<const scalar_t*>(params.pxsq);
    const auto* px0x = reinterpret_cast<const scalar_t*>(params.px0x);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          std::unique_ptr<char[]> work = std::make_unique<char[]>(workspace_size);
          for (int64_t i = 0; i < n; ++i) {
            auto* out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto* tau = reinterpret_cast<const scalar_t*>(data[1] + i * strides[1]);
            auto* omega = reinterpret_cast<const scalar_t*>(data[2] + i * strides[2]);
            auto* asymm = reinterpret_cast<const scalar_t*>(data[3] + i * strides[3]);
            auto* scaling = reinterpret_cast<const scalar_t*>(data[4] + i * strides[4]);
            auto* albedo = reinterpret_cast<const scalar_t*>(data[5] + i * strides[5]);
            auto* flux_factor = reinterpret_cast<const scalar_t*>(data[6] + i * strides[6]);
            auto* brdf_f0 = reinterpret_cast<const scalar_t*>(data[7] + i * strides[7]);
            auto* brdf_f = reinterpret_cast<const scalar_t*>(data[8] + i * strides[8]);
            auto* ubrdf_f = reinterpret_cast<const scalar_t*>(data[9] + i * strides[9]);
            auto* slterm_isotropic =
                reinterpret_cast<const scalar_t*>(data[10] + i * strides[10]);
            auto* slterm_f0 = reinterpret_cast<const scalar_t*>(data[11] + i * strides[11]);
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
                work.get());
          }
        },
        grain_size);
  });
}

void solar_2s_flux_cpu(at::TensorIterator& iter, const Solar2sParams& params) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_solar_2s_flux_cpu", [&] {
    const int nlay = last_dim_size(iter.input(0));
    const auto grain_size =
        std::max<int64_t>(1, iter.numel() / std::max<int>(1, at::get_num_threads()));
    const auto workspace_size =
        two_stream_flux_pair_packed_cols<scalar_t>(nlay) * sizeof(scalar_t) +
        solar_2s_workspace_bytes<scalar_t>(nlay);
    const auto* chapman = reinterpret_cast<const scalar_t*>(params.chapman);
    const auto* pxsq = reinterpret_cast<const scalar_t*>(params.pxsq);
    const auto* px0x = reinterpret_cast<const scalar_t*>(params.px0x);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          std::unique_ptr<char[]> work = std::make_unique<char[]>(workspace_size);
          for (int64_t i = 0; i < n; ++i) {
            auto* out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto* tau = reinterpret_cast<const scalar_t*>(data[1] + i * strides[1]);
            auto* omega = reinterpret_cast<const scalar_t*>(data[2] + i * strides[2]);
            auto* asymm = reinterpret_cast<const scalar_t*>(data[3] + i * strides[3]);
            auto* scaling = reinterpret_cast<const scalar_t*>(data[4] + i * strides[4]);
            auto* albedo = reinterpret_cast<const scalar_t*>(data[5] + i * strides[5]);
            auto* flux_factor = reinterpret_cast<const scalar_t*>(data[6] + i * strides[6]);
            auto* brdf_f0 = reinterpret_cast<const scalar_t*>(data[7] + i * strides[7]);
            auto* brdf_f = reinterpret_cast<const scalar_t*>(data[8] + i * strides[8]);
            auto* ubrdf_f = reinterpret_cast<const scalar_t*>(data[9] + i * strides[9]);
            auto* slterm_isotropic =
                reinterpret_cast<const scalar_t*>(data[10] + i * strides[10]);
            auto* slterm_f0 = reinterpret_cast<const scalar_t*>(data[11] + i * strides[11]);
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
                work.get());
          }
        },
        grain_size);
  });
}

void solar_2s_prop_flux_cpu(at::TensorIterator& iter, const Solar2sPropParams& params) {
  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_solar_2s_prop_flux_cpu", [&] {
    const int nlay = static_cast<int>(params.nlay);
    const int nprop = static_cast<int>(params.nprop);
    const auto grain_size =
        std::max<int64_t>(1, iter.numel() / std::max<int>(1, at::get_num_threads()));
    const auto staging_size = 4 * nlay * sizeof(scalar_t);
    const auto workspace_size = staging_size +
                                two_stream_flux_pair_packed_cols<scalar_t>(nlay) *
                                    sizeof(scalar_t) +
                                solar_2s_workspace_bytes<scalar_t>(nlay);
    const auto* chapman = reinterpret_cast<const scalar_t*>(params.chapman);
    const auto* pxsq = reinterpret_cast<const scalar_t*>(params.pxsq);
    const auto* px0x = reinterpret_cast<const scalar_t*>(params.px0x);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          std::unique_ptr<char[]> work = std::make_unique<char[]>(workspace_size);
          for (int64_t i = 0; i < n; ++i) {
            auto* out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto* prop = reinterpret_cast<const scalar_t*>(data[1] + i * strides[1]);
            auto* albedo = reinterpret_cast<const scalar_t*>(data[2] + i * strides[2]);
            auto* flux_factor = reinterpret_cast<const scalar_t*>(data[3] + i * strides[3]);
            auto* brdf_f0 = reinterpret_cast<const scalar_t*>(data[4] + i * strides[4]);
            auto* brdf_f = reinterpret_cast<const scalar_t*>(data[5] + i * strides[5]);
            auto* ubrdf_f = reinterpret_cast<const scalar_t*>(data[6] + i * strides[6]);
            auto* slterm_isotropic = reinterpret_cast<const scalar_t*>(data[7] + i * strides[7]);
            auto* slterm_f0 = reinterpret_cast<const scalar_t*>(data[8] + i * strides[8]);
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
                work.get());
          }
        },
        grain_size);
  });
}

at::Tensor thermal_2s_packed(
    at::Tensor tau,
    at::Tensor omega,
    at::Tensor asymm,
    at::Tensor scaling,
    at::Tensor planck,
    at::Tensor surfbb,
    at::Tensor emissivity,
    at::Tensor albedo,
    at::Tensor brdf_f,
    at::Tensor ubrdf_f,
    double stream_value,
    double user_stream,
    double thermal_tcutoff,
    bool return_profile,
    bool return_fluxes,
    bool do_upwelling,
    bool do_dnwelling,
    bool use_brdf) {
  check_thermal_inputs(tau, omega, asymm, scaling, planck, surfbb, emissivity, albedo, brdf_f, ubrdf_f);
  auto tau_c = tau.contiguous();
  auto omega_c = omega.contiguous();
  auto asymm_c = asymm.contiguous();
  auto scaling_c = scaling.contiguous();
  auto planck_c = planck.contiguous();
  auto surfbb_c = surfbb.contiguous();
  auto emissivity_c = emissivity.contiguous();
  auto albedo_c = albedo.contiguous();
  auto brdf_f_c = use_brdf ? brdf_f.contiguous() : albedo_c;
  auto ubrdf_f_c = use_brdf ? ubrdf_f.contiguous() : albedo_c;

  Thermal2sParams params{
      stream_value,
      user_stream,
      thermal_tcutoff,
      return_profile,
      return_fluxes,
      do_upwelling,
      do_dnwelling,
      use_brdf,
  };
  return run_thermal_2s(
      tau_c,
      omega_c,
      asymm_c,
      scaling_c,
      planck_c,
      surfbb_c,
      emissivity_c,
      albedo_c,
      brdf_f_c,
      ubrdf_f_c,
      params);
}

at::Tensor thermal_2s_flux(
    at::Tensor tau,
    at::Tensor omega,
    at::Tensor asymm,
    at::Tensor scaling,
    at::Tensor planck,
    at::Tensor surfbb,
    at::Tensor emissivity,
    at::Tensor albedo,
    at::Tensor brdf_f,
    at::Tensor ubrdf_f,
    double stream_value,
    double user_stream,
    double thermal_tcutoff,
    bool do_upwelling,
    bool do_dnwelling,
    bool use_brdf) {
  check_thermal_inputs(tau, omega, asymm, scaling, planck, surfbb, emissivity, albedo, brdf_f, ubrdf_f);
  auto tau_c = tau.contiguous();
  auto omega_c = omega.contiguous();
  auto asymm_c = asymm.contiguous();
  auto scaling_c = scaling.contiguous();
  auto planck_c = planck.contiguous();
  auto surfbb_c = surfbb.contiguous();
  auto emissivity_c = emissivity.contiguous();
  auto albedo_c = albedo.contiguous();
  auto brdf_f_c = use_brdf ? brdf_f.contiguous() : albedo_c;
  auto ubrdf_f_c = use_brdf ? ubrdf_f.contiguous() : albedo_c;

  Thermal2sParams params{
      stream_value,
      user_stream,
      thermal_tcutoff,
      false,
      true,
      do_upwelling,
      do_dnwelling,
      use_brdf,
  };
  return run_thermal_2s_flux(
      tau_c,
      omega_c,
      asymm_c,
      scaling_c,
      planck_c,
      surfbb_c,
      emissivity_c,
      albedo_c,
      brdf_f_c,
      ubrdf_f_c,
      params);
}

at::Tensor thermal_2s_prop_flux(
    at::Tensor prop,
    at::Tensor planck,
    at::Tensor surfbb,
    at::Tensor emissivity,
    at::Tensor albedo,
    at::Tensor brdf_f,
    at::Tensor ubrdf_f,
    double stream_value,
    double user_stream,
    double thermal_tcutoff,
    bool do_upwelling,
    bool do_dnwelling,
    bool use_brdf,
    bool flip_layers) {
  check_prop_inputs(prop, "thermal_2s_prop_flux");
  const auto rows = prop_rows(prop);
  const auto nlay = prop.size(2);
  check_matrix_shape(planck, rows, nlay + 1, "thermal_2s_prop_flux planck must have shape (rows, nlay + 1)");
  check_vector_size(surfbb, rows, "thermal_2s_prop_flux surfbb must have shape (rows,)");
  check_vector_size(emissivity, rows, "thermal_2s_prop_flux emissivity must have shape (rows,)");
  check_vector_size(albedo, rows, "thermal_2s_prop_flux albedo must have shape (rows,)");
  check_vector_size(brdf_f, rows, "thermal_2s_prop_flux brdf_f must have shape (rows,)");
  check_vector_size(ubrdf_f, rows, "thermal_2s_prop_flux ubrdf_f must have shape (rows,)");
  check_all_same_dtype_device(prop, "thermal_2s_prop_flux", planck, surfbb, emissivity, albedo, brdf_f, ubrdf_f);

  auto prop_c = prop.contiguous();
  auto planck_c = planck.contiguous();
  auto surfbb_c = surfbb.contiguous();
  auto emissivity_c = emissivity.contiguous();
  auto albedo_c = albedo.contiguous();
  auto brdf_f_c = use_brdf ? brdf_f.contiguous() : albedo_c;
  auto ubrdf_f_c = use_brdf ? ubrdf_f.contiguous() : albedo_c;
  Thermal2sPropParams params{
      prop_c.size(2),
      prop_c.size(3),
      stream_value,
      user_stream,
      thermal_tcutoff,
      do_upwelling,
      do_dnwelling,
      use_brdf,
      flip_layers,
  };
  return run_thermal_2s_prop_flux(
      prop_c, planck_c, surfbb_c, emissivity_c, albedo_c, brdf_f_c, ubrdf_f_c, params);
}

at::Tensor thermal_2s(
    at::Tensor tau,
    at::Tensor omega,
    at::Tensor asymm,
    at::Tensor scaling,
    at::Tensor planck,
    at::Tensor surfbb,
    at::Tensor emissivity,
    at::Tensor albedo,
    double stream_value,
    double user_stream,
    double thermal_tcutoff,
    bool return_profile) {
  check_thermal_inputs(tau, omega, asymm, scaling, planck, surfbb, emissivity, albedo, albedo, albedo);
  auto tau_c = tau.contiguous();
  auto omega_c = omega.contiguous();
  auto asymm_c = asymm.contiguous();
  auto scaling_c = scaling.contiguous();
  auto planck_c = planck.contiguous();
  auto surfbb_c = surfbb.contiguous();
  auto emissivity_c = emissivity.contiguous();
  auto albedo_c = albedo.contiguous();

  Thermal2sParams params{
      stream_value,
      user_stream,
      thermal_tcutoff,
      return_profile,
      false,
      true,
      false,
      false,
  };
  auto packed = run_thermal_2s(
      tau_c,
      omega_c,
      asymm_c,
      scaling_c,
      planck_c,
      surfbb_c,
      emissivity_c,
      albedo_c,
      albedo_c,
      albedo_c,
      params);
  return return_profile ? packed : packed.select(1, 0);
}

at::Tensor thermal_fo(
    at::Tensor tau,
    at::Tensor omega,
    at::Tensor scaling,
    at::Tensor planck,
    at::Tensor surfbb,
    at::Tensor emissivity,
    at::Tensor heights,
    at::Tensor xfine,
    at::Tensor wfine,
    at::Tensor cota,
    at::Tensor cotfine,
    at::Tensor csqfine,
    double rayconv,
    bool do_nadir,
    bool do_optical_deltam_scaling,
    bool do_source_deltam_scaling,
    bool return_components,
    bool return_profile) {
  check_thermal_fo_inputs(
      tau, omega, scaling, planck, surfbb, emissivity, heights, xfine, wfine, cota, cotfine, csqfine);
  auto tau_c = tau.contiguous();
  auto omega_c = omega.contiguous();
  auto scaling_c = scaling.contiguous();
  auto planck_c = planck.contiguous();
  auto surfbb_c = surfbb.contiguous();
  auto emissivity_c = emissivity.contiguous();
  auto heights_c = heights.contiguous();
  auto xfine_c = xfine.contiguous();
  auto wfine_c = wfine.contiguous();
  auto cota_c = cota.contiguous();
  auto cotfine_c = cotfine.contiguous();
  auto csqfine_c = csqfine.contiguous();

  const auto rows = tau_c.size(0);
  const auto nlay = tau_c.size(1);
  const auto nlev = nlay + 1;
  const auto output_cols =
      return_profile ? nlev * (return_components ? int64_t{3} : int64_t{1})
                     : (return_components ? int64_t{3} : int64_t{1});
  auto height_delta =
      at::sub(heights_c.slice(0, 0, nlay), heights_c.slice(0, 1, nlay + 1)).contiguous();
  auto output = at::empty({rows, output_cols}, tau_c.options());

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape({rows, output_cols}, /*squash_dims=*/{1})
                  .add_output(output)
                  .add_input(tau_c)
                  .add_input(omega_c)
                  .add_input(scaling_c)
                  .add_input(planck_c)
                  .add_owned_input(surfbb_c.view({rows, 1}))
                  .add_owned_input(emissivity_c.view({rows, 1}))
                  .build();

  ThermalFoParams params{
      static_cast<int>(xfine_c.size(0)),
      do_nadir,
      do_optical_deltam_scaling,
      do_source_deltam_scaling,
      return_components,
      return_profile,
      rayconv,
      height_delta.data_ptr(),
      xfine_c.data_ptr(),
      wfine_c.data_ptr(),
      cota_c.data_ptr(),
      cotfine_c.data_ptr(),
      csqfine_c.data_ptr(),
  };
  PY2SESS_DISPATCH_KERNEL(output, thermal_fo_cpu, thermal_fo_cuda, iter, params);
  return (return_components || return_profile) ? output : output.select(1, 0);
}

at::Tensor solar_fo(
    at::Tensor tau,
    at::Tensor omega,
    at::Tensor scaling,
    at::Tensor albedo,
    at::Tensor flux_factor,
    at::Tensor exact_scatter,
    at::Tensor inv_layer_thickness,
    at::Tensor sunpathsnl,
    at::Tensor cota,
    at::Tensor cotfine,
    at::Tensor csqfine,
    at::Tensor wfine,
    at::Tensor xfine,
    at::Tensor sunpathsfine,
    at::Tensor nfinedivs,
    at::Tensor ntraversefine,
    double mu0,
    double rayconv,
    int ntrav_nl,
    bool do_nadir,
    bool return_components,
    bool return_profile) {
  check_solar_fo_inputs(
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
      ntraversefine);
  auto tau_c = tau.contiguous();
  auto omega_c = omega.contiguous();
  auto scaling_c = scaling.contiguous();
  auto albedo_c = albedo.contiguous();
  auto flux_factor_c = flux_factor.contiguous();
  auto exact_scatter_c = exact_scatter.contiguous();
  auto inv_layer_thickness_c = inv_layer_thickness.contiguous();
  auto sunpathsnl_c = sunpathsnl.contiguous();
  auto cota_c = cota.contiguous();
  auto cotfine_c = cotfine.contiguous();
  auto csqfine_c = csqfine.contiguous();
  auto wfine_c = wfine.contiguous();
  auto xfine_c = xfine.contiguous();
  auto sunpathsfine_c = sunpathsfine.contiguous();
  auto nfinedivs_c = nfinedivs.contiguous();
  auto ntraversefine_c = ntraversefine.contiguous();

  const auto rows = tau_c.size(0);
  const auto nlev = tau_c.size(1) + 1;
  const auto output_cols =
      return_profile ? nlev * (return_components ? int64_t{3} : int64_t{1})
                     : (return_components ? int64_t{3} : int64_t{1});
  auto output = at::empty({rows, output_cols}, tau_c.options());

  auto iter = at::TensorIteratorConfig()
                  .resize_outputs(false)
                  .check_all_same_dtype(true)
                  .declare_static_shape({rows, output_cols}, /*squash_dims=*/{1})
                  .add_output(output)
                  .add_input(tau_c)
                  .add_input(omega_c)
                  .add_input(scaling_c)
                  .add_owned_input(albedo_c.view({rows, 1}))
                  .add_owned_input(flux_factor_c.view({rows, 1}))
                  .add_input(exact_scatter_c)
                  .build();

  SolarFoParams params{
      static_cast<int>(xfine_c.size(0)),
      ntrav_nl,
      static_cast<int>(sunpathsfine_c.size(0)),
      do_nadir,
      return_components,
      return_profile,
      mu0,
      rayconv,
      inv_layer_thickness_c.data_ptr(),
      sunpathsnl_c.data_ptr(),
      cota_c.data_ptr(),
      cotfine_c.data_ptr(),
      csqfine_c.data_ptr(),
      wfine_c.data_ptr(),
      xfine_c.data_ptr(),
      sunpathsfine_c.data_ptr(),
      nfinedivs_c.data_ptr(),
      ntraversefine_c.data_ptr(),
  };
  PY2SESS_DISPATCH_KERNEL(output, solar_fo_cpu, solar_fo_cuda, iter, params);
  return (return_components || return_profile) ? output : output.select(1, 0);
}

at::Tensor solar_2s_packed(
    at::Tensor tau,
    at::Tensor omega,
    at::Tensor asymm,
    at::Tensor scaling,
    at::Tensor albedo,
    at::Tensor flux_factor,
    at::Tensor brdf_f0,
    at::Tensor brdf_f,
    at::Tensor ubrdf_f,
    at::Tensor slterm_isotropic,
    at::Tensor slterm_f0,
    at::Tensor chapman,
    at::Tensor pxsq,
    at::Tensor px0x,
    double stream_value,
    double x0,
    double user_stream,
    double user_secant,
    double azmfac,
    double px11,
    double ulp,
    bool return_profile,
    bool return_fluxes,
    bool do_upwelling,
    bool do_dnwelling,
    bool use_brdf,
    bool use_surface_leaving,
    bool sl_isotropic) {
  check_solar_required_inputs(tau, omega, asymm, scaling, albedo, flux_factor, chapman, pxsq, px0x);
  if (use_brdf) {
    check_matrix_min_cols(brdf_f0, tau.size(0), 2, "solar_2s brdf_f0 must have shape (rows, 2)");
    check_matrix_min_cols(brdf_f, tau.size(0), 2, "solar_2s brdf_f must have shape (rows, 2)");
    check_matrix_min_cols(ubrdf_f, tau.size(0), 2, "solar_2s ubrdf_f must have shape (rows, 2)");
    check_all_same_dtype_device(tau, "solar_2s", brdf_f0, brdf_f, ubrdf_f);
  }
  if (use_surface_leaving) {
    check_vector_size(
        slterm_isotropic, tau.size(0), "solar_2s slterm_isotropic must have shape (rows,)");
    check_matrix_min_cols(slterm_f0, tau.size(0), 2, "solar_2s slterm_f0 must have shape (rows, 2)");
    check_all_same_dtype_device(tau, "solar_2s", slterm_isotropic, slterm_f0);
  }
  auto tau_c = tau.contiguous();
  auto omega_c = omega.contiguous();
  auto asymm_c = asymm.contiguous();
  auto scaling_c = scaling.contiguous();
  auto albedo_c = albedo.contiguous();
  auto flux_factor_c = flux_factor.contiguous();
  auto chapman_c = chapman.contiguous();
  auto pxsq_c = pxsq.contiguous();
  auto px0x_c = px0x.contiguous();
  auto optional_pair = albedo_c.view({albedo_c.size(0), 1}).expand({albedo_c.size(0), 2});
  auto brdf_f0_c = use_brdf ? brdf_f0.contiguous() : optional_pair;
  auto brdf_f_c = use_brdf ? brdf_f.contiguous() : optional_pair;
  auto ubrdf_f_c = use_brdf ? ubrdf_f.contiguous() : optional_pair;
  auto slterm_isotropic_c = use_surface_leaving ? slterm_isotropic.contiguous() : albedo_c;
  auto slterm_f0_c = use_surface_leaving ? slterm_f0.contiguous() : optional_pair;

  Solar2sParams params{
      stream_value,
      x0,
      user_stream,
      user_secant,
      azmfac,
      px11,
      ulp,
      return_profile,
      return_fluxes,
      do_upwelling,
      do_dnwelling,
      use_brdf,
      use_surface_leaving,
      sl_isotropic,
      chapman_c.data_ptr(),
      pxsq_c.data_ptr(),
      px0x_c.data_ptr(),
  };
  return run_solar_2s(
      tau_c,
      omega_c,
      asymm_c,
      scaling_c,
      albedo_c,
      flux_factor_c,
      brdf_f0_c,
      brdf_f_c,
      ubrdf_f_c,
      slterm_isotropic_c,
      slterm_f0_c,
      params);
}

at::Tensor solar_2s_flux(
    at::Tensor tau,
    at::Tensor omega,
    at::Tensor asymm,
    at::Tensor scaling,
    at::Tensor albedo,
    at::Tensor flux_factor,
    at::Tensor brdf_f0,
    at::Tensor brdf_f,
    at::Tensor ubrdf_f,
    at::Tensor slterm_isotropic,
    at::Tensor slterm_f0,
    at::Tensor chapman,
    at::Tensor pxsq,
    at::Tensor px0x,
    double stream_value,
    double x0,
    double user_stream,
    double user_secant,
    double azmfac,
    double px11,
    double ulp,
    bool do_upwelling,
    bool do_dnwelling,
    bool use_brdf,
    bool use_surface_leaving,
    bool sl_isotropic) {
  check_solar_required_inputs(tau, omega, asymm, scaling, albedo, flux_factor, chapman, pxsq, px0x);
  if (use_brdf) {
    check_matrix_min_cols(brdf_f0, tau.size(0), 2, "solar_2s brdf_f0 must have shape (rows, 2)");
    check_matrix_min_cols(brdf_f, tau.size(0), 2, "solar_2s brdf_f must have shape (rows, 2)");
    check_matrix_min_cols(ubrdf_f, tau.size(0), 2, "solar_2s ubrdf_f must have shape (rows, 2)");
    check_all_same_dtype_device(tau, "solar_2s", brdf_f0, brdf_f, ubrdf_f);
  }
  if (use_surface_leaving) {
    check_vector_size(
        slterm_isotropic, tau.size(0), "solar_2s slterm_isotropic must have shape (rows,)");
    check_matrix_min_cols(slterm_f0, tau.size(0), 2, "solar_2s slterm_f0 must have shape (rows, 2)");
    check_all_same_dtype_device(tau, "solar_2s", slterm_isotropic, slterm_f0);
  }
  auto tau_c = tau.contiguous();
  auto omega_c = omega.contiguous();
  auto asymm_c = asymm.contiguous();
  auto scaling_c = scaling.contiguous();
  auto albedo_c = albedo.contiguous();
  auto flux_factor_c = flux_factor.contiguous();
  auto chapman_c = chapman.contiguous();
  auto pxsq_c = pxsq.contiguous();
  auto px0x_c = px0x.contiguous();
  auto optional_pair = albedo_c.view({albedo_c.size(0), 1}).expand({albedo_c.size(0), 2});
  auto brdf_f0_c = use_brdf ? brdf_f0.contiguous() : optional_pair;
  auto brdf_f_c = use_brdf ? brdf_f.contiguous() : optional_pair;
  auto ubrdf_f_c = use_brdf ? ubrdf_f.contiguous() : optional_pair;
  auto slterm_isotropic_c = use_surface_leaving ? slterm_isotropic.contiguous() : albedo_c;
  auto slterm_f0_c = use_surface_leaving ? slterm_f0.contiguous() : optional_pair;

  Solar2sParams params{
      stream_value,
      x0,
      user_stream,
      user_secant,
      azmfac,
      px11,
      ulp,
      false,
      true,
      do_upwelling,
      do_dnwelling,
      use_brdf,
      use_surface_leaving,
      sl_isotropic,
      chapman_c.data_ptr(),
      pxsq_c.data_ptr(),
      px0x_c.data_ptr(),
  };
  return run_solar_2s_flux(
      tau_c,
      omega_c,
      asymm_c,
      scaling_c,
      albedo_c,
      flux_factor_c,
      brdf_f0_c,
      brdf_f_c,
      ubrdf_f_c,
      slterm_isotropic_c,
      slterm_f0_c,
      params);
}

at::Tensor solar_2s_prop_flux(
    at::Tensor prop,
    at::Tensor albedo,
    at::Tensor flux_factor,
    at::Tensor brdf_f0,
    at::Tensor brdf_f,
    at::Tensor ubrdf_f,
    at::Tensor slterm_isotropic,
    at::Tensor slterm_f0,
    at::Tensor chapman,
    at::Tensor pxsq,
    at::Tensor px0x,
    double stream_value,
    double x0,
    double user_stream,
    double user_secant,
    double azmfac,
    double px11,
    double ulp,
    bool do_upwelling,
    bool do_dnwelling,
    bool use_brdf,
    bool use_surface_leaving,
    bool sl_isotropic,
    bool flip_layers) {
  check_prop_inputs(prop, "solar_2s_prop_flux");
  const auto rows = prop_rows(prop);
  const auto nlay = prop.size(2);
  check_vector_size(albedo, rows, "solar_2s_prop_flux albedo must have shape (rows,)");
  check_vector_size(flux_factor, rows, "solar_2s_prop_flux flux_factor must have shape (rows,)");
  check_matrix_shape(chapman, nlay, nlay, "solar_2s_prop_flux chapman must have shape (nlay, nlay)");
  TORCH_CHECK(pxsq.numel() >= 2, "solar_2s_prop_flux pxsq must contain two Fourier values");
  TORCH_CHECK(px0x.numel() >= 2, "solar_2s_prop_flux px0x must contain two Fourier values");
  check_all_same_dtype_device(prop, "solar_2s_prop_flux", albedo, flux_factor, chapman, pxsq, px0x);
  if (use_brdf) {
    check_matrix_min_cols(brdf_f0, rows, 2, "solar_2s_prop_flux brdf_f0 must have shape (rows, 2)");
    check_matrix_min_cols(brdf_f, rows, 2, "solar_2s_prop_flux brdf_f must have shape (rows, 2)");
    check_matrix_min_cols(ubrdf_f, rows, 2, "solar_2s_prop_flux ubrdf_f must have shape (rows, 2)");
    check_all_same_dtype_device(prop, "solar_2s_prop_flux", brdf_f0, brdf_f, ubrdf_f);
  }
  if (use_surface_leaving) {
    check_vector_size(slterm_isotropic, rows, "solar_2s_prop_flux slterm_isotropic must have shape (rows,)");
    check_matrix_min_cols(slterm_f0, rows, 2, "solar_2s_prop_flux slterm_f0 must have shape (rows, 2)");
    check_all_same_dtype_device(prop, "solar_2s_prop_flux", slterm_isotropic, slterm_f0);
  }

  auto prop_c = prop.contiguous();
  auto albedo_c = albedo.contiguous();
  auto flux_factor_c = flux_factor.contiguous();
  auto chapman_c = flip_layers ? chapman.flip({0, 1}).contiguous() : chapman.contiguous();
  auto pxsq_c = pxsq.contiguous();
  auto px0x_c = px0x.contiguous();
  auto optional_pair = albedo_c.view({albedo_c.size(0), 1}).expand({albedo_c.size(0), 2});
  auto brdf_f0_c = use_brdf ? brdf_f0.contiguous() : optional_pair;
  auto brdf_f_c = use_brdf ? brdf_f.contiguous() : optional_pair;
  auto ubrdf_f_c = use_brdf ? ubrdf_f.contiguous() : optional_pair;
  auto slterm_isotropic_c = use_surface_leaving ? slterm_isotropic.contiguous() : albedo_c;
  auto slterm_f0_c = use_surface_leaving ? slterm_f0.contiguous() : optional_pair;

  Solar2sPropParams params{
      prop_c.size(2),
      prop_c.size(3),
      stream_value,
      x0,
      user_stream,
      user_secant,
      azmfac,
      px11,
      ulp,
      do_upwelling,
      do_dnwelling,
      use_brdf,
      use_surface_leaving,
      sl_isotropic,
      flip_layers,
      chapman_c.data_ptr(),
      pxsq_c.data_ptr(),
      px0x_c.data_ptr(),
  };
  return run_solar_2s_prop_flux(
      prop_c,
      albedo_c,
      flux_factor_c,
      brdf_f0_c,
      brdf_f_c,
      ubrdf_f_c,
      slterm_isotropic_c,
      slterm_f0_c,
      params);
}

at::Tensor solar_2s(
    at::Tensor tau,
    at::Tensor omega,
    at::Tensor asymm,
    at::Tensor scaling,
    at::Tensor albedo,
    at::Tensor flux_factor,
    at::Tensor chapman,
    at::Tensor pxsq,
    at::Tensor px0x,
    double stream_value,
    double x0,
    double user_stream,
    double user_secant,
    double azmfac,
    double px11,
    double ulp,
    bool return_profile) {
  check_solar_required_inputs(tau, omega, asymm, scaling, albedo, flux_factor, chapman, pxsq, px0x);
  auto tau_c = tau.contiguous();
  auto omega_c = omega.contiguous();
  auto asymm_c = asymm.contiguous();
  auto scaling_c = scaling.contiguous();
  auto albedo_c = albedo.contiguous();
  auto flux_factor_c = flux_factor.contiguous();
  auto chapman_c = chapman.contiguous();
  auto pxsq_c = pxsq.contiguous();
  auto px0x_c = px0x.contiguous();

  Solar2sParams params{
      stream_value,
      x0,
      user_stream,
      user_secant,
      azmfac,
      px11,
      ulp,
      return_profile,
      false,
      true,
      false,
      false,
      false,
      false,
      chapman_c.data_ptr(),
      pxsq_c.data_ptr(),
      px0x_c.data_ptr(),
  };
  auto optional_pair = albedo_c.view({albedo_c.size(0), 1}).expand({albedo_c.size(0), 2});
  auto packed = run_solar_2s(
      tau_c,
      omega_c,
      asymm_c,
      scaling_c,
      albedo_c,
      flux_factor_c,
      optional_pair,
      optional_pair,
      optional_pair,
      albedo_c,
      optional_pair,
      params);
  return return_profile ? packed : packed.select(1, 0);
}

at::Tensor tensoriterator_copy(at::Tensor input) {
  TORCH_CHECK(input.is_floating_point(), "tensoriterator_copy expects a floating tensor");
  auto contiguous = input.contiguous();
  auto output = at::empty_like(contiguous);

  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(true)
                  .add_output(output)
                  .add_input(contiguous)
                  .build();

  AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "py2sess_tensoriterator_copy_cpu", [&] {
    const auto grain_size =
        std::max<int64_t>(1, iter.numel() / std::max<int>(1, at::get_num_threads()));
    const auto workspace_size = rt_workspace_bytes(1);
    iter.for_each(
        [&](char** data, const int64_t* strides, int64_t n) {
          std::unique_ptr<char[]> work = std::make_unique<char[]>(workspace_size);
          (void)work;
          for (int64_t i = 0; i < n; ++i) {
            auto* out = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            auto* in = reinterpret_cast<const scalar_t*>(data[1] + i * strides[1]);
            *out = *in;
          }
        },
        grain_size);
  });

  return output;
}

std::size_t workspace_bytes(std::int64_t nlay) {
  return rt_workspace_bytes(nlay);
}

}  // namespace py2sess_native

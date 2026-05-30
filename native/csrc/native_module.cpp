#include "native_module.hpp"

#include <c10/util/Exception.h>

#include <initializer_list>
#include <utility>
#include <vector>

#include "native_dispatch.hpp"

namespace py2sess_native {
namespace {

at::Tensor ensure_prop4(at::Tensor prop) {
  while (prop.dim() < 4) {
    prop = prop.unsqueeze(0);
  }
  TORCH_CHECK(prop.dim() == 4, "TwoStreamEssNative expects prop with shape (nwave, ncol, nlyr, nprop)");
  TORCH_CHECK(prop.size(3) >= 3, "TwoStreamEssNative expects prop[..., 0:3] = tau, omega, asymm");
  TORCH_CHECK(prop.is_floating_point(), "TwoStreamEssNative expects floating prop tensors");
  return prop;
}

at::Tensor to_prop_options(at::Tensor tensor, const at::Tensor& prop) {
  if (tensor.scalar_type() == prop.scalar_type() && tensor.device() == prop.device()) {
    return tensor;
  }
  return tensor.to(prop.device(), prop.scalar_type());
}

at::Tensor flatten_levels(
    at::Tensor tensor,
    int64_t nwave,
    int64_t ncol,
    int64_t nlev,
    bool flip_levels,
    const char* label) {
  if (flip_levels) {
    tensor = tensor.flip({-1});
  }
  const auto rows = nwave * ncol;
  if (tensor.dim() == 2) {
    TORCH_CHECK(
        tensor.size(0) == rows && tensor.size(1) == nlev,
        label,
        " must have shape (rows, nlev) or (nwave, ncol, nlev)");
    return tensor.contiguous();
  }
  TORCH_CHECK(
      tensor.dim() == 3 && tensor.size(0) == nwave && tensor.size(1) == ncol && tensor.size(2) == nlev,
      label,
      " must have shape (rows, nlev) or (nwave, ncol, nlev)");
  return tensor.contiguous().view({rows, nlev});
}

at::Tensor flatten_field(
    at::Tensor tensor,
    int64_t nwave,
    int64_t ncol,
    const char* label) {
  const auto rows = nwave * ncol;
  if (tensor.dim() == 0) {
    return tensor.expand({rows}).contiguous();
  }
  if (tensor.dim() == 1) {
    if (tensor.size(0) == rows) {
      return tensor.contiguous();
    }
    if (tensor.size(0) == ncol) {
      return tensor.view({1, ncol}).expand({nwave, ncol}).contiguous().view({rows});
    }
    if (tensor.size(0) == 1) {
      return tensor.expand({rows}).contiguous();
    }
  }
  if (tensor.dim() == 2) {
    TORCH_CHECK(
        tensor.size(0) == nwave && tensor.size(1) == ncol,
        label,
        " must have shape (rows,), (ncol,), or (nwave, ncol)");
    return tensor.contiguous().view({rows});
  }
  TORCH_CHECK(false, label, " must have shape (rows,), (ncol,), or (nwave, ncol)");
}

at::Tensor flatten_pair_field(
    at::Tensor tensor,
    int64_t nwave,
    int64_t ncol,
    const char* label) {
  const auto rows = nwave * ncol;
  if (tensor.dim() == 2) {
    TORCH_CHECK(
        tensor.size(0) == rows && tensor.size(1) >= 2,
        label,
        " must have shape (rows, 2) or (nwave, ncol, 2)");
    return tensor.contiguous();
  }
  TORCH_CHECK(
      tensor.dim() == 3 && tensor.size(0) == nwave && tensor.size(1) == ncol && tensor.size(2) >= 2,
      label,
      " must have shape (rows, 2) or (nwave, ncol, 2)");
  return tensor.contiguous().view({rows, tensor.size(2)});
}

at::Tensor require_bc(
    std::map<std::string, at::Tensor>* bc,
    const at::Tensor& prop,
    const std::string& key) {
  auto item = bc->find(key);
  TORCH_CHECK(item != bc->end(), "TwoStreamEssNative missing boundary tensor '", key, "'");
  return to_prop_options(item->second, prop);
}

at::Tensor require_bc_any(
    std::map<std::string, at::Tensor>* bc,
    const at::Tensor& prop,
    std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    auto item = bc->find(key);
    if (item != bc->end()) {
      return to_prop_options(item->second, prop);
    }
  }
  std::string message = "TwoStreamEssNative missing boundary tensor ";
  bool first = true;
  for (const char* key : keys) {
    message += first ? "'" : " or '";
    message += key;
    message += "'";
    first = false;
  }
  TORCH_CHECK(false, message);
}

at::Tensor optional_pair_view(const at::Tensor& field) {
  return field.view({field.size(0), 1}).expand({field.size(0), 2});
}

}  // namespace

TwoStreamEssNativeImpl::TwoStreamEssNativeImpl() = default;

TwoStreamEssNativeImpl::TwoStreamEssNativeImpl(TwoStreamEssNativeOptions options)
    : options(std::move(options)) {}

void TwoStreamEssNativeImpl::reset() {}

at::Tensor TwoStreamEssNativeImpl::thermal_2s_flux(
    at::Tensor prop,
    std::map<std::string, at::Tensor>* bc) {
  prop = ensure_prop4(std::move(prop));
  const auto nwave = prop.size(0);
  const auto ncol = prop.size(1);
  const auto nlay = prop.size(2);
  const auto nlev = nlay + 1;

  auto planck = flatten_levels(require_bc(bc, prop, "planck"), nwave, ncol, nlev, false, "planck");
  auto surfbb = flatten_field(require_bc(bc, prop, "surfbb"), nwave, ncol, "surfbb");
  auto emissivity = flatten_field(require_bc(bc, prop, "emissivity"), nwave, ncol, "emissivity");
  auto albedo = flatten_field(require_bc(bc, prop, "albedo"), nwave, ncol, "albedo");
  auto brdf_f = options.use_brdf
                    ? flatten_field(require_bc(bc, prop, "brdf_f"), nwave, ncol, "brdf_f")
                    : albedo;
  auto ubrdf_f = options.use_brdf
                     ? flatten_field(require_bc(bc, prop, "ubrdf_f"), nwave, ncol, "ubrdf_f")
                     : albedo;

  return py2sess_native::thermal_2s_prop_flux(
      prop,
      planck,
      surfbb,
      emissivity,
      albedo,
      brdf_f,
      ubrdf_f,
      options.stream_value,
      options.user_stream,
      options.thermal_tcutoff,
      options.do_upwelling,
      options.do_dnwelling,
      options.use_brdf,
      options.flip_layers);
}

at::Tensor TwoStreamEssNativeImpl::solar_2s_flux(
    at::Tensor prop,
    std::map<std::string, at::Tensor>* bc) {
  prop = ensure_prop4(std::move(prop));
  const auto nwave = prop.size(0);
  const auto ncol = prop.size(1);

  auto albedo = flatten_field(require_bc(bc, prop, "albedo"), nwave, ncol, "albedo");
  auto flux_factor = flatten_field(require_bc_any(bc, prop, {"flux_factor", "fbeam"}), nwave, ncol, "flux_factor");
  auto chapman = require_bc(bc, prop, "chapman");
  auto pxsq = require_bc(bc, prop, "pxsq");
  auto px0x = require_bc(bc, prop, "px0x");

  auto optional_pair = optional_pair_view(albedo);
  auto brdf_f0 = options.use_brdf
                     ? flatten_pair_field(require_bc(bc, prop, "brdf_f0"), nwave, ncol, "brdf_f0")
                     : optional_pair;
  auto brdf_f = options.use_brdf
                    ? flatten_pair_field(require_bc(bc, prop, "brdf_f"), nwave, ncol, "brdf_f")
                    : optional_pair;
  auto ubrdf_f = options.use_brdf
                     ? flatten_pair_field(require_bc(bc, prop, "ubrdf_f"), nwave, ncol, "ubrdf_f")
                     : optional_pair;
  auto slterm_isotropic =
      options.use_surface_leaving
          ? flatten_field(require_bc(bc, prop, "slterm_isotropic"), nwave, ncol, "slterm_isotropic")
          : albedo;
  auto slterm_f0 =
      options.use_surface_leaving
          ? flatten_pair_field(require_bc(bc, prop, "slterm_f0"), nwave, ncol, "slterm_f0")
          : optional_pair;

  return py2sess_native::solar_2s_prop_flux(
      prop,
      albedo,
      flux_factor,
      brdf_f0,
      brdf_f,
      ubrdf_f,
      slterm_isotropic,
      slterm_f0,
      chapman,
      pxsq,
      px0x,
      options.stream_value,
      options.x0,
      options.user_stream,
      options.user_secant,
      options.azmfac,
      options.px11,
      options.ulp,
      options.do_upwelling,
      options.do_dnwelling,
      options.use_brdf,
      options.use_surface_leaving,
      options.sl_isotropic,
      options.flip_layers,
      options.plane_parallel_chapman);
}

}  // namespace py2sess_native

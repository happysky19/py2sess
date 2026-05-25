#pragma once

#include <ATen/Tensor.h>
#include <ATen/TensorIterator.h>

#include <cstdint>

namespace py2sess_native {

struct Thermal2sParams {
  double stream_value;
  double user_stream;
  double thermal_tcutoff;
  bool return_profile;
  bool return_fluxes;
  bool do_upwelling;
  bool do_dnwelling;
  bool use_brdf;
};

struct Thermal2sPropParams {
  std::int64_t nlay;
  std::int64_t nprop;
  double stream_value;
  double user_stream;
  double thermal_tcutoff;
  bool do_upwelling;
  bool do_dnwelling;
  bool use_brdf;
  bool flip_layers;
};

struct Solar2sParams {
  double stream_value;
  double x0;
  double user_stream;
  double user_secant;
  double azmfac;
  double px11;
  double ulp;
  bool return_profile;
  bool return_fluxes;
  bool do_upwelling;
  bool do_dnwelling;
  bool use_brdf;
  bool use_surface_leaving;
  bool sl_isotropic;
  const void* chapman;
  const void* pxsq;
  const void* px0x;
};

struct Solar2sPropParams {
  std::int64_t nlay;
  std::int64_t nprop;
  double stream_value;
  double x0;
  double user_stream;
  double user_secant;
  double azmfac;
  double px11;
  double ulp;
  bool do_upwelling;
  bool do_dnwelling;
  bool use_brdf;
  bool use_surface_leaving;
  bool sl_isotropic;
  bool flip_layers;
  const void* chapman;
  const void* pxsq;
  const void* px0x;
};

struct ThermalFoParams {
  int nfine;
  bool do_nadir;
  bool do_optical_deltam_scaling;
  bool do_source_deltam_scaling;
  bool return_components;
  bool return_profile;
  double rayconv;
  const void* height_delta;
  const void* xfine;
  const void* wfine;
  const void* cota;
  const void* cotfine;
  const void* csqfine;
};

struct ThermalFoFluxParams {
  double stream_value;
  bool do_optical_deltam_scaling;
  bool do_source_deltam_scaling;
  int n_mu;
  const void* mu_nodes;
  const void* mu_weights;
};

struct SolarFoParams {
  int nfine;
  int ntrav_nl;
  int ntrav_max;
  bool do_nadir;
  bool return_components;
  bool return_profile;
  double mu0;
  double rayconv;
  const void* inv_layer_thickness;
  const void* sunpathsnl;
  const void* cota;
  const void* cotfine;
  const void* csqfine;
  const void* wfine;
  const void* xfine;
  const void* sunpathsfine;
  const void* nfinedivs;
  const void* ntraversefine;
};

struct SolarFoPpParams {
  double mu0;
  double user_stream;
  bool return_profile;
};

struct SolarFoFluxParams {
  double stream_value;
  double mu0;
  bool do_optical_deltam_scaling;
};

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
    bool use_brdf);

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
    bool use_brdf);

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
    bool flip_layers);

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
    bool return_profile);

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
    bool return_profile);

at::Tensor thermal_fo_flux_correction(
    at::Tensor tau,
    at::Tensor omega,
    at::Tensor scaling,
    at::Tensor planck,
    at::Tensor surfbb,
    at::Tensor emissivity,
    at::Tensor mu_nodes,
    at::Tensor mu_weights,
    double stream_value,
    bool do_optical_deltam_scaling,
    bool do_source_deltam_scaling);

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
    bool sl_isotropic);

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
    bool sl_isotropic);

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
    bool flip_layers);

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
    bool return_profile);

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
    bool return_profile);

at::Tensor solar_fo_plane_parallel(
    at::Tensor tau,
    at::Tensor omega,
    at::Tensor scaling,
    at::Tensor surface_reflectance,
    at::Tensor flux_factor,
    at::Tensor exact_scatter,
    double mu0,
    double user_stream,
    bool return_profile);

at::Tensor solar_fo_flux_correction(
    at::Tensor tau,
    at::Tensor omega,
    at::Tensor scaling,
    at::Tensor surface_reflectance,
    at::Tensor flux_factor,
    double stream_value,
    double mu0,
    bool do_optical_deltam_scaling);

void thermal_2s_cpu(at::TensorIterator& iter, const Thermal2sParams& params);
void solar_2s_cpu(at::TensorIterator& iter, const Solar2sParams& params);
void thermal_2s_flux_cpu(at::TensorIterator& iter, const Thermal2sParams& params);
void solar_2s_flux_cpu(at::TensorIterator& iter, const Solar2sParams& params);
void thermal_2s_prop_flux_cpu(at::TensorIterator& iter, const Thermal2sPropParams& params);
void solar_2s_prop_flux_cpu(at::TensorIterator& iter, const Solar2sPropParams& params);
void thermal_fo_cpu(at::TensorIterator& iter, const ThermalFoParams& params);
void thermal_fo_flux_correction_cpu(
    at::TensorIterator& iter,
    const ThermalFoFluxParams& params);
void solar_fo_cpu(at::TensorIterator& iter, const SolarFoParams& params);
void solar_fo_plane_parallel_cpu(at::TensorIterator& iter, const SolarFoPpParams& params);
void solar_fo_flux_correction_cpu(at::TensorIterator& iter, const SolarFoFluxParams& params);

#ifdef PY2SESS_WITH_CUDA
void thermal_2s_cuda(at::TensorIterator& iter, const Thermal2sParams& params);
void solar_2s_cuda(at::TensorIterator& iter, const Solar2sParams& params);
void thermal_2s_flux_cuda(at::TensorIterator& iter, const Thermal2sParams& params);
void solar_2s_flux_cuda(at::TensorIterator& iter, const Solar2sParams& params);
void thermal_2s_prop_flux_cuda(at::TensorIterator& iter, const Thermal2sPropParams& params);
void solar_2s_prop_flux_cuda(at::TensorIterator& iter, const Solar2sPropParams& params);
void thermal_fo_cuda(at::TensorIterator& iter, const ThermalFoParams& params);
void thermal_fo_flux_correction_cuda(
    at::TensorIterator& iter,
    const ThermalFoFluxParams& params);
void solar_fo_cuda(at::TensorIterator& iter, const SolarFoParams& params);
void solar_fo_plane_parallel_cuda(at::TensorIterator& iter, const SolarFoPpParams& params);
void solar_fo_flux_correction_cuda(at::TensorIterator& iter, const SolarFoFluxParams& params);
#endif

}  // namespace py2sess_native

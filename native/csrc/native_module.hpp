#pragma once

#include <ATen/Tensor.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>

#include <map>
#include <string>

namespace py2sess_native {

struct TwoStreamEssNativeOptions {
  double stream_value = 0.5;
  double user_stream = 1.0;
  double thermal_tcutoff = 88.0;
  double x0 = 1.0;
  double user_secant = 1.0;
  double azmfac = 1.0;
  double px11 = 1.0;
  double ulp = 0.0;
  bool do_upwelling = true;
  bool do_dnwelling = false;
  bool use_brdf = false;
  bool use_surface_leaving = false;
  bool sl_isotropic = true;
  bool flip_layers = false;
};

class TwoStreamEssNativeImpl : public torch::nn::Cloneable<TwoStreamEssNativeImpl> {
 public:
  TwoStreamEssNativeOptions options;

  TwoStreamEssNativeImpl();
  explicit TwoStreamEssNativeImpl(TwoStreamEssNativeOptions options);
  void reset() override;

  at::Tensor thermal_2s_flux(at::Tensor prop, std::map<std::string, at::Tensor>* bc);
  at::Tensor solar_2s_flux(at::Tensor prop, std::map<std::string, at::Tensor>* bc);
};

TORCH_MODULE(TwoStreamEssNative);

}  // namespace py2sess_native

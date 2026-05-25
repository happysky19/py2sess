#include <torch/extension.h>

#include "native_dispatch.hpp"
#include "native_module.hpp"

#include <map>
#include <sstream>
#include <string>

namespace py2sess_native {
namespace {

std::map<std::string, at::Tensor> tensor_kwargs(const pybind11::kwargs& kwargs) {
  std::map<std::string, at::Tensor> bc;
  for (auto item : kwargs) {
    auto key = pybind11::cast<std::string>(item.first);
    auto value = pybind11::cast<at::Tensor>(item.second);
    bc.emplace(std::move(key), std::move(value));
  }
  return bc;
}

pybind11::dict backend_info() {
  pybind11::dict info;
  info["backend"] = "native-extension";
#ifdef PY2SESS_WITH_CUDA
  info["cuda"] = true;
#else
  info["cuda"] = false;
#endif
  info["memory_model"] = "tensoriterator-row-workspace";
  info["thermal_2s"] = true;
  info["thermal_fo"] = true;
  info["solar_2s"] = true;
  info["solar_fo"] = true;
  info["level_fluxes"] = true;
  info["solar_brdf"] = true;
  info["thermal_brdf"] = true;
  info["solar_surface_leaving"] = true;
  info["pyharp_style_module"] = true;
  return info;
}

}  // namespace
}  // namespace py2sess_native

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<py2sess_native::TwoStreamEssNativeOptions>(m, "TwoStreamEssNativeOptions")
      .def(pybind11::init<>())
      .def_readwrite("stream_value", &py2sess_native::TwoStreamEssNativeOptions::stream_value)
      .def_readwrite("user_stream", &py2sess_native::TwoStreamEssNativeOptions::user_stream)
      .def_readwrite("thermal_tcutoff", &py2sess_native::TwoStreamEssNativeOptions::thermal_tcutoff)
      .def_readwrite("x0", &py2sess_native::TwoStreamEssNativeOptions::x0)
      .def_readwrite("user_secant", &py2sess_native::TwoStreamEssNativeOptions::user_secant)
      .def_readwrite("azmfac", &py2sess_native::TwoStreamEssNativeOptions::azmfac)
      .def_readwrite("px11", &py2sess_native::TwoStreamEssNativeOptions::px11)
      .def_readwrite("ulp", &py2sess_native::TwoStreamEssNativeOptions::ulp)
      .def_readwrite("do_upwelling", &py2sess_native::TwoStreamEssNativeOptions::do_upwelling)
      .def_readwrite("do_dnwelling", &py2sess_native::TwoStreamEssNativeOptions::do_dnwelling)
      .def_readwrite("use_brdf", &py2sess_native::TwoStreamEssNativeOptions::use_brdf)
      .def_readwrite(
          "use_surface_leaving", &py2sess_native::TwoStreamEssNativeOptions::use_surface_leaving)
      .def_readwrite("sl_isotropic", &py2sess_native::TwoStreamEssNativeOptions::sl_isotropic)
      .def_readwrite("flip_layers", &py2sess_native::TwoStreamEssNativeOptions::flip_layers)
      .def("__repr__", [](const py2sess_native::TwoStreamEssNativeOptions& options) {
        std::ostringstream ss;
        ss << "TwoStreamEssNativeOptions("
           << "stream_value=" << options.stream_value
           << ", user_stream=" << options.user_stream
           << ", thermal_tcutoff=" << options.thermal_tcutoff
           << ", x0=" << options.x0
           << ", user_secant=" << options.user_secant
           << ", do_upwelling=" << options.do_upwelling
           << ", do_dnwelling=" << options.do_dnwelling
           << ", use_brdf=" << options.use_brdf
           << ", use_surface_leaving=" << options.use_surface_leaving
           << ", flip_layers=" << options.flip_layers
           << ")";
        return ss.str();
      });

  torch::python::bind_module<py2sess_native::TwoStreamEssNativeImpl>(m, "TwoStreamEssNative")
      .def(pybind11::init<>())
      .def(pybind11::init<py2sess_native::TwoStreamEssNativeOptions>(), pybind11::arg("options"))
      .def_readwrite("options", &py2sess_native::TwoStreamEssNativeImpl::options)
      .def(
          "thermal_2s_flux",
          [](py2sess_native::TwoStreamEssNativeImpl& self,
             at::Tensor prop,
             const pybind11::kwargs& kwargs) {
            auto bc = py2sess_native::tensor_kwargs(kwargs);
            return self.thermal_2s_flux(std::move(prop), &bc);
          },
          pybind11::arg("prop"))
      .def(
          "solar_2s_flux",
          [](py2sess_native::TwoStreamEssNativeImpl& self,
             at::Tensor prop,
             const pybind11::kwargs& kwargs) {
            auto bc = py2sess_native::tensor_kwargs(kwargs);
            return self.solar_2s_flux(std::move(prop), &bc);
          },
          pybind11::arg("prop"));

  m.def("backend_info", &py2sess_native::backend_info);
  m.def(
      "thermal_2s",
      &py2sess_native::thermal_2s,
      pybind11::arg("tau"),
      pybind11::arg("omega"),
      pybind11::arg("asymm"),
      pybind11::arg("scaling"),
      pybind11::arg("planck"),
      pybind11::arg("surfbb"),
      pybind11::arg("emissivity"),
      pybind11::arg("albedo"),
      pybind11::arg("stream_value"),
      pybind11::arg("user_stream"),
      pybind11::arg("thermal_tcutoff"),
      pybind11::arg("return_profile") = false);
  m.def(
      "thermal_2s_packed",
      &py2sess_native::thermal_2s_packed,
      pybind11::arg("tau"),
      pybind11::arg("omega"),
      pybind11::arg("asymm"),
      pybind11::arg("scaling"),
      pybind11::arg("planck"),
      pybind11::arg("surfbb"),
      pybind11::arg("emissivity"),
      pybind11::arg("albedo"),
      pybind11::arg("brdf_f"),
      pybind11::arg("ubrdf_f"),
      pybind11::arg("stream_value"),
      pybind11::arg("user_stream"),
      pybind11::arg("thermal_tcutoff"),
      pybind11::arg("return_profile") = false,
      pybind11::arg("return_fluxes") = false,
      pybind11::arg("do_upwelling") = true,
      pybind11::arg("do_dnwelling") = false,
      pybind11::arg("use_brdf") = false);
  m.def(
      "thermal_2s_flux",
      &py2sess_native::thermal_2s_flux,
      pybind11::arg("tau"),
      pybind11::arg("omega"),
      pybind11::arg("asymm"),
      pybind11::arg("scaling"),
      pybind11::arg("planck"),
      pybind11::arg("surfbb"),
      pybind11::arg("emissivity"),
      pybind11::arg("albedo"),
      pybind11::arg("brdf_f"),
      pybind11::arg("ubrdf_f"),
      pybind11::arg("stream_value"),
      pybind11::arg("user_stream"),
      pybind11::arg("thermal_tcutoff"),
      pybind11::arg("do_upwelling") = true,
      pybind11::arg("do_dnwelling") = false,
      pybind11::arg("use_brdf") = false);
  m.def(
      "thermal_fo",
      &py2sess_native::thermal_fo,
      pybind11::arg("tau"),
      pybind11::arg("omega"),
      pybind11::arg("scaling"),
      pybind11::arg("planck"),
      pybind11::arg("surfbb"),
      pybind11::arg("emissivity"),
      pybind11::arg("heights"),
      pybind11::arg("xfine"),
      pybind11::arg("wfine"),
      pybind11::arg("cota"),
      pybind11::arg("cotfine"),
      pybind11::arg("csqfine"),
      pybind11::arg("rayconv"),
      pybind11::arg("do_nadir"),
      pybind11::arg("do_optical_deltam_scaling") = true,
      pybind11::arg("do_source_deltam_scaling") = false,
      pybind11::arg("return_components") = false,
      pybind11::arg("return_profile") = false);
  m.def(
      "thermal_fo_flux_correction",
      &py2sess_native::thermal_fo_flux_correction,
      pybind11::arg("tau"),
      pybind11::arg("omega"),
      pybind11::arg("scaling"),
      pybind11::arg("planck"),
      pybind11::arg("surfbb"),
      pybind11::arg("emissivity"),
      pybind11::arg("mu_nodes"),
      pybind11::arg("mu_weights"),
      pybind11::arg("stream_value"),
      pybind11::arg("do_optical_deltam_scaling") = true,
      pybind11::arg("do_source_deltam_scaling") = false);
  m.def(
      "solar_2s",
      &py2sess_native::solar_2s,
      pybind11::arg("tau"),
      pybind11::arg("omega"),
      pybind11::arg("asymm"),
      pybind11::arg("scaling"),
      pybind11::arg("albedo"),
      pybind11::arg("flux_factor"),
      pybind11::arg("chapman"),
      pybind11::arg("pxsq"),
      pybind11::arg("px0x"),
      pybind11::arg("stream_value"),
      pybind11::arg("x0"),
      pybind11::arg("user_stream"),
      pybind11::arg("user_secant"),
      pybind11::arg("azmfac"),
      pybind11::arg("px11"),
      pybind11::arg("ulp"),
      pybind11::arg("return_profile") = false);
  m.def(
      "solar_fo",
      &py2sess_native::solar_fo,
      pybind11::arg("tau"),
      pybind11::arg("omega"),
      pybind11::arg("scaling"),
      pybind11::arg("albedo"),
      pybind11::arg("flux_factor"),
      pybind11::arg("exact_scatter"),
      pybind11::arg("inv_layer_thickness"),
      pybind11::arg("sunpathsnl"),
      pybind11::arg("cota"),
      pybind11::arg("cotfine"),
      pybind11::arg("csqfine"),
      pybind11::arg("wfine"),
      pybind11::arg("xfine"),
      pybind11::arg("sunpathsfine"),
      pybind11::arg("nfinedivs"),
      pybind11::arg("ntraversefine"),
      pybind11::arg("mu0"),
      pybind11::arg("rayconv"),
      pybind11::arg("ntrav_nl"),
      pybind11::arg("do_nadir"),
      pybind11::arg("return_components") = false,
      pybind11::arg("return_profile") = false);
  m.def(
      "solar_fo_plane_parallel",
      &py2sess_native::solar_fo_plane_parallel,
      pybind11::arg("tau"),
      pybind11::arg("omega"),
      pybind11::arg("scaling"),
      pybind11::arg("surface_reflectance"),
      pybind11::arg("flux_factor"),
      pybind11::arg("exact_scatter"),
      pybind11::arg("mu0"),
      pybind11::arg("user_stream"),
      pybind11::arg("return_profile") = false);
  m.def(
      "solar_fo_flux_correction",
      &py2sess_native::solar_fo_flux_correction,
      pybind11::arg("tau"),
      pybind11::arg("omega"),
      pybind11::arg("scaling"),
      pybind11::arg("surface_reflectance"),
      pybind11::arg("flux_factor"),
      pybind11::arg("stream_value"),
      pybind11::arg("mu0"),
      pybind11::arg("do_optical_deltam_scaling") = true);
  m.def(
      "solar_2s_packed",
      &py2sess_native::solar_2s_packed,
      pybind11::arg("tau"),
      pybind11::arg("omega"),
      pybind11::arg("asymm"),
      pybind11::arg("scaling"),
      pybind11::arg("albedo"),
      pybind11::arg("flux_factor"),
      pybind11::arg("brdf_f0"),
      pybind11::arg("brdf_f"),
      pybind11::arg("ubrdf_f"),
      pybind11::arg("slterm_isotropic"),
      pybind11::arg("slterm_f0"),
      pybind11::arg("chapman"),
      pybind11::arg("pxsq"),
      pybind11::arg("px0x"),
      pybind11::arg("stream_value"),
      pybind11::arg("x0"),
      pybind11::arg("user_stream"),
      pybind11::arg("user_secant"),
      pybind11::arg("azmfac"),
      pybind11::arg("px11"),
      pybind11::arg("ulp"),
      pybind11::arg("return_profile") = false,
      pybind11::arg("return_fluxes") = false,
      pybind11::arg("do_upwelling") = true,
      pybind11::arg("do_dnwelling") = false,
      pybind11::arg("use_brdf") = false,
      pybind11::arg("use_surface_leaving") = false,
      pybind11::arg("sl_isotropic") = true);
  m.def(
      "solar_2s_flux",
      &py2sess_native::solar_2s_flux,
      pybind11::arg("tau"),
      pybind11::arg("omega"),
      pybind11::arg("asymm"),
      pybind11::arg("scaling"),
      pybind11::arg("albedo"),
      pybind11::arg("flux_factor"),
      pybind11::arg("brdf_f0"),
      pybind11::arg("brdf_f"),
      pybind11::arg("ubrdf_f"),
      pybind11::arg("slterm_isotropic"),
      pybind11::arg("slterm_f0"),
      pybind11::arg("chapman"),
      pybind11::arg("pxsq"),
      pybind11::arg("px0x"),
      pybind11::arg("stream_value"),
      pybind11::arg("x0"),
      pybind11::arg("user_stream"),
      pybind11::arg("user_secant"),
      pybind11::arg("azmfac"),
      pybind11::arg("px11"),
      pybind11::arg("ulp"),
      pybind11::arg("do_upwelling") = true,
      pybind11::arg("do_dnwelling") = false,
      pybind11::arg("use_brdf") = false,
      pybind11::arg("use_surface_leaving") = false,
      pybind11::arg("sl_isotropic") = true);
}

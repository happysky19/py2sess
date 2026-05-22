"""Native C++/CUDA backend loader and capability helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any


@lru_cache(maxsize=1)
def _load_native_extension() -> Any | None:
    """Imports the optional compiled native extension when it is available."""
    try:
        from .backend import _load_torch

        if _load_torch() is None:
            return None
        return import_module("py2sess._native")
    except ImportError:
        return None


def native_extension_available() -> bool:
    """Returns whether the optional native extension has been built."""
    return _load_native_extension() is not None


def native_backend_info() -> dict[str, Any]:
    """Returns native backend build information without requiring the extension."""
    extension = _load_native_extension()
    if extension is None:
        return {
            "available": False,
            "backend": "torch-fallback",
            "cuda": False,
        }
    info = dict(extension.backend_info())
    info["available"] = True
    return info


def native_backend_supports_device(device_type: str) -> bool:
    """Returns whether the built native extension can dispatch on a torch device type."""
    extension = _load_native_extension()
    if extension is None:
        return False
    if device_type == "cpu":
        return True
    if device_type == "cuda":
        return bool(dict(extension.backend_info()).get("cuda", False))
    return False


def _require_native_extension() -> Any:
    extension = _load_native_extension()
    if extension is None:
        raise RuntimeError("py2sess._native is not built")
    return extension


def _torch_tensor_helpers(like):
    from .backend import _load_torch

    torch = _load_torch()
    if torch is None:  # pragma: no cover
        raise RuntimeError("backend='native' requires torch to be installed")

    def as_float_tensor(value):
        if torch.is_tensor(value):
            return value.to(dtype=like.dtype, device=like.device)
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)

    def as_int_tensor(value):
        if torch.is_tensor(value):
            return value.to(dtype=torch.int64, device=like.device)
        return torch.as_tensor(value, dtype=torch.int64, device=like.device)

    def scalar_float(value):
        if torch.is_tensor(value):
            return float(value.reshape(-1)[0].detach().cpu().item())
        return float(value)

    return as_float_tensor, as_int_tensor, scalar_float


def _first_panel(value, ndim: int):
    return value[..., 0] if value.ndim == ndim else value


def _zero_if_none(value, like, shape):
    return like.new_zeros(shape) if value is None else value


def _optional_pair_like(value):
    return value.reshape((value.shape[0], 1)).expand((value.shape[0], 2))


def _scalar_float(value) -> float:
    if hasattr(value, "detach"):
        return float(value.reshape(-1)[0].detach().cpu().item())
    return float(value[0] if hasattr(value, "__len__") else value)


def _split_packed_2s(packed, *, nlay: int, return_profile: bool) -> dict[str, Any]:
    """Splits a native packed radiance-plus-flux tensor into named views."""
    nlev = int(nlay) + 1
    radiance_cols = nlev if return_profile else 1
    radiance_block = packed[:, :radiance_cols]
    radiance = radiance_block if return_profile else radiance_block[:, 0]
    offset = radiance_cols
    return {
        "radiance": radiance,
        "flux_up": packed[:, offset : offset + nlev],
        "flux_down": packed[:, offset + nlev : offset + 2 * nlev],
        "flux_net": packed[:, offset + 2 * nlev : offset + 3 * nlev],
        "flux_mean": packed[:, offset + 3 * nlev : offset + 4 * nlev],
    }


def _unpack_fo_profile(
    packed,
    *,
    nlay: int,
    return_components: bool,
    component_names: tuple[str, str],
):
    nlev = int(nlay) + 1
    total_profile = packed[:, :nlev]
    if not return_components:
        return total_profile
    first_profile = packed[:, nlev : 2 * nlev]
    second_profile = packed[:, 2 * nlev : 3 * nlev]
    first_name, second_name = component_names
    return {
        "total": total_profile[:, 0],
        first_name: first_profile[:, 0],
        second_name: second_profile[:, 0],
        "total_profile": total_profile,
        f"{first_name}_profile": first_profile,
        f"{second_name}_profile": second_profile,
    }


def solve_thermal_2s(
    *,
    tau,
    omega,
    asymm,
    scaling,
    planck,
    surfbb,
    emissivity,
    albedo,
    brdf_f=None,
    ubrdf_f=None,
    stream_value: float,
    user_stream: float,
    thermal_tcutoff: float,
    return_profile: bool,
    return_fluxes: bool = False,
    do_upwelling: bool = True,
    do_dnwelling: bool = False,
    use_brdf: bool = False,
):
    """Runs the compiled thermal 2S native kernel.

    Raises
    ------
    RuntimeError
        If the optional native extension has not been built.
    """
    extension = _require_native_extension()
    if return_fluxes or use_brdf:
        if use_brdf:
            brdf_f = _zero_if_none(brdf_f, tau, (tau.shape[0],))
            ubrdf_f = _zero_if_none(ubrdf_f, tau, (tau.shape[0],))
        else:
            brdf_f = albedo
            ubrdf_f = albedo
        packed = extension.thermal_2s_packed(
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
            float(stream_value),
            float(user_stream),
            float(thermal_tcutoff),
            bool(return_profile),
            bool(return_fluxes),
            bool(do_upwelling),
            bool(do_dnwelling),
            bool(use_brdf),
        )
        if return_fluxes:
            return _split_packed_2s(packed, nlay=int(tau.shape[-1]), return_profile=return_profile)
        return packed if return_profile else packed[:, 0]
    return extension.thermal_2s(
        tau,
        omega,
        asymm,
        scaling,
        planck,
        surfbb,
        emissivity,
        albedo,
        float(stream_value),
        float(user_stream),
        float(thermal_tcutoff),
        bool(return_profile),
    )


def solve_thermal_fo(
    *,
    tau,
    omega,
    scaling,
    planck,
    surfbb,
    emissivity,
    heights,
    geometry: dict[str, Any],
    do_optical_deltam_scaling: bool = True,
    do_source_deltam_scaling: bool = False,
    return_components: bool = False,
    return_profile: bool = False,
):
    """Runs the compiled thermal FO endpoint native kernel."""
    extension = _require_native_extension()
    do_nadir = bool(geometry["do_nadir"][0])
    xfine = _first_panel(geometry["xfine"], 3)
    wfine = _first_panel(geometry["wfine"], 3)
    cota = _first_panel(geometry["cota"], 2)
    cotfine = _first_panel(geometry["cotfine"], 3)
    csqfine = _first_panel(geometry["csqfine"], 3)
    rayconv = _scalar_float(geometry.get("raycon", 0.0))
    packed = extension.thermal_fo(
        tau,
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
        csqfine,
        rayconv,
        do_nadir,
        bool(do_optical_deltam_scaling),
        bool(do_source_deltam_scaling),
        bool(return_components),
        bool(return_profile),
    )
    if not return_profile:
        return packed
    return _unpack_fo_profile(
        packed,
        nlay=int(tau.shape[-1]),
        return_components=return_components,
        component_names=("atmosphere", "surface"),
    )


def solve_solar_2s(
    *,
    tau,
    omega,
    asymm,
    scaling,
    albedo,
    flux_factor,
    chapman,
    pxsq,
    px0x,
    brdf_f0=None,
    brdf_f=None,
    ubrdf_f=None,
    slterm_isotropic=None,
    slterm_f0=None,
    stream_value: float,
    x0: float,
    user_stream: float,
    user_secant: float,
    azmfac: float,
    px11: float,
    ulp: float,
    return_profile: bool,
    return_fluxes: bool = False,
    do_upwelling: bool = True,
    do_dnwelling: bool = False,
    use_brdf: bool = False,
    use_surface_leaving: bool = False,
    sl_isotropic: bool = True,
):
    """Runs the compiled solar-observation 2S native kernel."""
    extension = _require_native_extension()
    if return_fluxes or use_brdf or use_surface_leaving:
        row_pair = (tau.shape[0], 2)
        optional_pair = _optional_pair_like(albedo)
        if use_brdf:
            brdf_f0 = _zero_if_none(brdf_f0, tau, row_pair)
            brdf_f = _zero_if_none(brdf_f, tau, row_pair)
            ubrdf_f = _zero_if_none(ubrdf_f, tau, row_pair)
        else:
            brdf_f0 = optional_pair
            brdf_f = optional_pair
            ubrdf_f = optional_pair
        if use_surface_leaving:
            slterm_isotropic = _zero_if_none(slterm_isotropic, tau, (tau.shape[0],))
            slterm_f0 = _zero_if_none(slterm_f0, tau, row_pair)
        else:
            slterm_isotropic = albedo
            slterm_f0 = optional_pair
        packed = extension.solar_2s_packed(
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
            chapman,
            pxsq,
            px0x,
            float(stream_value),
            float(x0),
            float(user_stream),
            float(user_secant),
            float(azmfac),
            float(px11),
            float(ulp),
            bool(return_profile),
            bool(return_fluxes),
            bool(do_upwelling),
            bool(do_dnwelling),
            bool(use_brdf),
            bool(use_surface_leaving),
            bool(sl_isotropic),
        )
        if return_fluxes:
            return _split_packed_2s(packed, nlay=int(tau.shape[-1]), return_profile=return_profile)
        return packed if return_profile else packed[:, 0]
    return extension.solar_2s(
        tau,
        omega,
        asymm,
        scaling,
        albedo,
        flux_factor,
        chapman,
        pxsq,
        px0x,
        float(stream_value),
        float(x0),
        float(user_stream),
        float(user_secant),
        float(azmfac),
        float(px11),
        float(ulp),
        bool(return_profile),
    )


def solve_solar_fo(
    *,
    tau,
    omega,
    scaling,
    albedo,
    flux_factor,
    exact_scatter,
    precomputed: Any,
    direct_surface_reflectance=None,
    return_components: bool = False,
    return_profile: bool = False,
):
    """Runs the compiled solar-observation FO endpoint native kernel."""
    extension = _require_native_extension()
    as_float_tensor, as_int_tensor, scalar_float = _torch_tensor_helpers(tau)

    xfine = _first_panel(as_float_tensor(precomputed.xfine), 3)
    wfine = _first_panel(as_float_tensor(precomputed.wfine), 3)
    cota = _first_panel(as_float_tensor(precomputed.cota), 2)
    cotfine = _first_panel(as_float_tensor(precomputed.cotfine), 3)
    csqfine = _first_panel(as_float_tensor(precomputed.csqfine), 3)
    sunpathsfine = _first_panel(as_float_tensor(precomputed.sunpathsfine), 4)
    ntraversefine = _first_panel(as_int_tensor(precomputed.ntraversefine), 3)
    nfinedivs = _first_panel(as_int_tensor(precomputed.nfinedivs), 2)
    surface_reflectance = (
        albedo if direct_surface_reflectance is None else direct_surface_reflectance
    )

    packed = extension.solar_fo(
        tau,
        omega,
        scaling,
        as_float_tensor(surface_reflectance),
        flux_factor,
        exact_scatter,
        as_float_tensor(precomputed.inv_layer_thickness),
        as_float_tensor(precomputed.sunpathsnl),
        cota,
        cotfine,
        csqfine,
        wfine,
        xfine,
        sunpathsfine,
        nfinedivs,
        ntraversefine,
        scalar_float(precomputed.mu0),
        scalar_float(precomputed.rayconv),
        int(precomputed.ntrav_nl),
        bool(precomputed.do_nadir),
        bool(return_components),
        bool(return_profile),
    )
    if not return_profile:
        return packed
    return _unpack_fo_profile(
        packed,
        nlay=int(tau.shape[-1]),
        return_components=return_components,
        component_names=("single_scatter", "direct_beam"),
    )

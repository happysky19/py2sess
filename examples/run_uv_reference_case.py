"""Run the packaged UV 2S benchmark fixture with NumPy and optional torch."""

from __future__ import annotations

import numpy as np

from py2sess.reference_cases import load_uv_benchmark_case
from py2sess.core.backend import has_torch, to_numpy
from py2sess.core.solar_obs_batch_numpy import solve_solar_obs_batch_numpy


def _relative_diff(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Returns a stable elementwise relative difference."""
    scale = np.maximum(np.abs(reference), 1.0e-15)
    return np.abs(value - reference) / scale


def _print_summary(label: str, two_stream: np.ndarray, reference) -> None:
    """Prints compact max-difference summaries."""
    print(label)
    print("  note: this packaged UV benchmark targets the saved 2S radiance")
    print(
        "  2S   max_abs={:.6e} max_rel={:.6e}".format(
            np.max(np.abs(two_stream - reference.ref_2s)),
            np.max(_relative_diff(two_stream, reference.ref_2s)),
        )
    )


def main() -> None:
    """Runs the packaged UV fixture."""
    case = load_uv_benchmark_case()

    numpy_2s = solve_solar_obs_batch_numpy(
        tau=case.tau,
        omega=case.omega,
        asymm=case.asymm,
        scaling=case.scaling,
        albedo=case.albedo,
        flux_factor=case.flux_factor,
        stream_value=case.stream_value,
        chapman=case.chapman,
        x0=case.x0,
        user_stream=case.user_stream,
        user_secant=case.user_secant,
        azmfac=case.azmfac,
        px11=case.px11,
        pxsq=case.pxsq,
        px0x=case.px0x,
        ulp=case.ulp,
    )
    _print_summary("numpy", numpy_2s, case)

    if has_torch():
        from py2sess.core.solar_obs_batch_torch import solve_solar_obs_batch_torch

        torch_2s = to_numpy(
            solve_solar_obs_batch_torch(
                tau=case.tau,
                omega=case.omega,
                asymm=case.asymm,
                scaling=case.scaling,
                albedo=case.albedo,
                flux_factor=case.flux_factor,
                stream_value=case.stream_value,
                chapman=case.chapman,
                x0=case.x0,
                user_stream=case.user_stream,
                user_secant=case.user_secant,
                azmfac=case.azmfac,
                px11=case.px11,
                pxsq=case.pxsq,
                px0x=case.px0x,
                ulp=case.ulp,
                device="cpu",
            )
        )
        _print_summary("torch cpu", torch_2s, case)
        print(f"torch-vs-numpy 2S max_abs={np.max(np.abs(torch_2s - numpy_2s)):.6e}")


if __name__ == "__main__":
    main()

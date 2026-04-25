"""Run the packaged TIR benchmark fixture with NumPy and optional torch."""

from __future__ import annotations

import numpy as np

from py2sess import TwoStreamEss, TwoStreamEssOptions
from py2sess.reference_cases import load_tir_benchmark_case
from py2sess.rtsolver.backend import has_torch, to_numpy


def _relative_diff(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Returns a stable elementwise relative difference."""
    scale = np.maximum(np.abs(reference), 1.0e-15)
    return np.abs(value - reference) / scale


def _print_summary(
    label: str, two_stream: np.ndarray, fo: np.ndarray, total: np.ndarray, reference
) -> None:
    """Prints compact max-difference summaries."""
    print(label)
    print("  note: this packaged TIR fixture uses stream=0.5")
    print(
        "  2S   max_abs={:.6e} max_rel={:.6e}".format(
            np.max(np.abs(two_stream - reference.ref_2s)),
            np.max(_relative_diff(two_stream, reference.ref_2s)),
        )
    )
    print(
        "  FO   max_abs={:.6e} max_rel={:.6e}".format(
            np.max(np.abs(fo - reference.ref_fo)),
            np.max(_relative_diff(fo, reference.ref_fo)),
        )
    )
    print(
        "  total max_abs={:.6e} max_rel={:.6e}".format(
            np.max(np.abs(total - reference.ref_total)),
            np.max(_relative_diff(total, reference.ref_total)),
        )
    )


def main() -> None:
    """Runs the packaged TIR fixture."""
    case = load_tir_benchmark_case()

    solver = TwoStreamEss(TwoStreamEssOptions(nlyr=case.tau_arr.shape[1], mode="thermal"))
    numpy_result = solver.forward(
        tau=case.tau_arr,
        ssa=case.omega_arr,
        g=case.asymm_arr,
        z=case.heights,
        angles=case.user_angle,
        stream=case.stream_value,
        albedo=case.albedo,
        delta_m_truncation_factor=case.d2s_scaling,
        planck=case.thermal_bb_input,
        surface_planck=case.surfbb,
        emissivity=case.emissivity,
        include_fo=True,
    )
    numpy_2s = numpy_result.radiance_2s
    numpy_fo = numpy_result.radiance_fo
    numpy_total = numpy_result.radiance_total
    _print_summary("numpy", numpy_2s, numpy_fo, numpy_total, case)

    if has_torch():
        torch_solver = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=case.tau_arr.shape[1],
                mode="thermal",
                backend="torch",
                torch_device="cpu",
                torch_dtype="float64",
                torch_enable_grad=False,
            )
        )
        torch_result = torch_solver.forward(
            tau=case.tau_arr,
            ssa=case.omega_arr,
            g=case.asymm_arr,
            z=case.heights,
            angles=case.user_angle,
            stream=case.stream_value,
            albedo=case.albedo,
            delta_m_truncation_factor=case.d2s_scaling,
            planck=case.thermal_bb_input,
            surface_planck=case.surfbb,
            emissivity=case.emissivity,
            include_fo=True,
        )
        torch_2s = to_numpy(torch_result.radiance_2s)
        torch_fo = to_numpy(torch_result.radiance_fo)
        torch_total = to_numpy(torch_result.radiance_total)
        _print_summary("torch cpu", torch_2s, torch_fo, torch_total, case)
        print(
            "torch-vs-numpy component parity"
            f"  2S max_abs={np.max(np.abs(torch_2s - numpy_2s)):.6e}"
            f"  FO max_abs={np.max(np.abs(torch_fo - numpy_fo)):.6e}"
            f"  total max_abs={np.max(np.abs(torch_total - numpy_total)):.6e}"
        )


if __name__ == "__main__":
    main()

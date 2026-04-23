"""Run the packaged TIR benchmark fixture with NumPy and optional torch."""

from __future__ import annotations

import numpy as np

from py2sess.reference_cases import load_tir_benchmark_case
from py2sess.core.backend import has_torch, to_numpy
from py2sess.core.thermal_batch_numpy import solve_thermal_batch_numpy


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

    numpy_result = solve_thermal_batch_numpy(
        tau_arr=case.tau_arr,
        omega_arr=case.omega_arr,
        asymm_arr=case.asymm_arr,
        d2s_scaling=case.d2s_scaling,
        thermal_bb_input=case.thermal_bb_input,
        surfbb=case.surfbb,
        albedo=case.albedo,
        emissivity=case.emissivity,
        heights=case.heights,
        user_angle_degrees=case.user_angle,
        stream_value=case.stream_value,
    )
    numpy_2s = numpy_result.two_stream_toa
    numpy_fo = numpy_result.fo_total_up_toa
    _print_summary("numpy", numpy_2s, numpy_fo, numpy_2s + numpy_fo, case)

    if has_torch():
        from py2sess.core.thermal_batch_torch import solve_thermal_batch_torch

        torch_result = solve_thermal_batch_torch(
            tau_arr=case.tau_arr,
            omega_arr=case.omega_arr,
            asymm_arr=case.asymm_arr,
            d2s_scaling=case.d2s_scaling,
            thermal_bb_input=case.thermal_bb_input,
            surfbb=case.surfbb,
            albedo=case.albedo,
            emissivity=case.emissivity,
            heights=case.heights,
            user_angle_degrees=case.user_angle,
            stream_value=case.stream_value,
            device="cpu",
        )
        torch_2s = to_numpy(torch_result.two_stream_toa)
        torch_fo = to_numpy(torch_result.fo_total_up_toa)
        _print_summary("torch cpu", torch_2s, torch_fo, torch_2s + torch_fo, case)
        print(
            "torch-vs-numpy component parity"
            f"  2S max_abs={np.max(np.abs(torch_2s - numpy_2s)):.6e}"
            f"  FO max_abs={np.max(np.abs(torch_fo - numpy_fo)):.6e}"
        )


if __name__ == "__main__":
    main()

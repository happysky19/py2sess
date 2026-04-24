"""Print the analytic benchmark cases bundled with the test suite."""

from __future__ import annotations

import math
import os
import sys

import numpy as np

from py2sess.analytic_oracles import (
    lambertian_surface_fo_radiance,
    solar_fo_single_scatter_isotropic_one_layer,
    thermal_atmosphere_only_down_profile,
    thermal_atmosphere_only_up_profile,
    thermal_fo_single_layer_uniform_source,
    thermal_surface_only_up_profile,
    twostream_upward_flux_pair_from_isotropic_intensity,
)
from py2sess import TwoStreamEss, TwoStreamEssOptions


_ANSI_RESET = "\033[0m"
_ANSI_RED = "\033[31m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_CYAN = "\033[36m"


def _supports_color() -> bool:
    """Returns whether ANSI colors should be emitted."""
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _colorize(text: str, color: str) -> str:
    """Wraps a string in ANSI color codes when supported."""
    if not _supports_color():
        return text
    return f"{color}{text}{_ANSI_RESET}"


def _color_status(status: str) -> str:
    """Returns a colored status label when the terminal supports it."""
    if status == "PASS":
        return _colorize(status, _ANSI_GREEN)
    if status == "FAIL":
        return _colorize(status, _ANSI_RED)
    return _colorize(status, _ANSI_YELLOW)


def _print_section(title: str, description: str) -> None:
    """Prints a section header for a group of analytic checks."""
    print()
    print(_colorize(title, _ANSI_CYAN))
    print("-" * 72)
    print(description)
    print()


def _print_section_summary(*, passed: int, total: int) -> None:
    """Prints the pass count for one section."""
    section_status = "PASS" if passed == total else "FAIL"
    print(f"  section summary: {passed}/{total} {_color_status(section_status)}")


def _print_scalar_case(
    name: str, *, actual: float, expected: float, tolerance: float, note: str | None = None
) -> bool:
    """Prints one scalar analytic comparison as a readable block."""
    abs_diff = abs(actual - expected)
    status = "PASS" if abs_diff <= tolerance else "FAIL"
    print(f"{name}")
    if note is not None:
        print(f"  note:       {note}")
    print(f"  actual:     {actual:.12e}")
    print(f"  expected:   {expected:.12e}")
    print(f"  abs diff:   {abs_diff:.12e}")
    print(f"  tolerance:  {tolerance:.12e}")
    print(f"  status:     {_color_status(status)}")
    print()
    return status == "PASS"


def _print_zero_case(name: str, *, value: float, tolerance: float, note: str | None = None) -> bool:
    """Prints one zero-identity analytic comparison as a readable block."""
    status = "PASS" if value <= tolerance else "FAIL"
    print(f"{name}")
    if note is not None:
        print(f"  note:       {note}")
    print(f"  max abs:    {value:.12e}")
    print(f"  tolerance:  {tolerance:.12e}")
    print(f"  status:     {_color_status(status)}")
    print()
    return status == "PASS"


def _print_profile_case(
    name: str,
    *,
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: float,
    note: str | None = None,
) -> bool:
    """Prints one profile analytic comparison as a readable block."""
    abs_diff = np.abs(actual - expected)
    status = "PASS" if float(np.max(abs_diff)) <= tolerance else "FAIL"
    print(f"{name}")
    if note is not None:
        print(f"  note:       {note}")
    print(f"  actual:     {np.array2string(actual, precision=10, separator=', ')}")
    print(f"  expected:   {np.array2string(expected, precision=10, separator=', ')}")
    print(f"  max diff:   {float(np.max(abs_diff)):.12e}")
    print(f"  tolerance:  {tolerance:.12e}")
    print(f"  status:     {_color_status(status)}")
    print()
    return status == "PASS"


def main() -> None:
    """Runs the analytic benchmark cases and prints actual versus expected values."""
    stream = 1.0 / math.sqrt(3.0)
    total_cases = 0
    total_passed = 0

    solar_solver = TwoStreamEss(TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True))
    thermal_solver = TwoStreamEss(
        TwoStreamEssOptions(nlyr=3, mode="thermal", output_levels=True, downwelling=True)
    )
    thermal_solver_flux = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=3,
            mode="thermal",
            output_levels=True,
            downwelling=True,
            additional_mvout=True,
        )
    )
    thermal_solver_irregular = TwoStreamEss(
        TwoStreamEssOptions(nlyr=5, mode="thermal", output_levels=True, downwelling=True)
    )

    solar_surface = solar_solver.forward_fo(
        tau=np.zeros(3, dtype=float),
        ssa=np.zeros(3, dtype=float),
        g=np.zeros(3, dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array([[30.0, 0.0, 0.0]], dtype=float),
        stream=stream,
        fbeam=1.0,
        albedo=0.3,
        delta_m_scaling=np.zeros(3, dtype=float),
        geometry="pseudo_spherical",
    )
    solar_surface_expected = lambertian_surface_fo_radiance(
        flux_factor=1.0,
        albedo=0.3,
        solar_zenith_degrees=30.0,
    )

    solar_oblique = solar_solver.forward_fo(
        tau=np.zeros(3, dtype=float),
        ssa=np.zeros(3, dtype=float),
        g=np.zeros(3, dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array([[35.0, 50.0, 120.0]], dtype=float),
        stream=stream,
        fbeam=2.5,
        albedo=0.4,
        delta_m_scaling=np.zeros(3, dtype=float),
        geometry="pseudo_spherical",
    )
    solar_oblique_expected = lambertian_surface_fo_radiance(
        flux_factor=2.5,
        albedo=0.4,
        solar_zenith_degrees=35.0,
    )

    solar_view_independent = solar_solver.forward_fo(
        tau=np.zeros(3, dtype=float),
        ssa=np.zeros(3, dtype=float),
        g=np.zeros(3, dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array(
            [
                [35.0, 0.0, 0.0],
                [35.0, 50.0, 120.0],
            ],
            dtype=float,
        ),
        stream=stream,
        fbeam=2.5,
        albedo=0.4,
        delta_m_scaling=np.zeros(3, dtype=float),
        geometry="pseudo_spherical",
    )
    solar_view_independent_expected = np.full(2, solar_oblique_expected, dtype=float)

    solar_no_surface = solar_solver.forward_fo(
        tau=np.zeros(3, dtype=float),
        ssa=np.zeros(3, dtype=float),
        g=np.zeros(3, dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array([[35.0, 50.0, 120.0]], dtype=float),
        stream=stream,
        fbeam=2.5,
        albedo=0.0,
        delta_m_scaling=np.zeros(3, dtype=float),
        geometry="pseudo_spherical",
    )

    solar_zero_flux = TwoStreamEss(
        TwoStreamEssOptions(nlyr=3, mode="solar", output_levels=True, downwelling=True)
    ).forward(
        tau=np.array([0.2, 0.3, 0.1], dtype=float),
        ssa=np.array([0.5, 0.4, 0.3], dtype=float),
        g=np.array([0.2, 0.1, 0.3], dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array([[30.0, 20.0, 40.0]], dtype=float),
        stream=stream,
        fbeam=0.0,
        albedo=0.3,
        delta_m_scaling=np.array([0.01, 0.02, 0.03], dtype=float),
        include_fo=True,
    )

    thermal_blackbody = thermal_solver.forward_fo(
        tau=np.array([0.2, 0.3, 0.1], dtype=float),
        ssa=np.zeros(3, dtype=float),
        g=np.zeros(3, dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array([20.0], dtype=float),
        stream=stream,
        fbeam=1.0,
        albedo=0.0,
        delta_m_scaling=np.zeros(3, dtype=float),
        planck=np.ones(4, dtype=float),
        surface_planck=1.0,
        emissivity=1.0,
    )
    thermal_blackbody_flux_expected = twostream_upward_flux_pair_from_isotropic_intensity(
        intensity=1.0,
        stream_value=stream,
    )
    thermal_blackbody_full = thermal_solver_flux.forward(
        tau=np.array([0.2, 0.3, 0.1], dtype=float),
        ssa=np.zeros(3, dtype=float),
        g=np.zeros(3, dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array([20.0], dtype=float),
        stream=stream,
        fbeam=1.0,
        albedo=0.0,
        delta_m_scaling=np.zeros(3, dtype=float),
        planck=np.ones(4, dtype=float),
        surface_planck=1.0,
        emissivity=1.0,
        include_fo=True,
    )

    thermal_surface_only = thermal_solver.forward_fo(
        tau=np.array([0.2, 0.3, 0.1], dtype=float),
        ssa=np.zeros(3, dtype=float),
        g=np.zeros(3, dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array([20.0], dtype=float),
        stream=stream,
        fbeam=1.0,
        albedo=0.2,
        delta_m_scaling=np.zeros(3, dtype=float),
        planck=np.zeros(4, dtype=float),
        surface_planck=5.0,
        emissivity=0.8,
    )
    thermal_surface_only_expected = thermal_surface_only_up_profile(
        np.array([0.2, 0.3, 0.1], dtype=float),
        user_angle_degrees=20.0,
        surfbb=5.0,
        emissivity=0.8,
    )

    thermal_atmosphere_only = thermal_solver.forward_fo(
        tau=np.array([0.2, 0.3, 0.1], dtype=float),
        ssa=np.zeros(3, dtype=float),
        g=np.zeros(3, dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array([20.0], dtype=float),
        stream=stream,
        fbeam=1.0,
        albedo=0.0,
        delta_m_scaling=np.zeros(3, dtype=float),
        planck=np.ones(4, dtype=float),
        surface_planck=0.0,
        emissivity=1.0,
    )
    thermal_atmosphere_only_expected = thermal_atmosphere_only_up_profile(
        np.array([0.2, 0.3, 0.1], dtype=float),
        user_angle_degrees=20.0,
        blackbody_value=1.0,
    )
    thermal_atmosphere_only_down_expected = thermal_atmosphere_only_down_profile(
        np.array([0.2, 0.3, 0.1], dtype=float),
        user_angle_degrees=20.0,
        blackbody_value=1.0,
    )

    thermal_zero = thermal_solver.forward(
        tau=np.array([0.2, 0.3, 0.1], dtype=float),
        ssa=np.zeros(3, dtype=float),
        g=np.zeros(3, dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array([20.0], dtype=float),
        stream=stream,
        fbeam=1.0,
        albedo=0.0,
        delta_m_scaling=np.zeros(3, dtype=float),
        planck=np.zeros(4, dtype=float),
        surface_planck=0.0,
        emissivity=1.0,
        include_fo=True,
    )

    thermal_zero_nonblack = thermal_solver.forward(
        tau=np.array([0.2, 0.3, 0.1], dtype=float),
        ssa=np.zeros(3, dtype=float),
        g=np.zeros(3, dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array([20.0], dtype=float),
        stream=stream,
        fbeam=1.0,
        albedo=0.5,
        delta_m_scaling=np.zeros(3, dtype=float),
        planck=np.zeros(4, dtype=float),
        surface_planck=0.0,
        emissivity=0.5,
        include_fo=True,
    )

    thermal_super_surface = thermal_solver.forward_fo(
        tau=np.array([0.2, 0.3, 0.1], dtype=float),
        ssa=np.zeros(3, dtype=float),
        g=np.zeros(3, dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array([20.0], dtype=float),
        stream=stream,
        fbeam=1.0,
        albedo=0.2,
        delta_m_scaling=np.zeros(3, dtype=float),
        planck=np.zeros(4, dtype=float),
        surface_planck=5.0,
        emissivity=0.8,
    )
    thermal_super_atmos = thermal_solver.forward_fo(
        tau=np.array([0.2, 0.3, 0.1], dtype=float),
        ssa=np.zeros(3, dtype=float),
        g=np.zeros(3, dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array([20.0], dtype=float),
        stream=stream,
        fbeam=1.0,
        albedo=0.0,
        delta_m_scaling=np.zeros(3, dtype=float),
        planck=np.full(4, 1.5, dtype=float),
        surface_planck=0.0,
        emissivity=1.0,
    )
    thermal_super_total = thermal_solver.forward_fo(
        tau=np.array([0.2, 0.3, 0.1], dtype=float),
        ssa=np.zeros(3, dtype=float),
        g=np.zeros(3, dtype=float),
        z=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
        angles=np.array([20.0], dtype=float),
        stream=stream,
        fbeam=1.0,
        albedo=0.2,
        delta_m_scaling=np.zeros(3, dtype=float),
        planck=np.full(4, 1.5, dtype=float),
        surface_planck=5.0,
        emissivity=0.8,
    )

    solar_scattering = TwoStreamEss(
        TwoStreamEssOptions(nlyr=1, mode="solar", output_levels=True, plane_parallel=True)
    ).forward_fo(
        tau=np.array([0.4], dtype=float),
        ssa=np.array([0.6], dtype=float),
        g=np.array([0.0], dtype=float),
        z=np.array([1.0, 0.0], dtype=float),
        angles=np.array([[30.0, 20.0, 0.0]], dtype=float),
        stream=stream,
        fbeam=1.0,
        albedo=0.0,
        delta_m_scaling=np.zeros(1, dtype=float),
        geometry="pseudo_spherical",
    )
    solar_scattering_expected = solar_fo_single_scatter_isotropic_one_layer(
        tau=0.4,
        omega=0.6,
        solar_zenith_degrees=30.0,
        view_zenith_degrees=20.0,
        flux_factor=1.0,
    )

    thermal_scattering = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=1,
            mode="thermal",
            output_levels=True,
            downwelling=True,
            plane_parallel=True,
        )
    ).forward_fo(
        tau=np.array([0.4], dtype=float),
        ssa=np.array([0.3], dtype=float),
        g=np.array([0.0], dtype=float),
        z=np.array([1.0, 0.0], dtype=float),
        angles=np.array([20.0], dtype=float),
        stream=stream,
        fbeam=1.0,
        albedo=0.0,
        delta_m_scaling=np.zeros(1, dtype=float),
        planck=np.array([2.0, 2.0], dtype=float),
        surface_planck=0.0,
        emissivity=1.0,
    )
    thermal_scattering_expected = thermal_fo_single_layer_uniform_source(
        tau=0.4,
        omega=0.3,
        user_angle_degrees=20.0,
        blackbody_value=2.0,
    )

    thermal_surface_irregular = thermal_solver_irregular.forward_fo(
        tau=np.array([0.05, 0.12, 0.27, 0.08, 0.19], dtype=float),
        ssa=np.zeros(5, dtype=float),
        g=np.zeros(5, dtype=float),
        z=np.arange(5, -1, -1, dtype=float),
        angles=np.array([37.0], dtype=float),
        stream=stream,
        fbeam=1.0,
        albedo=0.09,
        delta_m_scaling=np.zeros(5, dtype=float),
        planck=np.zeros(6, dtype=float),
        surface_planck=2.7,
        emissivity=0.91,
    )
    thermal_surface_irregular_expected = thermal_surface_only_up_profile(
        np.array([0.05, 0.12, 0.27, 0.08, 0.19], dtype=float),
        user_angle_degrees=37.0,
        surfbb=2.7,
        emissivity=0.91,
    )

    print("analytic benchmarks")
    print("=" * 72)
    print("Grouped by physics purpose so failures are easier to diagnose.")

    _print_section(
        "1. Boundary Conditions",
        "Zero-source and zero-boundary checks that verify the solver does not create energy numerically.",
    )
    section_total = 0
    section_passed = 0
    section_passed += int(
        _print_zero_case(
            "solar_fo_no_scattering_no_surface",
            value=float(np.max(np.abs(solar_no_surface.radiance))),
            tolerance=1.0e-14,
            note="Without scattering and without a reflecting surface, FO upwelling solar radiance is zero.",
        )
    )
    section_total += 1
    section_passed += int(
        _print_zero_case(
            "solar_forward_zero_flux",
            value=float(
                max(
                    np.max(np.abs(solar_zero_flux.radiance_2s)),
                    np.max(np.abs(solar_zero_flux.radiance_total)),
                )
            ),
            tolerance=1.0e-14,
            note="Full solver should return zero everywhere when solar flux is zero.",
        )
    )
    section_total += 1
    section_passed += int(
        _print_zero_case(
            "thermal_forward_zero_sources",
            value=float(
                max(
                    np.max(np.abs(thermal_zero.radiance_2s)),
                    np.max(np.abs(thermal_zero.radiance_fo)),
                )
            ),
            tolerance=1.0e-14,
            note="Full thermal solver should remain zero when all thermal sources are zero.",
        )
    )
    section_total += 1
    section_passed += int(
        _print_zero_case(
            "thermal_forward_zero_sources_nonblack_surface",
            value=float(
                max(
                    np.max(np.abs(thermal_zero_nonblack.radiance_2s)),
                    np.max(np.abs(thermal_zero_nonblack.radiance_fo)),
                )
            ),
            tolerance=1.0e-14,
            note="Zero thermal sources should still give zero even when the surface is not black.",
        )
    )
    section_total += 1
    _print_section_summary(passed=section_passed, total=section_total)
    total_passed += section_passed
    total_cases += section_total

    _print_section(
        "2. Clear-Sky Shortwave",
        "Surface-only shortwave cases with closed-form Lambertian answers and direct-view invariants.",
    )
    section_total = 0
    section_passed = 0
    section_passed += int(
        _print_scalar_case(
            "solar_fo_surface_only",
            actual=float(solar_surface.radiance[0]),
            expected=solar_surface_expected,
            tolerance=1.0e-12,
            note="Lambertian surface-only FO solution.",
        )
    )
    section_total += 1
    section_passed += int(
        _print_scalar_case(
            "solar_fo_oblique_surface_only",
            actual=float(solar_oblique.radiance[0]),
            expected=solar_oblique_expected,
            tolerance=1.0e-12,
            note="Same closed-form surface-only FO solution at oblique view and non-unit flux.",
        )
    )
    section_total += 1
    section_passed += int(
        _print_profile_case(
            "solar_fo_lambertian_view_independent",
            actual=solar_view_independent.radiance,
            expected=solar_view_independent_expected,
            tolerance=1.0e-12,
            note="Lambertian surface-only FO solution should not depend on viewing angle.",
        )
    )
    section_total += 1
    _print_section_summary(passed=section_passed, total=section_total)
    total_passed += section_passed
    total_cases += section_total

    _print_section(
        "3. Clear-Sky Longwave",
        "Pure-absorption longwave identities for surface emission, atmospheric emission, and blackbody preservation.",
    )
    section_total = 0
    section_passed = 0
    section_passed += int(
        _print_scalar_case(
            "thermal_fo_isothermal_absorbing_column",
            actual=float(thermal_blackbody.radiance_up_toa[0]),
            expected=1.0,
            tolerance=5.0e-10,
            note="FO thermal solution for an isothermal pure-absorption column with black surface.",
        )
    )
    section_total += 1
    section_passed += int(
        _print_profile_case(
            "thermal_full_isothermal_blackbody_toa_flux",
            actual=thermal_blackbody_full.fluxes_toa[:, 0],
            expected=thermal_blackbody_flux_expected,
            tolerance=2.0e-9,
            note="Full thermal solver TOA flux pair for the isothermal blackbody case.",
        )
    )
    section_total += 1
    section_passed += int(
        _print_profile_case(
            "thermal_fo_surface_only_profile",
            actual=thermal_surface_only.radiance_up_profile[0],
            expected=thermal_surface_only_expected,
            tolerance=8.0e-5,
            note="Closed-form surface emission attenuated by pure absorption.",
        )
    )
    section_total += 1
    section_passed += int(
        _print_profile_case(
            "thermal_fo_atmosphere_only_profile",
            actual=thermal_atmosphere_only.radiance_up_profile[0],
            expected=thermal_atmosphere_only_expected,
            tolerance=2.0e-5,
            note="Closed-form atmospheric emission through a pure-absorption column.",
        )
    )
    section_total += 1
    section_passed += int(
        _print_profile_case(
            "thermal_fo_atmosphere_only_down_profile",
            actual=thermal_atmosphere_only.intensity_atmos_dn_profile[0],
            expected=thermal_atmosphere_only_down_expected,
            tolerance=2.0e-5,
            note="Closed-form downward atmospheric emission through a pure-absorption column.",
        )
    )
    section_total += 1
    _print_section_summary(passed=section_passed, total=section_total)
    total_passed += section_passed
    total_cases += section_total

    _print_section(
        "4. Scattering Cases",
        "One-layer scattering checks with exact FO solutions in shortwave and longwave mode.",
    )
    section_total = 0
    section_passed = 0
    section_passed += int(
        _print_scalar_case(
            "solar_fo_one_layer_isotropic_scattering",
            actual=float(solar_scattering.radiance[0]),
            expected=solar_scattering_expected,
            tolerance=1.0e-12,
            note="One-layer plane-parallel isotropic single-scatter solar FO solution over a black surface.",
        )
    )
    section_total += 1
    section_passed += int(
        _print_scalar_case(
            "thermal_fo_one_layer_uniform_scattering",
            actual=float(thermal_scattering.radiance_up_toa[0]),
            expected=thermal_scattering_expected,
            tolerance=1.0e-12,
            note="One-layer plane-parallel thermal FO solution with uniform source and nonzero single-scattering albedo.",
        )
    )
    section_total += 1
    _print_section_summary(passed=section_passed, total=section_total)
    total_passed += section_passed
    total_cases += section_total

    _print_section(
        "5. Composite Exact Cases",
        "Multi-effect identities where the exact answer comes from superposition or an irregular but still analytic profile.",
    )
    section_total = 0
    section_passed = 0
    section_passed += int(
        _print_profile_case(
            "thermal_fo_superposition_surface_plus_atmosphere",
            actual=thermal_super_total.radiance_up_profile[0],
            expected=(
                thermal_super_surface.radiance_up_profile + thermal_super_atmos.radiance_up_profile
            )[0],
            tolerance=2.0e-10,
            note="Thermal FO total should equal the sum of independent surface-only and atmosphere-only runs.",
        )
    )
    section_total += 1
    section_passed += int(
        _print_profile_case(
            "thermal_fo_surface_only_irregular_profile",
            actual=thermal_surface_irregular.radiance_up_profile[0],
            expected=thermal_surface_irregular_expected,
            tolerance=2.0e-4,
            note="Pure-absorption surface-only thermal solution on an irregular multilayer profile.",
        )
    )
    section_total += 1
    _print_section_summary(passed=section_passed, total=section_total)
    total_passed += section_passed
    total_cases += section_total

    print()
    print("=" * 72)
    print(
        f"overall summary: {total_passed}/{total_cases} {_color_status('PASS' if total_passed == total_cases else 'FAIL')}"
    )


if __name__ == "__main__":
    main()

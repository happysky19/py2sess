"""Shared orchestration helpers for NumPy and torch solvers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


BoundaryTerms = dict[str, Any]


def solar_problem_size(prepared) -> tuple[int, int]:
    """Returns the geometry and layer count for a solar solve.

    Parameters
    ----------
    prepared
        Preprocessed solver inputs.

    Returns
    -------
    tuple of int
        Number of solar geometries and number of layers.
    """
    return int(prepared.user_obsgeoms.shape[0]), int(prepared.tau_arr.size)


def thermal_problem_size(prepared) -> tuple[int, int]:
    """Returns the user-angle and layer count for a thermal solve.

    Parameters
    ----------
    prepared
        Preprocessed solver inputs.

    Returns
    -------
    tuple of int
        Number of user angles and number of layers.
    """
    return int(prepared.geometry.user_streams.size), int(prepared.tau_arr.size)


def include_surface_term(fourier: int, *, albedo: float, do_brdf_surface: bool) -> bool:
    """Returns whether a solar surface contribution is active.

    Parameters
    ----------
    fourier
        Current Fourier index.
    albedo
        Lambertian surface albedo.
    do_brdf_surface
        Whether explicit BRDF coefficients are enabled.

    Returns
    -------
    bool
        ``True`` when either the zeroth-order Lambertian term or any BRDF term
        should contribute to the solve.
    """
    return (fourier == 0 and albedo != 0.0) or do_brdf_surface


def include_thermal_surface_term(*, albedo: float, do_brdf_surface: bool) -> bool:
    """Returns whether a thermal surface contribution is active.

    Parameters
    ----------
    albedo
        Lambertian surface albedo.
    do_brdf_surface
        Whether explicit BRDF coefficients are enabled.

    Returns
    -------
    bool
        ``True`` when the thermal boundary problem should include a surface
        reflection term.
    """
    return do_brdf_surface or albedo != 0.0


def accumulate_fourier_term(current, contribution, *, fourier: int, azmfac):
    """Accumulates one Fourier contribution into an output quantity.

    Parameters
    ----------
    current
        Current accumulated value.
    contribution
        Fourier contribution for the current mode.
    fourier
        Current Fourier index.
    azmfac
        Azimuthal weighting factor applied to nonzero Fourier terms.

    Returns
    -------
    Any
        Updated accumulated value with the current Fourier contribution applied.
    """
    if fourier == 0:
        return contribution
    return current + azmfac * contribution


def prepare_thermal_postprocessing(
    *,
    do_postprocessing,
    delta_tau,
    user_secants,
    n_users,
    nlay,
    build_user_solution,
    build_hmult,
    make_zero_array,
    exp_outer,
):
    """Builds thermal postprocessing arrays or matching zero placeholders.

    Parameters
    ----------
    do_postprocessing
        Whether user-angle postprocessing is active.
    delta_tau
        Layer optical thickness array.
    user_secants
        Secants of the user viewing angles.
    n_users
        Number of user angles.
    nlay
        Number of layers.
    build_user_solution
        Callback that returns the user-angle homogeneous solution.
    build_hmult
        Callback that returns thermal homogeneous multipliers.
    make_zero_array
        Backend-specific zero-array factory.
    exp_outer
        Backend-specific exponential outer-product helper.

    Returns
    -------
    tuple
        User-angle homogeneous solutions, multipliers, and user transmittances.
    """
    if do_postprocessing:
        t_delt_userm = exp_outer(delta_tau, user_secants)
        u_xpos, u_xneg = build_user_solution()
        hmult_1, hmult_2 = build_hmult(t_delt_userm)
        return u_xpos, u_xneg, hmult_1, hmult_2, t_delt_userm

    u_xpos = make_zero_array((n_users, nlay))
    u_xneg = make_zero_array((n_users, nlay))
    hmult_1 = make_zero_array((n_users, nlay))
    hmult_2 = make_zero_array((n_users, nlay))
    t_delt_userm = make_zero_array((nlay, n_users))
    return u_xpos, u_xneg, hmult_1, hmult_2, t_delt_userm


def prepare_solar_misc(
    *,
    do_plane_parallel,
    build_plane_parallel,
    build_spherical,
):
    """Selects the solar transmittance-preparation branch.

    Parameters
    ----------
    do_plane_parallel
        Whether the plane-parallel geometry path is active.
    build_plane_parallel
        Callback producing plane-parallel solar transmittance terms.
    build_spherical
        Callback producing spherical solar transmittance terms.

    Returns
    -------
    dict
        Solar transmittance and multiplier-preparation arrays for the selected
        geometry branch.
    """
    if do_plane_parallel:
        return build_plane_parallel()
    return build_spherical()


def prepare_solar_direct_beam_terms(
    *,
    ngeoms,
    do_include_surface,
    make_zero_vector,
    compute_direct_beam,
):
    """Builds solar direct-beam surface terms for every geometry.

    Parameters
    ----------
    ngeoms
        Number of geometries in the current solve.
    do_include_surface
        Whether the current Fourier mode includes a surface term.
    make_zero_vector
        Backend-specific zero-vector factory.
    compute_direct_beam
        Callback returning the direct-beam term for one geometry.

    Returns
    -------
    array-like
        One direct-beam surface term per geometry.
    """
    direct_beam_terms = make_zero_vector(ngeoms)
    if not do_include_surface:
        return direct_beam_terms
    for geometry_index in range(ngeoms):
        direct_beam_terms[geometry_index] = compute_direct_beam(geometry_index)
    return direct_beam_terms


def prepare_solar_fourier_postprocessing(
    *,
    build_hom_solution,
    build_user_solution,
    build_hmult,
):
    """Builds shared solar postprocessing terms for one Fourier mode.

    Returns
    -------
    tuple
        Eigenvalues, eigentrans, homogeneous solutions, and user-angle
        multiplier arrays for the current Fourier mode.
    """
    eigenvalue, eigentrans, xpos, norm_saved = build_hom_solution()
    u_xpos, u_xneg = build_user_solution(xpos)
    hmult_1, hmult_2 = build_hmult(eigenvalue, eigentrans)
    return eigenvalue, eigentrans, xpos, norm_saved, u_xpos, u_xneg, hmult_1, hmult_2


def prepare_solar_geometry_solution(
    *,
    build_gbeam_solution: Callable[[], tuple[Any, Any, Any, Any, Any, Any]],
    solve_bvp: Callable[[Any, Any], tuple[Any, Any]],
    extract_boundary_terms: Callable[[Any, Any, Any, Any], BoundaryTerms],
):
    """Builds the per-geometry solar Green-function and BVP solution bundle.

    Returns
    -------
    tuple
        Green-function source terms, BVP solutions, and extracted boundary
        terms reused by the output assembly routines.
    """
    gamma_m, gamma_p, aterm, bterm, wupper, wlower = build_gbeam_solution()
    lcon, mcon = solve_bvp(wupper, wlower)
    boundary_terms = extract_boundary_terms(lcon, mcon, wupper, wlower)
    return gamma_m, gamma_p, aterm, bterm, wupper, wlower, lcon, mcon, boundary_terms


def prepare_thermal_boundary_terms(
    *,
    lcon,
    mcon,
    xpos,
    eigentrans,
    wlower,
) -> BoundaryTerms:
    """Builds boundary terms reused by thermal output assembly.

    Returns
    -------
    dict
        Cached lower-boundary thermal terms used by flux and radiance assembly.
    """
    return {
        "eigentransnl": eigentrans[-1],
        "lcon_xvec1nl": lcon[-1] * xpos[0, -1],
        "mcon_xvec1nl": mcon[-1] * xpos[1, -1],
        "wlower1nl": wlower[0, -1],
    }


def accumulate_scalar_and_levels(
    *,
    scalar_store,
    scalar_value,
    level_store,
    level_value,
    index,
    fourier,
    azmfac,
    do_level_output,
):
    """Accumulates a scalar result and optional level array in place.

    Parameters
    ----------
    scalar_store
        Output array storing one scalar per geometry.
    scalar_value
        Current Fourier contribution for the scalar output.
    level_store
        Output array storing optional level profiles.
    level_value
        Current Fourier contribution for the level profile.
    index
        Geometry or user-angle index to update.
    fourier
        Current Fourier index.
    azmfac
        Azimuthal weighting factor for nonzero Fourier modes.
    do_level_output
        Whether level profiles are enabled for the solve.
    """
    scalar_store[index] = accumulate_fourier_term(
        scalar_store[index],
        scalar_value,
        fourier=fourier,
        azmfac=azmfac,
    )
    if do_level_output:
        level_store[index, :] = accumulate_fourier_term(
            level_store[index, :],
            level_value,
            fourier=fourier,
            azmfac=azmfac,
        )


def accumulate_flux_pair(
    *,
    fluxes_toa,
    fluxes_boa,
    toa,
    boa,
    index,
    fourier,
    azmfac,
):
    """Accumulates a TOA/BOA flux pair in place.

    Parameters
    ----------
    fluxes_toa, fluxes_boa
        Output flux arrays updated in place.
    toa, boa
        Current Fourier contributions for TOA and BOA fluxes.
    index
        Geometry or user-angle index to update.
    fourier
        Current Fourier index.
    azmfac
        Azimuthal weighting factor for nonzero Fourier modes.
    """
    fluxes_toa[:, index] = accumulate_fourier_term(
        fluxes_toa[:, index],
        toa,
        fourier=fourier,
        azmfac=azmfac,
    )
    fluxes_boa[:, index] = accumulate_fourier_term(
        fluxes_boa[:, index],
        boa,
        fourier=fourier,
        azmfac=azmfac,
    )

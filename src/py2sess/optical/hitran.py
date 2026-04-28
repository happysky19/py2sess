"""HITRAN line-by-line gas cross sections matching the GEOCAPE benchmark path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_C2 = 1.4387752
_T0 = 296.0
_VOIGT_EXTRA = 25.0
_RRTPI = 0.5641895835
_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)
_HUMLIK_Y0 = 1.5
_HUMLIK_Y0PY0 = 3.0
_HUMLIK_Y0Q = 2.25
_HUMLIK_C = np.array(
    [1.0117281, -0.75197147, 0.012557727, 0.010022008, -0.00024206814, 5.0084806e-7]
)
_HUMLIK_S = np.array([1.393237, 0.23115241, -0.15535147, 0.0062183662, 9.1908299e-5, -6.2752596e-7])
_HUMLIK_T = np.array([0.31424038, 0.94778839, 1.5976826, 2.2795071, 3.0206370, 3.8897249])

_MOLECULE_NUMBER = {
    "H2O": 1,
    "CO2": 2,
    "O3": 3,
    "N2O": 4,
    "CO": 5,
    "CH4": 6,
    "O2": 7,
    "NO": 8,
    "SO2": 9,
    "NO2": 10,
    "NH3": 11,
    "HNO3": 12,
    "H2CO": 20,
    "HCOOH": 32,
    "CH3OH": 39,
}

_FILE_VERSION = {
    1: "hit09",
    7: "hit10",
    9: "hit09",
    19: "hit09",
    24: "hit10",
    26: "hit11",
    27: "hit09",
}

_SKIP_ISOTOPES = {(2, 9), (2, 10), (6, 4), (27, 2)}

_ISOTOPE_MASS_AMU = {
    (1, 1): 18.010565,
    (1, 2): 20.014811,
    (1, 3): 19.014780,
    (1, 4): 19.016740,
    (1, 5): 21.020985,
    (1, 6): 20.020956,
    (2, 1): 43.989830,
    (2, 2): 44.993185,
    (2, 3): 45.994076,
    (2, 4): 44.994045,
    (2, 5): 46.997431,
    (2, 6): 45.997400,
    (2, 7): 47.998322,
    (2, 8): 46.998291,
    (3, 1): 47.984745,
    (3, 2): 49.988991,
    (3, 3): 49.988991,
    (3, 4): 48.988960,
    (3, 5): 48.988960,
    (4, 1): 44.001062,
    (4, 2): 44.998096,
    (4, 3): 44.998096,
    (4, 4): 46.005308,
    (4, 5): 45.005278,
    (5, 1): 27.994915,
    (5, 2): 28.998270,
    (5, 3): 29.999161,
    (5, 4): 28.999130,
    (5, 5): 31.002516,
    (5, 6): 30.002485,
    (6, 1): 16.031300,
    (6, 2): 17.034655,
    (6, 3): 17.037475,
    (7, 1): 31.989830,
    (7, 2): 33.994076,
    (7, 3): 32.994045,
    (8, 1): 29.997989,
    (8, 2): 30.995023,
    (8, 3): 32.002234,
    (9, 1): 63.961901,
    (9, 2): 65.957695,
    (10, 1): 45.992904,
    (11, 1): 17.026549,
    (11, 2): 18.023583,
    (12, 1): 62.995644,
    (20, 1): 30.010565,
    (20, 2): 31.013920,
    (20, 3): 32.014811,
    (32, 1): 46.005480,
    (39, 1): 32.026215,
}


@dataclass(frozen=True)
class HitranLineData:
    molecule_number: int
    isotope: np.ndarray
    center_cm_inv: np.ndarray
    strength: np.ndarray
    air_half_width: np.ndarray
    lower_state_energy: np.ndarray
    temperature_exponent: np.ndarray
    pressure_shift: np.ndarray
    molecular_mass_amu: np.ndarray

    @property
    def size(self) -> int:
        return int(self.center_cm_inv.size)


def hitran_molecule_number(molecule: str) -> int:
    """Return the HITRAN molecule number used by the Fortran benchmark code."""
    key = molecule.strip().upper()
    try:
        return _MOLECULE_NUMBER[key]
    except KeyError as exc:
        raise ValueError(f"unsupported HITRAN molecule {molecule!r}") from exc


def read_hitran_lines(
    hitran_dir: str | Path,
    molecule: str,
    wavenumber_cm_inv,
    *,
    margin_cm_inv: float = _VOIGT_EXTRA,
) -> HitranLineData:
    """Read HITRAN lines inside the Fortran benchmark margin."""
    grid = _as_1d_float(wavenumber_cm_inv, "wavenumber_cm_inv")
    wstart = float(np.min(grid))
    wend = float(np.max(grid))
    molnum = hitran_molecule_number(molecule)
    isotopes = []
    centers = []
    strengths = []
    widths = []
    elows = []
    coeffs = []
    shifts = []
    masses = []
    line_file = Path(hitran_dir) / f"{molnum:02d}_{_FILE_VERSION.get(molnum, 'hit08')}.par"
    with line_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_hitran_line(line)
            if parsed is None:
                continue
            line_mol, iso, center, strength, width, elow, coeff, shift = parsed
            if center > wend + margin_cm_inv:
                break
            if (line_mol, iso) in _SKIP_ISOTOPES or line_mol == 40:
                continue
            if line_mol != molnum or center <= wstart - margin_cm_inv:
                continue
            try:
                mass = _ISOTOPE_MASS_AMU[(line_mol, iso)]
            except KeyError as exc:
                raise ValueError(f"missing isotope mass for HITRAN ({line_mol}, {iso})") from exc
            isotopes.append(iso)
            centers.append(center)
            strengths.append(strength)
            widths.append(width)
            elows.append(elow)
            coeffs.append(coeff)
            shifts.append(shift)
            masses.append(mass)
    return HitranLineData(
        molecule_number=molnum,
        isotope=np.asarray(isotopes, dtype=int),
        center_cm_inv=np.asarray(centers, dtype=float),
        strength=np.asarray(strengths, dtype=float),
        air_half_width=np.asarray(widths, dtype=float),
        lower_state_energy=np.asarray(elows, dtype=float),
        temperature_exponent=np.asarray(coeffs, dtype=float),
        pressure_shift=np.asarray(shifts, dtype=float),
        molecular_mass_amu=np.asarray(masses, dtype=float),
    )


def load_hitran_partition_functions(hitran_dir: str | Path) -> dict[tuple[int, int], np.ndarray]:
    """Load ``hitran08-parsum.resorted`` into ``(mol, isotope) -> q[148:342]``."""
    path = Path(hitran_dir) / "hitran08-parsum.resorted"
    out: dict[tuple[int, int], np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            parts = header.split()
            if len(parts) < 2:
                continue
            molnum = int(parts[0])
            isotope = int(parts[1])
            values = [float(handle.readline()) for _ in range(195)]
            out[(molnum, isotope)] = np.asarray(values, dtype=float)
    return out


def humlicek_voigt(x, y) -> np.ndarray:
    """Real Faddeeva/Voigt approximation ported from Fortran ``HUMLIK``."""
    xarr, yarr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if np.any(yarr < 0.0):
        raise ValueError("y must be nonnegative")

    yq = yarr * yarr
    yrrtpi = yarr * _RRTPI
    abx = np.abs(xarr)
    xq = abx * abx
    out = np.empty_like(xarr, dtype=float)

    remaining = np.ones_like(xarr, dtype=bool)
    mask = yarr >= 70.55
    out[mask] = yrrtpi[mask] / (xq[mask] + yq[mask])
    remaining &= ~mask

    with np.errstate(invalid="ignore"):
        xlim0 = np.sqrt(15100.0 + yarr * (40.0 - yarr * 3.6))
        xlim1 = np.zeros_like(yarr, dtype=float)
        low_y = yarr < 8.425
        xlim1[low_y] = np.sqrt(164.0 - yarr[low_y] * (4.3 + yarr[low_y] * 1.8))
    xlim2 = 6.8 - yarr
    xlim3 = 2.4 * yarr
    xlim4 = 18.1 * yarr + 1.65
    near_zero = yarr <= 1.0e-6
    xlim1 = np.where(near_zero, xlim0, xlim1)
    xlim2 = np.where(near_zero, xlim0, xlim2)

    mask = remaining & (abx >= xlim0)
    out[mask] = yrrtpi[mask] / (xq[mask] + yq[mask])
    remaining &= ~mask

    mask = remaining & (abx >= xlim1)
    if np.any(mask):
        yy = yarr[mask]
        yqm = yq[mask]
        xx = xq[mask]
        a0 = yqm + 0.5
        d0 = a0 * a0
        d2 = yqm + yqm - 1.0
        d = _RRTPI / (d0 + xx * (d2 + xx))
        out[mask] = d * yy * (a0 + xx)
    remaining &= ~mask

    mask = remaining & (abx > xlim2)
    if np.any(mask):
        yy = yarr[mask]
        yqm = yq[mask]
        h0 = 0.5625 + yqm * (4.5 + yqm * (10.5 + yqm * (6.0 + yqm)))
        h2 = -4.5 + yqm * (9.0 + yqm * (6.0 + yqm * 4.0))
        h4 = 10.5 - yqm * (6.0 - yqm * 6.0)
        h6 = -6.0 + yqm * 4.0
        e0 = 1.875 + yqm * (8.25 + yqm * (5.5 + yqm))
        e2 = 5.25 + yqm * (1.0 + yqm * 3.0)
        e4 = 0.75 * h6
        xx = xq[mask]
        d = _RRTPI / (h0 + xx * (h2 + xx * (h4 + xx * (h6 + xx))))
        out[mask] = d * yy * (e0 + xx * (e2 + xx * (e4 + xx)))
    remaining &= ~mask

    mask = remaining & (abx < xlim3)
    if np.any(mask):
        yy = yarr[mask]
        xx = xq[mask]
        z0 = 272.1014 + yy * (
            1280.829
            + yy
            * (
                2802.870
                + yy
                * (
                    3764.966
                    + yy
                    * (
                        3447.629
                        + yy
                        * (
                            2256.981
                            + yy
                            * (1074.409 + yy * (369.1989 + yy * (88.26741 + yy * (13.39880 + yy))))
                        )
                    )
                )
            )
        )
        z2 = 211.678 + yy * (
            902.3066
            + yy
            * (
                1758.336
                + yy
                * (
                    2037.310
                    + yy
                    * (1549.675 + yy * (793.4273 + yy * (266.2987 + yy * (53.59518 + yy * 5.0))))
                )
            )
        )
        z4 = 78.86585 + yy * (
            308.1852
            + yy * (497.3014 + yy * (479.2576 + yy * (269.2916 + yy * (80.39278 + yy * 10.0))))
        )
        z6 = 22.03523 + yy * (55.02933 + yy * (92.75679 + yy * (53.59518 + yy * 10.0)))
        z8 = 1.496460 + yy * (13.39880 + yy * 5.0)
        p0 = 153.5168 + yy * (
            549.3954
            + yy
            * (
                919.4955
                + yy
                * (
                    946.8970
                    + yy
                    * (
                        662.8097
                        + yy
                        * (
                            328.2151
                            + yy * (115.3772 + yy * (27.93941 + yy * (4.264678 + yy * 0.3183291)))
                        )
                    )
                )
            )
        )
        p2 = -34.16955 + yy * (
            -1.322256
            + yy
            * (
                124.5975
                + yy
                * (189.7730 + yy * (139.4665 + yy * (56.81652 + yy * (12.79458 + yy * 1.2733163))))
            )
        )
        p4 = 2.584042 + yy * (
            10.46332 + yy * (24.01655 + yy * (29.81482 + yy * (12.79568 + yy * 1.9099744)))
        )
        p6 = -0.07272979 + yy * (0.9377051 + yy * (4.266322 + yy * 1.273316))
        p8 = 0.0005480304 + yy * 0.3183291
        d = 1.7724538 / (z0 + xx * (z2 + xx * (z4 + xx * (z6 + xx * (z8 + xx)))))
        out[mask] = d * (p0 + xx * (p2 + xx * (p4 + xx * (p6 + xx * p8))))
    remaining &= ~mask

    if np.any(remaining):
        xr = xarr[remaining]
        xrq = xq[remaining]
        yr = yarr[remaining]
        ypy0 = yr + _HUMLIK_Y0
        ypy0q = ypy0 * ypy0
        acc = np.zeros_like(xr, dtype=float)
        near = abx[remaining] <= xlim4[remaining]
        for j in range(6):
            dm = xr - _HUMLIK_T[j]
            mq = dm * dm
            mf = 1.0 / (mq + ypy0q)
            xm = mf * dm
            ym = mf * ypy0

            dp = xr + _HUMLIK_T[j]
            pq = dp * dp
            pf = 1.0 / (pq + ypy0q)
            xp = pf * dp
            yp = pf * ypy0

            term_near = _HUMLIK_C[j] * (ym + yp) - _HUMLIK_S[j] * (xm - xp)
            yf = yr + _HUMLIK_Y0PY0
            term_far = (_HUMLIK_C[j] * (mq * mf - _HUMLIK_Y0 * ym) + _HUMLIK_S[j] * yf * xm) / (
                mq + _HUMLIK_Y0Q
            ) + (_HUMLIK_C[j] * (pq * pf - _HUMLIK_Y0 * yp) - _HUMLIK_S[j] * yf * xp) / (
                pq + _HUMLIK_Y0Q
            )
            acc += np.where(near, term_near, term_far)
        far = abx[remaining] > xlim4[remaining]
        acc = np.where(far, yr * acc + np.exp(-xrq), acc)
        out[remaining] = acc
    return out


def hitran_cross_sections(
    *,
    hitran_dir: str | Path,
    molecule: str,
    spectral_grid,
    pressure_atm,
    temperature_k,
    is_wavenumber: bool = True,
    fwhm: float = 0.0,
    partition_functions: dict[tuple[int, int], np.ndarray] | None = None,
    lines: HitranLineData | None = None,
) -> np.ndarray:
    """Compute level gas cross sections for the benchmark no-convolution path.

    Returned shape is ``(nspec, nlevel)``. Pressure must be in atmospheres,
    matching the Fortran CreateProps call site.
    """
    if abs(float(fwhm)) > 1.0e-13:
        raise NotImplementedError("HITRAN Gaussian convolution is not implemented")
    grid = _as_1d_float(spectral_grid, "spectral_grid")
    pressure = _as_1d_float(pressure_atm, "pressure_atm")
    temperature = _as_1d_float(temperature_k, "temperature_k")
    if pressure.shape != temperature.shape:
        raise ValueError("pressure_atm and temperature_k must have the same shape")

    wavenumber, inverse_order = _wavenumber_grid(grid, is_wavenumber=is_wavenumber)
    if lines is None:
        lines = read_hitran_lines(hitran_dir, molecule, wavenumber)
    if partition_functions is None:
        partition_functions = load_hitran_partition_functions(hitran_dir)
    if lines.size == 0:
        return np.zeros((grid.size, pressure.size), dtype=float)

    q296 = _lookup_partition_matrix(
        partition_functions,
        molecule_number=lines.molecule_number,
        isotopes=lines.isotope,
        temperature_k=np.array([_T0], dtype=float),
    )[0]
    q = _lookup_partition_matrix(
        partition_functions,
        molecule_number=lines.molecule_number,
        isotopes=lines.isotope,
        temperature_k=temperature,
    )
    out = _cross_sections_all_levels(
        wavenumber=wavenumber,
        lines=lines,
        q296=q296,
        q=q,
        pressure_atm=pressure,
        temperature_k=temperature,
    )
    return np.maximum(out[inverse_order], 0.0)


def _cross_sections_all_levels(
    *,
    wavenumber: np.ndarray,
    lines: HitranLineData,
    q296: np.ndarray,
    q: np.ndarray,
    pressure_atm: np.ndarray,
    temperature_k: np.ndarray,
) -> np.ndarray:
    spec = np.zeros((wavenumber.size, pressure_atm.size), dtype=float)
    nvoigt, voigt_grid = _fortran_voigt_grid(wavenumber)
    rt0t = _T0 / temperature_k
    rc2t = _C2 / temperature_k
    rc2t0 = _C2 / _T0
    for i in range(lines.size):
        sigma = lines.center_cm_inv[i] + lines.pressure_shift[i] * pressure_atm
        nvlo, nvhi = _fortran_voigt_windows(voigt_grid, nvoigt, sigma)
        has_window = nvhi >= nvlo
        if not np.any(has_window):
            continue
        start = max(0, int(np.min(nvlo[has_window])) - nvoigt)
        stop = min(wavenumber.size, int(np.max(nvhi[has_window])) - nvoigt + 1)
        if stop <= start:
            continue

        vg = 4.30140e-7 * sigma * np.sqrt(temperature_k / lines.molecular_mass_amu[i])
        voigta = pressure_atm * lines.air_half_width[i] * rt0t ** lines.temperature_exponent[i] / vg
        ratio1 = np.exp(-lines.lower_state_energy[i] * rc2t) - np.exp(
            -(sigma + lines.lower_state_energy[i]) * rc2t
        )
        ratio2 = np.exp(-lines.lower_state_energy[i] * rc2t0) - np.exp(
            -(sigma + lines.lower_state_energy[i]) * rc2t0
        )
        ratio = ratio1 / ratio2 * q296[i] / q[:, i]
        vnorm = ratio * lines.strength[i] / vg * _INV_SQRT_PI
        local_wave = wavenumber[start:stop]
        central_index = nvoigt + np.arange(start, stop)
        active = has_window[:, np.newaxis] & (
            (central_index[np.newaxis, :] >= nvlo[:, np.newaxis])
            & (central_index[np.newaxis, :] <= nvhi[:, np.newaxis])
        )
        x = (local_wave[np.newaxis, :] - sigma[:, np.newaxis]) / vg[:, np.newaxis]
        contribution = vnorm[:, np.newaxis] * humlicek_voigt(x, voigta[:, np.newaxis])
        spec[start:stop] += np.where(active, contribution, 0.0).T
    return spec


def _fortran_voigt_grid(wavenumber: np.ndarray) -> tuple[int, np.ndarray]:
    if wavenumber.size < 2:
        return 0, wavenumber.copy()
    step = (wavenumber[-1] - wavenumber[0]) / (wavenumber.size - 1)
    if step <= 0.0:
        raise ValueError("wavenumber grid must be increasing")
    nvoigt = int(_VOIGT_EXTRA / step)
    grid = np.empty(wavenumber.size + 2 * nvoigt, dtype=float)
    grid[nvoigt : nvoigt + wavenumber.size] = wavenumber
    if nvoigt:
        grid[:nvoigt] = wavenumber[0] - step * np.arange(nvoigt, 0, -1)
        grid[nvoigt + wavenumber.size :] = wavenumber[-1] + step * np.arange(1, nvoigt + 1)
    return nvoigt, grid


def _fortran_voigt_windows(
    voigt_grid: np.ndarray,
    nvoigt: int,
    sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    npoints = voigt_grid.size
    insert = np.searchsorted(voigt_grid, sigma, side="left")
    idx = insert - 1
    valid = (idx >= 0) & (insert < npoints)

    nvlo = np.empty(sigma.shape, dtype=int)
    nvhi = np.empty(sigma.shape, dtype=int)
    nvlo[valid] = np.maximum(0, idx[valid] - nvoigt)
    nvhi[valid] = np.minimum(npoints - 1, idx[valid] + nvoigt)

    no_bin = ~valid
    outside = no_bin & (
        (sigma < voigt_grid[0] - _VOIGT_EXTRA) | (sigma > voigt_grid[-1] + _VOIGT_EXTRA)
    )
    below = no_bin & ~outside & (sigma < voigt_grid[0])
    above = no_bin & ~outside & ~below
    nvlo[outside] = 1
    nvhi[outside] = 0
    nvlo[below] = 0
    nvhi[below] = nvoigt - 1
    nvlo[above] = npoints - nvoigt
    nvhi[above] = npoints - 1
    return nvlo, nvhi


def _wavenumber_grid(grid: np.ndarray, *, is_wavenumber: bool) -> tuple[np.ndarray, np.ndarray]:
    wavenumber = grid if is_wavenumber else 1.0e7 / grid
    order = np.argsort(wavenumber)
    sorted_grid = np.asarray(wavenumber[order], dtype=float)
    if np.any(np.diff(sorted_grid) <= 0.0):
        raise ValueError("spectral grid must be unique")
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)
    return sorted_grid, inverse_order


def _lookup_partition_matrix(
    partition_functions: dict[tuple[int, int], np.ndarray],
    *,
    molecule_number: int,
    isotopes: np.ndarray,
    temperature_k: np.ndarray,
) -> np.ndarray:
    temperature = np.asarray(temperature_k, dtype=float)
    if np.any((temperature < 150.0) | (temperature > 340.0)):
        raise ValueError("HITRAN partition function temperature must be in [150, 340] K")
    unique_isotopes, inverse = np.unique(isotopes.astype(int), return_inverse=True)
    unique_values = np.empty((temperature.size, unique_isotopes.size), dtype=float)
    for index, isotope in enumerate(unique_isotopes):
        try:
            table = partition_functions[(molecule_number, int(isotope))]
        except KeyError as exc:
            raise ValueError(
                f"partition function is missing HITRAN ({molecule_number}, {int(isotope)})"
            ) from exc
        unique_values[:, index] = _lookup_q_table_vector(table, temperature)
    return unique_values[:, inverse]


def _lookup_q_table_vector(table: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    exact = temperature == np.floor(temperature)
    out = np.empty_like(temperature, dtype=float)
    if np.any(exact):
        out[exact] = table[temperature[exact].astype(int) - 148]
    if np.any(~exact):
        temp = temperature[~exact]
        ib = temp.astype(int)
        tcalc = temp - ib
        ia = ib - 1
        ic = ib + 1
        id_ = ib + 2
        ya = table[ia - 148]
        yb = table[ib - 148]
        yc = table[ic - 148]
        yd = table[id_ - 148]
        a = tcalc + 1.0
        b = tcalc
        c = tcalc - 1.0
        d = tcalc - 2.0
        out[~exact] = (
            -(ya * b * c * d / 6.0)
            + (yb * a * c * d / 2.0)
            - (yc * a * b * d / 2.0)
            + (yd * a * b * c / 6.0)
        )
    return out


def _parse_hitran_line(
    line: str,
) -> tuple[int, int, float, float, float, float, float, float] | None:
    if len(line) < 67:
        return None
    try:
        return (
            int(line[0:2]),
            int(line[2:3]),
            float(line[3:15]),
            float(line[15:25]),
            float(line[35:40]),
            float(line[45:55]),
            float(line[55:59]),
            float(line[59:67]),
        )
    except ValueError:
        return None


def _as_1d_float(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr

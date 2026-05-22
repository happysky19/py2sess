from __future__ import annotations

import unittest

import numpy as np

from py2sess import (
    TwoStreamEss,
    TwoStreamEssOptions,
    native_backend_info,
    native_extension_available,
)
from py2sess.rtsolver.backend import has_torch, to_numpy
from py2sess.rtsolver.native_backend import (
    native_backend_supports_device,
    solve_solar_fo,
    solve_thermal_fo,
)


class NativeBackendTests(unittest.TestCase):
    def test_native_backend_info_and_device_support(self) -> None:
        info = native_backend_info()
        self.assertIn("available", info)
        self.assertIn("backend", info)
        self.assertEqual(native_extension_available(), bool(info["available"]))
        if not native_extension_available():
            return
        self.assertTrue(native_backend_supports_device("cpu"))
        self.assertEqual(native_backend_supports_device("cuda"), bool(info.get("cuda", False)))
        self.assertFalse(native_backend_supports_device("mps"))

    @unittest.skipUnless(has_torch(), "torch is not installed")
    @unittest.skipUnless(native_extension_available(), "py2sess._native is not built")
    def test_native_solar_two_stream_matches_torch(self) -> None:
        kwargs = dict(
            tau=np.array([[0.01, 0.02, 0.03], [0.04, 0.03, 0.02]]),
            ssa=np.array([[0.2, 0.15, 0.1], [0.1, 0.2, 0.15]]),
            g=np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.05]]),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=np.array([[30.0, 20.0, 0.0], [45.0, 10.0, 30.0]]),
            albedo=np.array([0.1, 0.2]),
            fbeam=np.array([1.0, 0.8]),
        )
        native = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="solar",
                backend="native",
                torch_dtype="float64",
                output_levels=True,
            )
        ).forward(**kwargs)
        torch_result = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="solar",
                backend="torch",
                torch_dtype="float64",
                output_levels=True,
            )
        ).forward(**kwargs)
        for field in ("radiance_profile_2s", "radiance_total"):
            np.testing.assert_allclose(
                to_numpy(getattr(native, field)),
                to_numpy(getattr(torch_result, field)),
                rtol=1.0e-12,
                atol=1.0e-12,
            )

    @unittest.skipUnless(has_torch(), "torch is not installed")
    @unittest.skipUnless(native_extension_available(), "py2sess._native is not built")
    def test_native_thermal_two_stream_matches_torch(self) -> None:
        kwargs = dict(
            tau=np.array([[0.2, 0.3, 0.4], [0.1, 0.2, 0.25]]),
            ssa=np.array([[0.15, 0.1, 0.05], [0.05, 0.1, 0.15]]),
            g=np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.05]]),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[30.0, 60.0],
            planck=np.array([[1.0, 1.1, 1.2, 1.3], [0.9, 1.0, 1.1, 1.2]]),
            surface_planck=np.array([1.4, 1.3]),
            emissivity=np.array([0.9, 0.85]),
            albedo=np.array([0.05, 0.08]),
        )
        native = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="thermal",
                backend="native",
                torch_dtype="float64",
                output_levels=True,
            )
        ).forward(**kwargs)
        torch_result = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="thermal",
                backend="torch",
                torch_dtype="float64",
                output_levels=True,
            )
        ).forward(**kwargs)
        for field in ("radiance_profile_2s", "radiance_total"):
            np.testing.assert_allclose(
                to_numpy(getattr(native, field)),
                to_numpy(getattr(torch_result, field)),
                rtol=1.0e-12,
                atol=1.0e-12,
            )

    @unittest.skipUnless(has_torch(), "torch is not installed")
    @unittest.skipUnless(native_extension_available(), "py2sess._native is not built")
    def test_native_solar_fluxes_match_scalar_rows(self) -> None:
        kwargs = dict(
            tau=np.array([[0.01, 0.02, 0.03], [0.015, 0.025, 0.035]]),
            ssa=np.full((2, 3), 0.2),
            g=np.full((2, 3), 0.1),
            z=np.array([3.0, 2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=np.array([0.1, 0.2]),
        )
        native = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=3,
                mode="solar",
                backend="native",
                torch_dtype="float64",
                output_fluxes=True,
            )
        ).forward(**kwargs)
        for row in range(2):
            scalar = TwoStreamEss(
                TwoStreamEssOptions(nlyr=3, mode="solar", output_fluxes=True)
            ).forward(
                tau=kwargs["tau"][row],
                ssa=kwargs["ssa"][row],
                g=kwargs["g"][row],
                z=kwargs["z"],
                angles=kwargs["angles"],
                albedo=kwargs["albedo"][row],
            )
            np.testing.assert_allclose(
                to_numpy(native.flux_up)[row], scalar.flux_up[0], rtol=1.0e-12, atol=1.0e-12
            )
            np.testing.assert_allclose(
                to_numpy(native.flux_down)[row],
                scalar.flux_down[0],
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        np.testing.assert_allclose(
            to_numpy(native.flux_net),
            to_numpy(native.flux_up) - to_numpy(native.flux_down),
            rtol=0.0,
            atol=0.0,
        )

    @unittest.skipUnless(has_torch(), "torch is not installed")
    @unittest.skipUnless(native_extension_available(), "py2sess._native is not built")
    def test_native_solar_fo_matches_torch_batch_kernel(self) -> None:
        import torch

        from py2sess.rtsolver.fo_solar_obs_batch_numpy import fo_solar_obs_batch_precompute
        from py2sess.rtsolver.fo_solar_obs_batch_torch import solve_fo_solar_obs_eps_batch_torch

        dtype = torch.float64
        device = torch.device("cpu")
        tau = torch.tensor([[0.01, 0.02, 0.03], [0.015, 0.025, 0.035]], dtype=dtype)
        omega = torch.full((2, 3), 0.2, dtype=dtype)
        scaling = torch.zeros_like(tau)
        albedo = torch.tensor([0.1, 0.2], dtype=dtype)
        flux_factor = torch.tensor([1.0, 0.8], dtype=dtype)
        exact_scatter = torch.tensor([[0.02, 0.03, 0.04], [0.025, 0.035, 0.045]], dtype=dtype)
        precomputed = fo_solar_obs_batch_precompute(
            user_obsgeom=np.array([30.0, 20.0, 0.0], dtype=float),
            heights=np.array([3.0, 2.0, 1.0, 0.0], dtype=float),
            earth_radius=6371.0,
            nfine=3,
        )
        native = solve_solar_fo(
            tau=tau,
            omega=omega,
            scaling=scaling,
            albedo=albedo,
            flux_factor=flux_factor,
            exact_scatter=exact_scatter,
            precomputed=precomputed,
        )
        torch_result = solve_fo_solar_obs_eps_batch_torch(
            tau=tau,
            omega=omega,
            scaling=scaling,
            albedo=albedo,
            flux_factor=flux_factor,
            exact_scatter=exact_scatter,
            precomputed=precomputed,
            dtype=dtype,
            device=device,
        )
        np.testing.assert_allclose(to_numpy(native), to_numpy(torch_result), rtol=1.0e-12)

        direct_reflectance = torch.tensor([0.35, 0.05], dtype=dtype)
        native_direct = solve_solar_fo(
            tau=tau,
            omega=omega,
            scaling=scaling,
            albedo=albedo,
            direct_surface_reflectance=direct_reflectance,
            flux_factor=flux_factor,
            exact_scatter=exact_scatter,
            precomputed=precomputed,
        )
        torch_direct = solve_fo_solar_obs_eps_batch_torch(
            tau=tau,
            omega=omega,
            scaling=scaling,
            albedo=albedo,
            direct_surface_reflectance=direct_reflectance,
            flux_factor=flux_factor,
            exact_scatter=exact_scatter,
            precomputed=precomputed,
            dtype=dtype,
            device=device,
        )
        np.testing.assert_allclose(
            to_numpy(native_direct), to_numpy(torch_direct), rtol=1.0e-12, atol=1.0e-12
        )

    @unittest.skipUnless(has_torch(), "torch is not installed")
    @unittest.skipUnless(native_extension_available(), "py2sess._native is not built")
    def test_native_thermal_fo_matches_torch_batch_kernel(self) -> None:
        import torch

        from py2sess.rtsolver.thermal_batch_numpy import precompute_fo_thermal_geometry_numpy
        from py2sess.rtsolver.thermal_batch_torch import (
            _fo_thermal_toa_batch,
            fo_thermal_geometry_to_torch,
        )

        dtype = torch.float64
        device = torch.device("cpu")
        tau = torch.tensor([[0.2, 0.3, 0.4], [0.1, 0.2, 0.25]], dtype=dtype)
        omega = torch.tensor([[0.15, 0.1, 0.05], [0.05, 0.1, 0.15]], dtype=dtype)
        scaling = torch.zeros_like(tau)
        planck = torch.tensor([[1.0, 1.1, 1.2, 1.3], [0.9, 1.0, 1.1, 1.2]], dtype=dtype)
        surfbb = torch.tensor([1.4, 1.3], dtype=dtype)
        emissivity = torch.tensor([0.9, 0.85], dtype=dtype)
        heights = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=dtype)
        geometry_np = precompute_fo_thermal_geometry_numpy(
            heights=to_numpy(heights),
            user_angle_degrees=30.0,
            earth_radius=6371.0,
            nfine=3,
        )
        geometry = fo_thermal_geometry_to_torch(geometry_np, dtype=dtype, device=device)
        native = solve_thermal_fo(
            tau=tau,
            omega=omega,
            scaling=scaling,
            planck=planck,
            surfbb=surfbb,
            emissivity=emissivity,
            heights=heights,
            geometry=geometry,
        )
        torch_result = _fo_thermal_toa_batch(
            tau=tau,
            omega=omega,
            scaling=scaling,
            thermal_bb_input=planck,
            surfbb=surfbb,
            emissivity=emissivity,
            heights=heights,
            user_angle_degrees=30.0,
            earth_radius=6371.0,
            nfine=3,
            fo_geometry=geometry,
        )
        np.testing.assert_allclose(
            to_numpy(native), to_numpy(torch_result), rtol=1.0e-12, atol=1.0e-12
        )

    @unittest.skipUnless(has_torch(), "torch is not installed")
    @unittest.skipUnless(native_extension_available(), "py2sess._native is not built")
    def test_native_solar_surface_leaving_matches_scalar_rows(self) -> None:
        kwargs = dict(
            tau=np.array([[0.01, 0.02], [0.04, 0.03]]),
            ssa=np.full((2, 2), 0.1),
            g=np.zeros((2, 2)),
            z=np.array([2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 10.0],
            albedo=0.0,
            delta_m_truncation_factor=np.zeros((2, 2), dtype=float),
        )
        surface_leaving = {
            "slterm_isotropic": np.array([[0.2], [0.3]], dtype=float),
            "slterm_f_0": np.zeros((2, 1, 2), dtype=float),
        }
        native = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=2,
                mode="solar",
                backend="native",
                torch_dtype="float64",
                surface_leaving=True,
                output_levels=True,
                output_fluxes=True,
            )
        ).forward(**kwargs, surface_leaving=surface_leaving, include_fo=True)
        for row in range(2):
            scalar = TwoStreamEss(
                TwoStreamEssOptions(
                    nlyr=2,
                    mode="solar",
                    surface_leaving=True,
                    output_levels=True,
                    output_fluxes=True,
                )
            ).forward(
                tau=kwargs["tau"][row],
                ssa=kwargs["ssa"][row],
                g=kwargs["g"][row],
                z=kwargs["z"],
                angles=kwargs["angles"],
                albedo=0.0,
                delta_m_truncation_factor=kwargs["delta_m_truncation_factor"][row],
                surface_leaving={
                    "slterm_isotropic": surface_leaving["slterm_isotropic"][row],
                    "slterm_f_0": surface_leaving["slterm_f_0"][row],
                },
                include_fo=True,
            )
            for field in ("radiance_profile_2s", "radiance_profile_total", "radiance_total"):
                np.testing.assert_allclose(
                    to_numpy(getattr(native, field))[row],
                    getattr(scalar, field)[0],
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )
            np.testing.assert_allclose(
                to_numpy(native.flux_up)[row], scalar.flux_up[0], rtol=1.0e-12, atol=1.0e-12
            )


if __name__ == "__main__":
    unittest.main()

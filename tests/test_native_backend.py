from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from py2sess import (
    TwoStreamEss,
    TwoStreamEssOptions,
    native_backend_info,
    native_extension_available,
)
from py2sess.rtsolver import native_backend as native_backend_module
from py2sess.rtsolver.backend import has_torch, to_numpy
from py2sess.rtsolver.native_backend import native_backend_supports_device


class _FakeNativeExtension:
    def __init__(self, *, cuda: bool) -> None:
        self.cuda = cuda

    def backend_info(self) -> dict[str, object]:
        return {"backend": "native-extension", "cuda": self.cuda, "level_fluxes": True}


class _FakeDevice:
    def __init__(self, device_type: str) -> None:
        self.type = device_type


class _FakeTensor:
    def __init__(self, device_type: str) -> None:
        self.device = _FakeDevice(device_type)


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

    def test_cuda_extension_is_reported_and_selected_separately(self) -> None:
        cpu_extension = _FakeNativeExtension(cuda=False)
        cuda_extension = _FakeNativeExtension(cuda=True)
        with (
            mock.patch.object(
                native_backend_module, "_load_native_extension", return_value=cpu_extension
            ),
            mock.patch.object(
                native_backend_module,
                "_load_native_cuda_extension",
                return_value=cuda_extension,
            ),
        ):
            info = native_backend_module.native_backend_info()
            self.assertTrue(info["available"])
            self.assertTrue(info["cuda"])
            self.assertTrue(info["cuda_extension_available"])
            self.assertIs(
                native_backend_module._require_native_extension_for_tensor(_FakeTensor("cpu")),
                cpu_extension,
            )
            self.assertIs(
                native_backend_module._require_native_extension_for_tensor(_FakeTensor("cuda")),
                cuda_extension,
            )

    @unittest.skipUnless(has_torch(), "torch is not installed")
    @unittest.skipUnless(native_extension_available(), "py2sess._native is not built")
    def test_native_solar_two_stream_matches_torch(self) -> None:
        kwargs = dict(
            tau=np.array([[0.01, 0.02], [0.04, 0.03]]),
            ssa=np.array([[0.2, 0.15], [0.1, 0.2]]),
            g=np.array([[0.1, 0.2], [0.2, 0.1]]),
            z=np.array([2.0, 1.0, 0.0]),
            angles=np.array([[30.0, 20.0, 0.0], [45.0, 10.0, 30.0]]),
            albedo=np.array([0.1, 0.2]),
            fbeam=np.array([1.0, 0.8]),
        )
        native = TwoStreamEss(
            TwoStreamEssOptions(nlyr=2, mode="solar", backend="native", torch_dtype="float64")
        ).forward(**kwargs)
        torch_result = TwoStreamEss(
            TwoStreamEssOptions(nlyr=2, mode="solar", backend="torch", torch_dtype="float64")
        ).forward(**kwargs)
        np.testing.assert_allclose(
            to_numpy(native.radiance_total),
            to_numpy(torch_result.radiance_total),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    @unittest.skipUnless(has_torch(), "torch is not installed")
    @unittest.skipUnless(native_extension_available(), "py2sess._native is not built")
    def test_native_thermal_two_stream_matches_torch(self) -> None:
        kwargs = dict(
            tau=np.array([[0.2, 0.3], [0.1, 0.2]]),
            ssa=np.array([[0.15, 0.1], [0.05, 0.1]]),
            g=np.array([[0.1, 0.2], [0.2, 0.1]]),
            z=np.array([2.0, 1.0, 0.0]),
            angles=[30.0, 60.0],
            planck=np.array([[1.0, 1.1, 1.2], [0.9, 1.0, 1.1]]),
            surface_planck=np.array([1.4, 1.3]),
            emissivity=np.array([0.9, 0.85]),
            albedo=np.array([0.05, 0.08]),
        )
        native = TwoStreamEss(
            TwoStreamEssOptions(nlyr=2, mode="thermal", backend="native", torch_dtype="float64")
        ).forward(**kwargs)
        torch_result = TwoStreamEss(
            TwoStreamEssOptions(nlyr=2, mode="thermal", backend="torch", torch_dtype="float64")
        ).forward(**kwargs)
        np.testing.assert_allclose(
            to_numpy(native.radiance_total),
            to_numpy(torch_result.radiance_total),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    @unittest.skipUnless(has_torch(), "torch is not installed")
    @unittest.skipUnless(native_extension_available(), "py2sess._native is not built")
    def test_native_solar_level_fluxes_match_scalar_rows(self) -> None:
        kwargs = dict(
            tau=np.array([[0.01, 0.02], [0.015, 0.025]]),
            ssa=np.full((2, 2), 0.2),
            g=np.full((2, 2), 0.1),
            z=np.array([2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=np.array([0.1, 0.2]),
        )
        native = TwoStreamEss(
            TwoStreamEssOptions(
                nlyr=2,
                mode="solar",
                backend="native",
                torch_dtype="float64",
                output_fluxes=True,
            )
        ).forward(**kwargs)
        for row in range(2):
            scalar = TwoStreamEss(
                TwoStreamEssOptions(nlyr=2, mode="solar", output_fluxes=True)
            ).forward(
                tau=kwargs["tau"][row],
                ssa=kwargs["ssa"][row],
                g=kwargs["g"][row],
                z=kwargs["z"],
                angles=kwargs["angles"],
                albedo=kwargs["albedo"][row],
            )
            np.testing.assert_allclose(to_numpy(native.flux_up)[row], scalar.flux_up[0])
            np.testing.assert_allclose(to_numpy(native.flux_down)[row], scalar.flux_down[0])

    @unittest.skipUnless(has_torch(), "torch is not installed")
    @unittest.skipUnless(native_extension_available(), "py2sess._native is not built")
    def test_native_solar_fo_matches_torch_batch_kernel(self) -> None:
        kwargs = dict(
            tau=np.array([[0.01, 0.02], [0.04, 0.03]]),
            ssa=np.array([[0.2, 0.15], [0.1, 0.2]]),
            g=np.array([[0.1, 0.2], [0.2, 0.1]]),
            z=np.array([2.0, 1.0, 0.0]),
            angles=[30.0, 20.0, 0.0],
            albedo=np.array([0.1, 0.2]),
            fo_scatter_term=np.array([[0.15, 0.1], [0.08, 0.11]]),
            include_fo=True,
        )
        native = TwoStreamEss(
            TwoStreamEssOptions(nlyr=2, mode="solar", backend="native", torch_dtype="float64")
        ).forward(**kwargs)
        torch_result = TwoStreamEss(
            TwoStreamEssOptions(nlyr=2, mode="solar", backend="torch", torch_dtype="float64")
        ).forward(**kwargs)
        np.testing.assert_allclose(
            to_numpy(native.radiance_fo),
            to_numpy(torch_result.radiance_fo),
            rtol=1.0e-12,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()

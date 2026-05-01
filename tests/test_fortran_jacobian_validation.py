from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from py2sess import TwoStreamEss, TwoStreamEssOptions
from py2sess.rtsolver.backend import has_torch
from py2sess.scene import load_scene


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks" / "thermal_jacobian_profile1"
SOLAR_CASE = ROOT / "benchmarks" / "solar_jacobian_profile1"


def _comparison_module():
    import importlib.util

    path = ROOT / "examples" / "compare_fortran_jacobian.py"
    spec = importlib.util.spec_from_file_location("compare_fortran_jacobian", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FortranJacobianValidationTests(unittest.TestCase):
    def test_thermal_jacobian_scene_is_yaml_controlled(self) -> None:
        scene = load_scene(
            profile=CASE / "profile.csv",
            config=CASE / "scene.yaml",
            strict_runtime_inputs=True,
        )
        inputs = scene.to_forward_inputs()

        self.assertEqual(inputs.mode, "thermal")
        self.assertEqual(inputs.wavelengths.shape, (1000,))
        self.assertEqual(inputs.kwargs["tau"].shape, (1000, 123))
        self.assertIn("planck", inputs.kwargs)
        self.assertNotIn("fortran", inputs.kwargs)

    def test_solar_jacobian_case_is_yaml_controlled(self) -> None:
        import yaml

        config = yaml.safe_load((SOLAR_CASE / "scene.yaml").read_text())
        self.assertEqual(config["mode"], "solar")
        self.assertEqual(config["rt_inputs"]["path"], "rt_inputs.npz")
        self.assertEqual(config["jacobian_reference"]["path"], "fortran_jacobian_reference.npz")

        inputs = dict(np.load(SOLAR_CASE / config["rt_inputs"]["path"]))
        self.assertEqual(inputs["wavelength_nm"].shape, (1000,))
        self.assertEqual(inputs["tau"].shape, (1000, 114))
        self.assertEqual(inputs["ssa"].shape, (1000, 114))
        self.assertEqual(inputs["g"].shape, (1000, 114))
        self.assertEqual(inputs["delta_m_truncation_factor"].shape, (1000, 114))
        self.assertEqual(inputs["fo_scatter_term"].shape, (1000, 114))

    @unittest.skipUnless(has_torch(), "PyTorch is required")
    def test_radiance_and_surface_emissivity_gradients_match_fortran_reference(self) -> None:
        compare = _comparison_module()
        scene = load_scene(
            profile=CASE / "profile.csv",
            config=CASE / "scene.yaml",
            strict_runtime_inputs=True,
        )
        reference = dict(np.load(CASE / "fortran_jacobian_reference.npz"))
        result = compare.thermal_toa_jacobians(scene)
        indices = compare.matching_indices(result["wavelength_nm"], reference["wavelength_nm"])
        scale = compare.unit_scale(
            "auto",
            result["radiance_total"][indices],
            reference["radiance_total"],
        )

        np.testing.assert_allclose(
            scale * result["radiance_total"][indices],
            reference["radiance_total"],
            rtol=1.0e-5,
            atol=5.0e-7,
        )
        np.testing.assert_allclose(
            scale * result["surface_emissivity_jacobian_total"][indices],
            reference["surface_emissivity_jacobian_total"],
            rtol=1.0e-4,
            atol=5.0e-7,
        )
        np.testing.assert_allclose(
            scale * result["surface_temperature_jacobian_total_normalized"][indices],
            reference["surface_temperature_jacobian_total"],
            rtol=1.0e-4,
            atol=5.0e-7,
        )

    @unittest.skipUnless(has_torch(), "PyTorch is required")
    def test_solar_fo_2s_radiance_and_albedo_gradient_match_fortran_reference(
        self,
    ) -> None:
        compare = _comparison_module()
        result, reference = compare.solar_toa_jacobians(SOLAR_CASE / "scene.yaml")
        indices = compare.matching_indices(result["wavelength_nm"], reference["wavelength_nm"])

        for component in ("2s", "fo", "total"):
            np.testing.assert_allclose(
                result[f"radiance_{component}"][indices],
                reference[f"radiance_{component}"],
                rtol=5.0e-5,
                atol=5.0e-8,
            )
            np.testing.assert_allclose(
                result[f"surface_albedo_jacobian_{component}"][indices],
                reference[f"surface_albedo_jacobian_{component}"],
                rtol=5.0e-5,
                atol=5.0e-8,
            )

    @unittest.skipUnless(has_torch(), "PyTorch is required")
    def test_direct_rt_variable_autograd_matches_finite_difference(self) -> None:
        import torch

        scene = load_scene(
            profile=CASE / "profile.csv",
            config=CASE / "scene.yaml",
            spectral_limit=2,
            strict_runtime_inputs=True,
        )
        kwargs = scene.to_forward_inputs().kwargs
        checks = (
            ("tau", (0, 10)),
            ("ssa", (0, 10)),
            ("g", (0, 10)),
            ("planck", (0, 20)),
            ("surface_planck", (0,)),
            ("emissivity", (0,)),
        )
        for name, index in checks:
            with self.subTest(name=name):
                autograd_value, finite_difference = _gradient_check(kwargs, name, index)
                np.testing.assert_allclose(
                    autograd_value,
                    finite_difference,
                    rtol=5.0e-4,
                    atol=1.0e-10,
                )
        self.assertTrue(torch.is_grad_enabled())

    @unittest.skipUnless(has_torch(), "PyTorch is required")
    def test_direct_rt_component_gradients_match_finite_difference(self) -> None:
        scene = load_scene(
            profile=CASE / "profile.csv",
            config=CASE / "scene.yaml",
            spectral_limit=2,
            strict_runtime_inputs=True,
        )
        kwargs = scene.to_forward_inputs().kwargs
        checks = (
            ("tau", (0, 10)),
            ("ssa", (0, 10)),
            ("g", (0, 10)),
            ("planck", (0, 20)),
            ("surface_planck", (0,)),
            ("emissivity", (0,)),
        )
        for component in ("2s", "fo", "total"):
            for name, index in checks:
                with self.subTest(component=component, name=name, index=index):
                    autograd_value, finite_difference = _gradient_check(
                        kwargs, name, index, component=component
                    )
                    np.testing.assert_allclose(
                        autograd_value,
                        finite_difference,
                        rtol=8.0e-4,
                        atol=1.0e-10,
                    )

    @unittest.skipUnless(has_torch(), "PyTorch is required")
    def test_thermal_scattering_gradients_match_finite_difference_at_multiple_points(
        self,
    ) -> None:
        scene = load_scene(
            profile=CASE / "profile.csv",
            config=CASE / "scene.yaml",
            spectral_limit=3,
            strict_runtime_inputs=True,
        )
        kwargs = scene.to_forward_inputs().kwargs
        checks = (
            ("ssa", (0, 0)),
            ("ssa", (0, 10)),
            ("ssa", (1, 25)),
            ("ssa", (2, 80)),
            ("g", (0, 10)),
            ("g", (1, 25)),
            ("g", (2, 80)),
        )
        for name, index in checks:
            with self.subTest(name=name, index=index):
                autograd_value, finite_difference = _gradient_check(kwargs, name, index)
                np.testing.assert_allclose(
                    autograd_value,
                    finite_difference,
                    rtol=2.0e-4,
                    atol=1.0e-11,
                )


def _gradient_check(
    kwargs: dict[str, np.ndarray],
    name: str,
    index: tuple[int, ...],
    *,
    component: str = "total",
):
    import torch

    tensors = {
        key: torch.tensor(np.asarray(kwargs[key]), dtype=torch.float64)
        for key in (
            "tau",
            "ssa",
            "g",
            "delta_m_truncation_factor",
            "planck",
            "surface_planck",
            "emissivity",
        )
    }
    tensors[name] = tensors[name].detach().clone().requires_grad_(True)
    total = _thermal_sum(kwargs, tensors, component=component)
    if total.requires_grad:
        total.backward()
        grad = tensors[name].grad
        autograd_value = 0.0 if grad is None else float(grad[index])
    else:
        autograd_value = 0.0

    base = torch.tensor(np.asarray(kwargs[name]), dtype=torch.float64)
    step = _finite_difference_step(name, float(base[index]))
    plus = {key: value.detach().clone() for key, value in tensors.items()}
    minus = {key: value.detach().clone() for key, value in tensors.items()}
    plus[name][index] += step
    minus[name][index] -= step
    plus[name].requires_grad_(True)
    minus[name].requires_grad_(True)
    finite_difference = float(
        (
            _thermal_sum(kwargs, plus, component=component)
            - _thermal_sum(kwargs, minus, component=component)
        )
        / (2 * step)
    )
    return autograd_value, finite_difference


def _finite_difference_step(name: str, value: float) -> float:
    if name in {"ssa", "g"}:
        return max(1.0e-8, 0.1 * max(abs(value), 1.0e-12))
    return 1.0e-6 * max(abs(value), 1.0)


def _thermal_sum(
    kwargs: dict[str, np.ndarray],
    tensors: dict,
    *,
    component: str = "total",
):
    solver = TwoStreamEss(
        TwoStreamEssOptions(
            nlyr=tensors["tau"].shape[-1],
            mode="thermal",
            backend="torch",
            torch_dtype="float64",
            torch_enable_grad=True,
        )
    )
    emissivity = tensors["emissivity"]
    result = solver.forward(
        tau=tensors["tau"],
        ssa=tensors["ssa"],
        g=tensors["g"],
        z=kwargs["z"],
        angles=kwargs["angles"],
        planck=tensors["planck"],
        surface_planck=tensors["surface_planck"],
        emissivity=emissivity,
        albedo=1.0 - emissivity,
        delta_m_truncation_factor=tensors["delta_m_truncation_factor"],
        stream=kwargs.get("stream"),
        include_fo=True,
    )
    if component == "2s":
        return result.radiance_2s.sum()
    if component == "fo":
        return result.radiance_fo.sum()
    if component == "total":
        return result.radiance_total.sum()
    raise ValueError(f"unknown component {component!r}")


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
EXPERIMENTS_DIR = SRC_DIR / "experiments"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR))
import bc_multiprocess_benchmark as bench  # noqa: E402


class FakeParams:
    def __init__(self, marker):
        self.marker = marker

    def to_dict(self):
        return {"marker": self.marker}


def make_ds(values, var_names, coords):
    return xr.Dataset(
        {name: (("year", "month", "day", "lat", "lon"), values[i]) for i, name in enumerate(var_names)},
        coords=coords,
    )


class BenchmarkHarnessTests(unittest.TestCase):
    """Fast, synthetic-data tests of the benchmark's own comparison logic.

    These do not exercise the real Fortran correction or a real multi-worker
    Client -- they only guard the harness's pass/fail reporting against
    regressions. See bc_multiprocess_benchmark.py's --synthetic-self-test /
    real-data modes for the actual investigation.
    """

    def setUp(self):
        self.var_names = ["hus", "ta"]
        self.coords = {
            "year": [2000],
            "month": [1],
            "day": [1, 2],
            "lat": [-10.0, 10.0],
            "lon": [100.0, 110.0],
        }
        shape = (1, 1, 2, 2, 2)
        rng = np.random.default_rng(0)
        self.reference_values = [rng.random(shape).astype(np.float32) for _ in self.var_names]

    def test_identical_candidate_passes_every_check(self):
        reference_ds = make_ds(self.reference_values, self.var_names, self.coords)
        candidate_ds = make_ds([v.copy() for v in self.reference_values], self.var_names, self.coords)
        params = np.full((2, 2), None, dtype=object)
        for idx in np.ndindex(params.shape):
            params[idx] = FakeParams(1.0).to_dict()

        checks = bench.compare_to_reference(
            reference_ds, params, candidate_ds, params, self.var_names, lower_limit=None, tolerance=1e-9
        )
        for name, (ok, detail) in checks.items():
            self.assertTrue(ok, f"{name} unexpectedly failed: {detail}")

    def test_shape_mismatch_is_detected(self):
        reference_ds = make_ds(self.reference_values, self.var_names, self.coords)
        bad_values = [v[:, :, :1, :, :] for v in self.reference_values]
        bad_coords = dict(self.coords)
        bad_coords["day"] = [1]
        candidate_ds = make_ds(bad_values, self.var_names, bad_coords)
        params = np.full((2, 2), None, dtype=object)

        checks = bench.compare_to_reference(
            reference_ds, params, candidate_ds, params, self.var_names, lower_limit=None, tolerance=1e-9
        )
        self.assertFalse(checks["shapes_match"][0])

    def test_max_abs_diff_beyond_tolerance_fails(self):
        reference_ds = make_ds(self.reference_values, self.var_names, self.coords)
        perturbed = [v.copy() for v in self.reference_values]
        perturbed[0][0, 0, 0, 0, 0] += 5.0
        candidate_ds = make_ds(perturbed, self.var_names, self.coords)
        params = np.full((2, 2), None, dtype=object)

        checks = bench.compare_to_reference(
            reference_ds, params, candidate_ds, params, self.var_names, lower_limit=None, tolerance=1e-6
        )
        self.assertFalse(checks["max_abs_diff"][0])

    def test_lower_bound_pin_count_detects_known_symptom(self):
        # Simulate the documented symptom: a whole column pinned to the
        # configured physical lower bound in the candidate but not the
        # reference.
        reference_values = [np.full((1, 1, 2, 2, 2), 5.0, dtype=np.float32) for _ in self.var_names]
        reference_ds = make_ds(reference_values, self.var_names, self.coords)

        candidate_values = [v.copy() for v in reference_values]
        candidate_values[0][0, 0, :, 0, 0] = 0.001  # one lat/lon column pinned to the lower bound
        candidate_ds = make_ds(candidate_values, self.var_names, self.coords)

        params = np.full((2, 2), None, dtype=object)
        checks = bench.compare_to_reference(
            reference_ds, params, candidate_ds, params,
            self.var_names, lower_limit=[0.001, 137], tolerance=1e-9,
        )
        self.assertFalse(checks["lower_bound_pin_count_matches"][0])

    def test_make_synthetic_tile_shapes_are_consistent(self):
        gcm_ds, obs_ds, var_names = bench.make_synthetic_tile(n_year=2, n_month=2, n_day=3, n_lat=4, n_lon=4)
        for name in var_names:
            self.assertEqual(gcm_ds[name].shape, (2, 2, 3, 4, 4))
            self.assertEqual(obs_ds[name].shape, (2, 2, 3, 4, 4))
        self.assertTrue(np.isfinite(gcm_ds[var_names[0]].values).all())


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dask
import numpy as np
import xarray as xr

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
import bc_grid_function


class FakeParams:
    def __init__(self, marker):
        self.marker = marker

    def to_dict(self):
        return {"marker": self.marker}


def make_dataset(values):
    return xr.Dataset(
        {
            name: (
                ("year", "month", "day", "lat", "lon"),
                values[index],
            )
            for index, name in enumerate(("w", "ta", "hus"))
        },
        coords={
            "year": [2000, 2001],
            "month": [1, 2],
            "day": [1, 2, 3],
            "lat": [-10.0, 10.0],
            "lon": [100.0, 110.0],
        },
    )


class GridCellExecutionTests(unittest.TestCase):
    def setUp(self):
        shape = (3, 2, 2, 3, 2, 2)
        self.gcm_values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        self.obs_values = self.gcm_values + 10.0
        self.config = SimpleNamespace(dask_cell_batch_size=2)

    def test_keeps_complete_time_history_per_horizontal_cell(self):
        seen_shapes = []

        def fake_correction(config, gcm_cell, obs_cell):
            seen_shapes.append((gcm_cell.shape, obs_cell.shape))
            return gcm_cell + obs_cell, FakeParams(float(gcm_cell[0, 0, 0, 0]))

        with dask.config.set(scheduler="synchronous"), patch.object(
            bc_grid_function, "correction_wrapper", side_effect=fake_correction
        ):
            corrected, params = bc_grid_function._correct_hist_grid_cells(
                self.config,
                make_dataset(self.gcm_values),
                make_dataset(self.obs_values),
                ["w", "ta", "hus"],
            )

        self.assertEqual(seen_shapes, [((3, 2, 2, 3), (3, 2, 2, 3))] * 4)
        expected = self.gcm_values + self.obs_values
        for index, name in enumerate(("w", "ta", "hus")):
            np.testing.assert_array_equal(corrected[name].values, expected[index])
        self.assertTrue(all(param is not None for param in params.flat))

    def test_rejects_non_finite_input_before_native_call(self):
        values = self.obs_values.copy()
        values[2, 0, 0, 0, 1, 1] = np.nan

        with self.assertRaisesRegex(ValueError, "Reference tile contains NaN"):
            bc_grid_function._correct_hist_grid_cells(
                self.config,
                make_dataset(self.gcm_values),
                make_dataset(values),
                ["w", "ta", "hus"],
            )

    def test_rejects_non_finite_native_output_with_cell_location(self):
        def fake_correction(config, gcm_cell, obs_cell):
            corrected = gcm_cell.copy()
            corrected[0, 0, 0, 0] = np.nan
            return corrected, FakeParams(0.0)

        with dask.config.set(scheduler="synchronous"), patch.object(
            bc_grid_function, "correction_wrapper", side_effect=fake_correction
        ), self.assertRaisesRegex(
            FloatingPointError, "lat index 0, lon index 0"
        ):
            bc_grid_function._correct_hist_grid_cells(
                self.config,
                make_dataset(self.gcm_values),
                make_dataset(self.obs_values),
                ["w", "ta", "hus"],
            )


class GridCellMultiWorkerDispatchTests(unittest.TestCase):
    """Guards the dispatch/aggregation logic under a real multi-worker Client.

    correction_wrapper is mocked (as in GridCellExecutionTests above) so this
    stays fast and does not require the real Fortran extension -- it does NOT
    reproduce the documented multi-worker corruption bug (that requires the
    real Fortran call; see experiments/bc_multiprocess_benchmark.py for the
    actual investigation). This only checks that dispatching the same work
    across 1 vs 2 real distributed workers still produces identical output,
    which would catch a regression in the dispatch/aggregation code itself.
    """

    def setUp(self):
        shape = (3, 2, 2, 3, 2, 2)
        self.gcm_values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        self.obs_values = self.gcm_values + 10.0
        self.config = SimpleNamespace(dask_cell_batch_size=2)

    def _run_with_client(self, n_workers):
        from dask.distributed import Client

        def fake_correction(config, gcm_cell, obs_cell):
            return gcm_cell + obs_cell, FakeParams(float(gcm_cell[0, 0, 0, 0]))

        client = Client(n_workers=n_workers, threads_per_worker=1, processes=True)
        try:
            with patch.object(bc_grid_function, "correction_wrapper", side_effect=fake_correction):
                return bc_grid_function._correct_hist_grid_cells(
                    self.config,
                    make_dataset(self.gcm_values),
                    make_dataset(self.obs_values),
                    ["w", "ta", "hus"],
                )
        finally:
            client.close()

    def test_one_vs_two_worker_dispatch_agree_with_mocked_correction(self):
        corrected_1, params_1 = self._run_with_client(1)
        corrected_2, params_2 = self._run_with_client(2)

        for name in ("w", "ta", "hus"):
            np.testing.assert_array_equal(corrected_1[name].values, corrected_2[name].values)
        self.assertEqual(list(params_1.flat), list(params_2.flat))


if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

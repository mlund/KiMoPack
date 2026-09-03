"""Replacing or removing corrupted points.

Detectors mark saturated or dead readings with absurd values — the Pascher
software writes 21 where a real signal is of order 1e-3 — so those have to be
replaced or their time points dropped before anything is fitted.
"""

import unittest

import numpy as np

import KiMoPack.plot_func as pf

from .synthetic import make_dataset


def _project_with_outliers():
    ds, _ = make_dataset()
    ds = ds.copy()
    ds.iloc[5, 5] = 999.0
    ds.iloc[7, 7] = -999.0
    return pf.TA("synthetic", ds=ds), ds


class ReplacingBadValues(unittest.TestCase):
    def test_outliers_are_replaced(self):
        ta, _ = _project_with_outliers()
        ta.Filter_data(value=10)
        self.assertLess(float(np.abs(ta.ds.values).max()), 10.0)

    def test_both_the_working_and_the_master_copy_are_filtered(self):
        """ds_ori is the reference every later correction is rebuilt from."""
        ta, _ = _project_with_outliers()
        ta.Filter_data(value=10)
        self.assertLess(float(np.abs(ta.ds_ori.values).max()), 10.0)

    def test_good_data_is_left_alone(self):
        ta, original = _project_with_outliers()
        ta.Filter_data(value=10)
        untouched = ta.ds.iloc[20:30, 20:30].values
        self.assertTrue(np.allclose(untouched, original.iloc[20:30, 20:30].values))

    def test_the_replacement_value_is_used(self):
        ta, _ = _project_with_outliers()
        ta.Filter_data(value=10, replace_bad_values=-1.0)
        self.assertAlmostEqual(float(ta.ds.iloc[5, 5]), -1.0)

    def test_no_replacement_value_means_nan(self):
        ta, _ = _project_with_outliers()
        ta.Filter_data(value=10, replace_bad_values=None)
        self.assertTrue(np.isnan(ta.ds.iloc[5, 5]))

    def test_separate_upper_and_lower_bounds(self):
        ta, _ = _project_with_outliers()
        ta.Filter_data(uppervalue=10, lowervalue=-2000)
        self.assertAlmostEqual(float(ta.ds.iloc[5, 5]), 0.0)
        self.assertAlmostEqual(float(ta.ds.iloc[7, 7]), -999.0,
                               msg="a value inside the lower bound should survive")


class FilteringWithCropsConfigured(unittest.TestCase):
    """Filtering used to do nothing at all once any crop was set.

    The cropped frame was assigned over the loop variable, so every later
    change landed on a copy that was discarded — and the docs recommend
    setting a bordercut before filtering.
    """

    def test_filtering_still_happens_with_a_bordercut(self):
        ta, _ = _project_with_outliers()
        ta.bordercut = [420, 680]
        ta.Filter_data(value=10)
        self.assertLess(float(np.abs(ta.ds.values).max()), 10.0)

    def test_filtering_still_happens_with_timelimits(self):
        ta, _ = _project_with_outliers()
        ta.timelimits = [-5, 900]
        ta.Filter_data(value=10)
        self.assertLess(float(np.abs(ta.ds.values).max()), 10.0)


class DroppingBadTimePoints(unittest.TestCase):
    def test_damaged_time_points_are_removed(self):
        ta, original = _project_with_outliers()
        before = ta.ds.shape[0]
        ta.Filter_data(value=10, cut_bad_times=True, replace_bad_values=None)
        self.assertLess(ta.ds.shape[0], before)

    def test_the_surviving_times_keep_their_data(self):
        ta, original = _project_with_outliers()
        ta.Filter_data(value=10, cut_bad_times=True, replace_bad_values=None)
        self.assertNotIn(original.index[5], list(ta.ds.index))
        self.assertIn(original.index[50], list(ta.ds.index))


class FilteringAFrameDirectly(unittest.TestCase):
    def test_a_supplied_frame_is_returned_filtered(self):
        ta, ds = _project_with_outliers()
        result = ta.Filter_data(ds=ds, value=10)
        self.assertLess(float(np.abs(result.values).max()), 10.0)

    def test_a_supplied_frame_leaves_the_project_alone(self):
        ta, ds = _project_with_outliers()
        ta.Filter_data(ds=ds, value=10)
        self.assertGreater(float(np.abs(ta.ds.values).max()), 100.0)

    def test_a_read_only_frame_can_be_filtered(self):
        """Frames read back from HDF5 do not own their data."""
        ta, ds = _project_with_outliers()
        frozen = ds.copy()
        frozen.values.flags.writeable = False
        result = ta.Filter_data(ds=frozen, value=10)
        self.assertLess(float(np.abs(result.values).max()), 10.0)

"""Cropping, binning and slicing the measurement matrix.

``sub_ds`` is the single path every plot and the fit take to get from the full
matrix to the part they want, so its behaviour is what "the data" means
everywhere downstream.
"""

import numpy as np
import pandas

from KiMoPack import shaping

from .support import NumericTestCase
from .synthetic import make_dataset


class DoesNotMutateTheCaller(NumericTestCase):
    """Every option, checked against the input frame it was handed.

    Cropping used to reach back into the caller's DataFrame, so plotting a
    slice silently changed the data the next fit would use.
    """

    def setUp(self):
        self.ds, _ = make_dataset()

    def _unchanged(self, **kwargs):
        before = self.ds.copy()
        shaping.sub_ds(self.ds, **kwargs)
        self.assertUnchanged(self.ds, before, msg=f"sub_ds({kwargs}) altered its input")

    def test_bordercut(self):
        self._unchanged(bordercut=[450, 650])

    def test_timelimits(self):
        self._unchanged(timelimits=[0.5, 100])

    def test_wave_nm_bin(self):
        self._unchanged(wave_nm_bin=20)

    def test_scattercut(self):
        self._unchanged(scattercut=[500, 520])

    def test_ignore_time_region(self):
        self._unchanged(ignore_time_region=[1, 5])

    def test_no_options_at_all(self):
        self._unchanged()

    def test_single_wavelength(self):
        self._unchanged(wavelength=500, wavelength_bin=10)

    def test_single_time(self):
        self._unchanged(times=[1.0, 10.0])


class Cropping(NumericTestCase):
    def setUp(self):
        self.ds, _ = make_dataset()

    def test_bordercut_keeps_only_the_named_range(self):
        out = shaping.sub_ds(self.ds, bordercut=[450, 650])
        self.assertGreaterEqual(float(out.columns.min()), 450.0)
        self.assertLessEqual(float(out.columns.max()), 650.0)

    def test_timelimits_keep_only_the_named_range(self):
        out = shaping.sub_ds(self.ds, timelimits=[0.5, 100])
        self.assertGreaterEqual(float(out.index.min()), 0.5)
        self.assertLessEqual(float(out.index.max()), 100.0)

    def test_axis_names_survive(self):
        out = shaping.sub_ds(self.ds, bordercut=[450, 650], timelimits=[0.5, 100])
        self.assertEqual(out.index.name, self.ds.index.name)
        self.assertEqual(out.columns.name, self.ds.columns.name)


class Binning(NumericTestCase):
    def setUp(self):
        self.ds, _ = make_dataset()

    def test_wavelength_binning_reduces_the_column_count(self):
        out = shaping.sub_ds(self.ds, wave_nm_bin=25)
        self.assertLess(out.shape[1], self.ds.shape[1])
        self.assertEqual(out.shape[0], self.ds.shape[0])

    def test_binning_preserves_the_mean_level(self):
        """Averaging into wider bins must not change the overall magnitude."""
        out = shaping.sub_ds(self.ds, wave_nm_bin=25)
        self.assertAlmostEqual(float(out.values.mean()), float(self.ds.values.mean()), places=2)

    def test_a_bin_narrower_than_the_data_is_rejected(self):
        with self.assertRaises(ValueError):
            shaping.sub_ds(self.ds, wave_nm_bin=0.001)

    def test_time_binning_reduces_the_row_count(self):
        out = shaping.sub_ds(self.ds, time_bin=4)
        self.assertLess(out.shape[0], self.ds.shape[0])


class Masking(NumericTestCase):
    def setUp(self):
        self.ds, _ = make_dataset()

    def test_scattercut_zeroes_the_named_region(self):
        out = shaping.sub_ds(self.ds, scattercut=[500, 540])
        masked = out.loc[:, 505:535]
        self.assertAllClose(masked.values, np.zeros_like(masked.values))

    def test_scattercut_can_drop_instead_of_zero(self):
        out = shaping.sub_ds(self.ds, scattercut=[500, 540], drop_scatter=True)
        self.assertFalse(((out.columns > 505) & (out.columns < 535)).any())

    def test_several_scatter_regions_are_all_masked(self):
        out = shaping.sub_ds(self.ds, scattercut=[[450, 470], [600, 620]])
        for low, high in [(455, 465), (605, 615)]:
            with self.subTest(region=(low, high)):
                masked = out.loc[:, low:high]
                self.assertAllClose(masked.values, np.zeros_like(masked.values))

    def test_ignore_time_region_zeroes_those_rows(self):
        out = shaping.sub_ds(self.ds, ignore_time_region=[1, 5])
        masked = out.loc[1.1:4.9, :]
        self.assertAllClose(masked.values, np.zeros_like(masked.values))

    def test_ignore_time_region_can_drop_instead(self):
        out = shaping.sub_ds(self.ds, ignore_time_region=[1, 5], drop_ignore=True)
        self.assertFalse(((out.index > 1.05) & (out.index < 4.95)).any())


class Slicing(NumericTestCase):
    def setUp(self):
        self.ds, _ = make_dataset()

    def test_a_single_wavelength_becomes_one_column(self):
        out = shaping.sub_ds(self.ds, wavelength=500, wavelength_bin=10)
        self.assertEqual(out.shape[1], 1)
        self.assertEqual(out.shape[0], self.ds.shape[0])

    def test_several_wavelengths_become_several_columns(self):
        out = shaping.sub_ds(self.ds, wavelength=[450, 500, 600], wavelength_bin=10)
        self.assertEqual(out.shape[1], 3)

    def test_a_sliced_wavelength_averages_its_window(self):
        out = shaping.sub_ds(self.ds, wavelength=[500], wavelength_bin=20)
        expected = self.ds.loc[:, 490:510].mean(axis="columns")
        self.assertAllClose(out.values.ravel(), expected.values)

    def test_times_become_columns_of_spectra(self):
        out = shaping.sub_ds(self.ds, times=[1.0, 10.0])
        self.assertEqual(out.shape[1], 2)
        self.assertEqual(out.shape[0], self.ds.shape[1])

    def test_asking_for_both_axes_at_once_is_refused(self):
        with self.assertRaises(ValueError):
            shaping.sub_ds(self.ds, times=[1.0], wavelength=[500])


class EqualEnergyBinning(NumericTestCase):
    def test_columns_become_electronvolts_in_descending_order(self):
        ds, _ = make_dataset()
        out = shaping.sub_ds(ds, equal_energy_bin=0.05)
        self.assertEqual(out.columns.name, "Energy in eV")
        self.assertTrue((np.diff(out.columns.values) < 0).all(),
                        "energy axis should run high to low, matching wavelength order")


class ReturnsAFrame(NumericTestCase):
    def test_nan_values_are_filled(self):
        ds, _ = make_dataset()
        ds = ds.copy()
        ds.iloc[3, 4] = np.nan
        out = shaping.sub_ds(ds)
        self.assertFalse(pandas.isna(out.values).any())

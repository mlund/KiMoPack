"""The crop and binning settings, carried as one value.

The same eight or so settings are threaded through every plot and the fit.
Grouping them means a caller states them once and they cannot drift apart
between the data that is fitted and the data that is drawn.
"""

import dataclasses
import unittest

from KiMoPack.shaping import DataSelection, sub_ds

from .support import NumericTestCase
from .synthetic import make_dataset


class Defaults(unittest.TestCase):
    def test_an_empty_selection_changes_nothing(self):
        ds, _ = make_dataset()
        result = DataSelection().apply(ds)
        self.assertEqual(result.shape, ds.shape)

    def test_it_is_immutable(self):
        """Settings are shared between callers, so nobody may edit one in place."""
        selection = DataSelection(bordercut=[420, 680])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            selection.bordercut = [400, 700]

    def test_replacing_a_field_gives_a_new_value(self):
        original = DataSelection(bordercut=[420, 680])
        wider = original.replace(bordercut=[400, 700])
        self.assertEqual(original.bordercut, [420, 680])
        self.assertEqual(wider.bordercut, [400, 700])


class MatchesSubDs(NumericTestCase):
    """apply() must be exactly the call it replaces."""

    def setUp(self):
        self.ds, _ = make_dataset()

    def _same(self, selection, **kwargs):
        self.assertFrameAllClose(selection.apply(self.ds, **kwargs),
                                 sub_ds(self.ds, **{**selection.as_kwargs(), **kwargs}))

    def test_cropping(self):
        self._same(DataSelection(bordercut=[420, 680], timelimits=[0.5, 100]))

    def test_binning(self):
        self._same(DataSelection(wave_nm_bin=20, time_bin=3))

    def test_masking(self):
        self._same(DataSelection(scattercut=[[450, 470], [600, 620]], ignore_time_region=[1, 5]))

    def test_extracting_kinetics(self):
        self._same(DataSelection(bordercut=[420, 680], wavelength_bin=10), wavelength=[500, 600])

    def test_extracting_spectra(self):
        self._same(DataSelection(bordercut=[420, 680], baseunit="ps"),
                   times=[1.0, 10.0], time_width_percent=10)

    def test_energy_binning(self):
        self._same(DataSelection(equal_energy_bin=0.05))


class BuiltFromAProject(NumericTestCase):
    def test_it_reads_the_settings_off_a_TA_object(self):
        import KiMoPack.plot_func as pf

        ds, _ = make_dataset()
        ta = pf.TA("synthetic", ds=ds)
        ta.bordercut = [420, 680]
        ta.timelimits = [0.5, 100]
        ta.wave_nm_bin = 20
        ta.scattercut = [500, 520]
        selection = DataSelection.from_project(ta)
        self.assertEqual(selection.bordercut, [420, 680])
        self.assertEqual(selection.timelimits, [0.5, 100])
        self.assertEqual(selection.wave_nm_bin, 20)
        self.assertEqual(selection.scattercut, [500, 520])

    def test_the_result_matches_what_the_project_would_produce(self):
        import KiMoPack.plot_func as pf

        ds, _ = make_dataset()
        ta = pf.TA("synthetic", ds=ds)
        ta.bordercut = [420, 680]
        ta.wave_nm_bin = 20
        self.assertFrameAllClose(
            DataSelection.from_project(ta).apply(ta.ds),
            sub_ds(ta.ds, bordercut=ta.bordercut, wave_nm_bin=ta.wave_nm_bin,
                   timelimits=ta.timelimits, scattercut=ta.scattercut,
                   ignore_time_region=ta.ignore_time_region, time_bin=ta.time_bin,
                   equal_energy_bin=ta.equal_energy_bin, wavelength_bin=ta.wavelength_bin,
                   baseunit=ta.baseunit))

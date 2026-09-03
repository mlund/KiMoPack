"""The TA object's settings surface.

Users configure an analysis by assigning attributes — ``ta.timelimits``,
``ta.mod``, ``ta.bordercut`` — rather than by passing arguments, because a
session is iterative: crop, re-plot, adjust, re-plot. That makes assignment
the main interface, so a mistyped or misshapen assignment has to be caught at
the point it is made rather than surfacing as a strange plot much later.
"""

import unittest

import KiMoPack.plot_func as pf

from .synthetic import make_dataset, make_sparse_dataset


def _project(ds=None):
    if ds is None:
        ds, _ = make_dataset()
    return pf.TA("synthetic", ds=ds)


class KnownSettings(unittest.TestCase):
    def test_ordinary_assignment_works(self):
        ta = _project()
        ta.timelimits = [0.5, 100]
        ta.bordercut = [420, 680]
        ta.wave_nm_bin = 10
        self.assertEqual(ta.timelimits, [0.5, 100])
        self.assertEqual(ta.wave_nm_bin, 10)

    def test_results_and_fit_settings_can_be_assigned(self):
        ta = _project()
        ta.par = None
        ta.mod = "consecutive"
        ta.log_fit = True
        self.assertEqual(ta.mod, "consecutive")


class RejectsTypos(unittest.TestCase):
    """A misspelled setting used to do nothing at all, silently."""

    def test_a_misspelled_setting_is_refused(self):
        ta = _project()
        with self.assertRaises(AttributeError):
            ta.timelimit = [0.5, 100]

    def test_the_message_suggests_the_real_name(self):
        ta = _project()
        with self.assertRaises(AttributeError) as caught:
            ta.timelimit = [0.5, 100]
        self.assertIn("timelimits", str(caught.exception))

    def test_other_near_misses_are_caught_too(self):
        ta = _project()
        for wrong, right in [("bordercuts", "bordercut"), ("scatter_cut", "scattercut"),
                             ("rel_waves", "rel_wave"), ("intensity_ranges", "intensity_range")]:
            with self.subTest(wrong=wrong):
                with self.assertRaises(AttributeError) as caught:
                    setattr(ta, wrong, [1, 2])
                self.assertIn(right, str(caught.exception))


class ValidatesShape(unittest.TestCase):
    def test_a_range_needs_two_bounds(self):
        ta = _project()
        with self.assertRaises(ValueError) as caught:
            ta.bordercut = 350
        self.assertIn("bordercut", str(caught.exception))

    def test_a_reversed_range_is_refused(self):
        """Slicing high-to-low silently returns nothing."""
        ta = _project()
        with self.assertRaises(ValueError):
            ta.timelimits = [100, 1]

    def test_region_lists_may_be_a_pair_or_a_list_of_pairs(self):
        ta = _project()
        ta.scattercut = [500, 520]
        ta.scattercut = [[450, 470], [600, 620]]
        ta.scattercut = None

    def test_an_unknown_model_lists_the_available_ones(self):
        ta = _project()
        with self.assertRaises(ValueError) as caught:
            ta.mod = "nonsense"
        message = str(caught.exception)
        self.assertIn("nonsense", message)
        self.assertIn("consecutive", message)

    def test_a_user_model_function_is_accepted(self):
        ta = _project()

        def my_model(times, pardf):
            raise NotImplementedError

        ta.mod = my_model
        self.assertIs(ta.mod, my_model)


class Defaults(unittest.TestCase):
    def test_a_user_setting_survives_re_derivation(self):
        """error_matrix_amplification was reset every time, from a typo in hasattr."""
        ta = _project()
        ta.error_matrix_amplification = 3
        ta._TA__make_standard_parameter()
        self.assertEqual(ta.error_matrix_amplification, 3)

    def test_sparse_data_defaults_to_the_measured_wavelengths(self):
        """Only eight channels exist, so the standard 300-1000 grid is useless."""
        ds, _ = make_sparse_dataset()
        ta = _project(ds)
        self.assertEqual(sorted(ta.rel_wave), sorted(float(c) for c in ds.columns))
        self.assertEqual(ta.wavelength_bin, 0.0)

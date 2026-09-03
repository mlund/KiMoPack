"""The parameter table that sits between lmfit and the kinetic models.

Which parameters are rate constants is decided by their *name*, and both the
models and the results tables depend on that convention, so it is pinned here.
"""

import lmfit
import pandas

from KiMoPack.kinetics import parameters

from ..support import NumericTestCase


def _example_parameters():
    par = lmfit.Parameters()
    par.add("k0", value=0.5, min=0.0, max=10.0, vary=True)
    par.add("k1", value=0.02, min=0.0, max=1.0, vary=True)
    par.add("t0", value=0.1, min=-0.5, max=0.5, vary=False)
    par.add("resolution", value=0.086, min=0.04, max=0.5, vary=False)
    return par


class RateNaming(NumericTestCase):
    def test_k_prefixed_names_are_rates(self):
        for name in ["k0", "k1", "k12"]:
            with self.subTest(name=name):
                self.assertTrue(parameters.is_rate(name))

    def test_tk_prefixed_names_are_rates(self):
        self.assertTrue(parameters.is_rate("tk0"))

    def test_the_structural_parameters_are_not_rates(self):
        for name in ["t0", "resolution", "background", "infinite", "explicit_GS",
                     "sub_steps", "ext_spectra_shift"]:
            with self.subTest(name=name):
                self.assertFalse(parameters.is_rate(name))


class ParToPardf(NumericTestCase):
    def test_carries_every_bound_across(self):
        pardf = parameters.par_to_pardf(_example_parameters())
        self.assertEqual(list(pardf.index), ["k0", "k1", "t0", "resolution"])
        self.assertEqual(pardf.loc["k0", "value"], 0.5)
        self.assertEqual(pardf.loc["k0", "max"], 10.0)
        self.assertEqual(pardf.loc["t0", "vary"], False)

    def test_marks_the_rates(self):
        pardf = parameters.par_to_pardf(_example_parameters())
        self.assertEqual(list(pardf["is_rate"]), [True, True, False, False])

    def test_round_trips_through_lmfit(self):
        original = _example_parameters()
        restored = parameters.pardf_to_par(parameters.par_to_pardf(original))
        self.assertEqual(set(restored.keys()), set(original.keys()))
        for key in original:
            with self.subTest(key=key):
                self.assertAlmostEqual(restored[key].value, original[key].value)
                self.assertEqual(restored[key].vary, original[key].vary)
                self.assertAlmostEqual(restored[key].min, original[key].min)
                self.assertAlmostEqual(restored[key].max, original[key].max)


class PardfToTimedf(NumericTestCase):
    def test_rates_become_lifetimes(self):
        pardf = parameters.par_to_pardf(_example_parameters())
        timedf = parameters.pardf_to_timedf(pardf)
        self.assertAlmostEqual(timedf.loc["k0", "value"], 1 / 0.5)
        self.assertAlmostEqual(timedf.loc["k1", "value"], 1 / 0.02)

    def test_bounds_swap_because_inversion_reverses_order(self):
        """A fast rate is a short time: the upper rate bound is the lower time bound."""
        pardf = parameters.par_to_pardf(_example_parameters())
        timedf = parameters.pardf_to_timedf(pardf)
        self.assertAlmostEqual(timedf.loc["k0", "min"], 1 / 10.0)

    def test_non_rates_pass_through_untouched(self):
        pardf = parameters.par_to_pardf(_example_parameters())
        timedf = parameters.pardf_to_timedf(pardf)
        self.assertAlmostEqual(timedf.loc["t0", "value"], 0.1)
        self.assertAlmostEqual(timedf.loc["resolution", "value"], 0.086)

    def test_a_zero_rate_becomes_an_infinite_time(self):
        par = lmfit.Parameters()
        par.add("k0", value=0.0, min=0.0, max=1.0)
        timedf = parameters.pardf_to_timedf(parameters.par_to_pardf(par))
        self.assertEqual(timedf.loc["k0", "value"], "inf")

    def test_leaves_the_input_table_alone(self):
        pardf = parameters.par_to_pardf(_example_parameters())
        before = pardf.copy()
        parameters.pardf_to_timedf(pardf)
        pandas.testing.assert_frame_equal(pardf, before)

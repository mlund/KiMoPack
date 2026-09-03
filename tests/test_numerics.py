"""Behaviour of the numeric helpers, checked against closed-form answers."""

import numpy as np
import pandas
from scipy.special import erf

from KiMoPack import numerics

from .support import NumericTestCase


class FindNearest(NumericTestCase):
    def test_returns_closest_value(self):
        grid = np.array([100.0, 200.0, 300.0])
        self.assertEqual(numerics.find_nearest(grid, 190.0), 200.0)
        self.assertEqual(numerics.find_nearest_index(grid, 190.0), 1)

    def test_ties_take_the_first(self):
        self.assertEqual(numerics.find_nearest_index(np.array([0.0, 10.0]), 5.0), 0)

    def test_string_grids_compare_numerically(self):
        """con_str converts before comparing.

        Wavelength axes read from ASCII arrive as strings; lexicographic
        comparison would put '1000' next to '100'.
        """
        grid = np.array(["100", "200", "1000"])
        self.assertEqual(numerics.find_nearest_index(grid, 900.0, con_str=True), 2)
        self.assertEqual(numerics.find_nearest(grid, 900.0, con_str=True), "1000")


class Rebin(NumericTestCase):
    def test_interpolates_a_frame_onto_a_new_index(self):
        original = pandas.DataFrame({"a": [0.0, 2.0], "b": [1.0, 3.0]}, index=[0.0, 2.0])
        result = numerics.rebin(original, np.array([0.0, 1.0, 2.0]))
        self.assertAllClose(result["a"].values, [0.0, 1.0, 2.0])
        self.assertAllClose(result["b"].values, [1.0, 2.0, 3.0])

    def test_keeps_column_order_and_names(self):
        original = pandas.DataFrame({"b": [1.0, 3.0], "a": [0.0, 2.0]}, index=[0.0, 2.0])
        result = numerics.rebin(original, np.array([0.0, 2.0]))
        self.assertEqual(list(result.columns), ["b", "a"])

    def test_a_linear_ramp_survives_resampling(self):
        x = np.linspace(0.0, 10.0, 11)
        original = pandas.Series(3.0 * x + 1.0, index=x)
        finer = np.linspace(0.0, 10.0, 51)
        self.assertAllClose(numerics.rebin(original, finer).values, 3.0 * finer + 1.0)

    def test_rejects_types_it_cannot_handle(self):
        """Returning None here used to surface as an error much further away."""
        with self.assertRaises(TypeError):
            numerics.rebin([1, 2, 3], np.array([0.0, 1.0]))


class SavitzkyGolay(NumericTestCase):
    def test_reproduces_a_polynomial_of_equal_order(self):
        """A filter of order n leaves polynomials up to degree n untouched."""
        x = np.arange(40.0)
        y = 2.0 + 0.5 * x - 0.01 * x**2
        self.assertAllClose(numerics.savitzky_golay(y, 11, 3), y, rtol=1e-9, atol=1e-9)

    def test_suppresses_noise_without_shifting_the_signal(self):
        x = np.linspace(0.0, 4.0 * np.pi, 400)
        clean = np.sin(x)
        noisy = clean + np.random.default_rng(0).normal(0.0, 0.05, x.size)
        smoothed = numerics.savitzky_golay(noisy, 31, 3)
        self.assertLess(np.abs(smoothed - clean).mean(), np.abs(noisy - clean).mean())


class FrameGolay(NumericTestCase):
    def test_smooths_every_column(self):
        x = np.arange(40.0)
        frame = pandas.DataFrame({"a": 2.0 + x, "b": 5.0 - 0.5 * x}, index=x)
        result = numerics.Frame_golay(frame, window=11, order=3)
        self.assertFrameAllClose(result, frame, rtol=1e-9, atol=1e-9)

    def test_leaves_the_caller_frame_alone(self):
        """Smoothing in place silently altered data the caller still needed."""
        x = np.arange(40.0)
        frame = pandas.DataFrame({"a": np.sin(x / 5.0)}, index=x)
        before = frame.copy()
        numerics.Frame_golay(frame, window=11, order=3)
        self.assertUnchanged(frame, before)


class Shift(NumericTestCase):
    def test_resamples_values_as_if_the_data_had_moved(self):
        """The index stays put; the values are re-read from the shifted curve."""
        x = np.linspace(0.0, 10.0, 11)
        frame = pandas.DataFrame({"a": x}, index=x)
        result = numerics.shift(frame, name="a", shift=1.0)
        self.assertAllClose(result.index.values, x)
        # Interior points read f(x - 1); the ends clamp to the original range.
        self.assertAllClose(result["a"].values[2:-1], x[2:-1] - 1.0)

    def test_leaves_the_caller_frame_alone(self):
        frame = pandas.DataFrame({"a": [1.0, 2.0, 3.0]}, index=[0.0, 1.0, 2.0])
        before = frame.copy()
        numerics.shift(frame, name="a", shift=1.0)
        self.assertUnchanged(frame, before)


class Norm(NumericTestCase):
    def test_maps_each_column_onto_zero_to_one(self):
        frame = pandas.DataFrame({"a": [1.0, 3.0, 5.0], "b": [-2.0, 0.0, 2.0]})
        result = numerics.norm(frame)
        self.assertAllClose(result["a"].values, [0.0, 0.5, 1.0])
        self.assertAllClose(result["b"].values, [0.0, 0.5, 1.0])


class InstrumentResponse(NumericTestCase):
    def test_rise_is_half_height_one_sigma_after_the_start(self):
        """The documented definition: sigma is the width after which it is 50%."""
        self.assertAlmostEqual(float(numerics.rise(np.array([0.3]), sigma=0.3, begin=0.0)[0]), 0.5)

    def test_rise_climbs_from_zero_to_one(self):
        x = np.linspace(-5.0, 5.0, 501)
        y = numerics.rise(x, sigma=0.4, begin=0.0)
        self.assertAlmostEqual(float(y[0]), 0.0, places=6)
        self.assertAlmostEqual(float(y[-1]), 1.0, places=6)
        self.assertMonotonicDecreasing(-y, msg="the response must rise monotonically")

    def test_rise_matches_the_error_function_it_is_built_from(self):
        x = np.linspace(-2.0, 2.0, 101)
        expected = (erf((x - 0.0 - 0.25) * np.sqrt(2) / 0.25) + 1) / 2
        self.assertAllClose(numerics.rise(x, sigma=0.25), expected)

    def test_gauss_is_a_normalised_density(self):
        x = np.linspace(-10.0, 10.0, 20001)
        y = numerics.gauss(x, sigma=0.7, mu=1.0)
        self.assertAlmostEqual(np.trapezoid(y, x), 1.0, places=6)
        self.assertAlmostEqual(x[np.argmax(y)], 1.0, places=3)


class Flatten(NumericTestCase):
    def test_unpacks_one_level_of_nesting(self):
        self.assertEqual(numerics.flatten([[1, 2], [3], [4, 5]]), [1, 2, 3, 4, 5])


class LogAnd(NumericTestCase):
    def test_combines_any_number_of_masks(self):
        a = np.array([True, True, False, True])
        b = np.array([True, False, True, True])
        c = np.array([True, True, True, False])
        self.assertAllClose(numerics.log_and(a, b, c), [True, False, False, False])


class VarianceRatio(NumericTestCase):
    def test_ratio_exceeds_one_and_falls_with_more_data(self):
        """The F-test threshold tightens as degrees of freedom grow."""
        few = numerics.s2_vs_smin2(Spectral_points=64, Time_points=40)
        many = numerics.s2_vs_smin2(Spectral_points=64, Time_points=400)
        self.assertGreater(few, 1.0)
        self.assertGreater(many, 1.0)
        self.assertLess(many, few)

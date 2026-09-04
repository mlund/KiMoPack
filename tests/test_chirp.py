"""Finding and undoing the wavelength dependence of time zero.

In a white-light probe, blue light arrives at the sample after red, so each
wavelength has its own zero time. The offset follows a smooth polynomial in
wavelength, and correcting for it is a prerequisite for any global fit.
"""

import numpy as np

import KiMoPack.plot_func as pf
from KiMoPack import chirp

from .support import NumericTestCase
from .synthetic import convolved_decay, make_chirped_dataset


class ApplyChirp(NumericTestCase):
    def test_zero_coefficients_change_nothing(self):
        ds, _ = make_chirped_dataset(coeffs=(0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertAllClose(chirp.apply_chirp(ds, [0, 0, 0, 0, 0]).values, ds.values)

    def test_a_constant_offset_shifts_every_column_alike(self):
        ds, _ = make_chirped_dataset(coeffs=(0.0, 0.0, 0.0, 0.0, 0.0))
        shifted = chirp.apply_chirp(ds, [0, 0, 0, 0, 2.0])
        # Undoing a +2 shift with a -2 shift returns the original, except at
        # the edges where interpolation has run out of data.
        back = chirp.apply_chirp(shifted, [0, 0, 0, 0, -2.0])
        interior = (ds.index.values > -2.0) & (ds.index.values < 40.0)
        self.assertAllClose(back.values[interior], ds.values[interior], atol=2e-2)

    def test_the_axes_are_preserved(self):
        ds, coeffs = make_chirped_dataset()
        out = chirp.apply_chirp(ds, coeffs)
        self.assertEqual(out.index.name, ds.index.name)
        self.assertEqual(out.columns.name, ds.columns.name)
        self.assertAllClose(out.columns.values, ds.columns.values)

    def test_the_input_is_not_modified(self):
        ds, coeffs = make_chirped_dataset()
        before = ds.copy()
        chirp.apply_chirp(ds, coeffs)
        self.assertUnchanged(ds, before)


class OnsetDetection(NumericTestCase):
    """Each detector locates the rise of a single kinetic trace."""

    def setUp(self):
        self.times = np.linspace(-5.0, 20.0, 500)
        self.t0 = 1.3
        self.signal = convolved_decay(self.times, tau=1e6, t0=self.t0, resolution=0.5)

    def test_the_sigmoid_fit_finds_the_onset(self):
        self.assertAlmostEqual(chirp.fit_sigmoid_onset(self.times, self.signal), self.t0, places=1)

    def test_the_threshold_crossing_finds_the_half_height(self):
        self.assertAlmostEqual(chirp.find_threshold_crossing(self.times, self.signal),
                               self.t0, places=1)

    def test_the_steepest_slope_is_at_the_onset(self):
        self.assertAlmostEqual(chirp.find_max_derivative(self.times, self.signal),
                               self.t0, places=1)

    def test_a_flat_trace_has_no_onset(self):
        flat = np.ones_like(self.times)
        self.assertIsNone(chirp.fit_sigmoid_onset(self.times, flat))
        self.assertIsNone(chirp.find_threshold_crossing(self.times, flat))


class FindChirpSparse(NumericTestCase):
    def test_recovers_a_known_polynomial(self):
        coeffs = (0.0, 0.0, 0.0, -4e-3, 2.2)
        ds, expected = make_chirped_dataset(coeffs=coeffs)
        _, fitcoeff, t0_values = chirp.find_chirp_sparse(ds, t_range=(-4, 6), plot=False)
        recovered = [np.polyval(fitcoeff, w) for w in ds.columns.values.astype(float)]
        wanted = [np.polyval(expected, w) for w in ds.columns.values.astype(float)]
        self.assertAllClose(recovered, wanted, atol=0.15)

    def test_finds_an_onset_for_every_channel(self):
        ds, _ = make_chirped_dataset()
        _, _, t0_values = chirp.find_chirp_sparse(ds, t_range=(-4, 6), plot=False)
        self.assertEqual(len(t0_values), ds.shape[1])

    def test_the_correction_flattens_the_onset_across_wavelength(self):
        ds, _ = make_chirped_dataset(coeffs=(0.0, 0.0, 0.0, -4e-3, 2.2))
        corrected, _, _ = chirp.find_chirp_sparse(ds, t_range=(-4, 6), plot=False)
        _, _, residual = chirp.find_chirp_sparse(corrected, t_range=(-4, 6), plot=False)
        spread = np.ptp(list(residual.values()))
        self.assertLess(spread, 0.4, "onsets should line up after correction")

    def test_coefficients_are_returned_in_the_five_slot_form(self):
        """The rest of the package stores chirp as five polynomial coefficients."""
        ds, _ = make_chirped_dataset()
        _, fitcoeff, _ = chirp.find_chirp_sparse(ds, t_range=(-4, 6), plot=False)
        self.assertEqual(len(fitcoeff), 5)

    def test_an_unknown_method_is_rejected(self):
        ds, _ = make_chirped_dataset()
        with self.assertRaises(ValueError):
            chirp.find_chirp_sparse(ds, method="guesswork", plot=False)

    def test_a_flat_dataset_falls_back_to_a_constant(self):
        """With no detectable onset anywhere there is nothing to fit a curve to."""
        ds, _ = make_chirped_dataset()
        flat = ds.copy()
        flat.loc[:, :] = 1.0
        _, fitcoeff, t0_values = chirp.find_chirp_sparse(flat, plot=False)
        self.assertEqual(len(t0_values), 0)
        self.assertEqual(len(fitcoeff), 5)


class ReplayingAStoredCorrection(NumericTestCase):
    """Fix_Chirp re-applying coefficients saved with a project.

    This path runs on every project load, and it wrote through
    ``DataFrame.values``, which pandas makes read-only under copy-on-write.
    """

    def setUp(self):
        self.ds, self.coeffs = make_chirped_dataset(coeffs=(0.0, 0.0, 0.0, -4e-3, 2.2))

    def test_replay_actually_corrects_the_data(self):
        """It used to return the input untouched.

        The correction was written into ``DataFrame.values``, which hands back
        a fresh array each time it is read, so every write landed in a
        temporary that was discarded. Loading a saved project restored the
        coefficients, reported success, and left the data uncorrected.
        """
        replayed = pf.Fix_Chirp(self.ds, fitcoeff=list(self.coeffs))
        moved = (abs(replayed.values - self.ds.fillna(0).values) > 1e-12).sum()
        self.assertGreater(moved, replayed.size // 2,
                           "replaying a stored chirp must change the data")

    def test_replay_matches_the_tested_correction(self):
        replayed = pf.Fix_Chirp(self.ds, fitcoeff=list(self.coeffs))
        self.assertAllClose(replayed.values, chirp.apply_chirp(self.ds, self.coeffs).values)

    def test_replay_works_on_a_read_only_frame(self):
        """Frames read back from HDF5 do not own their data."""
        frozen = self.ds.copy()
        frozen.values.flags.writeable = False
        replayed = pf.Fix_Chirp(frozen, fitcoeff=list(self.coeffs))
        self.assertEqual(replayed.shape, self.ds.shape)

    def test_six_coefficient_projects_still_load(self):
        """Older projects stored six coefficients; the last two collapse into one."""
        old_style = [0.0, 0.0, 0.0, -4e-3, 2.2, 0.2]
        handed_in = list(old_style)
        pf.Fix_Chirp(self.ds, fitcoeff=old_style)
        self.assertEqual(old_style, handed_in, "the caller's coefficients were modified")

    def test_replay_does_not_modify_the_input(self):
        before = self.ds.copy()
        pf.Fix_Chirp(self.ds, fitcoeff=list(self.coeffs))
        self.assertUnchanged(self.ds, before)


class PolynomialOrder(NumericTestCase):
    """The curve cannot have more freedom than the data supports."""

    def test_the_order_is_limited_by_the_detected_onsets(self):
        """Three points cannot pin down a fourth-order polynomial."""
        ds, _ = make_chirped_dataset(coeffs=(0.0, 0.0, 0.0, -4e-3, 2.2))
        flat = ds.copy()
        for column in list(flat.columns)[3:]:
            flat[column] = 1.0
        fit = chirp.detect_chirp(flat, t_range=(-4, 6))
        self.assertLessEqual(fit.poly_order, max(1, len(fit.onsets) - 1))

    def test_never_more_coefficients_than_the_format_holds(self):
        """Chirp is stored and replayed as five coefficients throughout."""
        ds, _ = make_chirped_dataset()
        for order in (2, 4, 6, 9):
            with self.subTest(order=order):
                fit = chirp.detect_chirp(ds, t_range=(-4, 6), poly_order=order)
                self.assertEqual(len(fit.coefficients), chirp.CHIRP_COEFFICIENTS)

    def test_a_high_order_request_still_corrects(self):
        ds, expected = make_chirped_dataset(coeffs=(0.0, 0.0, 0.0, -4e-3, 2.2))
        corrected, coefficients, _ = chirp.find_chirp_sparse(ds, t_range=(-4, 6),
                                                             poly_order=9, plot=False)
        self.assertEqual(len(coefficients), 5)
        self.assertEqual(corrected.shape, ds.shape)

"""Solving for the species spectra given concentrations and a measurement.

The measurement is modelled as ``A = c @ spectra``. With the concentrations
fixed, the spectra follow from a linear least-squares solve, so a dataset
built from known spectra must return exactly those spectra.
"""

import numpy as np
import pandas

from KiMoPack.kinetics import amplitudes

from ..support import NumericTestCase
from ..synthetic import make_dataset


class FillInt(NumericTestCase):
    def setUp(self):
        self.ds, self.truth = make_dataset(taus=(1.0, 30.0), centres=(480.0, 600.0))
        self.c = self.truth.concentrations

    def test_recovers_the_spectra_it_was_built_from(self):
        result = amplitudes.fill_int(ds=self.ds, c=self.c, final=True)
        self.assertAllClose(result["DAC"].values, self.truth.spectra.values.T, atol=1e-9)

    def test_an_exact_decomposition_leaves_no_error(self):
        result = amplitudes.fill_int(ds=self.ds, c=self.c, final=True)
        self.assertAlmostEqual(float(result["error"]), 0.0, places=12)

    def test_the_error_is_the_summed_squared_residual(self):
        noisy, truth = make_dataset(noise=0.01, seed=1)
        result = amplitudes.fill_int(ds=noisy, c=truth.concentrations, final=True)
        self.assertAlmostEqual(float(result["error"]),
                               float((result["AE"] ** 2).values.sum()), places=9)

    def test_the_reconstruction_and_residual_add_back_to_the_data(self):
        result = amplitudes.fill_int(ds=self.ds, c=self.c, final=True)
        self.assertAllClose((result["AC"] + result["AE"]).values, self.ds.values, atol=1e-9)

    def test_final_results_keep_the_axis_labels(self):
        result = amplitudes.fill_int(ds=self.ds, c=self.c, final=True)
        for key in ["A", "AC", "AE"]:
            with self.subTest(key=key):
                self.assertEqual(result[key].index.name, self.ds.index.name)
                self.assertEqual(result[key].columns.name, self.ds.columns.name)
        self.assertEqual(result["DAC"].index.name, self.ds.columns.name)

    def test_spectra_are_labelled_with_the_species(self):
        c = self.c.copy()
        c.columns = ["fast", "slow"]
        result = amplitudes.fill_int(ds=self.ds, c=c, final=True)
        self.assertEqual(list(result["DAC"].columns), ["fast", "slow"])

    def test_a_plain_call_reports_only_the_error(self):
        """The optimiser calls this on every iteration and needs nothing else."""
        result = amplitudes.fill_int(ds=self.ds, c=self.c, final=False)
        self.assertEqual(set(result), {"error"})

    def test_shapes_can_be_requested_without_a_final_pass(self):
        """Used by the fit's shape dumping, where it raised NameError on DAC."""
        result = amplitudes.fill_int(ds=self.ds, c=self.c, final=False, return_shapes=True)
        self.assertIn("DAC", result)
        self.assertAllClose(result["DAC"].values, self.truth.spectra.values.T, atol=1e-9)

    def test_non_finite_amplitudes_are_zeroed(self):
        """A species with no signal must not poison the whole reconstruction."""
        c = self.c.copy()
        c["dead"] = 0.0
        result = amplitudes.fill_int(ds=self.ds, c=c, final=True)
        self.assertTrue(np.isfinite(result["DAC"].values).all())

    def test_a_single_species_still_works(self):
        times = np.linspace(0.0, 10.0, 40)
        waves = np.linspace(400.0, 500.0, 10)
        c = pandas.DataFrame({"only": np.exp(-times)}, index=times)
        spectrum = np.linspace(1.0, 2.0, 10)
        ds = pandas.DataFrame(np.outer(np.exp(-times), spectrum), index=times, columns=waves)
        result = amplitudes.fill_int(ds=ds, c=c, final=True)
        self.assertAllClose(result["DAC"].values.ravel(), spectrum, atol=1e-9)

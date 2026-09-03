"""Regressions against the datasets committed to the repository.

These read the bundled experiments and re-run real fits end to end, rather
than working from generated data. Set KIMOPACK_SLOW_TESTS=1 to include them.
"""

import pathlib

import lmfit
import matplotlib

matplotlib.use("Agg")

import KiMoPack.plot_func as pf  # noqa: E402

from ..support import NumericTestCase, slow  # noqa: E402

DATA = pathlib.Path(__file__).parents[2] / "Tutorial_Notebooks" / "Data"


@slow
class SavedProjects(NumericTestCase):
    """Projects saved by earlier versions must keep loading."""

    def test_a_stored_fit_comes_back_intact(self):
        ta = pf.TA("con_1_solved.hdf5", path=str(DATA / "Introduction"))
        self.assertEqual(ta.mod, "full_consecutive")
        rates = dict(ta.re["fit_results_rates"]["value"])
        for name in ["k0", "k1", "k2", "t0", "resolution"]:
            with self.subTest(name=name):
                self.assertIn(name, rates)
        self.assertGreater(ta.re["r2"], 0.9)

    def test_the_shaping_settings_are_restored(self):
        ta = pf.TA("full_consecutive_fit.hdf5", path=str(DATA / "Introduction"))
        self.assertIsNotNone(ta.ds)
        self.assertIsNotNone(ta.ds_ori)


@slow
class RefittingRealData(NumericTestCase):
    def test_a_two_component_fit_converges(self):
        ta = pf.TA("TA_Ru-dppz_400nm_ACN.SIA", path=str(DATA / "Fitting-2"))
        ta.Cor_Chirp(chirp_file="TA_Ru-dppz_400nm_ACN_chirp.dat")
        ta.timelimits = [0.3, 1000]
        ta.bordercut = [350, 700]
        ta.wave_nm_bin = 10
        ta.par = lmfit.Parameters()
        ta.par.add("k0", value=1.0)
        ta.par.add("k1", value=0.02)
        ta.mod = "paral"
        ta.Fit_Global()
        self.assertGreater(ta.re["r2"], 0.99)
        # Recorded from this dataset; a change here means the fit moved.
        self.assertAlmostEqual(dict(ta.re["fit_results_rates"]["value"])["k0"], 0.234051, places=5)

    def test_the_sequential_model_runs_on_real_data(self):
        ta = pf.TA("TA_Ru-dppz_400nm_ACN.SIA", path=str(DATA / "Fitting-2"))
        ta.timelimits = [0.3, 1000]
        ta.bordercut = [350, 700]
        ta.wave_nm_bin = 20
        ta.par = lmfit.Parameters()
        ta.par.add("k0", value=1.0)
        ta.par.add("k1", value=0.02)
        ta.mod = "consecutive"
        ta.Fit_Global()
        self.assertGreater(ta.re["r2"], 0.99)


@slow
class FullPipeline(NumericTestCase):
    """The whole chain, with the numbers pinned.

    These values agree bit for bit on pandas 2.3 and 3.0; a change here means
    a step in the pipeline moved, not that a library did.
    """

    def _prepared(self):
        ta = pf.TA("TA_Ru-dppz_400nm_ACN.SIA", path=str(DATA / "Fitting-2"))
        ta.Cor_Chirp(chirp_file="TA_Ru-dppz_400nm_ACN_chirp.dat")
        return ta

    def test_each_correction_moves_the_data_by_a_known_amount(self):
        import numpy as np

        ta = self._prepared()
        self.assertAlmostEqual(float(np.nansum(np.abs(ta.ds.values))), 139227.316702, places=4)
        ta.Filter_data(value=10)
        self.assertAlmostEqual(float(np.nansum(np.abs(ta.ds.values))), 2367.417003, places=4)
        ta.Background(lowlimit=-5, uplimit=-1)
        self.assertAlmostEqual(float(np.nansum(np.abs(ta.ds.values))), 1675.156372, places=4)

    def test_the_fit_after_the_full_chain(self):
        ta = self._prepared()
        ta.Filter_data(value=10)
        ta.Background(lowlimit=-5, uplimit=-1)
        ta.timelimits = [0.3, 1000]
        ta.bordercut = [350, 700]
        ta.wave_nm_bin = 10
        ta.par = lmfit.Parameters()
        ta.par.add("k0", value=1.0)
        ta.par.add("k1", value=0.02)
        ta.mod = "paral"
        ta.Fit_Global()
        self.assertAlmostEqual(dict(ta.re["fit_results_rates"]["value"])["k0"], 0.2335388121,
                               places=8)
        self.assertAlmostEqual(ta.re["r2"], 0.998754772806, places=10)

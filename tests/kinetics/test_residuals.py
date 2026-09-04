"""Fitting several datasets together.

A joint fit varies one set of rate constants across measurements that were
recorded separately — different solvents, different excitation powers — so
that a shared mechanism is tested against all of them at once.
"""

import lmfit
import numpy as np

import KiMoPack.plot_func as pf

from ..support import NumericTestCase
from ..synthetic import make_dataset


def _projects():
    projects = []
    for taus, centres in [((1.0, 30.0), (480.0, 600.0)), ((1.5, 25.0), (500.0, 620.0))]:
        ds, _ = make_dataset(taus=taus, centres=centres)
        ta = pf.TA("synthetic", ds=ds)
        ta.timelimits = [-1, 500]
        ta.bordercut = [420, 680]
        ta.wave_nm_bin = 20
        ta.par = _parameters()
        projects.append(ta)
    return projects


def _parameters(extras=()):
    par = lmfit.Parameters()
    par.add("k0", value=1.0)
    par.add("k1", value=1 / 30.0)
    par.add("t0", value=0.0, vary=False)
    par.add("resolution", value=0.2, vary=False)
    for name in extras:
        par.add(name, value=0.0, vary=False)
    return par


class SharedSpectra(NumericTestCase):
    """same_DAS: one set of spectra has to explain every dataset."""

    def test_each_project_reports_its_own_error(self):
        """They used to be handed the combined error instead.

        error_total carries the combined value; error is the project's own,
        and r2 is computed from it, so reporting the total made every r2 wrong.
        """
        results = pf.err_func_multi(_parameters(), mod="paral", final=True,
                                    multi_project=_projects(), same_DAS=True)
        errors = [float(r["error"]) for r in results]
        self.assertNotAlmostEqual(errors[0], errors[1],
                                  msg="two different datasets cannot fit equally well")

    def test_the_project_errors_add_up_to_the_total(self):
        results = pf.err_func_multi(_parameters(), mod="paral", final=True,
                                    multi_project=_projects(), same_DAS=True)
        total = float(results[0]["error_total"])
        self.assertAlmostEqual(sum(float(r["error"]) for r in results), total, places=6)

    def test_every_project_gets_the_same_spectra(self):
        """That is what sharing the spectra means."""
        results = pf.err_func_multi(_parameters(), mod="paral", final=True,
                                    multi_project=_projects(), same_DAS=True)
        self.assertAllClose(results[0]["DAC"].values, results[1]["DAC"].values)

    def test_each_project_keeps_its_own_matrices(self):
        projects = _projects()
        results = pf.err_func_multi(_parameters(), mod="paral", final=True,
                                    multi_project=projects, same_DAS=True)
        for result in results:
            with self.subTest():
                self.assertEqual(result["A"].shape, result["AC"].shape)
                self.assertEqual(result["A"].shape, result["AE"].shape)

    def test_searching_returns_a_single_number(self):
        error = pf.err_func_multi(_parameters(), mod="paral", final=False,
                                  multi_project=_projects(), same_DAS=True)
        self.assertIsInstance(float(error), float)


class SeparateSpectra(NumericTestCase):
    """Each dataset gets its own spectra; only the kinetics are shared."""

    def test_the_combined_error_is_a_root_mean_square(self):
        projects = _projects()
        combined = float(pf.err_func_multi(_parameters(), mod="paral", final=False,
                                           multi_project=projects))
        singles = [float(pf.err_func(_parameters(), pf.sub_ds(
            ta.ds, timelimits=ta.timelimits, bordercut=ta.bordercut, wave_nm_bin=ta.wave_nm_bin),
            mod="paral")) for ta in projects]
        self.assertAlmostEqual(combined, float(np.sqrt(np.mean(np.array(singles) ** 2))), places=6)

    def test_weights_may_omit_the_first_project(self):
        """The interface offers 'the others relative to this one'."""
        projects = _projects()
        explicit = pf.err_func_multi(_parameters(), mod="paral", final=False,
                                     multi_project=projects, weights=[1.0, 3.0])
        implied = pf.err_func_multi(_parameters(), mod="paral", final=False,
                                    multi_project=projects, weights=[3.0])
        self.assertAlmostEqual(float(explicit), float(implied), places=9)

    def test_a_wrong_number_of_weights_is_refused(self):
        with self.assertRaises(ValueError):
            pf.err_func_multi(_parameters(), mod="paral", final=False,
                              multi_project=_projects(), weights=[1.0, 2.0, 3.0])

    def test_a_unique_parameter_comes_from_each_project(self):
        projects = _projects()
        projects[1].par["t0"].value = 0.5
        shared = pf.err_func_multi(_parameters(), mod="paral", final=False,
                                   multi_project=projects)
        per_project = pf.err_func_multi(_parameters(), mod="paral", final=False,
                                        multi_project=projects, unique_parameter=["t0"])
        self.assertNotAlmostEqual(float(shared), float(per_project),
                                  msg="the second project's own t0 should have been used")


class SharedSpectraConcentrations(NumericTestCase):
    """Each project's concentrations must be its own.

    The joint solve stacks every dataset into one tall matrix; the result has
    to be cut back apart along the same boundaries.
    """

    def test_each_project_gets_its_own_concentrations(self):
        projects = _projects()
        results = pf.err_func_multi(_parameters(), mod="paral", final=True,
                                    multi_project=projects, same_DAS=True)
        for result in results:
            with self.subTest():
                self.assertEqual(len(result["c"].index), len(result["A"].index))

    def test_the_concentrations_line_up_with_the_matrices(self):
        projects = _projects()
        results = pf.err_func_multi(_parameters(), mod="paral", final=True,
                                    multi_project=projects, same_DAS=True)
        for result in results:
            with self.subTest():
                self.assertAllClose(result["c"].index.values.astype(float),
                                    result["A"].index.values.astype(float))

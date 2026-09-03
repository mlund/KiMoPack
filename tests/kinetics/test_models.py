"""Concentration profiles produced by the kinetic models.

The parallel model is checked against the exponential it claims to be; the
sequential model against the analytic Bateman solution for a decay chain.
Neither reference comes from KiMoPack.
"""

import lmfit
import numpy as np
import pandas

from KiMoPack.kinetics import models
from KiMoPack.kinetics.parameters import par_to_pardf

from ..support import NumericTestCase
from ..synthetic import bateman_concentrations


def _pardf(rates, t0=0.0, resolution=0.1, extra=()):
    par = lmfit.Parameters()
    for i, rate in enumerate(rates):
        par.add(f"k{i}", value=rate)
    par.add("t0", value=t0, vary=False)
    par.add("resolution", value=resolution, vary=False)
    for name in extra:
        par.add(name, value=0.0, vary=False)
    return par_to_pardf(par)


class ParallelModel(NumericTestCase):
    def test_decays_exponentially_once_the_response_has_finished(self):
        times = np.linspace(-1.0, 40.0, 800)
        c = models.build_c(times=times, mod="paral", pardf=_pardf([1 / 3.0], resolution=0.1))
        late = times > 1.0
        self.assertAllClose(c.values[late, 0], np.exp(-times[late] / 3.0), rtol=1e-6)

    def test_each_species_decays_with_its_own_rate(self):
        times = np.linspace(-1.0, 40.0, 800)
        c = models.build_c(times=times, mod="paral", pardf=_pardf([1 / 2.0, 1 / 20.0]))
        late = times > 1.0
        self.assertAllClose(c.values[late, 0], np.exp(-times[late] / 2.0), rtol=1e-6)
        self.assertAllClose(c.values[late, 1], np.exp(-times[late] / 20.0), rtol=1e-6)

    def test_is_half_risen_one_resolution_after_t0(self):
        """The response reaches 50% one 'resolution' after the onset."""
        c = models.build_c(times=np.array([0.3]), mod="paral",
                           pardf=_pardf([1e-9], t0=0.0, resolution=0.3))
        self.assertAlmostEqual(float(c.values[0, 0]), 0.5, places=6)

    def test_nothing_has_arrived_well_before_t0(self):
        times = np.linspace(-5.0, -2.0, 20)
        c = models.build_c(times=times, mod="paral", pardf=_pardf([1 / 3.0], resolution=0.1))
        self.assertAllClose(c.values, np.zeros_like(c.values), atol=1e-9)

    def test_the_index_is_the_time_axis(self):
        times = np.linspace(-1.0, 10.0, 50)
        c = models.build_c(times=times, mod="paral", pardf=_pardf([1 / 3.0]))
        self.assertEqual(c.index.name, "time")
        self.assertAllClose(c.index.values, times)

    def test_a_background_species_is_flat(self):
        times = np.linspace(-1.0, 10.0, 50)
        c = models.build_c(times=times, mod="paral", pardf=_pardf([1 / 3.0], extra=["background"]))
        self.assertIn("background", c.columns)
        self.assertAllClose(c["background"].values, np.ones(times.size))

    def test_an_explicit_ground_state_starts_empty(self):
        times = np.linspace(-1.0, 10.0, 50)
        c = models.build_c(times=times, mod="paral", pardf=_pardf([1 / 3.0], extra=["explicit_GS"]))
        self.assertAllClose(c["GS"].values, np.zeros(times.size))

    def test_an_infinite_species_never_decays(self):
        times = np.linspace(-1.0, 100.0, 400)
        c = models.build_c(times=times, mod="paral", pardf=_pardf([1 / 3.0], extra=["infinite"]))
        late = times > 2.0
        self.assertAllClose(c["infinite"].values[late], np.ones(late.sum()), rtol=1e-6)

    def test_every_alias_gives_the_same_answer(self):
        times = np.linspace(-1.0, 20.0, 100)
        pardf = _pardf([1 / 3.0])
        reference = models.build_c(times=times, mod="paral", pardf=pardf)
        for alias in ["parallel", "decays", "exponential"]:
            with self.subTest(alias=alias):
                self.assertFrameAllClose(models.build_c(times=times, mod=alias, pardf=pardf),
                                         reference)


class SequentialModel(NumericTestCase):
    """A -> B -> C, integrated forward from a Gaussian excitation pulse."""

    def setUp(self):
        self.times = np.linspace(-2.0, 80.0, 3000)
        self.taus = (2.0, 15.0)
        self.pardf = _pardf([1 / t for t in self.taus], resolution=0.2)
        self.reference = bateman_concentrations(self.times, self.taus)
        # Bateman assumes an instantaneous pulse, so the two only agree once
        # the instrument response has finished.
        self.settled = self.times > 2.0

    def test_matches_the_analytic_chain_after_the_pulse(self):
        c = models.build_c(times=self.times, mod="consecutive", pardf=self.pardf, sub_steps=20)
        self.assertAllClose(c.values[self.settled], self.reference.values[self.settled],
                            atol=5e-3, msg="sequential model should follow the Bateman solution")

    def test_the_integration_has_converged(self):
        """More sub-steps must not move the answer."""
        coarse = models.build_c(times=self.times, mod="consecutive", pardf=self.pardf, sub_steps=5)
        fine = models.build_c(times=self.times, mod="consecutive", pardf=self.pardf, sub_steps=80)
        self.assertAllClose(coarse.values[self.settled], fine.values[self.settled], atol=1e-3)

    def test_population_is_conserved_when_nothing_can_leave(self):
        """With a non-decaying final species the chain must retain all of it."""
        times = np.linspace(-2.0, 200.0, 4000)
        pardf = _pardf([1 / 2.0, 1 / 15.0], resolution=0.2, extra=["infinite"])
        c = models.build_c(times=times, mod="consecutive", pardf=pardf, sub_steps=20)
        self.assertAlmostEqual(float(c.iloc[-1].sum()), 1.0, places=6)

    def test_the_non_decaying_species_is_named(self):
        pardf = _pardf([1 / 2.0], resolution=0.2, extra=["infinite"])
        c = models.build_c(times=self.times, mod="consecutive", pardf=pardf, sub_steps=10)
        self.assertEqual(list(c.columns)[-1], "Non Decaying")

    def test_sub_steps_can_be_carried_in_the_parameter_table(self):
        """Documented as a parameter, but it was read as a column, not a row."""
        pardf = _pardf([1 / 2.0], resolution=0.2)
        pardf.loc["sub_steps", :] = {"value": 25, "is_rate": False, "min": 25,
                                     "max": 25, "vary": False, "expr": None}
        from_table = models.build_c(times=self.times, mod="consecutive", pardf=pardf)
        explicit = models.build_c(times=self.times, mod="consecutive",
                                  pardf=_pardf([1 / 2.0], resolution=0.2), sub_steps=25)
        self.assertAllClose(from_table.values, explicit.values)

    def test_concentrations_never_go_negative(self):
        c = models.build_c(times=self.times, mod="consecutive", pardf=self.pardf, sub_steps=20)
        self.assertGreaterEqual(float(c.values.min()), 0.0)

    def test_every_alias_gives_the_same_answer(self):
        reference = models.build_c(times=self.times, mod="consecutive", pardf=self.pardf,
                                   sub_steps=10)
        for alias in ["sequential", "full_consecutive", "full_sequential"]:
            with self.subTest(alias=alias):
                self.assertFrameAllClose(
                    models.build_c(times=self.times, mod=alias, pardf=self.pardf, sub_steps=10),
                    reference,
                )


class Registry(NumericTestCase):
    def test_lists_the_built_in_models(self):
        available = models.available_models()
        for name in ["paral", "exponential", "consecutive", "full_consecutive"]:
            with self.subTest(name=name):
                self.assertIn(name, available)

    def test_parallel_species_are_decay_associated(self):
        self.assertEqual(models.resolve_model("paral").species_are, "DAS")

    def test_sequential_species_are_species_associated(self):
        self.assertEqual(models.resolve_model("consecutive").species_are, "SAS")

    def test_consecutive_is_optimised_through_the_cheap_parallel_model(self):
        """Documented speed trick: fit the rates as decays, integrate only at the end."""
        self.assertEqual(models.resolve_model("consecutive").optimise_with, "paral")

    def test_the_full_variants_are_optimised_as_themselves(self):
        self.assertIsNone(models.resolve_model("full_consecutive").optimise_with)

    def test_an_unknown_name_says_what_is_available(self):
        with self.assertRaises(ValueError) as caught:
            models.resolve_model("nonsense")
        message = str(caught.exception)
        self.assertIn("nonsense", message)
        self.assertIn("consecutive", message)

    def test_a_user_function_is_accepted_as_a_model(self):
        """Users pass their own target models; the library must not care."""

        def two_state(times, pardf):
            # The user contract hands over the value column as a Series.
            self.assertIsInstance(pardf, pandas.Series)
            decay = np.exp(-np.asarray(times) * pardf["k0"])
            return pandas.DataFrame({"A": decay, "B": 1 - decay}, index=times)

        model = models.resolve_model(two_state)
        c = model.build(times=np.linspace(0.0, 10.0, 20), pardf=_pardf([0.5]))
        self.assertEqual(list(c.columns), ["A", "B"])
        self.assertEqual(model.species_are, "SAS")

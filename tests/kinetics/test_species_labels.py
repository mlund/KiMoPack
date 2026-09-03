"""Naming the species a fit reports.

The labels travel with the spectra into every figure and CSV export, so a
wrong one renames a physical species in a published figure. Which species
exist is decided by the model and the optional parameters, so the names have
to follow the columns the model actually produced.
"""

import lmfit
import numpy as np

import KiMoPack.plot_func as pf

from ..support import NumericTestCase
from ..synthetic import make_dataset


def _fit_once(mod, extras):
    ds, _ = make_dataset(taus=(1.0, 30.0))
    par = lmfit.Parameters()
    par.add("k0", value=1.0)
    par.add("k1", value=1 / 30.0)
    par.add("t0", value=0.0, vary=False)
    par.add("resolution", value=0.2, vary=False)
    for name in extras:
        par.add(name, value=0.0, vary=False)
    return pf.err_func(par, ds, mod=mod, final=True)


class SpeciesNames(NumericTestCase):
    def test_a_plain_fit_names_species_by_number(self):
        for mod in ["paral", "consecutive"]:
            with self.subTest(mod=mod):
                re = _fit_once(mod, [])
                self.assertEqual([str(c) for c in re["DAC"].columns], ["0", "1"])

    def test_the_non_decaying_species_is_named(self):
        for mod in ["paral", "consecutive"]:
            with self.subTest(mod=mod):
                re = _fit_once(mod, ["infinite"])
                self.assertEqual(list(re["DAC"].columns)[-1], "Non Decaying")

    def test_background_and_non_decaying_keep_their_own_names(self):
        """Both exist at once; neither may take the other's name."""
        for mod in ["paral", "consecutive"]:
            with self.subTest(mod=mod):
                re = _fit_once(mod, ["background", "infinite"])
                names = [str(c) for c in re["DAC"].columns]
                self.assertIn("Non Decaying", names)
                self.assertIn("background", names)
                self.assertEqual(len(set(names)), len(names), "names must be unique")

    def test_an_explicit_ground_state_keeps_its_name(self):
        for mod in ["paral", "consecutive"]:
            with self.subTest(mod=mod):
                re = _fit_once(mod, ["explicit_GS", "background", "infinite"])
                names = [str(c) for c in re["DAC"].columns]
                for expected in ["GS", "background", "Non Decaying"]:
                    self.assertIn(expected, names)
                self.assertEqual(len(set(names)), len(names), "names must be unique")

    def test_the_concentrations_carry_the_same_names(self):
        """Spectra and concentrations are plotted against each other."""
        for mod in ["paral", "consecutive"]:
            with self.subTest(mod=mod):
                re = _fit_once(mod, ["background", "infinite"])
                self.assertEqual([str(c) for c in re["DAC"].columns],
                                 [str(c) for c in re["c"].columns])

    def test_the_named_species_behave_as_their_names_claim(self):
        """A label is only right if the column does what the name says.

        Both sit at 1 long after excitation, so they are only distinguishable
        beforehand: a background is present the whole time, while a
        non-decaying species does not exist until the pulse creates it.
        """
        for mod in ["paral", "consecutive"]:
            with self.subTest(mod=mod):
                c = _fit_once(mod, ["background", "infinite"])["c"]
                before = c.index.values < -1.0
                self.assertAllClose(
                    c["background"].values[before], np.ones(before.sum()), rtol=1e-9,
                    msg="a background is there before the pulse arrives")
                self.assertAllClose(
                    c["Non Decaying"].values[before], np.zeros(before.sum()), atol=1e-6,
                    msg="a non-decaying species cannot exist before it is created")

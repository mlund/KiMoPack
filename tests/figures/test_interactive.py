"""The hover-to-inspect plot.

Moving the pointer over the 2D map redraws a spectrum at that delay and a
kinetic trace at that wavelength. It has never been usable: the colour
argument fell back to an attribute nothing ever set.
"""

import types

import matplotlib.pyplot as plt

import KiMoPack.plot_func as pf

from ..support import NumericTestCase
from ..synthetic import make_dataset


def _hover(x, y):
    """The part of a matplotlib motion event the callback reads."""
    return types.SimpleNamespace(xdata=x, ydata=y)


class Opening(NumericTestCase):
    def setUp(self):
        ds, _ = make_dataset()
        self.ta = pf.TA("synthetic", ds=ds)
        self.ta.bordercut = [420, 680]
        self.ta.wave_nm_bin = 20
        self.addCleanup(plt.close, "all")

    def test_it_opens_without_arguments(self):
        mover, cursor = self.ta.Plot_Interactive()
        self.assertTrue(hasattr(mover, "move"))

    def test_it_draws_the_map_and_both_side_panels(self):
        self.ta.Plot_Interactive()
        self.assertGreaterEqual(len(plt.gcf().get_axes()), 3)

    def test_hovering_fills_the_side_panels(self):
        """The callback is the whole point; it slices the data at the cursor."""
        mover, _cursor = self.ta.Plot_Interactive()
        mover.move(_hover(500.0, 10.0))
        self.assertTrue(mover.ax_time.get_lines(), "no spectrum drawn at the hovered delay")
        self.assertTrue(mover.ax_kinetic.get_lines(), "no kinetics drawn at the hovered wavelength")

    def test_hovering_twice_does_not_accumulate(self):
        mover, _cursor = self.ta.Plot_Interactive()
        mover.move(_hover(500.0, 10.0))
        first = len(mover.ax_time.get_lines())
        mover.move(_hover(550.0, 20.0))
        self.assertEqual(len(mover.ax_time.get_lines()), first)

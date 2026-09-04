"""Cases a review found that the behaviour-preserving tests missed.

The differential tests proved the refactor drew the same pictures for the
inputs it was given. These are the inputs it was not given.
"""

import matplotlib.pyplot as plt
import numpy as np

import KiMoPack.plot_func as pf

from ..support import NumericTestCase
from ..synthetic import make_dataset


class ShortSpans(NumericTestCase):
    """Smoothing is applied per masked span, and a span can be tiny.

    A cut near the edge of the spectrum leaves a handful of channels; a
    smoothing window wider than that cannot be applied, and losing the whole
    figure over it is worse than drawing that piece unsmoothed.
    """

    def setUp(self):
        self.ds, _ = make_dataset(waves=np.arange(400.0, 701.0, 2.0))
        self.addCleanup(plt.close, "all")

    def test_a_cut_near_the_edge_still_draws(self):
        pf.plot_time(self.ds, rel_time=[1, 5], scattercut=[690, 695], title="x")
        self.assertTrue(plt.get_fignums())

    def test_two_close_cuts_still_draw(self):
        pf.plot_time(self.ds, rel_time=[1, 5], scattercut=[[500, 504], [510, 514]], title="x")
        self.assertTrue(plt.get_fignums())

    def test_the_short_piece_is_still_drawn(self):
        """Unsmoothed is fine; missing is not."""
        fig = pf.plot_time(self.ds, rel_time=[1], scattercut=[690, 695], title="x")
        ax = fig.get_axes()[0]
        # One trace either side of the cut, plus the zero baseline.
        self.assertGreaterEqual(len([ln for ln in ax.get_lines() if len(ln.get_xdata()) > 1]), 2)


class DefaultArguments(NumericTestCase):
    """The documented defaults have to work."""

    def setUp(self):
        self.ds, _ = make_dataset()
        self.addCleanup(plt.close, "all")

    def test_spectra_without_chosen_delays_says_so(self):
        """It crashed on a comparison against None, here and upstream."""
        with self.assertRaises(ValueError) as caught:
            pf.plot_time(self.ds, title="x")
        self.assertIn("rel_time", str(caught.exception))

    def test_an_unknown_line_mode_says_what_is_allowed(self):
        with self.assertRaises(ValueError) as caught:
            pf.plot_time(self.ds, rel_time=[1], lines_are="nonsense", title="x")
        self.assertIn("smoothed", str(caught.exception))


class LegendMatchesTheTraces(NumericTestCase):
    """Labels and curves are derived from one list, not two."""

    def setUp(self):
        self.ds, _ = make_dataset()
        self.addCleanup(plt.close, "all")

    def test_a_delay_outside_the_data_is_not_labelled(self):
        fig = pf.plot_time(self.ds, rel_time=[1, 10, 1e9], title="x")
        ax = fig.get_axes()[0]
        drawn = [ln for ln in ax.get_lines() if ln.get_label() != "_nolegend_"]
        self.assertEqual(len(ax.get_legend().get_texts()), len(drawn))

    def test_every_labelled_curve_is_listed(self):
        fig = pf.plot_time(self.ds, rel_time=[1, 10], title="x")
        ax = fig.get_axes()[0]
        drawn = [ln for ln in ax.get_lines() if ln.get_label() != "_nolegend_"]
        self.assertEqual(len(ax.get_legend().get_texts()), len(drawn))

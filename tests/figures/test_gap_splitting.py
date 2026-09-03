"""Lines must break at masked regions, and each piece must be drawn once.

Scattered pump light is blanked out of the spectrum, and unusable delays out
of the kinetics. A trace crossing such a gap has to be drawn as separate
pieces, or the plot claims data where there is none.
"""

import matplotlib.pyplot as plt
import numpy as np

import KiMoPack.plot_func as pf

from ..support import NumericTestCase
from ..synthetic import make_dataset


def _segments(ax):
    """The x-ranges of every line drawn on an axis."""
    spans = []
    for line in ax.get_lines():
        x = np.asarray(line.get_xdata(), dtype=float)
        if x.size:
            spans.append((x.min(), x.max()))
    return spans


class SpectraAcrossScatterCuts(NumericTestCase):
    """plot_time draws spectra; scattercut splits them along wavelength."""

    def setUp(self):
        self.ds, _ = make_dataset()
        self.addCleanup(plt.close, "all")

    def _draw(self, scattercut):
        fig, ax = plt.subplots()
        pf.plot_time(self.ds, ax=ax, rel_time=[1.0], scattercut=scattercut, subplot=True)
        return ax

    def test_one_region_gives_two_pieces(self):
        self.assertEqual(len(_segments(self._draw([500, 540]))), 2)

    def test_each_region_adds_one_piece(self):
        for count in range(1, 4):
            with self.subTest(regions=count):
                cuts = [[450 + 40 * i, 460 + 40 * i] for i in range(count)]
                ax = self._draw(cuts)
                self.assertEqual(len(_segments(ax)), count + 1,
                                 "one piece per gap, each drawn exactly once")

    def test_no_piece_is_drawn_twice(self):
        ax = self._draw([[450, 470], [600, 620]])
        spans = _segments(ax)
        self.assertEqual(len(set(spans)), len(spans), f"duplicated segments: {spans}")

    def test_no_piece_spans_a_masked_region(self):
        ax = self._draw([[450, 470], [600, 620]])
        for low, high in _segments(ax):
            for cut_low, cut_high in [(450, 470), (600, 620)]:
                with self.subTest(span=(low, high), cut=(cut_low, cut_high)):
                    self.assertFalse(low < cut_low and high > cut_high,
                                     "a line was drawn straight across the gap")

    def test_no_cuts_gives_one_piece(self):
        self.assertEqual(len(_segments(self._draw(None))), 1)


class KineticsAcrossIgnoredTimes(NumericTestCase):
    """plot1d draws kinetics; ignore_time_region splits them along time."""

    def setUp(self):
        self.ds, _ = make_dataset()
        self.addCleanup(plt.close, "all")

    def _draw(self, ignore):
        fig, ax = plt.subplots()
        pf.plot1d(self.ds, ax=ax, wavelength=[500], width=10, ignore_time_region=ignore,
                  subplot=True)
        return ax

    def test_one_region_gives_two_pieces(self):
        self.assertEqual(len(_segments(self._draw([1, 5]))), 2)

    def test_each_region_adds_one_piece(self):
        for count in range(1, 4):
            with self.subTest(regions=count):
                cuts = [[10**i, 2 * 10**i] for i in range(count)]
                self.assertEqual(len(_segments(self._draw(cuts))), count + 1)

    def test_no_piece_is_drawn_twice(self):
        spans = _segments(self._draw([[1, 5], [50, 100]]))
        self.assertEqual(len(set(spans)), len(spans), f"duplicated segments: {spans}")

"""The plot methods on a project.

These marshal a project's settings into the drawing functions. Nothing
exercised them, so a wrong keyword surfaced only when someone made a figure.
"""

import matplotlib.pyplot as plt

import KiMoPack.plot_func as pf

from ..support import NumericTestCase
from ..synthetic import make_dataset


def _fitted():
    import lmfit

    ds, _ = make_dataset(taus=(1.0, 30.0))
    ta = pf.TA("synthetic", ds=ds)
    ta.timelimits = [-1, 500]
    ta.bordercut = [420, 680]
    ta.wave_nm_bin = 20
    ta.rel_wave = [450, 550, 650]
    ta.rel_time = [1, 10, 100]
    ta.par = lmfit.Parameters()
    ta.par.add("k0", value=1.0)
    ta.par.add("k1", value=1 / 30.0)
    ta.mod = "paral"
    ta.Fit_Global()
    return ta


class PlotRaw(NumericTestCase):
    def setUp(self):
        self.ta = _fitted()
        self.addCleanup(plt.close, "all")

    def test_every_panel_draws(self):
        for panel in range(4):
            with self.subTest(panel=panel):
                plt.close("all")
                self.ta.Plot_RAW(plotting=[panel])
                self.assertTrue(plt.get_fignums(), "no figure produced")

    def test_all_panels_together(self):
        self.ta.Plot_RAW(plotting=range(4))
        self.assertEqual(len(plt.get_fignums()), 4)

    def test_the_display_settings_reach_the_figure(self):
        """A dropped keyword between the project and the plot is invisible."""
        self.ta.log_scale = True
        self.ta.lintresh = 0.5
        self.ta.Plot_RAW(plotting=[0])
        ax = plt.gcf().get_axes()[0]
        self.assertEqual(ax.get_yscale(), "symlog")

    def test_handles_come_back_when_asked(self):
        handles = self.ta.Plot_RAW(plotting=range(4), return_figures_handles=True)
        self.assertIsInstance(handles, dict)
        self.assertTrue(handles)


class PlotFitOutput(NumericTestCase):
    def setUp(self):
        self.ta = _fitted()
        self.addCleanup(plt.close, "all")

    def test_every_panel_draws(self):
        for panel in range(7):
            with self.subTest(panel=panel):
                plt.close("all")
                self.ta.Plot_fit_output(plotting=[panel])
                self.assertTrue(plt.get_fignums(), "no figure produced")

    def test_all_panels_together(self):
        self.ta.Plot_fit_output(plotting=range(7))
        self.assertEqual(len(plt.get_fignums()), 7)

    def test_handles_come_back_when_asked(self):
        handles = self.ta.Plot_fit_output(plotting=range(7), return_figures_handles=True)
        self.assertIsInstance(handles, dict)
        self.assertTrue(handles)

    def test_it_refuses_politely_without_a_fit(self):
        ds, _ = make_dataset()
        unfitted = pf.TA("synthetic", ds=ds)
        self.assertIs(unfitted.Plot_fit_output(), False)


class LegendLayout(NumericTestCase):
    """The legend uses the column count the code asks for.

    plot1d computes ncol from the number of entries, but the marker pass that
    followed went through pandas, which rebuilt the existing legend with its
    own defaults and silently dropped that choice.
    """

    def setUp(self):
        self.addCleanup(plt.close, "all")

    def test_the_requested_column_count_survives(self):
        ds, _ = make_dataset()
        fig = pf.plot1d(ds, wavelength=[450, 550, 650], width=10, lines_are="smoothed")
        legend = fig.get_axes()[0].get_legend()
        self.assertEqual(legend._ncols, 2)

    def test_a_following_marker_pass_does_not_reset_it(self):
        ds, _ = make_dataset()
        fig, ax = plt.subplots()
        pf.plot1d(ds, ax=ax, wavelength=[450, 550, 650], width=10, lines_are="smoothed")
        pf.plot1d(ds, ax=ax, wavelength=[450, 550, 650], width=10, lines_are="data", subplot=True)
        self.assertEqual(ax.get_legend()._ncols, 2)


class FitOverviewPanels(NumericTestCase):
    """The three-panel comparison: measured, modelled, and the difference.

    All three show the same matrix shape on the same axes; only the residual
    is exaggerated, or it would look uniformly blank beside the other two.
    """

    def setUp(self):
        self.ta = _fitted()
        self.addCleanup(plt.close, "all")

    def _panels(self, **kwargs):
        fig = pf.plot2d_fit(self.ta.re, cmap=plt.get_cmap("seismic"),
                            error_matrix_amplification=20, **kwargs)
        # The three map axes are created first; colour bars are appended after.
        return fig.get_axes()[:3]

    def test_three_maps_are_drawn(self):
        self.assertEqual(len(self._panels()), 3)

    def test_the_residual_is_shown_on_an_exaggerated_scale(self):
        maps = self._panels(intensity_range=4e-3)
        measured = max(abs(v) for v in maps[0].collections[0].get_clim())
        residual = max(abs(v) for v in maps[2].collections[0].get_clim())
        self.assertAlmostEqual(measured / residual, 20, places=6)

    def test_the_first_two_share_one_scale(self):
        maps = self._panels(intensity_range=4e-3)
        self.assertEqual(maps[0].collections[0].get_clim(), maps[1].collections[0].get_clim())

    def test_panels_are_titled(self):
        titles = [a.get_title() for a in self._panels()]
        self.assertEqual(titles, ["Measured", "Calculated", "Difference"])

    def test_labels_move_inside_when_stacked(self):
        """With patches the name goes in the corner, so no vertical space is lost."""
        maps = self._panels(patches=True)
        self.assertEqual([a.get_title() for a in maps], ["", "", ""])
        for ax in maps:
            with self.subTest():
                self.assertTrue(any(t.get_text() in ("measured", "calculated", "difference")
                                    for t in ax.texts))

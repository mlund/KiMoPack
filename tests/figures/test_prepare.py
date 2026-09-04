"""What a spectra panel decides, checked without drawing anything."""

import matplotlib.pyplot as plt

from KiMoPack.figures.mpl import draw_panel
from KiMoPack.figures.prepare import spectra_panel
from KiMoPack.figures.settings import ViewSettings
from KiMoPack.shaping import DataSelection

from ..support import NumericTestCase
from ..synthetic import make_dataset

COLORS = ["red", "green", "blue"]


class SpectraPanel(NumericTestCase):
    def setUp(self):
        self.ds, _ = make_dataset()
        self.view = ViewSettings(data_type="differential Absorption")

    def _panel(self, selection=None, **kwargs):
        return spectra_panel(self.ds, selection or DataSelection(), self.view, COLORS,
                             rel_time=kwargs.pop("rel_time", [1.0, 10.0]), **kwargs)

    def test_one_trace_per_requested_delay(self):
        self.assertEqual(len(self._panel().traces), 2)

    def test_delays_outside_the_measurement_are_dropped(self):
        """Keeping them would silently plot the nearest delay instead."""
        panel = self._panel(rel_time=[1.0, 10.0, 1e9])
        self.assertEqual(len(panel.traces), 2)

    def test_each_delay_gets_its_own_colour(self):
        colours = [t.color for t in self._panel().traces]
        self.assertEqual(colours, ["red", "green"])

    def test_a_masked_region_breaks_every_trace(self):
        selection = DataSelection(scattercut=[[450, 470], [600, 620]])
        panel = self._panel(selection=selection)
        # Two delays, three surviving spans each.
        self.assertEqual(len(panel.traces), 6)

    def test_a_broken_trace_is_listed_once(self):
        selection = DataSelection(scattercut=[[450, 470], [600, 620]])
        panel = self._panel(selection=selection)
        self.assertEqual(len(panel.legend_labels()), 2)

    def test_masked_regions_are_reported_when_asked(self):
        selection = DataSelection(scattercut=[[450, 470], [600, 620]])
        panel = self._panel(selection=selection, shade_masked=True)
        self.assertEqual(list(panel.shaded), [(450.0, 470.0), (600.0, 620.0)])

    def test_line_plots_do_not_shade_by_default(self):
        """They never have; shading them would change every existing figure."""
        selection = DataSelection(scattercut=[[450, 470], [600, 620]])
        self.assertEqual(self._panel(selection=selection).shaded, ())

    def test_the_crop_becomes_the_axis_limits(self):
        panel = self._panel(selection=DataSelection(bordercut=[430, 670]))
        self.assertEqual(panel.x.limits, [430, 670])

    def test_a_single_intensity_becomes_a_symmetric_range(self):
        view = ViewSettings(intensity_range=3e-3)
        panel = spectra_panel(self.ds, DataSelection(), view, COLORS, rel_time=[1.0])
        self.assertAllClose(panel.y.limits, [-3e-3, 3e-3])

    def test_the_axes_are_named_after_the_data(self):
        panel = self._panel()
        self.assertEqual(panel.y.label, "differential Absorption")
        self.assertIsNotNone(panel.x.label)

    def test_data_is_drawn_as_markers_and_fits_as_lines(self):
        self.assertEqual(self._panel(lines_are="data").traces[0].style, "markers")
        self.assertEqual(self._panel(lines_are="fitted").traces[0].style, "solid")

    def test_a_fit_is_drawn_slightly_transparent(self):
        """So the measurement underneath stays visible."""
        self.assertLess(self._panel(lines_are="fitted").traces[0].alpha, 1.0)

    def test_the_legend_says_what_the_lines_are(self):
        self.assertEqual(self._panel(lines_are="smoothed").legend_title, "lines = smoothed")
        self.assertEqual(self._panel(lines_are="fitted").legend_title, "lines = fit")

    def test_nothing_is_drawn_during_preparation(self):
        """The point of the split: no figure exists until a renderer runs."""
        plt.close("all")
        self._panel()
        self.assertEqual(plt.get_fignums(), [])


class Rendering(NumericTestCase):
    def setUp(self):
        self.ds, _ = make_dataset()
        self.addCleanup(plt.close, "all")

    def test_every_trace_reaches_the_axis(self):
        panel = spectra_panel(self.ds, DataSelection(scattercut=[[450, 470]]),
                              ViewSettings(), COLORS, rel_time=[1.0, 10.0])
        fig, ax = plt.subplots()
        draw_panel(panel, ax)
        self.assertEqual(len(ax.get_lines()), len(panel.traces))

    def test_masked_regions_are_shaded(self):
        panel = spectra_panel(self.ds, DataSelection(scattercut=[[450, 470], [600, 620]]),
                              ViewSettings(), COLORS, rel_time=[1.0], shade_masked=True)
        fig, ax = plt.subplots()
        draw_panel(panel, ax)
        self.assertEqual(len(ax.patches), 2)

    def test_only_labelled_traces_enter_the_legend(self):
        panel = spectra_panel(self.ds, DataSelection(scattercut=[[450, 470], [600, 620]]),
                              ViewSettings(), COLORS, rel_time=[1.0, 10.0])
        fig, ax = plt.subplots()
        draw_panel(panel, ax)
        handles, labels = ax.get_legend_handles_labels()
        self.assertEqual(len(labels), 2)

    def test_the_limits_are_applied(self):
        panel = spectra_panel(self.ds, DataSelection(bordercut=[430, 670]),
                              ViewSettings(intensity_range=2e-3), COLORS, rel_time=[1.0])
        fig, ax = plt.subplots()
        draw_panel(panel, ax)
        self.assertAllClose(ax.get_xlim(), [430, 670])
        self.assertAllClose(ax.get_ylim(), [-2e-3, 2e-3])


class KineticsPanel(NumericTestCase):
    """The mirror of a spectra panel: signal against delay."""

    def setUp(self):
        self.ds, _ = make_dataset()
        self.view = ViewSettings(data_type="differential Absorption", lintresh=0.3, linscale=1)

    def _panel(self, selection=None, **kwargs):
        from KiMoPack.figures.prepare import kinetics_panel

        return kinetics_panel(self.ds, selection or DataSelection(wavelength_bin=10), self.view,
                              COLORS, wavelength=kwargs.pop("wavelength", [450, 550]), **kwargs)

    def test_one_trace_per_requested_wavelength(self):
        self.assertEqual(len(self._panel().traces), 2)

    def test_an_ignored_region_breaks_every_trace(self):
        selection = DataSelection(wavelength_bin=10, ignore_time_region=[[1, 5], [50, 100]])
        self.assertEqual(len(self._panel(selection=selection).traces), 6)

    def test_a_broken_trace_is_listed_once(self):
        selection = DataSelection(wavelength_bin=10, ignore_time_region=[[1, 5], [50, 100]])
        self.assertEqual(len(self._panel(selection=selection).legend_labels()), 2)

    def test_the_time_axis_is_symmetric_log_by_default(self):
        """Transient data crosses zero, so a plain log axis cannot show it."""
        panel = self._panel()
        self.assertEqual(panel.x.scale, "symlog")
        self.assertEqual(panel.x.linthresh, 0.3)

    def test_a_log_axis_cannot_start_at_or_below_zero(self):
        panel = self._panel(plot_type="log", timelimits=[-5, 500])
        self.assertEqual(panel.x.scale, "log")
        self.assertGreater(panel.x.limits[0], 0)

    def test_a_linear_axis_keeps_the_limits_it_was_given(self):
        panel = self._panel(plot_type="lin", timelimits=[-1, 200])
        self.assertEqual(panel.x.scale, "linear")
        self.assertEqual(panel.x.limits, [-1, 200])

    def test_limits_default_to_the_measured_range(self):
        panel = self._panel()
        self.assertAllClose(panel.x.limits,
                            [self.ds.index.values.min(), self.ds.index.values.max()])

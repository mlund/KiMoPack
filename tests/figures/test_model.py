"""The description of a figure, before anything is drawn.

Separating what to draw from how to draw it is what lets a second front end
exist, and it is what makes the decisions testable: limits, scales, colours
and which pieces of a masked trace survive are all settled here, with no
display involved.
"""

import unittest

import numpy as np

from KiMoPack.figures.model import AxisSpec, Figure, Image, Panel, Trace


class Traces(unittest.TestCase):
    def test_a_trace_carries_its_points_and_how_to_draw_them(self):
        trace = Trace(x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]),
                      label="450 nm", color="red", style="solid")
        self.assertEqual(trace.label, "450 nm")
        self.assertEqual(len(trace.x), 2)

    def test_a_trace_is_immutable(self):
        import dataclasses

        trace = Trace(x=np.array([1.0]), y=np.array([2.0]), label="a")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            trace.label = "b"

    def test_a_trace_without_a_label_stays_out_of_the_legend(self):
        """Continuation pieces of a broken trace must not be listed again."""
        self.assertFalse(Trace(x=np.array([1.0]), y=np.array([2.0])).in_legend)
        self.assertTrue(Trace(x=np.array([1.0]), y=np.array([2.0]), label="a").in_legend)

    def test_the_point_count_must_match(self):
        with self.assertRaises(ValueError):
            Trace(x=np.array([1.0, 2.0]), y=np.array([3.0]))


class Axes(unittest.TestCase):
    def test_an_axis_states_its_scale(self):
        axis = AxisSpec(label="Time in ps", scale="symlog", linthresh=0.3)
        self.assertEqual(axis.scale, "symlog")

    def test_only_known_scales_are_accepted(self):
        with self.assertRaises(ValueError):
            AxisSpec(label="x", scale="logarithmic")

    def test_a_symlog_axis_needs_its_threshold(self):
        """Without it the renderer would have to invent one."""
        with self.assertRaises(ValueError):
            AxisSpec(label="x", scale="symlog", linthresh=None)


class Panels(unittest.TestCase):
    def _panel(self, **kwargs):
        return Panel(x=AxisSpec(label="nm"), y=AxisSpec(label="OD"), **kwargs)

    def test_a_panel_holds_its_traces(self):
        panel = self._panel(traces=(Trace(x=np.array([1.0]), y=np.array([2.0]), label="a"),))
        self.assertEqual(len(panel.traces), 1)

    def test_shaded_regions_are_pairs(self):
        panel = self._panel(shaded=((450.0, 470.0), (600.0, 620.0)))
        self.assertEqual(len(panel.shaded), 2)

    def test_the_legend_lists_only_labelled_traces(self):
        panel = self._panel(traces=(
            Trace(x=np.array([1.0]), y=np.array([2.0]), label="450 nm"),
            Trace(x=np.array([1.0]), y=np.array([2.0])),
        ))
        self.assertEqual(panel.legend_labels(), ["450 nm"])


class Figures(unittest.TestCase):
    def test_a_figure_holds_panels_and_a_name(self):
        panel = Panel(x=AxisSpec(label="nm"), y=AxisSpec(label="OD"))
        figure = Figure(panels=(panel,), name="spectra")
        self.assertEqual(figure.name, "spectra")
        self.assertEqual(len(figure.panels), 1)

    def test_the_layout_defaults_to_a_single_column(self):
        panel = Panel(x=AxisSpec(label="nm"), y=AxisSpec(label="OD"))
        self.assertEqual(Figure(panels=(panel, panel), name="x").layout, (2, 1))

    def test_an_explicit_layout_is_kept(self):
        panel = Panel(x=AxisSpec(label="nm"), y=AxisSpec(label="OD"))
        self.assertEqual(Figure(panels=(panel,) * 4, name="x", layout=(2, 2)).layout, (2, 2))

    def test_the_layout_must_fit_the_panels(self):
        panel = Panel(x=AxisSpec(label="nm"), y=AxisSpec(label="OD"))
        with self.assertRaises(ValueError):
            Figure(panels=(panel,) * 5, name="x", layout=(2, 2))


class Images(unittest.TestCase):
    def test_an_image_carries_its_grid_and_colour_limits(self):
        image = Image(values=np.zeros((3, 4)), x=np.arange(4.0), y=np.arange(3.0),
                      limits=(-1, 1), colormap="seismic")
        self.assertEqual(image.values.shape, (3, 4))
        self.assertEqual(image.limits, (-1, 1))

    def test_the_grid_must_match_the_axes(self):
        """A transposed array would otherwise draw a plausible wrong picture."""
        with self.assertRaises(ValueError):
            Image(values=np.zeros((3, 4)), x=np.arange(3.0), y=np.arange(4.0))

    def test_a_map_can_be_masked_in_both_directions(self):
        panel = Panel(x=AxisSpec(label="nm"), y=AxisSpec(label="ps"),
                      shaded=((450.0, 470.0),), shaded_y=((1.0, 5.0),))
        self.assertEqual(len(panel.shaded), 1)
        self.assertEqual(len(panel.shaded_y), 1)

    def test_a_colour_scale_can_be_logarithmic(self):
        image = Image(values=np.zeros((2, 2)), x=np.arange(2.0), y=np.arange(2.0),
                      log_scale=True, linscale=2)
        self.assertTrue(image.log_scale)
        self.assertEqual(image.linscale, 2)

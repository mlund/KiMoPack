"""How a figure should look, and where it should go.

These settings are read off a project and handed down through every drawing
layer. Grouping them keeps the 40-odd keyword forwarding out of the code and
makes it impossible to pass the colour map but forget the colour scale.
"""

import unittest

import KiMoPack.plot_func as pf
from KiMoPack.figures.settings import OutputSpec, ViewSettings

from ..synthetic import make_dataset


def _project():
    ds, _ = make_dataset()
    return pf.TA("synthetic", ds=ds)


class ReadFromAProject(unittest.TestCase):
    def test_it_takes_the_display_settings(self):
        ta = _project()
        ta.intensity_range = 3e-3
        ta.log_scale = True
        ta.lintresh = 0.5
        view = ViewSettings.from_project(ta)
        self.assertEqual(view.intensity_range, 3e-3)
        self.assertTrue(view.log_scale)
        self.assertEqual(view.lintresh, 0.5)

    def test_the_colour_maps_come_across(self):
        ta = _project()
        view = ViewSettings.from_project(ta)
        self.assertIs(view.cmap, ta.cmap)
        self.assertIs(view.line_colors, ta.line_colors)

    def test_an_override_wins_over_the_project(self):
        """Plot calls may pass their own colour map for one figure."""
        ta = _project()
        view = ViewSettings.from_project(ta, cmap="viridis")
        self.assertEqual(view.cmap, "viridis")
        self.assertIs(view.line_colors, ta.line_colors)

    def test_optional_metadata_is_absent_by_default(self):
        """'values' is only set by callers that summarise several scans."""
        self.assertIsNone(ViewSettings.from_project(_project()).values)

    def test_optional_metadata_is_picked_up_when_present(self):
        ta = _project()
        ta.values = [1, 2, 3]
        self.assertEqual(ViewSettings.from_project(ta).values, [1, 2, 3])


class Immutability(unittest.TestCase):
    def test_settings_cannot_be_edited_in_place(self):
        import dataclasses

        view = ViewSettings(log_scale=True)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            view.log_scale = False

    def test_replacing_gives_a_new_value(self):
        view = ViewSettings(log_scale=True)
        self.assertFalse(view.replace(log_scale=False).log_scale)
        self.assertTrue(view.log_scale)


class Output(unittest.TestCase):
    def test_it_takes_where_and_whether_to_save(self):
        ta = _project()
        ta.save_figures_to_folder = True
        spec = OutputSpec.from_project(ta, path="results", savetype="pdf")
        self.assertTrue(spec.save_figures_to_folder)
        self.assertEqual(spec.savetype, "pdf")

    def test_the_filename_falls_back_to_the_project(self):
        ta = _project()
        self.assertEqual(OutputSpec.from_project(ta).filename, ta.filename)

    def test_an_explicit_filename_wins(self):
        self.assertEqual(OutputSpec.from_project(_project(), filename="run7").filename, "run7")

    def test_the_title_falls_back_to_the_filename(self):
        """Every figure is titled after whatever it will be saved as."""
        spec = OutputSpec.from_project(_project(), filename="run7")
        self.assertEqual(spec.title, "run7")

    def test_an_explicit_title_wins(self):
        spec = OutputSpec.from_project(_project(), filename="run7", title="Sample A")
        self.assertEqual(spec.title, "Sample A")

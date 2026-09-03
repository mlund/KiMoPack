"""Guards the public API against accidental change.

KiMoPack is installed from PyPI and driven from notebooks that call functions
by name with keyword arguments, so a rename or a dropped keyword breaks users
silently and only at their next run. This compares the live surface against a
committed snapshot; regenerate it with ``python -m tests._surface`` when the
API is meant to change, and the diff becomes part of the review.
"""

import difflib
import pathlib
import unittest

import KiMoPack.plot_func as plot_func

from ._surface import dump

GOLDEN = pathlib.Path(__file__).with_name("facade_surface.txt")


class PublicSurface(unittest.TestCase):
    def test_matches_committed_snapshot(self):
        expected = GOLDEN.read_text().splitlines()
        actual = dump(plot_func)
        if actual == expected:
            return
        diff = "\n".join(difflib.unified_diff(expected, actual, "committed", "live", lineterm=""))
        self.fail(
            "The public API of plot_func changed.\n"
            "If this was intended, run `python -m tests._surface` and commit the result.\n\n" + diff
        )

    def test_notebook_entry_points_are_importable(self):
        """The names the bundled notebooks actually call.

        Counted across the 18 notebooks in Workflow_tools and
        Tutorial_Notebooks; these carry the most breakage risk.
        """
        for name in [
            "TA",
            "cm",
            "changefonts",
            "Summarize_scans",
            "GUI_open",
            "halfsize",
            "Frame_golay",
            "pardf_to_par",
            "Species_Spectra",
        ]:
            with self.subTest(name=name):
                self.assertTrue(hasattr(plot_func, name))

    def test_documented_members_still_resolve(self):
        """Every name in docs/source/plot_func.rst's autodoc allowlist.

        The allowlist is hand-maintained, and Sphinx only warns when an entry
        vanishes, so nothing else would notice a rename until readthedocs
        quietly dropped the page.
        """
        rst = pathlib.Path(__file__).parents[1] / "docs" / "source" / "plot_func.rst"
        for line in rst.read_text().splitlines():
            line = line.strip()
            if not line.startswith((":members:", ":private-members:")):
                continue
            for name in line.split(":", 2)[2].split(","):
                name = name.strip()
                if not name:
                    continue
                with self.subTest(name=name):
                    self.assertTrue(_resolves(name), f"{name} is documented but missing")


def _resolves(name):
    """Find a documented name on the module or on TA.

    Leading-underscore names in the allowlist are private TA methods, which
    Python mangles to ``_TA__name`` — except dunders like ``__init__``, which
    it leaves alone.
    """
    if hasattr(plot_func, name) or hasattr(plot_func.TA, name):
        return True
    if name.startswith("__") and not name.endswith("__"):
        return hasattr(plot_func.TA, "_TA" + name)
    return False

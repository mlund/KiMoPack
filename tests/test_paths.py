"""Path resolution and the small file-rewriting helper.

``check_folder`` is called from 22 places to answer "where does this go?", and
it creates the directory as it answers, so its rules are worth pinning down.
"""

import os
import tempfile
import unittest
from pathlib import Path

from KiMoPack import paths


class CheckFolder(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name).resolve()
        self._cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(os.chdir, self._cwd)

    def test_everything_none_gives_the_working_directory(self):
        self.assertEqual(paths.check_folder().resolve(), self.tmp)

    def test_absolute_path_is_used_as_is(self):
        target = self.tmp / "results"
        self.assertEqual(paths.check_folder(path=target).resolve(), target)

    def test_relative_path_hangs_off_the_working_directory(self):
        self.assertEqual(paths.check_folder(path="figures").resolve(), self.tmp / "figures")

    def test_relative_path_hangs_off_an_absolute_current_path(self):
        base = self.tmp / "base"
        result = paths.check_folder(path="figures", current_path=base)
        self.assertEqual(result.resolve(), base / "figures")

    def test_relative_current_path_falls_back_to_the_working_directory(self):
        result = paths.check_folder(path="figures", current_path="somewhere")
        self.assertEqual(result.resolve(), self.tmp / "figures")

    def test_filename_is_appended(self):
        result = paths.check_folder(path="out", filename="fit.png")
        self.assertEqual(result.resolve(), self.tmp / "out" / "fit.png")
        self.assertTrue((self.tmp / "out").is_dir(), "the directory should exist")

    def test_the_directory_is_created_including_parents(self):
        paths.check_folder(path="a/b/c")
        self.assertTrue((self.tmp / "a" / "b" / "c").is_dir())

    def test_a_filename_does_not_become_a_directory(self):
        paths.check_folder(path="out", filename="fit.png")
        self.assertFalse((self.tmp / "out" / "fit.png").exists())

    def test_byte_paths_decode_to_real_paths(self):
        """Byte paths are documented as supported.

        Formatting them with %s yields the string "b'out'", so the data would
        land in a directory whose name contains the quotes.
        """
        result = paths.check_folder(path=b"out", filename=b"fit.png")
        self.assertEqual(result.resolve(), self.tmp / "out" / "fit.png")


class CleanDoubleString(unittest.TestCase):
    def test_collapses_repeated_dashes_and_dots(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "chirp.dat"
            target.write_text("1.5--2.5 and 3..4\n")
            paths.clean_double_string("chirp.dat", path=tmp)
            self.assertEqual(target.read_text(), "1.5-2.5 and 3.4\n")

    def test_shortening_does_not_leave_a_tail_behind(self):
        """The file is rewritten in place, so it must be truncated."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "chirp.dat"
            target.write_text("a..........b")
            paths.clean_double_string("chirp.dat", path=tmp)
            self.assertEqual(target.read_text(), "a.b")

    def test_dashes_collapse_once_per_pair(self):
        """A single pass, which covers the '1.5--2.5' case it exists for."""
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "chirp.dat"
            target.write_text("a----b")
            paths.clean_double_string("chirp.dat", path=tmp)
            self.assertEqual(target.read_text(), "a--b")

"""Recognising and reading datasets with only a few wavelength channels."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas

from KiMoPack import sparse

from .support import NumericTestCase
from .synthetic import make_dataset, make_sparse_dataset


class IsSparseWavelength(NumericTestCase):
    def test_a_handful_of_channels_is_sparse(self):
        ds, _ = make_sparse_dataset()
        self.assertTrue(sparse.is_sparse_wavelength(ds))

    def test_a_dense_evenly_spaced_spectrum_is_not(self):
        ds, _ = make_dataset()
        self.assertFalse(sparse.is_sparse_wavelength(ds))

    def test_one_large_gap_makes_a_dense_axis_sparse(self):
        """Two detector bands with nothing in between are not a spectrum."""
        ds, _ = make_dataset()
        keep = [c for c in ds.columns if c < 450 or c > 650]
        self.assertTrue(sparse.is_sparse_wavelength(ds.loc[:, keep]))

    def test_a_single_channel_is_sparse(self):
        ds, _ = make_dataset()
        self.assertTrue(sparse.is_sparse_wavelength(ds.iloc[:, :1]))


class ReadSparseSIA(unittest.TestCase):
    """This reader referenced an undefined name and had never run."""

    def _write(self, tmp, sep="\t"):
        target = Path(tmp) / "sample.SIA"
        frame = pandas.DataFrame(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            index=[0.0, 1.0, 2.0], columns=[500.0, 600.0])
        frame.to_csv(target, sep=sep)
        return target

    def test_reads_a_matrix_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp)
            ds, data_type, baseunit = sparse.read_sparse_SIA("sample.SIA", path=tmp)
            self.assertEqual(ds.shape, (3, 2))
            self.assertEqual(list(ds.columns), [500.0, 600.0])
            self.assertEqual(baseunit, "ps")
            self.assertIn("Absorption", data_type)

    def test_labels_the_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp)
            ds, _, _ = sparse.read_sparse_SIA("sample.SIA", path=tmp, units="nm")
            self.assertEqual(ds.index.name, "Time in ps")
            self.assertEqual(ds.columns.name, "nm")

    def test_the_time_axis_can_be_shifted_and_scaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp)
            ds, _, _ = sparse.read_sparse_SIA("sample.SIA", path=tmp,
                                              shift_times_by=10.0, divide_times_by=2.0)
            np.testing.assert_allclose(ds.index.values, [5.0, 5.5, 6.0])

    def test_works_without_a_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = self._write(tmp)
            ds, _, _ = sparse.read_sparse_SIA(str(target))
            self.assertEqual(ds.shape, (3, 2))

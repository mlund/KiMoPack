"""Datasets with only a few wavelength channels.

Transient X-ray emission and single-channel experiments produce a handful of
widely separated detection energies rather than a continuous spectrum. Most of
the package assumes a dense axis it can bin and interpolate across, so these
have to be recognised and handled differently.
"""

import os

import numpy as np
import pandas


def is_sparse_wavelength(ds, max_columns=20, gap_ratio=5.0):
    """True when the wavelength axis is too sparse to treat as a spectrum.

    Either there are very few channels, or the axis has a hole in it far
    larger than its typical spacing — two detector bands with nothing between
    them is not a spectrum, however many points each band has.

    Parameters
    ----------
    ds : pandas.DataFrame
        Data matrix, times by wavelength.
    max_columns : int, optional
        At or below this many channels the axis is sparse regardless of spacing.
    gap_ratio : float, optional
        How many times the median spacing the largest gap must exceed.
    """
    wl = np.sort(ds.columns.values.astype(float))
    if len(wl) <= max_columns:
        return True
    gaps = np.diff(wl)
    if len(gaps) == 0:
        return True
    median_gap = np.median(gaps)
    if median_gap == 0:
        return True
    return gaps.max() / median_gap > gap_ratio


def read_sparse_SIA(filename, path=None, sep="\t", decimal=".", baseunit="ps", units="nm",
                    data_type="diff. Absorption in OD", shift_times_by=None,
                    divide_times_by=None, **kwargs):
    """Read a matrix whose wavelength columns are sparse or unevenly spaced.

    Returns the three-tuple that plugs into the ``conversion_function``
    interface of :class:`KiMoPack.plot_func.TA`.

    Parameters
    ----------
    filename : str
        File to read; joined with ``path`` when that is given.
    path : str, optional
        Directory containing the file.
    sep, decimal : str, optional
        CSV dialect.
    baseunit, units, data_type : str, optional
        Axis labels.
    shift_times_by, divide_times_by : float, optional
        Applied to the time axis after reading, in that order.

    Returns
    -------
    tuple
        ``(DataFrame, data_type, baseunit)``
    """
    filepath = os.path.join(path, filename) if path else filename
    ds = pandas.read_csv(filepath, sep=sep, decimal=decimal, index_col=0, header=0)
    ds.columns = ds.columns.values.astype(float)
    ds.index = ds.index.values.astype(float)
    ds = ds.dropna(how="all", axis=0).dropna(how="all", axis=1)
    ds = ds.sort_index(axis=0).sort_index(axis=1)
    if shift_times_by is not None:
        ds.index = ds.index.values + shift_times_by
    if divide_times_by is not None:
        ds.index = ds.index.values / divide_times_by
    ds.index.name = f"Time in {baseunit}" if baseunit in ("ps", "ns", "fs") else baseunit
    ds.columns.name = units
    return ds, data_type, baseunit

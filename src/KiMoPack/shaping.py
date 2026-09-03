"""Getting from the full measurement matrix to the part you want.

Every plot and the fit itself reach the data through :func:`sub_ds`, so what
it returns is what "the data" means everywhere downstream. It applies, in a
fixed order: spectral crop, spectral binning, time binning, time crop, time
masking, spectral masking, and finally extraction of individual traces or
spectra.

The order matters and is not negotiable — binning before cropping would
average across the crop edge — which is why this lives in one place rather
than being reassembled at each call site.
"""

import numbers

import numpy as np
import pandas
from scipy.stats import binned_statistic

from .numerics import find_nearest_index, nm_to_ev
from .regions import normalise_cuts


def _bin_axis(ds, x, y, width, what):
    """Average columns into bins of ``width`` along the already-converted axis ``x``.

    Detectors are often dense at one end of the range and sparse at the other.
    Where the native spacing is already wider than the requested bin there is
    nothing to average, so that tail is passed through untouched and only the
    dense head is rebinned.
    """
    if (x[1:] - x[:-1] > width).all():
        raise ValueError(f"{what} bins are to small for the data")

    rebin_max = np.argmin((x[1:] - x[:-1]) < width)
    if rebin_max == 0:
        # argmin returns 0 when every spacing is fine, so rebin everything.
        rebin_max = len(x)

    if rebin_max < len(x):
        bins = np.arange(x.min(), x[rebin_max], width)
        bin_means, bin_edges = binned_statistic(
            x[:rebin_max], ds.values[:, :rebin_max], statistic="mean", bins=bins)[:2]
        centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        return pandas.concat(
            (pandas.DataFrame(bin_means, index=y, columns=centres), ds.iloc[:, rebin_max:]),
            axis=1, join="outer")

    bins = np.arange(x.min(), x.max() + width, width)
    bin_means, bin_edges = binned_statistic(x, ds.values, statistic="mean", bins=bins)[:2]
    centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    return pandas.DataFrame(bin_means, index=y, columns=centres)


def _mask_columns(ds, cuts, drop, to_energy):
    """Blank the named spectral regions, by value lookup on the column axis."""
    x = ds.columns.values.astype("float")
    for low, high in _cut_pairs(cuts, to_energy):
        lower = find_nearest_index(x, low)
        upper = find_nearest_index(x, high)
        ds.iloc[:, lower:upper] = np.nan if drop else 0
    return ds


def _cut_pairs(cuts, to_energy):
    """Cut regions as pairs, converted to eV when the axis is energy.

    Wavelength and energy run in opposite directions, so a pair given in nm
    comes back reversed and has to be re-sorted.
    """
    if to_energy:
        if isinstance(cuts[0], numbers.Number):
            cuts = [list(nm_to_ev(cuts))[::-1]]
        else:
            cuts = [list(nm_to_ev(pair))[::-1] for pair in cuts]
    return normalise_cuts(cuts)


def sub_ds(ds, times=None, time_width_percent=0, ignore_time_region=None, drop_ignore=False,
           wave_nm_bin=None, baseunit=None, scattercut=None, drop_scatter=False, bordercut=None,
           timelimits=None, wavelength_bin=None, wavelength=None, time_bin=None,
           equal_energy_bin=None, from_fit=False):
    """Crop, bin, mask and slice the measurement matrix.

    Returns a new frame; the input is never modified.

    Parameters
    ----------
    ds : pandas.DataFrame
        Times down the index, wavelengths across the columns.
    times : list, optional
        Extract spectra at these delays. Mutually exclusive with ``wavelength``.
    time_width_percent : float, optional
        Average each extracted spectrum over this percentage around its delay.
    ignore_time_region : list, optional
        Delay region(s) to blank, as a pair or a list of pairs.
    drop_ignore : bool, optional
        Remove the ignored rows instead of zeroing them.
    wave_nm_bin : float, optional
        Average the spectrum into bins this wide, in nm.
    baseunit : str, optional
        Time unit used when labelling extracted spectra.
    scattercut : list, optional
        Spectral region(s) to blank, as a pair or a list of pairs.
    drop_scatter : bool, optional
        Remove the masked columns instead of zeroing them.
    bordercut : list, optional
        Keep only this spectral range.
    timelimits : list, optional
        Keep only this delay range.
    wavelength_bin : float, optional
        Width averaged around each extracted wavelength.
    wavelength : float or list, optional
        Extract kinetics at these wavelengths. Mutually exclusive with ``times``.
    time_bin : int, optional
        Average this many neighbouring delays together.
    equal_energy_bin : float, optional
        Bin in eV instead of nm; the column axis becomes energy.
    from_fit : bool, optional
        The frame already came out of a fit, so it is binned and needs no crop.

    Returns
    -------
    pandas.DataFrame
    """
    if (wavelength is not None) and (times is not None):
        raise ValueError("can not get wavelength and times back")

    ds = ds.copy()
    time_label = ds.index.name
    energy_label = ds.columns.name

    if (bordercut is not None) and not from_fit:
        ds.columns = ds.columns.astype("float")
        ds = ds.loc[:, bordercut[0]:bordercut[1]]

    if (equal_energy_bin is not None) and (wavelength is None):
        x = nm_to_ev(ds.columns.values.astype("float"))
        y = ds.index.values.astype("float")
        energy_label = "Energy in eV"
        if from_fit:
            ds.columns = x
        else:
            ds = _bin_axis(ds, x, y, equal_energy_bin, "equal_energy_bin")
    elif (wave_nm_bin is not None) and (wavelength is None):
        x = ds.columns.values.astype("float")
        y = ds.index.values.astype("float")
        ds = _bin_axis(ds, x, y, wave_nm_bin, "wavelength_nm_bins")

    if time_bin is not None:
        time = ds.index.values.astype("float")
        y = ds.columns.values.astype("float")
        time_bins = time[::int(time_bin)]
        bin_means, bin_edges = binned_statistic(time, ds.values.T, statistic="mean",
                                                bins=time_bins)[:2]
        centres = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        ds = pandas.DataFrame(bin_means, index=y, columns=centres).T

    if timelimits is not None:
        ds.index = ds.index.astype("float")
        ds = ds.loc[timelimits[0]:timelimits[1], :]

    if ignore_time_region is not None:
        ds = ds.fillna(value=0)
        ds.index = ds.index.astype("float")
        for low, high in normalise_cuts(ignore_time_region):
            ds.loc[low:high, :] = np.nan if drop_ignore else 0
        ds = ds.dropna(axis=0)

    if scattercut is not None:
        ds = ds.fillna(value=0)
        ds = _mask_columns(ds, scattercut, drop_scatter, equal_energy_bin is not None)
        ds = ds.dropna(axis=1)

    ds.index.name = time_label
    ds.columns.name = energy_label

    if wavelength is not None:
        ds = _extract_wavelengths(ds, wavelength, wavelength_bin, equal_energy_bin, from_fit,
                                  time_label, energy_label)
    if times is not None:
        ds = _extract_times(ds, times, time_width_percent, baseunit, time_label, energy_label)

    ds = ds.fillna(value=0)
    if equal_energy_bin is not None:
        ds = ds.sort_index(axis=1, ascending=False)
    return ds


def _extract_wavelengths(ds, wavelength, wavelength_bin, equal_energy_bin, from_fit,
                         time_label, energy_label):
    """Average a window around each wavelength into one kinetic trace."""
    if not hasattr(wavelength, "__iter__"):
        wavelength = np.array([wavelength])
    if len(wavelength) > 1:
        wavelength = np.sort(np.asarray(wavelength))

    col_min = float(ds.columns.min())
    col_max = float(ds.columns.max())
    columns = {}
    for wave in wavelength:
        upper = wave + wavelength_bin / 2
        lower = wave - wavelength_bin / 2
        if equal_energy_bin is not None and from_fit:
            lower, upper = nm_to_ev(upper), nm_to_ev(lower)
            wave = nm_to_ev(wave)
        lower = max(lower, col_min)
        upper = min(upper, col_max)
        # Silently skip anything the crop already removed.
        if lower > upper or wave in columns:
            continue
        columns[wave] = ds.loc[:, lower:upper].mean(axis="columns")

    if not columns:
        return ds
    out = pandas.DataFrame(columns)
    out.columns = out.columns.astype("float")
    out.columns.name = energy_label
    out.index.name = time_label
    return out


def _extract_times(ds, times, time_width_percent, baseunit, time_label, energy_label):
    """Take one spectrum per requested delay, optionally averaged over a window."""
    if not hasattr(times, "__iter__"):
        times = np.array([times])
    if baseunit is None:
        baseunit = "ps"
    time_scale = ds.index.values

    columns = {}
    for time in times:
        if time_width_percent > 0:
            margin = abs(time) * time_width_percent / 100.0
            first = find_nearest_index(time_scale, time - margin)
            last = find_nearest_index(time_scale, time + margin)
            lower, upper = time_scale[first], time_scale[last]
            middle = (lower + upper) / 2
            label = f"{middle:.3g} {baseunit} ({lower:.3g} - {upper:.3g} {baseunit})"
            columns[label] = ds.iloc[first:last, :].mean(axis="rows")
        else:
            index = find_nearest_index(time_scale, time)
            columns[f"{time_scale[index]:.3g} {baseunit}"] = ds.iloc[index, :]

    out = pandas.DataFrame(columns)
    out.columns.name = time_label
    out.index.name = energy_label
    return out

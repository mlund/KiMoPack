"""Array and DataFrame helpers with no I/O and no plotting.

Everything here is deterministic and side-effect free, which is what lets the
rest of the package be tested: the layers above compose these rather than
reimplementing interpolation and smoothing inline.
"""

import numpy as np
import pandas
import scipy.stats
from scipy.signal import savgol_filter
from scipy.special import erf


def flatten(mainlist):
    """Unpack one level of nesting."""
    return [entry for sublist in mainlist for entry in sublist]


def nearest_neighbor_method3(X, q):
    """Index of the column of ``X`` closest to the point ``q``."""
    X = X.T
    return np.argmin(np.sum((X - q) ** 2, axis=1))


def log_and(x, y, *args):
    """Logical and of two or more masks."""
    result = np.logical_and(x, y)
    for a in args:
        result = np.logical_and(result, a)
    return result


def find_nearest_index(arr, value, con_str=False):
    """Index of the entry closest to ``value``; ties take the first.

    ``con_str`` converts first, for axes read from ASCII that arrive as
    strings — comparing those as text would rank '1000' next to '100'.
    """
    if con_str:
        arr = np.array(arr, dtype="float")
    return int((np.abs(arr - value)).argmin())


def find_nearest(arr, value, con_str=False):
    """The entry of ``arr`` closest to ``value``."""
    return arr[find_nearest_index(arr, value, con_str=con_str)]


def rebin(ori_df, new_x):
    """Linearly interpolate onto a new index, preserving column order."""
    if isinstance(ori_df, pandas.DataFrame):
        old_x = ori_df.index.values.astype("float")
        return pandas.DataFrame(
            {col: np.interp(new_x, old_x, ori_df[col].values) for col in ori_df.columns},
            index=new_x,
            columns=ori_df.columns,
        )
    if isinstance(ori_df, pandas.Series):
        old_x = ori_df.index.values.astype("float")
        return pandas.Series(np.interp(new_x, old_x, ori_df.values), index=new_x)
    raise TypeError(f"rebin needs a DataFrame or Series, got {type(ori_df).__name__}")


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Savitzky-Golay filter, in the argument order the rest of the package uses."""
    return savgol_filter(x=y, window_length=window_size, polyorder=order, deriv=deriv, delta=rate)


def Frame_golay(df, window=5, order=3, transpose=False):
    """Savitzky-Golay smoothing of every column, or of a Series.

    Returns a new object: smoothing in place would corrupt the raw data that
    callers still plot alongside the smoothed trace.

    Parameters
    ----------
    df : pandas.DataFrame, pandas.Series
        Data to smooth.
    window : int, optional
        Number of points in the smoothing window, clamped to the data length
        and forced odd, as the filter requires.
    order : int, optional
        Polynomial order. ``order=1`` reduces this to a moving average.
    transpose : bool, optional
        Smooth across rows instead of down columns.

    Returns
    -------
    pandas.DataFrame or pandas.Series
    """
    if transpose:
        df = df.T
    window = min(len(df.index.values), window)
    order = min(len(df.index.values), order)
    if window % 2 == 0:
        window -= 1

    if isinstance(df, pandas.DataFrame):
        smoothed = pandas.DataFrame(
            {col: savitzky_golay(df.loc[:, col].values, window, order) for col in df.columns},
            index=df.index,
            columns=df.columns,
        )
        return smoothed.T if transpose else smoothed
    if isinstance(df, pandas.Series):
        return pandas.Series(savitzky_golay(df.values, window, order), index=df.index)
    raise TypeError(f"Frame_golay needs a DataFrame or Series, got {type(df).__name__}")


def shift(df, name=None, shift=None):
    """Re-read the named columns from a curve displaced by ``shift``.

    The index is left alone; the values become ``f(x - shift)``. Used to line
    up spectra that sit on slightly different energy axes.
    """
    if name is None:
        name = df.columns
    if isinstance(name, str):
        name = [name]
    result = df.copy()
    ori_en = np.array(df.index, dtype="float")
    for nam in name:
        ori_dat = df[nam].values
        if ori_en[0] > ori_en[1]:
            # np.interp needs an increasing sample axis; reverse and undo.
            dat = np.interp(ori_en[::-1], ori_en[::-1] + shift, ori_dat[::-1])[::-1]
        else:
            dat = np.interp(ori_en, ori_en + shift, ori_dat)
        result[nam] = dat
    return result


def norm(df):
    """Scale every column onto the interval 0 to 1."""
    return df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))


def rise(x, sigma=0.1, begin=0):
    """Instrument response as an error function climbing from 0 to 1.

    ``sigma`` is the width after which the response reaches 50%, and ``begin``
    marks the onset. This is a ramp applied to a decay, not a true convolution.
    """
    return (erf((x - begin - sigma) * np.sqrt(2) / (sigma)) + 1) / 2


def gauss(t, sigma=0.1, mu=0):
    """Normalised Gaussian density."""
    y = np.exp(-0.5 * ((t - mu) ** 2) / sigma**2)
    y /= sigma * np.sqrt(2 * np.pi)
    return y


def s2_vs_smin2(Spectral_points=512, Time_points=130, number_of_species=3, fitted_kinetic_pars=7,
                target_quality=0.95):
    """Variance ratio separating a significantly worse fit from the best one.

    An F-test against the null hypothesis that the extra parameters buy
    nothing. Used to turn a target confidence level into the chi-square
    threshold that bounds a parameter's confidence interval.
    """
    data_points = Spectral_points * Time_points
    fitted_parameter = Spectral_points * number_of_species + fitted_kinetic_pars
    Free_points = data_points - fitted_parameter
    f_stat = scipy.stats.f.ppf(q=target_quality, dfn=fitted_parameter, dfd=Free_points)
    return 1 + (fitted_parameter * f_stat / Free_points)

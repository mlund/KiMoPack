"""Turning parameters into a modelled dataset.

This is the inner loop of every fit. Given a measurement and a parameter set
it builds the concentration profile, solves for the species spectra, and
reports how far the result sits from the data.

It is deliberately free of reporting: no printing, no progress throttling, no
parameter dumps, no files. Those belong to the optimiser's bookkeeping, and
keeping them out is what makes the physics testable on its own.
"""

import numpy as np
import pandas

from ..numerics import rebin
from .amplitudes import fill_int
from .models import resolve_model


def sample_times(times_ori, pardf, pulse_sample=None, sub_sample=None):
    """Add integration points the measurement does not provide.

    The sequential model integrates forward through the excitation pulse, so
    it needs time points inside that pulse — which is usually far narrower
    than the delay steps the experiment recorded. ``pulse_sample`` densifies
    the axis around t0, ``sub_sample`` subdivides every measured interval.
    The extra points are dropped again once the concentrations are known.

    Parameters
    ----------
    times_ori : numpy.ndarray
        The measured delays.
    pardf : pandas.DataFrame
        Parameter table; ``t0`` and ``resolution`` locate the pulse.
    pulse_sample : bool, int or iterable, optional
        Densify around t0. An iterable is used as the pump region directly.
    sub_sample : int, optional
        Subdivide each measured interval into this many steps.

    Returns
    -------
    numpy.ndarray
        Sorted, unique times including every measured delay.
    """
    times = times_ori
    if pulse_sample is not None:
        t0 = float(pardf.loc["t0", "value"])
        resolution = float(pardf.loc["resolution", "value"])
        if hasattr(pulse_sample, "__iter__"):
            pump_region = pulse_sample
        else:
            pump_region = np.linspace(t0 - 4 * resolution, t0 + 4 * resolution, 20)
        if np.max(pump_region) < times_ori.min():
            # The pulse sits entirely before the first measured point; bridge
            # the gap so the integration is not asked to leap across it.
            connection = np.arange(np.max(pump_region), times_ori.min(), resolution / 10)
            times = np.unique(np.sort(np.hstack((pump_region, connection, times_ori))))
        else:
            times = np.unique(np.sort(np.hstack((pump_region, times_ori))))
    if sub_sample is not None:
        pieces = [times]
        for i in range(1, sub_sample, 1):
            pieces.append(times_ori[:-1] + ((times_ori[1:] - times_ori[:-1]) * i / sub_sample))
        times = np.unique(np.hstack(pieces))
        times.sort()
    return times


def build_concentrations(ds, pardf, mod, final=False, sub_sample=None, pulse_sample=None,
                         optimise_fast=True):
    """Concentration of every species at each measured delay.

    While the optimiser is searching, a model may name a cheaper stand-in that
    gives the same rate constants far faster; the expensive one is used for
    the final evaluation that produces the reported result. Joint fits across
    several datasets pass ``optimise_fast=False`` and integrate throughout,
    which is what they have always done.
    """
    times_ori = ds.index.values.astype("float")
    times = sample_times(times_ori, pardf, pulse_sample, sub_sample)

    model = resolve_model(mod)
    if optimise_fast and not final and model.optimise_with is not None:
        model = resolve_model(model.optimise_with)

    c = model.build(times=times, pardf=pardf)
    c = c.loc[times_ori, :]
    c.index.name = ds.index.name
    return c


def _outer_product(c_column, spectrum):
    """The matrix one species with a known spectrum contributes."""
    A, B = np.meshgrid(c_column.values, spectrum.values)
    return pandas.DataFrame((A * B).T, index=c_column.index, columns=spectrum.index.values)


def _prepare_external_spectra(ds, c, ext_spectra, pardf):
    """Subtract species whose spectra are already known.

    Some species have been measured separately — a known photoproduct, a
    reference compound. Their contribution is removed from the data before
    solving, so the solve only has to explain what is left. Unless the
    spectrum is only a guide, the species is also dropped from the solve.
    """
    ext_spectra = ext_spectra.sort_index()
    if "ext_spectra_shift" in list(pardf.index.values):
        ext_spectra.index = ext_spectra.index.values + pardf.loc["ext_spectra_shift", "value"]
    ext_spectra = rebin(ext_spectra, ds.columns.values.astype(float))
    if "ext_spectra_scale" in list(pardf.index.values):
        ext_spectra = ext_spectra * pardf.loc["ext_spectra_scale", "value"]

    guided = "ext_spectra_guide" in list(pardf.index.values)
    c_for_solve = c.copy()
    for col in ext_spectra.columns.values:
        ds = ds - _outer_product(c.loc[:, col], ext_spectra.loc[:, col])
        if not guided:
            c_for_solve = c_for_solve.drop(col, axis=1)
    return ds, c_for_solve, ext_spectra


def _restore_external_spectra(re, c, ext_spectra, pardf):
    """Put the externally known species back into the reported result.

    The matrices are only rebuilt when they are present: a mid-fit shape dump
    carries the spectra but not the full reconstruction.
    """
    guided = "ext_spectra_guide" in list(pardf.index.values)
    for col in ext_spectra.columns.values:
        if guided:
            re["DAC"][col] = re["DAC"][col] + ext_spectra.loc[:, col].values
        else:
            re["DAC"][col] = ext_spectra.loc[:, col].values
            re["c"][col] = c.loc[:, col].values
        if "A" in re:
            contribution = _outer_product(c.loc[:, col], ext_spectra.loc[:, col])
            re["A"] = re["A"] + contribution
            re["AC"] = re["AC"] + contribution


def simulate(ds, pardf, mod, final=False, sub_sample=None, pulse_sample=None,
             ext_spectra=None, return_shapes=False, optimise_fast=True):
    """Model the dataset and measure the mismatch.

    Parameters
    ----------
    ds : pandas.DataFrame
        Measurement, delays down the index and wavelengths across the columns.
    pardf : pandas.DataFrame
        Parameter table, with rate constants already in linear space.
    mod : str or callable
        Kinetic model name, or a user function taking ``(times, pardf)``.
    final : bool, optional
        Produce the full reconstruction rather than only the error. The
        optimiser leaves this off until it has converged.
    sub_sample, pulse_sample : optional
        Extra integration points; see :func:`sample_times`.
    ext_spectra : pandas.DataFrame, optional
        Spectra of species that are already known.
    return_shapes : bool, optional
        Include the spectra without the full reconstruction.
    optimise_fast : bool, optional
        Allow a model to substitute a cheaper stand-in while searching.

    Returns
    -------
    dict
        Always ``error``. With ``final``, also ``A``, ``AC``, ``AE``, ``DAC``,
        ``c`` and ``r2``.
    """
    c = build_concentrations(ds, pardf, mod, final=final, sub_sample=sub_sample,
                             pulse_sample=pulse_sample, optimise_fast=optimise_fast)

    if ext_spectra is None:
        re = fill_int(ds=ds, c=c, final=final, return_shapes=return_shapes)
    else:
        ds, c_for_solve, ext_spectra = _prepare_external_spectra(ds, c, ext_spectra, pardf)
        re = fill_int(ds=ds, c=c_for_solve, final=final, return_shapes=return_shapes)

    if ext_spectra is not None and (final or return_shapes):
        _restore_external_spectra(re, c, ext_spectra, pardf)
    if final:
        total = ((re["A"] - re["A"].mean().mean()) ** 2).sum().sum()
        re["r2"] = 1 - re["error"] / total
    return re


def project_dataset(ta, slice_settings):
    """The slice of one project's data that a joint fit should see."""
    from ..shaping import sub_ds

    return sub_ds(ds=ta.ds, scattercut=slice_settings.scattercut,
                  bordercut=slice_settings.bordercut, timelimits=slice_settings.timelimits,
                  wave_nm_bin=slice_settings.wave_nm_bin, time_bin=slice_settings.time_bin,
                  ignore_time_region=slice_settings.ignore_time_region)


def project_parameters(shared, ta, unique_parameter=None, log_fit=False):
    """The shared parameters, with this project's own values where they differ.

    A joint fit varies one set of parameters across every dataset, except for
    those named in ``unique_parameter`` — typically an amplitude or a time
    zero that genuinely differs between measurements.
    """
    from .parameters import par_to_pardf

    pardf = shared.copy()
    if unique_parameter is not None:
        try:
            own = par_to_pardf(ta.par)
        except Exception:
            own = getattr(ta, "pardf", shared).copy()
        for key in unique_parameter:
            pardf.loc[key, "value"] = own.loc[key, "value"]
    if log_fit:
        pardf.loc[pardf.is_rate, "value"] = pardf.loc[pardf.is_rate, "value"].apply(lambda x: 10**x)
    return pardf


def combine_errors(errors, weights=None):
    """Root-mean-square of the per-project errors.

    ``weights`` may omit the first project, which then counts as 1 — that is
    how the fitting interface presents "the others relative to this one".
    """
    if weights is None:
        return float(np.sqrt((np.array(errors) ** 2).mean()))
    weights = list(weights)
    if len(weights) == len(errors) - 1:
        weights.insert(0, 1)
    elif len(weights) != len(errors):
        raise ValueError(
            "The number of entries i the list must either be the number of all elements "
            '(including "TA" or the number of elements in other. '
            "In this case the element ta gets the weight=1"
        )
    return float(np.sqrt(((np.array(errors) * np.array(weights)) ** 2).mean()))

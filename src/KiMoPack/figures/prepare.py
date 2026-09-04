"""Deciding what a figure shows, without drawing it.

Each function here turns data and settings into a :mod:`KiMoPack.figures.model`
description. No matplotlib, no files, no display — which means the decisions
can be checked directly, and a second renderer can consume the same answer.
"""

import numpy as np

from ..numerics import Frame_golay, nm_to_ev
from ..regions import cut_pairs, frame_spans
from .model import AxisSpec, Panel, Trace

#: How the three line modes are drawn. 'data' shows the measurement as
#: markers; the other two are continuous curves.
LINE_STYLES = {"smoothed": "solid", "data": "markers", "fitted": "solid"}

#: Marker traces keep their own weight rather than the curve's, and sit behind
#: the curve when both share an axis.
MARKER_WIDTH = 1.5

LEGEND_TITLES = {"smoothed": "lines = smoothed", "data": None, "fitted": "lines = fit"}


def line_mode(lines_are):
    """The drawing style named by ``lines_are``."""
    for known in LINE_STYLES:
        if known in lines_are:
            return known
    raise ValueError(
        f"unknown lines_are {lines_are!r}; expected one of {', '.join(LINE_STYLES)}")


def delays_within(ds, rel_time):
    """The requested delays the measurement actually covers.

    Asking for one outside the range would otherwise draw the nearest delay
    instead, labelled as the one that was asked for.
    """
    if rel_time is None:
        return []
    if not hasattr(rel_time, "__iter__"):
        rel_time = [rel_time]
    times = ds.index.values.astype(float)
    return [t for t in rel_time if times.min() <= t <= times.max()]


def _traces_of(frame, cuts, colors, style, alpha, linewidth, to_energy, transform=None,
               zorder=None):
    """One trace per column per unbroken run of the index.

    Only the first run of each column is labelled, so a curve interrupted by
    two masked regions appears once in the legend rather than three times.

    ``transform`` is applied to each run separately, not to the whole frame:
    smoothing across a gap would pull values from the far side of a region
    that was masked precisely because they are not to be trusted.
    """
    traces = []
    for piece, first in frame_spans(frame, cuts, to_energy):
        if transform is not None:
            piece = transform(piece)
        x = piece.index.values.astype(float)
        for position, column in enumerate(piece.columns):
            traces.append(Trace(
                x=x, y=piece[column].values.astype(float),
                label=str(column) if first else None,
                color=colors[position % len(colors)] if len(colors) else None,
                style=style, alpha=alpha, width=linewidth, zorder=zorder))
    return traces


def spectra_panel(ds, selection, view, colors, rel_time=None, time_width_percent=0,
                  lines_are="smoothed", linewidth=1, from_fit=False, title=None,
                  shade_masked=False, fit_alpha=0.7, behind=False):
    """Spectra at chosen delays: signal against wavelength.

    ``selection`` says which part of the measurement to use, ``view`` how it
    should look; both are applied here so the renderer has nothing left to
    decide. ``shade_masked`` reports the masked regions for shading; line
    plots have never shaded them, so it is off unless asked for.
    """
    rel_time = delays_within(ds, rel_time)
    sliced = selection.apply(ds, times=rel_time, time_width_percent=time_width_percent,
                             drop_scatter=True, from_fit=from_fit)
    to_energy = selection.equal_energy_bin is not None
    style = LINE_STYLES[line_mode(lines_are)]

    if "smoothed" in lines_are:
        traces = _traces_of(sliced, selection.scattercut, colors, style, 1.0, linewidth,
                            to_energy,
                            transform=lambda p: Frame_golay(p, window=5, order=3, transpose=False))
    elif "data" in lines_are:
        # Markers are drawn whole: a gap in markers is already visible.
        traces = _traces_of(sliced, None, colors, style, 1.0, MARKER_WIDTH, to_energy,
                            zorder=0 if behind else None)
    else:
        traces = _traces_of(sliced, selection.scattercut, colors, style, fit_alpha, linewidth,
                            to_energy)

    limits = selection.bordercut
    if limits is not None and to_energy:
        limits = sorted(nm_to_ev(limits))
    intensity = view.intensity_range
    if intensity is not None and not hasattr(intensity, "__iter__"):
        intensity = np.array([-intensity, intensity])

    return Panel(
        x=AxisSpec(label=sliced.index.name, limits=limits),
        y=AxisSpec(label=view.data_type, limits=intensity),
        traces=tuple(traces),
        title=title,
        shaded=tuple(cut_pairs(selection.scattercut, to_energy)) if shade_masked else (),
        legend_title=LEGEND_TITLES[line_mode(lines_are)],
    )


def _time_axis(plot_type, timelimits, lintresh, linscale, label):
    """The time axis a kinetic trace is drawn against.

    Transient data spans femtoseconds to nanoseconds and crosses zero, so the
    default is symmetric-log: linear through the pulse, logarithmic after.
    """
    if "symlog" in plot_type:
        return AxisSpec(label=label, scale="symlog", limits=timelimits,
                        linthresh=lintresh, linscale=linscale)
    if "log" in plot_type:
        low, high = timelimits
        # A log axis cannot show zero or the negative delays before excitation.
        return AxisSpec(label=label, scale="log", limits=[max(1e-6, low), high])
    return AxisSpec(label=label, scale="linear", limits=timelimits)


def kinetics_panel(ds, selection, view, colors, wavelength=None, lines_are="smoothed",
                   linewidth=1, from_fit=False, plot_type="symlog", timelimits=None,
                   title=None, shade_masked=False, fit_alpha=1.0, behind=False):
    """Kinetics at chosen wavelengths: signal against delay.

    The mirror of :func:`spectra_panel` — same decisions, other axis. Traces
    break at ignored delay regions rather than at masked wavelengths.
    """
    if not hasattr(wavelength, "__iter__"):
        wavelength = [wavelength]
    sliced = selection.apply(ds, wavelength=wavelength, drop_ignore=True, from_fit=from_fit)
    style = LINE_STYLES[line_mode(lines_are)]

    if "smoothed" in lines_are:
        traces = _traces_of(sliced, selection.ignore_time_region, colors, style, 1.0, linewidth,
                            False, transform=lambda p: Frame_golay(p, window=5, order=3))
    elif "data" in lines_are:
        traces = _traces_of(sliced, None, colors, style, 1.0, MARKER_WIDTH, False,
                            zorder=0 if behind else None)
    else:
        traces = _traces_of(sliced, selection.ignore_time_region, colors, style, fit_alpha,
                            linewidth, False)

    times = sliced.index.values.astype(float)
    if timelimits is None:
        timelimits = [times.min(), times.max()]
    intensity = view.intensity_range
    if intensity is not None and not hasattr(intensity, "__iter__"):
        intensity = np.array([-intensity, intensity])

    return Panel(
        x=_time_axis(plot_type, timelimits, view.lintresh, view.linscale, sliced.index.name),
        y=AxisSpec(label=view.data_type, limits=intensity),
        traces=tuple(traces),
        title=title,
        shaded=tuple(cut_pairs(selection.ignore_time_region)) if shade_masked else (),
        legend_title=LEGEND_TITLES[line_mode(lines_are)],
    )

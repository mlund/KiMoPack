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
#: markers; the other two are continuous curves, with the fit drawn slightly
#: transparent so the data it sits on stays visible.
LINE_STYLES = {"smoothed": ("solid", 1.0), "data": ("markers", 1.0), "fitted": ("solid", 0.7)}

LEGEND_TITLES = {"smoothed": "lines = smoothed", "data": None, "fitted": "lines = fit"}


def _traces_of(frame, cuts, colors, style, alpha, linewidth, to_energy):
    """One trace per column per unbroken run of the index.

    Only the first run of each column is labelled, so a curve interrupted by
    two masked regions appears once in the legend rather than three times.
    """
    traces = []
    for piece, first in frame_spans(frame, cuts, to_energy):
        x = piece.index.values.astype(float)
        for position, column in enumerate(piece.columns):
            traces.append(Trace(
                x=x, y=piece[column].values.astype(float),
                label=str(column) if first else None,
                color=colors[position % len(colors)] if len(colors) else None,
                style=style, alpha=alpha, width=linewidth))
    return traces


def spectra_panel(ds, selection, view, colors, rel_time=None, time_width_percent=0,
                  lines_are="smoothed", linewidth=1, from_fit=False, title=None,
                  shade_masked=False):
    """Spectra at chosen delays: signal against wavelength.

    ``selection`` says which part of the measurement to use, ``view`` how it
    should look; both are applied here so the renderer has nothing left to
    decide. ``shade_masked`` reports the masked regions for shading; line
    plots have never shaded them, so it is off unless asked for.
    """
    if not hasattr(rel_time, "__iter__"):
        rel_time = [rel_time]
    times = ds.index.values.astype(float)
    # Asking for a delay the measurement does not cover would silently
    # produce the nearest one instead, which is worse than leaving it out.
    rel_time = [t for t in rel_time if times.min() <= t <= times.max()]

    sliced = selection.apply(ds, times=rel_time, time_width_percent=time_width_percent,
                             drop_scatter=True, from_fit=from_fit)
    to_energy = selection.equal_energy_bin is not None
    style, alpha = LINE_STYLES[next(k for k in LINE_STYLES if k in lines_are)]

    if "smoothed" in lines_are:
        source = Frame_golay(sliced, window=5, order=3, transpose=False)
        traces = _traces_of(source, selection.scattercut, colors, style, alpha, linewidth,
                            to_energy)
    elif "data" in lines_are:
        # Markers are drawn whole: a gap in markers is already visible.
        traces = _traces_of(sliced, None, colors, style, alpha, linewidth, to_energy)
    else:
        traces = _traces_of(sliced, selection.scattercut, colors, style, alpha, linewidth,
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
        legend_title=LEGEND_TITLES[next(k for k in LEGEND_TITLES if k in lines_are)],
    )

"""Drawing a figure description with matplotlib.

One of possibly several renderers. It knows about matplotlib; nothing that
decides *what* to draw does.
"""

import matplotlib.pyplot as plt
import numpy as np

#: Trace styles as matplotlib format strings.
_STYLES = {"solid": "-", "dashed": "--", "markers": "*"}


def draw_traces(panel, ax):
    """Draw a panel's content, leaving the axis itself alone.

    For callers that still manage their own labels, scales and limits.
    """
    for trace in panel.traces:
        ax.plot(trace.x, trace.y, _STYLES.get(trace.style, "-"),
                color=trace.color, alpha=trace.alpha, lw=trace.width,
                label=trace.label if trace.in_legend else "_nolegend_",
                **({} if trace.zorder is None else {"zorder": trace.zorder}))

    for low, high in panel.shaded:
        ax.axvspan(low, high, color="0.85", zorder=0, label="_nolegend_")
    return ax


def draw_panel(panel, ax):
    """Draw one panel onto an existing axis, decoration included."""
    draw_traces(panel, ax)
    for axis, spec in ((ax.xaxis, panel.x), (ax.yaxis, panel.y)):
        if spec.label is not None:
            axis.set_label_text(spec.label)
    _apply_scale(ax, "x", panel.x)
    _apply_scale(ax, "y", panel.y)
    if panel.title:
        ax.set_title(panel.title, pad=10)
    return ax


def _apply_scale(ax, which, spec):
    setter = ax.set_xscale if which == "x" else ax.set_yscale
    limiter = ax.set_xlim if which == "x" else ax.set_ylim
    if spec.scale == "symlog":
        setter("symlog", linthresh=spec.linthresh, linscale=spec.linscale)
    elif spec.scale == "log":
        setter("log")
    if spec.limits is not None:
        limiter(np.asarray(spec.limits, dtype=float))
    else:
        ax.autoscale(axis=which, tight=True)


def render(figure, size=None, dpi=100):
    """Draw a whole figure description into a new matplotlib figure."""
    rows, columns = figure.layout
    fig, axes = plt.subplots(rows, columns, figsize=size or figure.size or (10, 6), dpi=dpi)
    for panel, ax in zip(figure.panels, np.atleast_1d(axes).ravel(), strict=False):
        draw_panel(panel, ax)
    return fig

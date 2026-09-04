"""A figure described, not drawn.

Every decision that makes a plot correct — which pieces of a masked trace
survive, what the axes are called, which scale they use, what is shaded out —
is settled here, in plain data with no drawing library involved. A renderer
then turns that description into pixels.

Two things follow. The decisions become testable without a display, which is
most of what these figures get wrong. And a second front end becomes possible,
because it consumes the same description matplotlib does.
"""

import dataclasses

import numpy as np

#: How an axis maps values to position. 'symlog' is linear near zero and
#: logarithmic beyond, which is what transient data needs: the interesting
#: part spans femtoseconds to nanoseconds but crosses zero.
SCALES = ("linear", "log", "symlog")


@dataclasses.dataclass(frozen=True)
class Trace:
    """One line or set of markers."""

    x: np.ndarray
    y: np.ndarray
    #: None for the continuation pieces of a trace broken by a masked region,
    #: so one measurement is listed once however many times it is interrupted.
    label: object = None
    color: object = None
    style: str = "solid"
    width: float = 1.0
    alpha: float = 1.0
    #: Draw order. Measured points sit behind the curve drawn through them.
    zorder: object = None

    def __post_init__(self):
        if len(self.x) != len(self.y):
            raise ValueError(f"a trace needs matching x and y, got {len(self.x)} and {len(self.y)}")

    @property
    def in_legend(self):
        return self.label is not None


@dataclasses.dataclass(frozen=True)
class AxisSpec:
    """What one axis shows and how it is scaled."""

    label: object = None
    scale: str = "linear"
    limits: object = None
    #: Where a symmetric-log axis switches from linear to logarithmic.
    linthresh: object = None
    #: How much room the linear part gets relative to the logarithmic part.
    linscale: object = 1

    def __post_init__(self):
        if self.scale not in SCALES:
            raise ValueError(f"unknown axis scale {self.scale!r}, expected one of {SCALES}")
        if self.scale == "symlog" and self.linthresh is None:
            raise ValueError("a symlog axis needs a linthresh; the renderer cannot invent one")


@dataclasses.dataclass(frozen=True)
class Image:
    """A 2D map: values on a grid, with the colour limits to show them with."""

    values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    limits: object = None
    colormap: object = None
    colorbar_label: object = None
    #: Compress the colour scale logarithmically either side of zero. Transient
    #: signals span orders of magnitude and change sign, so a linear scale
    #: shows the strongest feature and nothing else.
    log_scale: bool = False
    #: How much of the colour range the near-zero linear part gets.
    linscale: float = 1

    def __post_init__(self):
        if self.values.shape != (len(self.y), len(self.x)):
            raise ValueError(
                f"image is {self.values.shape} but the axes are "
                f"{len(self.y)} by {len(self.x)}")


@dataclasses.dataclass(frozen=True)
class Panel:
    """One set of axes."""

    x: AxisSpec
    y: AxisSpec
    traces: tuple = ()
    image: object = None
    title: object = None
    #: Regions masked out along x, shaded so the panel does not imply
    #: measurements where none were kept.
    shaded: tuple = ()
    #: The same along y. A map masks in both directions; a line plot only in x.
    shaded_y: tuple = ()
    legend_title: object = None

    def legend_labels(self):
        return [t.label for t in self.traces if t.in_legend]


@dataclasses.dataclass(frozen=True)
class Figure:
    """A named set of panels, and how they sit together."""

    panels: tuple
    name: str
    layout: object = None
    size: object = None

    def __post_init__(self):
        if self.layout is None:
            object.__setattr__(self, "layout", (len(self.panels), 1))
        rows, columns = self.layout
        if rows * columns < len(self.panels):
            raise ValueError(
                f"layout {self.layout} has no room for {len(self.panels)} panels")

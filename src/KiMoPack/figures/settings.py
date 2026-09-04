"""What a figure should look like, and where it should go.

These are read off a project once and handed down whole. Previously each
layer forwarded them one keyword at a time, which is how a call ended up
passing the colour map but not the colour scale.

They are deliberately separate from
:class:`KiMoPack.shaping.DataSelection`: which part of the data to use is a
decision about the measurement, while these are decisions about the picture.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class ViewSettings:
    """How to draw it."""

    #: Colour limits. A single number means symmetric about zero.
    intensity_range: object = None
    log_scale: object = False
    #: Where a symmetric-log axis switches from linear to logarithmic, and how
    #: much room the linear part gets.
    lintresh: object = 0.3
    linscale: object = 1
    cmap: object = None
    line_colors: object = None
    baseunit: object = "ps"
    units: object = "nm"
    data_type: object = None
    legend_inside: object = True
    #: How much the residual map is exaggerated so it is visible next to the data.
    error_matrix_amplification: object = 10
    #: Optional per-scan metadata shown on the colour bar; only some callers set it.
    values: object = None

    _FROM_PROJECT = ("intensity_range", "log_scale", "lintresh", "linscale", "cmap",
                     "line_colors", "baseunit", "units", "data_type", "legend_inside",
                     "error_matrix_amplification", "values")

    @classmethod
    def from_project(cls, ta, **overrides):
        """Read a project's display settings; ``overrides`` win where given."""
        settings = {name: getattr(ta, name, None) for name in cls._FROM_PROJECT}
        settings.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**settings)

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)

    def as_kwargs(self, *names):
        """The named fields as a keyword dict.

        Drawing functions accept different subsets, so callers ask for what
        the function they are calling actually takes.
        """
        return {name: getattr(self, name) for name in (names or
                [f.name for f in dataclasses.fields(self)])}


@dataclasses.dataclass(frozen=True)
class OutputSpec:
    """Where the figure goes, and what it is called."""

    path: object = None
    filename: object = None
    title: object = None
    savetype: object = "png"
    save_figures_to_folder: object = False

    @classmethod
    def from_project(cls, ta, path=None, filename=None, title=None, savetype=None):
        """Fill in from the project what the caller did not state.

        The title falls back to the filename, because a figure saved as
        something should say so.
        """
        filename = filename if filename is not None else getattr(ta, "filename", None)
        return cls(
            path=path,
            filename=filename,
            title=title if title is not None else filename,
            savetype=savetype if savetype is not None else "png",
            save_figures_to_folder=getattr(ta, "save_figures_to_folder", False),
        )

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)

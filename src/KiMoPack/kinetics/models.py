"""Kinetic models: rate constants in, concentration profiles out.

Two families ship with the package. The parallel model treats every species
as decaying independently, which has a closed form and is therefore cheap.
The sequential model integrates a decay chain forward from a Gaussian
excitation pulse, which is far more expensive but describes species that feed
one another.

Models are looked up through :func:`resolve_model` rather than by comparing
strings at the call site. That matters because a model's name used to be
tested in nine scattered places — for dispatch, for whether the optimiser may
substitute a cheaper stand-in, and for whether the resulting spectra are
decay- or species-associated. Those three facts now travel with the model.

User-supplied models are ordinary callables and are wrapped into the same
shape, so nothing downstream needs to know where a model came from.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas

from ..numerics import gauss, rise

#: Conversion between a Gaussian standard deviation and its full width at half
#: maximum. The sequential model quotes ``resolution`` as a FWHM; the parallel
#: model uses it directly as the width of its error-function ramp.
FWHM = 2.35482


@dataclass(frozen=True)
class KineticModel:
    """A way of turning parameters into concentrations."""

    name: str
    build: Callable[..., pandas.DataFrame]
    #: 'DAS' when independent decays make the spectra decay-associated,
    #: 'SAS' when species feed one another and the spectra belong to species.
    species_are: str
    #: A cheaper model the optimiser may substitute while searching, or None
    #: to always use this one. The rates come out the same either way; only
    #: the final evaluation needs the expensive integration.
    optimise_with: Optional[str] = None


def _structural(pardf, name):
    """Optional behaviour is switched on by the presence of a named parameter."""
    return name in list(pardf.index.values)


def _rates(pardf):
    return pardf.loc[pardf.is_rate, "value"].values.astype(float)


def build_parallel(times, pardf, sub_steps=None):
    """Independently decaying species, each multiplied by the response ramp."""
    param = _rates(pardf)
    t0 = float(pardf.loc["t0", "value"])
    resolution = float(pardf.loc["resolution", "value"])

    c = np.exp(-1 * np.tile(times - t0, (len(param), 1)).T * param)
    # Before t0 the decay has not started; the ramp below suppresses it anyway.
    c[(times - t0) < 0] = 1
    c *= np.tile(rise(x=times, sigma=resolution, begin=t0), (len(param), 1)).T

    c = pandas.DataFrame(c, index=times)
    c.index.name = "time"
    if _structural(pardf, "explicit_GS"):
        c["GS"] = np.zeros(len(times), dtype="float")
    if _structural(pardf, "background"):
        c["background"] = 1
    if _structural(pardf, "infinite"):
        # Named to match the sequential model: the fit reports species by
        # these labels, and they end up in figures and exported spectra.
        c["Non Decaying"] = rise(x=times, sigma=resolution, begin=t0)
    return c


def build_sequential(times, pardf, sub_steps=None):
    """A -> B -> C -> ..., integrated forward through the excitation pulse.

    Forward Euler with ``sub_steps`` subdivisions per data point, rather than
    an ODE solver, because the concentrations must land exactly on the
    measured time axis and that axis is very unevenly spaced.

    The pulse enters as a Gaussian sampled on the time grid, so the grid has
    to resolve it: when the response is narrower than the spacing between time
    points the injected population is misestimated. Densifying the axis around
    t0 (``pulse_sample`` in the fitting layer) is what keeps that honest.
    """
    if _structural(pardf, "sub_steps"):
        sub_steps = pardf.loc["sub_steps", "value"]
    elif sub_steps is None:
        sub_steps = 10
    param = _rates(pardf)
    t0 = float(pardf.loc["t0", "value"])
    resolution = float(pardf.loc["resolution", "value"])

    infinite = _structural(pardf, "infinite")
    explicit_gs = _structural(pardf, "explicit_GS")
    n_decays = len(param) + (1 if infinite else 0)

    g = gauss(times, sigma=resolution / FWHM, mu=t0)
    c = np.zeros((len(times), n_decays), dtype="float")
    gs = np.zeros((len(times), 1), dtype="float") if explicit_gs else None

    for i in range(1, len(times)):
        dt = (times[i] - times[i - 1]) / sub_steps
        c_temp = c[i - 1, :]
        for _ in range(int(sub_steps)):
            dc = np.zeros(n_decays, dtype="float")
            for level in range(n_decays):
                if level == 0:
                    # The pulse feeds the first species only.
                    if infinite and n_decays == 1:
                        dc[level] = g[i] * dt
                    else:
                        dc[level] = g[i] * dt - param[level] * dt * c_temp[level]
                elif infinite and level == n_decays - 1:
                    # The last species is a sink: it fills but never empties.
                    dc[level] = param[level - 1] * dt * c_temp[level - 1]
                else:
                    dc[level] = param[level - 1] * dt * c_temp[level - 1] - param[level] * dt * c_temp[level]
            c_temp = c_temp + dc
            c_temp[c_temp < 0] = 0
        c[i, :] = c_temp
        if explicit_gs:
            # Whatever is excited is missing from the ground state.
            gs[i] = -(c[i, :].sum() if not infinite else c[i, :-1].sum())

    c = pandas.DataFrame(c, index=times)
    c.index.name = "time"
    if infinite:
        labels = list(c.columns.values)
        labels[-1] = "Non Decaying"
        c.columns = labels
    if _structural(pardf, "background"):
        c["background"] = 1
    if explicit_gs:
        c["GS"] = gs
    return c


MODELS = {}


def register(model, aliases=()):
    """Make a model reachable by name."""
    MODELS[model.name] = model
    for alias in aliases:
        MODELS[alias] = model


register(KineticModel(name="paral", build=build_parallel, species_are="DAS"),
         aliases=["parallel", "decays", "exponential"])
register(KineticModel(name="consecutive", build=build_sequential, species_are="SAS",
                      optimise_with="paral"),
         aliases=["sequential"])
register(KineticModel(name="full_consecutive", build=build_sequential, species_are="SAS"),
         aliases=["full_sequential"])


def available_models():
    """Every name accepted by :func:`resolve_model`, sorted."""
    return sorted(MODELS)


def _wrap_external(func):
    """Adapt a user function to the internal model interface.

    The published contract is ``f(times, pardf) -> DataFrame`` where pardf is
    the value column as a Series, so that is what gets passed. Their species
    feed one another, so the spectra are species-associated.
    """

    def build(times, pardf, sub_steps=None):
        return func(times=times, pardf=pardf.loc[:, "value"])

    return KineticModel(name=getattr(func, "__name__", "user model"), build=build,
                        species_are="SAS")


def resolve_model(mod):
    """Return the :class:`KineticModel` for a name or a user callable."""
    if callable(mod):
        return _wrap_external(mod)
    try:
        return MODELS[mod]
    except KeyError:
        raise ValueError(
            f"unknown kinetic model {mod!r}. Available: {', '.join(available_models())}. "
            "A callable taking (times, pardf) is also accepted."
        ) from None


def build_c(times, mod="paral", pardf=None, sub_steps=None):
    """Concentration profile over ``times`` for the given model and parameters.

    Parameters
    ----------
    times : numpy.ndarray
        Time axis the result is reported on.
    mod : str or callable, optional
        Model name, or a user function taking ``(times, pardf)``.
    pardf : pandas.DataFrame
        Parameter table from :func:`KiMoPack.kinetics.parameters.par_to_pardf`.
        Rate constants are the rows flagged ``is_rate``; ``t0`` and
        ``resolution`` are required. The optional rows ``background``,
        ``infinite`` and ``explicit_GS`` each add a species.
    sub_steps : int, optional
        Integration subdivisions per time point, for the sequential model.

    Returns
    -------
    pandas.DataFrame
        One column per species, indexed by time.
    """
    return resolve_model(mod).build(times=times, pardf=pardf, sub_steps=sub_steps)

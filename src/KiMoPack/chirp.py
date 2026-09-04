"""Wavelength-dependent time zero, and how to undo it.

A white-light probe is chirped: blue light traverses the optics more slowly
than red, so each detection wavelength sees the excitation at a slightly
different delay. The offset varies smoothly across the spectrum and is
described by a polynomial in wavelength. Removing it is a prerequisite for any
global fit, which assumes one shared time axis.

Detection is separated from drawing here. The detector returns a
:class:`ChirpFit` describing what it found; the diagnostic figure is one way
of looking at that, and a future interface can be another.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erf

#: Chirp is stored throughout the package as five polynomial coefficients in
#: numpy.polyval order, whatever order was actually fitted.
CHIRP_COEFFICIENTS = 5


@dataclass(frozen=True)
class ChirpFit:
    """The chirp curve found in a dataset."""

    coefficients: list
    #: Detected time zero per wavelength; channels with no visible rise are absent.
    onsets: dict
    poly_order: int
    window: tuple

    def delay_at(self, wavelength):
        """Time offset this fit predicts for a wavelength."""
        return np.polyval(self.coefficients, wavelength)


def apply_chirp(ds, fitcoeff):
    """Shift each wavelength column onto a common time axis.

    Returns a new frame. Points shifted in from outside the measured range are
    filled with zero rather than extrapolated.
    """
    times = ds.index.values.astype(float)
    ds_new = ds.copy()
    for col in ds.columns:
        delay = np.polyval(fitcoeff, float(col))
        f = interp1d(times - delay, ds[col].values, bounds_error=False, fill_value=0)
        ds_new[col] = f(times)
    ds_new.index.name = ds.index.name
    ds_new.columns.name = ds.columns.name
    return ds_new


def _edge_levels(signal):
    """Signal level before and after the rise, from the outer fifths."""
    n = max(3, len(signal) // 5)
    return np.median(signal[:n]), np.median(signal[-n:])


def find_threshold_crossing(times, signal, fraction=0.5):
    """Where the trace first crosses a fraction of its total step.

    Returns None when the trace has no step to cross.
    """
    baseline, late = _edge_levels(signal)
    if abs(late - baseline) < 1e-10:
        return None
    shifted = signal - (baseline + fraction * (late - baseline))
    crossings = np.where(np.diff(np.sign(shifted)))[0]
    if len(crossings) == 0:
        return None
    i = crossings[0]
    # Linear interpolation between the two samples that straddle the level.
    return float(times[i] + (times[i + 1] - times[i])
                 * abs(shifted[i]) / (abs(shifted[i]) + abs(shifted[i + 1])))


def find_max_derivative(times, signal):
    """Time of steepest change. Cheap, but sensitive to noise."""
    idx = np.argmax(np.abs(np.diff(signal) / np.diff(times)))
    return float(0.5 * (times[idx] + times[idx + 1]))


def fit_sigmoid_onset(times, signal, t0_guess=0.0):
    """Fit an error-function step and report its centre.

    The most robust of the three detectors, because it uses the whole trace
    rather than one crossing. Falls back to the threshold crossing if the fit
    does not converge, and returns None when there is no step at all.
    """
    import lmfit

    baseline, late = _edge_levels(signal)
    amplitude = late - baseline
    if abs(amplitude) < 1e-10:
        return None

    def residual(params):
        return signal - (
            params["bg"]
            + params["amp"] * 0.5 * (1 + erf((times - params["t0"])
                                             / (params["sigma"] * np.sqrt(2))))
        )

    par = lmfit.Parameters()
    par.add("t0", value=t0_guess, min=times.min(), max=times.max())
    par.add("sigma", value=0.1, min=0.01, max=(times.max() - times.min()) / 2)
    par.add("amp", value=amplitude)
    par.add("bg", value=baseline)
    try:
        result = lmfit.minimize(residual, par, method="nelder")
        if result.success or result.nfev > 10:
            return float(result.params["t0"].value)
    except Exception:
        pass
    return find_threshold_crossing(times, signal, 0.5)


DETECTORS = {
    "sigmoid": fit_sigmoid_onset,
    "threshold": find_threshold_crossing,
    "max_derivative": find_max_derivative,
}


def detect_chirp(ds, t_range=(-2, 2), method="sigmoid", poly_order=4, t0_guess=0.0,
                 threshold_fraction=0.5):
    """Locate time zero in every channel and fit a curve through them.

    Pure: no drawing, no file access. With too few detected onsets to define a
    curve there is nothing to fit, so a constant offset is returned instead.
    """
    if method not in DETECTORS:
        raise ValueError(
            f"unknown onset detection method {method!r}. "
            f"Available: {', '.join(sorted(DETECTORS))}"
        )

    wavelengths = ds.columns.values.astype(float)
    times = ds.index.values.astype(float)
    # The stored format holds five coefficients, so a fourth-order curve is
    # the most that can be replayed.
    poly_order = min(poly_order, CHIRP_COEFFICIENTS - 1, max(1, len(wavelengths) - 2))

    mask = (times >= t_range[0]) & (times <= t_range[1])
    if mask.sum() < 5:
        # The requested window missed the data; fall back to all of it.
        mask = np.ones(len(times), dtype=bool)
    window = times[mask]

    onsets = {}
    for wl in wavelengths:
        trace = ds.loc[mask, wl].values.astype(float)
        if method == "threshold":
            t0 = find_threshold_crossing(window, trace, threshold_fraction)
        elif method == "sigmoid":
            t0 = fit_sigmoid_onset(window, trace, t0_guess)
        else:
            t0 = find_max_derivative(window, trace)
        if t0 is not None:
            onsets[wl] = t0

    coefficients = [0.0] * CHIRP_COEFFICIENTS
    if len(onsets) < 2:
        coefficients[-1] = np.median(list(onsets.values())) if onsets else t0_guess
    else:
        # Channels where no onset was found cannot support the fit, so the
        # order is limited by what was actually detected, not by how many
        # channels were looked at.
        poly_order = min(poly_order, len(onsets) - 1)
        fitted = np.polyfit(np.array(list(onsets)), np.array(list(onsets.values())), poly_order)
        coefficients[CHIRP_COEFFICIENTS - len(fitted):] = list(fitted)

    return ChirpFit(coefficients=coefficients, onsets=onsets, poly_order=poly_order,
                    window=(float(window.min()), float(window.max())))


def plot_chirp_fit(ds, corrected, fit, t_range):
    """Four-panel diagnostic: raw traces, the curve, corrected traces, numbers."""
    import matplotlib.pyplot as plt

    wavelengths = ds.columns.values.astype(float)
    times = ds.index.values.astype(float)
    mask = (times >= t_range[0]) & (times <= t_range[1])
    if mask.sum() < 5:
        mask = np.ones(len(times), dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for wl in wavelengths:
        ax.plot(times[mask], ds.loc[mask, wl].values, "o-", label=f"{wl:.0f} nm", ms=3)
    for t0 in fit.onsets.values():
        ax.axvline(t0, ls="--", alpha=0.5)
    ax.set(xlabel="Time (ps)", ylabel="Signal", title="Raw traces + detected t0")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    if fit.onsets:
        ax.plot(list(fit.onsets), list(fit.onsets.values()), "ro", ms=8, label="detected t0")
    grid = np.linspace(wavelengths.min(), wavelengths.max(), 200)
    ax.plot(grid, np.polyval(fit.coefficients, grid), "b-", label=f"poly order {fit.poly_order}")
    ax.set(xlabel="Wavelength (nm)", ylabel="t0 (ps)", title="Chirp curve")
    ax.legend()

    ax = axes[1, 0]
    corrected_times = corrected.index.values
    m2 = (corrected_times >= t_range[0]) & (corrected_times <= t_range[1])
    for wl in wavelengths:
        ax.plot(corrected_times[m2], corrected.loc[m2, wl].values, "o-",
                label=f"{wl:.0f} nm", ms=3)
    ax.set(xlabel="Time (ps)", ylabel="Signal", title="Corrected traces")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    text = "Chirp correction coefficients:\n"
    for label, value in zip(["a4", "a3", "a2", "a1", "a0"], fit.coefficients, strict=True):
        text += f"  {label} = {value:.6e}\n"
    text += "\nDetected t0 per wavelength:\n"
    for wl, t0 in sorted(fit.onsets.items()):
        text += f"  {wl:.0f} nm: {t0:.4f} ps\n"
    ax.text(0.1, 0.9, text, transform=ax.transAxes, va="top", fontfamily="monospace", fontsize=11)
    plt.tight_layout()
    plt.show()
    return fig


def find_chirp_sparse(ds, t_range=(-2, 2), method="sigmoid", poly_order=4, t0_guess=0.0,
                      plot=True, threshold_fraction=0.5):
    """Detect the chirp of a sparse dataset and correct it.

    Returns
    -------
    tuple
        ``(corrected DataFrame, five coefficients, {wavelength: t0})``
    """
    fit = detect_chirp(ds, t_range=t_range, method=method, poly_order=poly_order,
                       t0_guess=t0_guess, threshold_fraction=threshold_fraction)
    corrected = apply_chirp(ds, fit.coefficients)
    if plot:
        plot_chirp_fit(ds, corrected, fit, t_range)
    return corrected, fit.coefficients, fit.onsets

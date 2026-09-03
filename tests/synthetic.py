"""Datasets whose exact decomposition is known.

Every fast test works from these rather than from the bundled experiments, so
assertions can be exact rather than "close to what it did last time", and the
suite stays free of multi-megabyte fixtures. The generator deliberately does
not import KiMoPack: a test that builds its expectation with the code under
test proves nothing.
"""

from dataclasses import dataclass

import numpy as np
import pandas
from scipy.special import erf

#: Conversion between a Gaussian standard deviation and its full width at half
#: maximum. KiMoPack quotes instrument response as FWHM.
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass
class Truth:
    """What a synthetic dataset was built from."""

    taus: np.ndarray
    rates: np.ndarray
    t0: float
    resolution: float
    concentrations: pandas.DataFrame
    spectra: pandas.DataFrame  # species x wavelength
    times: np.ndarray
    waves: np.ndarray


def default_times(start=-2.0, stop=1000.0, n=180):
    """A log-ish time axis: dense through the rise, sparse in the long tail."""
    early = np.linspace(start, 1.0, n // 3)
    late = np.geomspace(1.05, stop, n - n // 3)
    return np.unique(np.concatenate([early, late]))


def default_waves(start=400.0, stop=700.0, n=60):
    return np.linspace(start, stop, n)


def gaussian_bands(waves, centres, widths=None, heights=None):
    """Species spectra as Gaussian bands — one row per species."""
    centres = np.atleast_1d(centres).astype(float)
    widths = np.full(centres.shape, 40.0) if widths is None else np.atleast_1d(widths).astype(float)
    heights = np.ones_like(centres) if heights is None else np.atleast_1d(heights).astype(float)
    rows = [h * np.exp(-0.5 * ((waves - c) / w) ** 2) for c, w, h in zip(centres, widths, heights, strict=True)]
    return pandas.DataFrame(np.array(rows), index=list(range(len(centres))), columns=waves)


def convolved_decay(times, tau, t0=0.0, resolution=0.1):
    """Exponential decay convolved with a Gaussian instrument response.

    This is the textbook analytic result, not KiMoPack's approximation, so it
    serves as an independent reference for the parallel model.
    """
    sigma = resolution / FWHM
    k = 1.0 / tau
    arg = (times - t0 - k * sigma**2) / (sigma * np.sqrt(2.0))
    return 0.5 * np.exp(-k * (times - t0) + 0.5 * (k * sigma) ** 2) * (1.0 + erf(arg))


def parallel_concentrations(times, taus, t0=0.0, resolution=0.1):
    """Independently decaying species, each convolved with the same response."""
    data = np.column_stack([convolved_decay(times, tau, t0, resolution) for tau in taus])
    frame = pandas.DataFrame(data, index=times, columns=list(range(len(taus))))
    frame.index.name = "time"
    return frame


def bateman_concentrations(times, taus, t0=0.0):
    """Analytic solution of A->B->C->... with an instantaneous response.

    The reference for the sequential model. Requires distinct rate constants;
    the closed form has a removable singularity at equal rates.
    """
    k = 1.0 / np.asarray(taus, dtype=float)
    if len(set(np.round(k, 12))) != len(k):
        raise ValueError("Bateman reference needs distinct rate constants")
    t = np.asarray(times, dtype=float) - t0
    out = np.zeros((len(t), len(k)))
    for n in range(len(k)):
        total = np.zeros_like(t)
        for i in range(n + 1):
            denominator = np.prod([k[j] - k[i] for j in range(n + 1) if j != i])
            total += np.exp(-k[i] * t) / denominator
        out[:, n] = np.prod(k[:n]) * total
    out[t < 0] = 0.0
    frame = pandas.DataFrame(out, index=times, columns=list(range(len(k))))
    frame.index.name = "time"
    return frame


def make_dataset(
    taus=(1.0, 30.0), centres=(480.0, 600.0), t0=0.0, resolution=0.1, times=None, waves=None, noise=0.0, seed=0
):
    """A dataset that is exactly ``concentrations @ spectra``.

    With ``noise=0`` the factorisation is exact to machine precision, which is
    what lets the amplitude solver be checked against an exact answer.
    """
    times = default_times() if times is None else np.asarray(times, dtype=float)
    waves = default_waves() if waves is None else np.asarray(waves, dtype=float)
    concentrations = parallel_concentrations(times, taus, t0, resolution)
    spectra = gaussian_bands(waves, centres)
    values = concentrations.values @ spectra.values
    if noise:
        values = values + np.random.default_rng(seed).normal(0.0, noise, values.shape)
    ds = pandas.DataFrame(values, index=times, columns=waves)
    ds.index.name = "time"
    ds.columns.name = "wavelength"
    truth = Truth(
        taus=np.asarray(taus, dtype=float),
        rates=1.0 / np.asarray(taus, dtype=float),
        t0=t0,
        resolution=resolution,
        concentrations=concentrations,
        spectra=spectra,
        times=times,
        waves=waves,
    )
    return ds, truth


def make_chirped_dataset(coeffs=(0.0, 0.0, 0.0, -2e-3, 1.0), taus=(5.0,), centres=(500.0,),
                         waves=None, times=None, resolution=0.3):
    """A dataset whose zero time drifts with wavelength by a known polynomial.

    ``coeffs`` are in numpy.polyval order, giving the delay in time units at a
    given wavelength in nm. Each column is generated with its own onset, which
    is what chirp correction has to undo.
    """
    times = default_times(-5.0, 50.0, 400) if times is None else np.asarray(times, dtype=float)
    waves = np.linspace(400.0, 700.0, 24) if waves is None else np.asarray(waves, dtype=float)
    spectrum = gaussian_bands(waves, centres).values[0]
    values = np.zeros((times.size, waves.size))
    for j, wave in enumerate(waves):
        onset = np.polyval(coeffs, wave)
        values[:, j] = spectrum[j] * convolved_decay(times, taus[0], onset, resolution)
    ds = pandas.DataFrame(values, index=times, columns=waves)
    ds.index.name = "time"
    ds.columns.name = "wavelength"
    return ds, np.asarray(coeffs, dtype=float)


def make_sparse_dataset(n_waves=8, taus=(5.0,)):
    """A dataset with only a handful of widely separated wavelength channels.

    What an X-ray emission or single-channel experiment produces, where the
    dense-spectrum assumptions elsewhere do not hold.
    """
    waves = np.array([400.0, 430.0, 520.0, 560.0, 610.0, 700.0, 780.0, 850.0])[:n_waves]
    times = default_times(-5.0, 50.0, 200)
    return make_dataset(taus=taus, centres=(600.0,) * len(taus), times=times, waves=waves)

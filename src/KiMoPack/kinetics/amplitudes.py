"""Species spectra from a measurement and a set of concentration profiles.

The whole fit rests on a separation: the measurement is ``A = c @ spectra``,
so once the kinetics fix ``c`` the spectra follow from a linear solve rather
than from the optimiser. Only the handful of kinetic parameters are searched;
the hundreds of spectral amplitudes are solved exactly at every step.
"""

import numpy as np
import pandas


def fill_int(ds, c, final=True, baseunit="ps", return_shapes=False):
    """Solve for the species spectra and report how well they reproduce the data.

    Parameters
    ----------
    ds : pandas.DataFrame
        Measurement, times down the index and wavelengths across the columns.
    c : pandas.DataFrame
        Concentration of each species over the same times.
    final : bool, optional
        Return the full reconstruction as well as the error. The optimiser
        leaves this off, since it only needs a number to minimise.
    baseunit : str, optional
        Unused; retained because callers pass it positionally.
    return_shapes : bool, optional
        Include the spectra without paying for the full reconstruction.

    Returns
    -------
    dict
        Always ``error``, the summed squared residual. With ``final``, also
        ``A`` (data), ``AC`` (model), ``AE`` (residual), ``DAC`` (the spectra)
        and ``c``.
    """
    time = ds.index.values.astype("float")
    wl = ds.columns.values.astype("float")
    time_label = ds.index.name
    energy_label = ds.columns.name

    A = ds.values
    er = c.values
    ert = er.T
    try:
        eps = np.linalg.lstsq(np.matmul(ert, er), np.matmul(ert, A), rcond=-1)[0]
    except np.linalg.LinAlgError:
        # A degenerate concentration matrix means these parameters are
        # unusable; report a large error so the optimiser walks away.
        return {"error": 1000}

    # A species with no distinguishable contribution leaves the solve
    # underdetermined; drop it rather than propagate nan through the model.
    eps[np.isnan(eps)] = 0
    eps[np.isinf(eps)] = 0

    AC = np.matmul(er, eps)
    AE = A - AC
    fit_error = (AE**2).sum()

    if not (final or return_shapes):
        return {"error": fit_error}

    DAC = pandas.DataFrame(eps.T, index=wl)
    DAC.index.name = energy_label
    if len(c.columns) == DAC.shape[1]:
        DAC.columns = c.columns.values

    if not final:
        return {"DAC": DAC, "error": fit_error, "c": c}

    frames = {}
    for key, values in (("A", A), ("AC", AC), ("AE", AE)):
        frame = pandas.DataFrame(values, index=time, columns=wl)
        frame.index.name = time_label
        frame.columns.name = energy_label
        frames[key] = frame
    return {**frames, "DAC": DAC, "error": fit_error, "c": c}

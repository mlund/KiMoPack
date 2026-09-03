"""The parameter table shared by the models, the fit, and the results tables.

lmfit carries the values and bounds; the models need to know which of those
are rate constants and which describe the experiment. That distinction is
made by name, and the convention lives here so the models, the plots, and the
saved projects all read it the same way.
"""

import lmfit
import pandas

#: A parameter is a rate constant if its name starts with one of these. It is
#: a naming convention rather than a declaration, so a parameter called
#: 'kappa' would be inverted into a lifetime along with the real rates.
RATE_PREFIXES = ("k", "tk")


def is_rate(name):
    """True when this parameter name denotes a rate constant."""
    return str(name).startswith(RATE_PREFIXES)


def par_to_pardf(par):
    """Convert lmfit Parameters into a table, tagging the rate constants."""
    rows = {}
    for key in par.keys():
        rows[key] = {
            "value": par[key].value,
            "is_rate": is_rate(key),
            "min": par[key].min,
            "max": par[key].max,
            "vary": par[key].vary,
            "expr": par[key].expr,
        }
    return pandas.DataFrame(rows).T


def pardf_to_par(par_df):
    """Convert the table back into lmfit Parameters."""
    par = lmfit.Parameters()
    for key in par_df.index.values:
        par.add(
            key,
            value=par_df.loc[key, "value"],
            vary=par_df.loc[key, "vary"],
            min=par_df.loc[key, "min"],
            max=par_df.loc[key, "max"],
            expr=par_df.loc[key, "expr"],
        )
    return par


#: Inverting a rate reverses the order of its bounds, so they swap.
_INVERTED_PAIRS = {
    "min": "max",
    "max": "min",
    "lower_limit": "upper_limit",
    "upper_limit": "lower_limit",
}


def pardf_to_timedf(pardf):
    """Report rate constants as lifetimes, leaving everything else alone.

    Rates are what the fit optimises; lifetimes are what gets published, so
    this exists purely for presentation.
    """
    timedf = pardf.copy()
    has_confidence = "upper_limit" in pardf.keys()
    keys = ["init_value", "value", "min", "max"]
    if has_confidence:
        keys += ["lower_limit", "upper_limit"]

    for key in keys:
        if key not in pardf.keys():
            # init_value is absent from projects loaded off disk.
            continue
        target = _INVERTED_PAIRS.get(key, key)
        for row in pardf.index.values:
            if not pardf.loc[row, "is_rate"]:
                continue
            value = pardf.loc[row, key]
            if value is None:
                continue
            timedf.loc[row, target] = 1 / value if value != 0 else "inf"
    return timedf

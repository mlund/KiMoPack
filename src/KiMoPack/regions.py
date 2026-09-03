"""Cut regions and the spans left between them.

Users mask parts of an axis — scattered pump light in the spectrum, unusable
delays in time — by naming the regions to remove. Drawing then needs the
complement: the runs that survive, so a line breaks at the gap rather than
being drawn straight across it.

The cut list arrives in whichever shape was convenient at the call site: a
single ``[low, high]`` pair, a list of such pairs, or those pairs already
flattened. Normalising once here means callers stop caring which, and stop
needing to know whether the values happen to be sorted.
"""

import numbers

from .numerics import nm_to_ev


def normalise_cuts(cuts):
    """Return cuts as sorted, non-overlapping ``(low, high)`` pairs.

    Accepts ``None``, a single pair, a list of pairs, or a flat list of
    alternating bounds. Overlapping or touching regions are merged, so the
    number of spans they produce always matches the number of gaps on screen.
    """
    if cuts is None:
        return []

    values = []
    for entry in cuts:
        if isinstance(entry, numbers.Number):
            values.append(float(entry))
        else:
            values.extend(float(x) for x in entry)

    if len(values) % 2:
        raise ValueError(
            f"cut regions must be given as low/high pairs, got {len(values)} bounds: {values}"
        )

    pairs = sorted((min(a, b), max(a, b)) for a, b in zip(values[::2], values[1::2], strict=True))

    merged = []
    for low, high in pairs:
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return [tuple(pair) for pair in merged]


def contiguous_spans(cuts):
    """Spans between the cuts, as ``(start, stop)`` bounds for ``.loc``.

    ``None`` at either end means "open" — run to the edge of the data. With no
    cuts the result is a single fully open span, so a caller can loop over the
    result unconditionally instead of special-casing the common case.
    """
    pairs = normalise_cuts(cuts)
    if not pairs:
        return [(None, None)]

    spans = [(None, pairs[0][0])]
    for (_, previous_high), (next_low, _) in zip(pairs, pairs[1:], strict=False):
        spans.append((previous_high, next_low))
    spans.append((pairs[-1][1], None))
    return spans


def cut_pairs(cuts, to_energy=False):
    """Cut regions as sorted pairs, converted to eV when the axis is energy.

    Ranges are given in nm even when the plot is drawn against energy, so they
    have to be converted and re-sorted before they can be compared with the
    axis they mask.
    """
    if cuts is None:
        return []
    if to_energy:
        if isinstance(cuts[0], numbers.Number):
            cuts = [list(nm_to_ev(cuts))]
        else:
            cuts = [list(nm_to_ev(pair)) for pair in cuts]
    return normalise_cuts(cuts)


def frame_spans(frame, cuts, to_energy=False):
    """Yield ``(piece, is_first)`` for each run of the index between cuts.

    Drawing a masked trace means drawing it in pieces. ``is_first`` marks the
    one piece that should carry the legend entry, so a trace broken into three
    does not appear three times in the legend.
    """
    pairs = cut_pairs(cuts, to_energy)
    spans = [(None, None)] if not pairs else (
        [(None, pairs[0][0])]
        + [(a[1], b[0]) for a, b in zip(pairs, pairs[1:], strict=False)]
        + [(pairs[-1][1], None)]
    )
    first = True
    for start, stop in spans:
        piece = frame.loc[start:stop]
        if len(piece.index) == 0:
            continue
        yield piece, first
        first = False

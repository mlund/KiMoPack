"""Where output goes.

Output paths are assembled from a project directory, a subfolder and a
filename supplied at three different levels, so the rules for combining them
live here rather than being re-derived at each of the call sites.
"""

import os
import re
from pathlib import Path


def _as_path(value):
    """Accept str, bytes, or any PathLike.

    Byte paths come back from some filesystem APIs; formatting them as text
    would produce the literal ``b'out'``, quotes and all, and create a
    directory by that name.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        return Path(os.fsdecode(value))
    return Path(value)


def check_folder(path=None, current_path=None, filename=None):
    """Resolve a target directory, creating it, and optionally append a file.

    The directory is created as a side effect, so a caller that only wants to
    compute a name will still leave it behind.

    Parameters
    ----------
    path : str, bytes, Path, optional
        Absolute paths are used unchanged; relative ones hang off
        ``current_path``.
    current_path : str, bytes, Path, optional
        Base for a relative ``path``. Ignored with a warning unless absolute.
    filename : str, bytes, Path, optional
        Appended to the resolved directory. Not created.

    Returns
    -------
    pathlib.Path
    """
    path = _as_path(path)
    current_path = _as_path(current_path)
    filename = _as_path(filename)

    if current_path is not None and not current_path.is_absolute():
        print("attention, current_path was given but not absolute, replaced by cwd")
        current_path = None

    if path is None:
        directory = current_path if current_path is not None else Path.cwd()
    elif path.is_absolute():
        directory = path
    else:
        directory = (current_path if current_path is not None else Path.cwd()).joinpath(path)

    directory.mkdir(parents=True, exist_ok=True)
    return directory if filename is None else directory.joinpath(filename)


def clean_double_string(filename, path=None):
    """Collapse doubled dashes and dots in a file, rewriting it in place.

    Some instruments write '1.5--2.5' or '3..4' into their ASCII exports,
    which no float parser accepts.
    """
    if path is None:
        path = os.path.dirname(os.path.realpath(__file__))
    with open(Path(os.sep.join([str(path), str(filename)])), "r+") as f:
        text = f.read()
        text = re.sub("--", "-", text)
        text = re.sub(r"\.+", ".", text)
        f.seek(0)
        f.write(text)
        f.truncate()

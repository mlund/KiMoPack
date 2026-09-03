"""Renders the public API of ``plot_func`` as stable text.

Kept separate from the test that consumes it so the golden file can be
regenerated with ``python -m tests._surface`` after a deliberate API change.
Defaults are sanitised because reprs of matplotlib colormaps and numpy arrays
embed memory addresses, which would make the dump differ on every run.
"""

import inspect
import re

_ADDRESS = re.compile(r"<([\w.]+?)(?: object)? at 0x[0-9a-f]+>")


def _default(value):
    text = repr(value)
    text = _ADDRESS.sub(lambda m: f"<obj:{m.group(1).rsplit('.', 1)[-1]}>", text)
    return " ".join(text.split())


def _signature(obj):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return "(?)"
    parts = []
    for name, par in sig.parameters.items():
        if par.kind is par.VAR_POSITIONAL:
            parts.append("*" + name)
        elif par.kind is par.VAR_KEYWORD:
            parts.append("**" + name)
        elif par.default is par.empty:
            parts.append(name)
        else:
            parts.append(f"{name}={_default(par.default)}")
    return "({})".format(", ".join(parts))


def _is_ours(obj):
    """True for things KiMoPack defines, as opposed to re-exported dependencies.

    Signatures of matplotlib/numpy objects would churn on every upstream
    release, so those are recorded by name alone: their presence in the
    namespace is the contract (notebooks use ``pf.cm``), their internals are not.
    """
    module = getattr(obj, "__module__", "") or ""
    return module.startswith("KiMoPack") or module == "plot_func"


def dump(module):
    """Yield one ``name(signature)`` line per public callable, sorted."""
    lines = []
    for name in sorted(dir(module)):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and not _is_ours(obj):
            lines.append(f"{name}: reexport {type(obj).__name__}")
        elif inspect.isclass(obj):
            lines.append(f"class {name}")
            for attr in sorted(dir(obj)):
                # Name-mangled privates are part of the documented surface
                # (docs/source/plot_func.rst lists them under :private-members:).
                if attr.startswith("__") or (attr.startswith("_") and "__" not in attr):
                    continue
                member = getattr(obj, attr)
                if callable(member):
                    lines.append(f"    {name}.{attr}{_signature(member)}")
        elif inspect.isfunction(obj):
            lines.append(f"def {name}{_signature(obj)}")
        else:
            lines.append(f"{name}: {type(obj).__name__}")
    return lines


if __name__ == "__main__":
    import pathlib

    import matplotlib

    matplotlib.use("Agg")
    import KiMoPack.plot_func as plot_func

    target = pathlib.Path(__file__).with_name("facade_surface.txt")
    target.write_text("\n".join(dump(plot_func)) + "\n")
    print(f"wrote {target}")

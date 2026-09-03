"""Shared assertions and skip rules.

Numeric code needs assertions that report *what* diverged rather than dumping
two arrays, because a failure here usually means a physical quantity is wrong,
not that a container differs.
"""

import os
import unittest

import numpy as np

#: Tests reading the repository's real (multi-megabyte) datasets. A full fit on
#: one of those takes seconds to minutes, so they stay off unless asked for.
slow = unittest.skipUnless(
    os.environ.get("KIMOPACK_SLOW_TESTS"),
    "set KIMOPACK_SLOW_TESTS=1 to run tests against the bundled datasets",
)


class NumericTestCase(unittest.TestCase):
    def assertAllClose(self, actual, desired, rtol=1e-7, atol=0.0, msg=""):
        actual = np.asarray(actual, dtype=float)
        desired = np.asarray(desired, dtype=float)
        self.assertEqual(actual.shape, desired.shape, f"shape mismatch. {msg}")
        if np.allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=True):
            return
        bad = ~np.isclose(actual, desired, rtol=rtol, atol=atol, equal_nan=True)
        worst = np.unravel_index(np.argmax(np.abs(actual - desired)), actual.shape)
        self.fail(
            f"{msg}\n{bad.sum()} of {bad.size} values differ (rtol={rtol:g}, atol={atol:g}).\n"
            f"worst at {worst}: got {actual[worst]!r}, expected {desired[worst]!r}"
        )

    def assertFrameAllClose(self, actual, desired, rtol=1e-7, atol=0.0, msg=""):
        self.assertEqual(list(actual.columns), list(desired.columns), f"columns differ. {msg}")
        self.assertAllClose(actual.index.values, desired.index.values, msg=f"index differs. {msg}")
        self.assertAllClose(actual.values, desired.values, rtol=rtol, atol=atol, msg=msg)

    def assertMonotonicDecreasing(self, values, msg=""):
        values = np.asarray(values, dtype=float)
        rising = np.where(np.diff(values) > 1e-12)[0]
        if rising.size:
            self.fail(f"{msg}\nrises at index {rising[0]}: {values[rising[0]]!r} -> {values[rising[0] + 1]!r}")

    def assertUnchanged(self, frame, before, msg=""):
        """Guard against a function mutating its caller's DataFrame."""
        self.assertEqual(list(frame.columns), list(before.columns), f"columns were mutated. {msg}")
        self.assertAllClose(frame.index.values, before.index.values, msg=f"index was mutated. {msg}")
        self.assertAllClose(frame.values, before.values, msg=f"values were mutated. {msg}")

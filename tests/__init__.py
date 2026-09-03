"""Test suite for KiMoPack.

Selects a non-interactive matplotlib backend before anything can import
pyplot, so the suite runs headless and never opens a window.
"""

import matplotlib

matplotlib.use("Agg")

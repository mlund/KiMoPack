"""Turning results into figures.

Split in two so that a figure can be described without being drawn: the
settings and the drawing model here are free of matplotlib, and the renderer
is the one place that knows about it.
"""

"""Turning cut regions into the spans that remain drawable.

Users mask out scattered pump light and unusable time ranges by naming the
regions to remove. Every plot needs the complement — the runs between them —
so lines break at the gap instead of being drawn straight across it.
"""

import unittest

from KiMoPack import regions


class ContiguousSpans(unittest.TestCase):
    def test_no_cuts_leaves_one_open_span(self):
        self.assertEqual(regions.contiguous_spans(None), [(None, None)])

    def test_a_single_region_splits_into_two(self):
        self.assertEqual(regions.contiguous_spans([400, 450]), [(None, 400), (450, None)])

    def test_two_regions_split_into_three(self):
        self.assertEqual(
            regions.contiguous_spans([[400, 450], [600, 620]]),
            [(None, 400), (450, 600), (620, None)],
        )

    def test_flat_and_nested_forms_agree(self):
        """Callers pass either; sub_ds and the plots accept both."""
        self.assertEqual(
            regions.contiguous_spans([400, 450, 600, 620]),
            regions.contiguous_spans([[400, 450], [600, 620]]),
        )

    def test_three_regions_split_into_four(self):
        self.assertEqual(
            regions.contiguous_spans([[10, 20], [30, 40], [50, 60]]),
            [(None, 10), (20, 30), (40, 50), (60, None)],
        )

    def test_each_span_appears_exactly_once(self):
        """The hand-rolled loops redrew the trailing span once per extra region."""
        for count in range(1, 6):
            with self.subTest(regions=count):
                cuts = [[10 * i, 10 * i + 5] for i in range(1, count + 1)]
                spans = regions.contiguous_spans(cuts)
                self.assertEqual(len(spans), count + 1)
                self.assertEqual(len(set(spans)), count + 1)

    def test_a_reversed_pair_is_normalised(self):
        self.assertEqual(regions.contiguous_spans([450, 400]), [(None, 400), (450, None)])

    def test_regions_given_out_of_order_are_sorted(self):
        self.assertEqual(
            regions.contiguous_spans([[600, 620], [400, 450]]),
            [(None, 400), (450, 600), (620, None)],
        )

    def test_an_odd_number_of_bounds_is_rejected(self):
        """Silently dropping the stray value would quietly mask the wrong range."""
        with self.assertRaises(ValueError) as caught:
            regions.contiguous_spans([400, 450, 600])
        self.assertIn("pairs", str(caught.exception))

    def test_overlapping_regions_merge(self):
        self.assertEqual(
            regions.contiguous_spans([[400, 500], [450, 550]]),
            [(None, 400), (550, None)],
        )

    def test_touching_regions_merge(self):
        self.assertEqual(
            regions.contiguous_spans([[400, 500], [500, 550]]),
            [(None, 400), (550, None)],
        )


class NormaliseCuts(unittest.TestCase):
    def test_returns_sorted_disjoint_pairs(self):
        self.assertEqual(regions.normalise_cuts([[600, 620], [450, 400]]), [(400, 450), (600, 620)])

    def test_none_means_no_cuts(self):
        self.assertEqual(regions.normalise_cuts(None), [])

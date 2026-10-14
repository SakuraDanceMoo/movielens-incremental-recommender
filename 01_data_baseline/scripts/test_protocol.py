import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from core import metrics, rank_items, sample_negatives, split_sizes, temporal_scores
from prepare import fit


class ProtocolTests(unittest.TestCase):
    def test_ndcg_hand_calculation(self):
        actual, recall = metrics([1, 0, 1], 2)
        expected = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
        self.assertAlmostEqual(actual, expected)
        self.assertEqual(recall, 1)
        self.assertEqual(metrics([0] * 10, 7), (0, 0))
        self.assertAlmostEqual(metrics([1] * 10, 12)[0], 1)
        with self.assertRaises(ValueError): metrics([0], 0)

    def test_split_residual_and_ties(self):
        self.assertEqual(split_sizes(21), (16, 2, 3))
        self.assertEqual(split_sizes(20), (16, 2, 2))
        df = pl.DataFrame({"userId": [2, 1, 1], "timestamp": [1, 5, 5], "rowId": [0, 2, 1]})
        self.assertEqual(df.sort(["userId", "timestamp", "rowId"])["rowId"].to_list(), [1, 2, 0])

    def test_fixed_negatives_and_disjointness(self):
        catalog = np.arange(1, 501, dtype=np.int32)
        rated = np.arange(1, 80)
        a = sample_negatives(catalog, rated, np.random.default_rng(20260907))
        b = sample_negatives(catalog, rated, np.random.default_rng(20260907))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(np.unique(a)), 100)
        self.assertFalse(np.isin(a, rated).any())
        with self.assertRaises(ValueError): sample_negatives(np.arange(10), np.arange(5), np.random.default_rng(0))

    def test_future_and_equal_time_events_excluded(self):
        items = np.array([1, 1, 2, 2, 2])
        times = np.array([1, 2, 3, 5, 6])
        np.testing.assert_array_equal(temporal_scores(items, times, np.array([1, 2, 3]), 5), [2, 1, 0])

    def test_deterministic_rank_ties(self):
        items = np.array([90, 2, 3, 1])
        scores = np.array([5, 5, 1, 5])
        np.testing.assert_array_equal(items[rank_items(items, scores)], [1, 2, 90, 3])

    def test_fit_counts_positive_train_only(self):
        train = pl.DataFrame({"movieId": [1, 1, 2, 9], "rating": [4., 5., 3., 4.], "timestamp": [1, 2, 3, 4]})
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(fit(train, Path(t), 4), 3)
            self.assertEqual(pl.read_csv(Path(t) / "popularity.csv").to_dicts(),
                             [{"movieId": 1, "positive_count": 2}, {"movieId": 9, "positive_count": 1}])


if __name__ == "__main__":
    unittest.main(verbosity=2)

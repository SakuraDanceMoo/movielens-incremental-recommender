"""Pure protocol helpers shared by preparation, evaluation and regression tests."""
import numpy as np


def split_sizes(n):
    a, b = int(n * 8 // 10), int(n * 9 // 10)
    return a, b - a, n - b


def sample_negatives(catalog, rated, rng, size=100):
    """Uniform rejection sampling; catalog must be sorted and unique."""
    rated = np.unique(np.asarray(rated))
    locations = np.searchsorted(catalog, rated)
    inside = locations < len(catalog)
    matches = inside & (catalog[np.minimum(locations, len(catalog) - 1)] == rated)
    available = len(catalog) - int(matches.sum())
    if available < size:
        raise ValueError("Too few unrated catalog items")
    selected = np.empty(0, dtype=np.int32)
    while len(selected) < size:
        draw = rng.choice(catalog, size=max(2 * (size - len(selected)), 128), replace=True)
        good = draw[~np.isin(draw, rated) & ~np.isin(draw, selected)]
        # Keep first occurrences, preserving random draw order.
        _, indices = np.unique(good, return_index=True)
        selected = np.concatenate([selected, good[np.sort(indices)]])[:size]
    return selected.astype(np.int32)


def rank_items(items, scores, k=10):
    return np.lexsort((items, -scores))[:k]


def metrics(relevance, total_positives, k=10):
    relevance = np.asarray(relevance, dtype=np.float64)[:k]
    if total_positives <= 0:
        raise ValueError("No-positive users are excluded from metric denominator")
    discount = 1.0 / np.log2(np.arange(2, len(relevance) + 2))
    ideal = np.sum(1.0 / np.log2(np.arange(2, min(k, total_positives) + 2)))
    return float(np.dot(relevance, discount) / ideal), float(relevance.sum() / total_positives)


def temporal_scores(train_items, train_times, query_items, cutoff):
    """Small independent reference for tests; production uses a cumulative sweep."""
    values, counts = np.unique(train_items[train_times < cutoff], return_counts=True)
    lookup = dict(zip(values.tolist(), counts.tolist()))
    return np.array([lookup.get(int(i), 0) for i in query_items], dtype=np.int64)

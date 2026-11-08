"""Independently recompute graph-stage top-k metrics and PPR convergence checks."""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "05_graph_recommendation"
BASE = ROOT / "01_data_baseline"
K = 10


def recompute(path, users, positives, offsets):
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            user = int(row["userId"])
            rows.setdefault(user, []).append((int(row["movieId"]), int(row["rank"]), float(row["score"])))
    if set(rows) != set(map(int, users)):
        raise AssertionError("Graph result user coverage mismatch")
    ndcgs, recalls = [], []
    for index, user_value in enumerate(users):
        user = int(user_value)
        recs = rows[user]
        if len(recs) != K or [r[1] for r in recs] != list(range(1, K + 1)):
            raise AssertionError("Expected ten ordered rows per user")
        ids = [r[0] for r in recs]
        if len(set(ids)) != K:
            raise AssertionError("Duplicate movie in Top10")
        if any(a[2] < b[2] or (a[2] == b[2] and a[0] > b[0]) for a, b in zip(recs, recs[1:])):
            raise AssertionError("Top10 score order invalid")
        pos = set(map(int, positives[offsets[index]:offsets[index + 1]]))
        rel = np.asarray([int(movie in pos) for movie in ids], dtype=np.float64)
        ideal = np.sum(1.0 / np.log2(np.arange(2, min(K, len(pos)) + 2)))
        dcg = float(np.dot(rel, 1.0 / np.log2(np.arange(2, K + 2))))
        ndcgs.append(dcg / ideal)
        recalls.append(int(rel.sum()) / len(pos))
    return {"users": len(users), "rows": len(users) * K,
            "ndcg_at_10": float(np.mean(ndcgs)), "recall_at_10": float(np.mean(recalls))}


def main():
    report = json.loads((STAGE / "logs/graph_metrics.json").read_text(encoding="utf-8"))
    with np.load(BASE / "data/processed/candidates.npz") as archive:
        users = archive["user_ids"][:report["graph"]["users"]]
        positives, offsets = archive["positive_items"], archive["positive_offsets"]
    graph = recompute(STAGE / report["ppr"]["top10_file"], users, positives, offsets)
    fused = recompute(STAGE / report["graph_als_rank_fusion"]["top10_file"], users, positives, offsets)
    for actual, expected in ((graph, report["ppr"]), (fused, report["graph_als_rank_fusion"])):
        for key in ("ndcg_at_10", "recall_at_10"):
            if not math.isclose(actual[key], expected[key], abs_tol=1e-11):
                raise AssertionError(f"Independent {key} mismatch")
    if not report["acceptance"]["ppr_residual_within_tolerance"]:
        raise AssertionError("PPR did not reach configured convergence tolerance")
    result = {"status": "passed", "graph_ppr": graph, "graph_als_fusion": fused,
              "ppr_residual_within_tolerance": True,
              "checks": ["candidate user population", "rank sequence", "unique Top10 items",
                         "score ordering", "independent NDCG@10", "independent Recall@10", "PPR residual"]}
    (STAGE / "logs/verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

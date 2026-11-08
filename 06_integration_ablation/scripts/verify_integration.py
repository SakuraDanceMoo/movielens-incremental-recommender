"""Recompute integration and ablation macro metrics from per-user results."""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "06_integration_ablation"
VARIANTS = {"full", "without_incremental", "without_graph", "without_lsh", "als_only"}


def main():
    report = json.loads((STAGE / "logs/integration_metrics.json").read_text(encoding="utf-8"))
    by_variant = {name: [] for name in VARIANTS}
    with gzip.open(STAGE / report["outputs"]["ablation_per_user"], "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            by_variant[row["variant"]].append(row)
    with np.load(ROOT / "01_data_baseline/data/processed/candidates.npz") as archive:
        users = archive["user_ids"][:report["workset_users"]]
        positives, offsets = archive["positive_items"], archive["positive_offsets"]
        negatives = archive["negatives"][:report["workset_users"]]
    candidates_by_user = {}
    for index, user_value in enumerate(users):
        user = int(user_value)
        candidates_by_user[user] = set(map(int, positives[offsets[index]:offsets[index + 1]])) | set(map(int, negatives[index]))

    top_by_variant = {name: {} for name in VARIANTS}
    with gzip.open(STAGE / report["outputs"]["ablation_top10"], "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            top_by_variant[row["variant"]].setdefault(int(row["userId"]), []).append(
                (int(row["movieId"]), int(row["rank"]), float(row["score"])))
    result_rows = {}
    for name in VARIANTS:
        rows = by_variant[name]
        if len(rows) != report["workset_users"]:
            raise AssertionError(f"Wrong per-user population for {name}")
        ndcg = float(np.mean([float(row["ndcg_at_10"]) for row in rows]))
        recall = float(np.mean([float(row["recall_at_10"]) for row in rows]))
        expected = report["variants"][name]
        if not math.isclose(ndcg, expected["ndcg_at_10"], abs_tol=1e-11):
            raise AssertionError(f"NDCG mismatch for {name}")
        if not math.isclose(recall, expected["recall_at_10"], abs_tol=1e-11):
            raise AssertionError(f"Recall mismatch for {name}")
        ranked_rows = top_by_variant[name]
        if set(ranked_rows) != set(map(int, users)):
            raise AssertionError(f"Top-10 user population mismatch for {name}")
        top_ndcgs, top_recalls = [], []
        for index, user_value in enumerate(users):
            user = int(user_value)
            recommendations = ranked_rows[user]
            if len(recommendations) != 10 or [item[1] for item in recommendations] != list(range(1, 11)):
                raise AssertionError(f"Invalid Top-10 ranks for {name}, user {user}")
            item_ids = [item[0] for item in recommendations]
            if len(set(item_ids)) != 10 or not set(item_ids).issubset(candidates_by_user[user]):
                raise AssertionError(f"Duplicate or out-of-candidate item for {name}, user {user}")
            if any(a[2] < b[2] or (a[2] == b[2] and a[0] > b[0])
                   for a, b in zip(recommendations, recommendations[1:])):
                raise AssertionError(f"Score ordering mismatch for {name}, user {user}")
            pos = set(map(int, positives[offsets[index]:offsets[index + 1]]))
            relevant = np.asarray([int(item in pos) for item in item_ids], dtype=np.float64)
            ideal = np.sum(1 / np.log2(np.arange(2, min(10, len(pos)) + 2)))
            top_ndcgs.append(float(np.dot(relevant, 1 / np.log2(np.arange(2, 12))) / ideal))
            top_recalls.append(int(relevant.sum()) / len(pos))
        top_ndcg, top_recall = float(np.mean(top_ndcgs)), float(np.mean(top_recalls))
        if not math.isclose(top_ndcg, ndcg, abs_tol=1e-11) or not math.isclose(top_recall, recall, abs_tol=1e-11):
            raise AssertionError(f"Top10 metrics do not independently recompute for {name}")
        result_rows[name] = {"users": len(rows), "ndcg_at_10": top_ndcg, "recall_at_10": top_recall,
                             "candidate_pool_mean": float(np.mean([int(row["candidate_count"]) for row in rows])),
                             "candidate_positive_recall": float(np.mean([float(row["candidate_positive_recall"]) for row in rows]))}
    top10_rows = 0
    seen = set()
    with gzip.open(STAGE / report["outputs"]["integrated_top10"], "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["userId"]), int(row["rank"]))
            if key in seen:
                raise AssertionError("Duplicate user/rank in integrated output")
            seen.add(key)
            top10_rows += 1
    if top10_rows != report["workset_users"] * 10:
        raise AssertionError("Integrated Top10 row count mismatch")
    result = {"status": "passed", "variants": result_rows, "integrated_top10_rows": top10_rows,
              "full_ndcg_gain_at_least_10_percent": report["acceptance"]["full_ndcg_gain_at_least_10_percent"],
              "end_to_end_service_p99_verified": False,
              "checks": ["five ablation populations", "independent per-user macro metrics",
                         "candidate-pool summaries", "Top10 candidate membership", "unique Top10 items",
                         "ranking order", "independent Top10 NDCG@10 and Recall@10",
                         "integrated Top10 user/rank uniqueness"]}
    (STAGE / "logs/verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

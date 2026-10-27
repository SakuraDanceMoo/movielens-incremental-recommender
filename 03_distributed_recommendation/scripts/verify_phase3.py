"""Independently validate Phase 3 metrics and top-10 candidate membership."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "03_distributed_recommendation"
DATA = ROOT / "01_data_baseline/data/processed"
METRICS = STAGE / "logs/spark_metrics.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(relevant, positives):
    relevant = np.asarray(relevant, dtype=np.float64)
    discounts = 1 / np.log2(np.arange(2, len(relevant) + 2))
    ideal = np.sum(1 / np.log2(np.arange(2, min(10, positives) + 2)))
    hits = int(np.count_nonzero(relevant))
    return float(np.dot(relevant, discounts) / ideal), hits / int(positives)


def read_groups(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        current_user, rows = None, []
        for row in reader:
            user = int(row["userId"])
            if current_user is not None and user != current_user:
                yield current_user, rows
                rows = []
            current_user = user
            rows.append((int(row["movieId"]), int(row["rank"]), float(row["score"])))
        if current_user is not None:
            yield current_user, rows


def validate_rankings(path, indices, users, positives, offsets, negatives):
    expected = iter(indices)
    ndcgs, recalls, checked_rows = [], [], 0
    for user, rows in read_groups(path):
        index = next(expected, None)
        if index is None or user != int(users[index]):
            raise AssertionError(f"Unexpected or out-of-order user in {path}: {user}")
        if len(rows) != 10 or [row[1] for row in rows] != list(range(1, 11)):
            raise AssertionError(f"Expected ranks 1..10 for user {user}")
        pos = set(map(int, positives[offsets[index]:offsets[index + 1]]))
        candidates = pos.union(map(int, negatives[index]))
        items = [row[0] for row in rows]
        if len(set(items)) != 10 or not set(items).issubset(candidates):
            raise AssertionError(f"Top-10 contains duplicate or non-candidate items for user {user}")
        scores = [row[2] for row in rows]
        if any(a < b or (a == b and rows[i][0] > rows[i + 1][0])
               for i, (a, b) in enumerate(zip(scores, scores[1:]))):
            raise AssertionError(f"Ranking order is invalid for user {user}")
        relevance = [int(item in pos) for item in items]
        ndcg, recall = score(relevance, len(pos))
        ndcgs.append(ndcg)
        recalls.append(recall)
        checked_rows += len(rows)
    if next(expected, None) is not None:
        raise AssertionError(f"Missing user recommendations in {path}")
    return {"users": len(ndcgs), "rows": checked_rows,
            "ndcg_at_10": float(np.mean(ndcgs)), "recall_at_10": float(np.mean(recalls))}


def main():
    report = json.loads(METRICS.read_text(encoding="utf-8"))
    with np.load(DATA / "candidates.npz") as arrays:
        users, positives, offsets, negatives = (arrays[name] for name in
            ("user_ids", "positive_items", "positive_offsets", "negatives"))
        if sha256(DATA / "candidates.npz") != report["candidate_sha256"]:
            raise AssertionError("Candidate file hash changed")

        scale_checks = []
        for result in report["als_scale_results"]:
            scale = result["scale"]
            indices = np.flatnonzero(users % 100 < int(round(scale * 100)))
            with gzip.open(STAGE / result["per_user_file"], "rt", encoding="utf-8", newline="") as stream:
                values = list(csv.DictReader(stream))
            if len(values) != len(indices) or [int(row["userId"]) for row in values] != list(map(int, users[indices])):
                raise AssertionError(f"Per-user ALS rows do not match candidates at scale {scale}")
            ndcg = float(np.mean([float(row["ndcg_at_10"]) for row in values]))
            recall = float(np.mean([float(row["recall_at_10"]) for row in values]))
            if not math.isclose(ndcg, result["ndcg_at_10"], abs_tol=1e-10):
                raise AssertionError(f"ALS NDCG mismatch at scale {scale}")
            if not math.isclose(recall, result["recall_at_10"], abs_tol=1e-10):
                raise AssertionError(f"ALS Recall mismatch at scale {scale}")
            scale_checks.append({"scale": scale, "users": len(indices),
                                "per_user_ndcg_recomputes": True,
                                "per_user_recall_recomputes": True})

        full = report["als_scale_results"][-1]
        full_result = validate_rankings(STAGE / full["top10_file"], np.arange(len(users)),
                                        users, positives, offsets, negatives)
        if not math.isclose(full_result["ndcg_at_10"], full["ndcg_at_10"], abs_tol=1e-10):
            raise AssertionError("Full ALS Top-10 NDCG does not independently recompute")
        if not math.isclose(full_result["recall_at_10"], full["recall_at_10"], abs_tol=1e-10):
            raise AssertionError("Full ALS Top-10 Recall does not independently recompute")

        itemcf = report["distributed_itemcf"]
        itemcf_result = validate_rankings(STAGE / itemcf["top10_file"], np.arange(itemcf["workset_users"]),
                                          users, positives, offsets, negatives)
        if not math.isclose(itemcf_result["ndcg_at_10"], itemcf["ndcg_at_10"], abs_tol=1e-10):
            raise AssertionError("ItemCF Top-10 NDCG does not independently recompute")
        if not math.isclose(itemcf_result["recall_at_10"], itemcf["recall_at_10"], abs_tol=1e-10):
            raise AssertionError("ItemCF Top-10 Recall does not independently recompute")

    result = {
        "status": "passed",
        "candidate_sha256": report["candidate_sha256"],
        "als_per_user_scale_checks": scale_checks,
        "als_full_top10_recomputation": full_result,
        "itemcf_workset_top10_recomputation": itemcf_result,
        "checks": ["candidate hash", "scale user populations", "per-user macro metrics",
                   "top-10 candidate membership", "unique items", "rank sequence", "sort order",
                   "independent NDCG@10", "independent Recall@10"],
    }
    output = STAGE / "logs/verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

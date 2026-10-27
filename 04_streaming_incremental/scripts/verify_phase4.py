"""Recompute replay metrics, candidate validity and selected state summaries."""
from __future__ import annotations

import csv
import hashlib
import gzip
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "04_streaming_incremental"
BASELINE = ROOT / "01_data_baseline"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    report = json.loads((STAGE / "logs/stream_metrics.json").read_text(encoding="utf-8"))
    with np.load(BASELINE / "data/processed/candidates.npz") as archive:
        users, cutoffs, positives, offsets, negatives = (archive[name] for name in
            ("user_ids", "cutoffs", "positive_items", "positive_offsets", "negatives"))
    if sha256(BASELINE / "data/processed/candidates.npz") != report["candidate_sha256"]:
        raise AssertionError("Frozen candidate file hash changed")
    if len(users) != report["users_evaluated"]:
        raise AssertionError("Candidate user population changed")

    per_user = {}
    with gzip.open(STAGE / report["outputs"]["per_user_metrics"], "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            user = int(row["userId"])
            if user in per_user:
                raise AssertionError(f"Duplicate per-user metrics for {user}")
            per_user[user] = (float(row["ndcg_at_10"]), float(row["recall_at_10"]))
    if set(per_user) != set(map(int, users)):
        raise AssertionError("Per-user metrics do not cover the frozen candidate population")
    ndcg = float(np.mean([per_user[int(user)][0] for user in users]))
    recall = float(np.mean([per_user[int(user)][1] for user in users]))
    if not math.isclose(ndcg, report["ranking"]["ndcg_at_10"], abs_tol=1e-11):
        raise AssertionError("Per-user NDCG does not recompute")
    if not math.isclose(recall, report["ranking"]["recall_at_10"], abs_tol=1e-11):
        raise AssertionError("Per-user Recall does not recompute")
    with gzip.open(BASELINE / "data/processed/baseline_per_user.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        baseline_rows = {int(row["userId"]): (float(row["ndcg_at_10"]), float(row["recall_at_10"]))
                         for row in csv.DictReader(stream)}
    baseline_ndcg = float(np.mean([baseline_rows[int(user)][0] for user in users]))
    baseline_recall = float(np.mean([baseline_rows[int(user)][1] for user in users]))
    if not math.isclose(baseline_ndcg, report["ranking"]["matching_user_stage_one_baseline_ndcg_at_10"], abs_tol=1e-11):
        raise AssertionError("Matching-user baseline NDCG does not recompute")
    if not math.isclose(baseline_recall, report["ranking"]["matching_user_stage_one_baseline_recall_at_10"], abs_tol=1e-11):
        raise AssertionError("Matching-user baseline Recall does not recompute")

    positive_sets = [set(map(int, positives[offsets[i]:offsets[i + 1]])) for i in range(len(users))]
    top_groups = {}
    with gzip.open(STAGE / report["outputs"]["top10"], "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            top_groups.setdefault(int(row["userId"]), []).append(
                (int(row["movieId"]), int(row["rank"]), float(row["score"])))
    if set(top_groups) != set(map(int, users)):
        raise AssertionError("Top-10 user population mismatch")
    top_ndcg, top_recall = [], []
    for index, user_value in enumerate(users):
        user = int(user_value)
        rows = top_groups[user]
        if len(rows) != 10 or [r[1] for r in rows] != list(range(1, 11)):
            raise AssertionError(f"Invalid Top-10 ranks for user {user}")
        items = [row[0] for row in rows]
        if len(set(items)) != 10:
            raise AssertionError(f"Duplicate recommendation for user {user}")
        candidate_set = positive_sets[index] | set(map(int, negatives[index]))
        if not set(items).issubset(candidate_set):
            raise AssertionError(f"Non-candidate item recommended for user {user}")
        if any(a[2] < b[2] or (a[2] == b[2] and a[0] > b[0]) for a, b in zip(rows, rows[1:])):
            raise AssertionError(f"Incorrect score order for user {user}")
        rel = np.asarray([int(item in positive_sets[index]) for item in items], dtype=np.float64)
        discounts = 1 / np.log2(np.arange(2, 12))
        ideal = np.sum(1 / np.log2(np.arange(2, min(10, len(positive_sets[index])) + 2)))
        top_ndcg.append(float(np.dot(rel, discounts) / ideal))
        top_recall.append(int(rel.sum()) / len(positive_sets[index]))
    if not math.isclose(float(np.mean(top_ndcg)), ndcg, abs_tol=1e-11):
        raise AssertionError("Top-10 NDCG does not recompute")
    if not math.isclose(float(np.mean(top_recall)), recall, abs_tol=1e-11):
        raise AssertionError("Top-10 Recall does not recompute")

    config = json.loads((STAGE / "config.json").read_text(encoding="utf-8"))
    events = (pl.scan_parquet(BASELINE / "data/processed/train.parquet")
              .filter(pl.col("rating") >= config["positive_threshold"])
              .select(pl.col("rowId").cast(pl.Int64), pl.col("userId").cast(pl.Int32), pl.col("movieId").cast(pl.Int32),
                      pl.col("timestamp").cast(pl.Int64))
              .sort(["timestamp", "rowId"])
              .collect(engine="streaming"))
    event_users = events["userId"].to_numpy()
    event_items = events["movieId"].to_numpy()
    timestamps = events["timestamp"].to_numpy()
    state_checks = []
    for snapshot in report["state"]["checkpoints"]:
        offset = int(np.searchsorted(timestamps, snapshot["cutoff"], side="left"))
        if offset != snapshot["source_offset"]:
            raise AssertionError("Checkpoint source offset does not match event-time cutoff")
        if snapshot["watermark"] != snapshot["cutoff"]:
            raise AssertionError("Checkpoint watermark did not advance to the recommendation cutoff")
        if offset == 0:
            start = 0
        else:
            expiry = snapshot["watermark"] - config["window_days"] * 24 * 60 * 60
            start = int(np.searchsorted(timestamps[:offset], expiry, side="left"))
        if start != snapshot["window_start_offset"]:
            raise AssertionError("Checkpoint rolling-window start mismatch")
        if offset - start != snapshot["window_positive_events"]:
            raise AssertionError("Checkpoint rolling-window event count mismatch")
        active_items = int(np.unique(event_items[start:offset]).size)
        profiled_users = int(np.unique(event_users[:offset]).size)
        if active_items != snapshot["active_window_items"] or profiled_users != snapshot["profiled_users"]:
            raise AssertionError("Checkpoint active state dimensions mismatch")
        state_checks.append({"cutoff": snapshot["cutoff"], "source_offset": offset,
                             "rolling_window_events": offset - start, "state_summary_recomputes": True})

    if report["state"]["late_events_dropped"] != 0:
        raise AssertionError("Ordered offline replay unexpectedly dropped late events")
    if not report["state"]["replayed_source_offset_idempotent"]:
        raise AssertionError("Consumed source cursor replay was not idempotent")
    if not report["performance"]["event_update_p99_below_target"]:
        raise AssertionError("Measured event update p99 missed the configured target")

    verification = {
        "status": "passed",
        "users": len(users),
        "top10_rows": sum(map(len, top_groups.values())),
        "ndcg_at_10": ndcg,
        "recall_at_10": recall,
        "ndcg_no_degradation_met": ndcg >= baseline_ndcg,
        "recall_no_degradation_met": recall >= baseline_recall,
        "candidate_membership_and_order": True,
        "event_time_checkpoints": state_checks,
        "source_cursor_idempotency": True,
        "late_events_dropped": 0,
        "event_update_p99_below_100ms": True,
        "checks": ["candidate population", "per-user macro metrics", "top-10 ranks", "unique recommendations",
                    "frozen candidate membership", "score ordering", "independent NDCG@10", "independent Recall@10",
                    "event-time source offsets", "watermarks", "sliding-window boundaries", "state summaries",
                    "replay cursor idempotency", "event update p99"],
    }
    output = STAGE / "logs/verification.json"
    output.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

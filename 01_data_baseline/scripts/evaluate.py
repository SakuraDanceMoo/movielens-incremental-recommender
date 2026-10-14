"""Evaluate temporal global Most Popular using train events and frozen test candidates."""
import argparse
import csv
import gzip
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from core import metrics, rank_items
from download import STAGE, sha256
from prepare import save_json

MODEL_VERSION = "most-popular-temporal-train-v1"


def evaluate(output_prefix="baseline"):
    started = time.perf_counter()
    processed, logs = STAGE / "data/processed", STAGE / "logs"
    cfg = json.loads((STAGE / "config.json").read_text(encoding="utf-8"))
    with np.load(processed / "candidates.npz") as z:
        user_ids, cutoffs, positives, offsets, negatives = [z[k] for k in
            ["user_ids", "cutoffs", "positive_items", "positive_offsets", "negatives"]]
    events = pl.read_parquet(processed / "train_positive_events.parquet")
    train_items, train_times = events["movieId"].to_numpy(), events["timestamp"].to_numpy()
    assert np.all(train_times[:-1] <= train_times[1:])
    counts = np.zeros(int(max(train_items.max(), negatives.max(), positives.max())) + 1, dtype=np.int64)
    order = np.argsort(cutoffs, kind="stable")
    ndcgs, recalls = np.zeros(len(user_ids)), np.zeros(len(user_ids))
    latencies = np.zeros(len(user_ids))
    recommendations = np.zeros((len(user_ids), cfg["k"]), dtype=np.int32)
    recommendation_scores = np.zeros_like(recommendations, dtype=np.int64)
    relevance_hash = hashlib.sha256()
    cursor, zero_score, advanced_seconds = 0, 0, 0.0
    for index in order:
        prep = time.perf_counter()
        end = int(np.searchsorted(train_times, cutoffs[index], side="left"))
        np.add.at(counts, train_items[cursor:end], 1)
        cursor = end
        advanced_seconds += time.perf_counter() - prep
        rank_started = time.perf_counter()
        pos = positives[offsets[index]:offsets[index + 1]]
        candidates = np.concatenate([pos, negatives[index]])
        scores = counts[candidates]
        ranked = rank_items(candidates, scores, cfg["k"])
        relevance = (ranked < len(pos)).astype(np.uint8)
        ndcgs[index], recalls[index] = metrics(relevance, len(pos), cfg["k"])
        recommendations[index] = candidates[ranked]
        recommendation_scores[index] = scores[ranked]
        latencies[index] = (time.perf_counter() - rank_started) * 1000
        zero_score += int(np.all(scores == 0))
        relevance_hash.update(relevance.tobytes())
    elapsed = time.perf_counter() - started
    generated = datetime.now(timezone(timedelta(hours=8))).isoformat()
    top_path = processed / f"{output_prefix}_top10.csv.gz"
    # Deterministic gzip header; generated_at is run-specific metadata.
    with top_path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as binary:
        import io
        with io.TextIOWrapper(binary, encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["userId", "movieId", "rank", "score", "model_version", "generated_at"])
            for i, u in enumerate(user_ids):
                for r, (movie, score) in enumerate(zip(recommendations[i], recommendation_scores[i]), 1):
                    writer.writerow([int(u), int(movie), r, int(score), MODEL_VERSION, generated])
    per_user = pl.DataFrame({"userId": user_ids, "ndcg_at_10": ndcgs, "recall_at_10": recalls,
                            "test_positives": np.diff(offsets), "ranking_ms": latencies})
    with gzip.open(processed / f"{output_prefix}_per_user.csv.gz", "wb") as f:
        f.write(per_user.write_csv().encode("utf-8"))
    # Hash only scientific outputs, excluding clock-dependent metadata/latencies.
    semantic = hashlib.sha256()
    for array in [user_ids, recommendations, recommendation_scores, ndcgs, recalls]:
        semantic.update(array.tobytes())
    result = {"model": MODEL_VERSION, "generated_at": generated, "dataset": cfg["dataset"],
              "ndcg_at_10": float(ndcgs.mean()), "recall_at_10": float(recalls.mean()),
              "evaluated_users": len(user_ids), "users_with_no_positive_recommendation": int(np.sum(ndcgs == 0)),
              "users_with_all_zero_candidate_scores": zero_score,
              "seconds_load_and_evaluate": elapsed, "users_per_second_load_and_evaluate": len(user_ids) / elapsed,
              "temporal_count_update_seconds": advanced_seconds,
              "ranking_latency_ms": {"mean": float(latencies.mean()), "p50": float(np.quantile(latencies, .5)),
                                     "p95": float(np.quantile(latencies, .95)), "p99": float(np.quantile(latencies, .99))},
              "latency_scope": "in-process candidate scoring, sorting and metric computation; excludes I/O and temporal count updates; not service latency",
              "candidate_sha256": sha256(processed / "candidates.npz"),
              "scientific_output_sha256": semantic.hexdigest(),
              "top10_file": top_path.relative_to(STAGE).as_posix(), "top10_sha256": sha256(top_path),
              "fit_policy": cfg["popularity"], "teacher_alignment": cfg["teacher_script_alignment"]}
    save_json(logs / f"{output_prefix}_metrics.json", result)
    if output_prefix == "baseline":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        axes[0].hist(ndcgs, bins=np.linspace(0, 1, 41), color="#3568a4")
        axes[0].set_xlabel("NDCG@10 per user"); axes[0].set_ylabel("Users")
        axes[1].bar(["NDCG@10", "Recall@10"], [ndcgs.mean(), recalls.mean()], color=["#3568a4", "#438763"])
        axes[1].set_ylim(0, 1); axes[1].set_ylabel("Macro mean")
        fig.savefig(STAGE / "curves/baseline_metrics.png", dpi=160); plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="baseline")
    evaluate(parser.parse_args().output_prefix)

"""Event-time incremental explicit-rating regression for hybrid data."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "04_streaming_incremental"
MIN_RATING, MAX_RATING = 0.5, 5.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(data_dir: Path):
    ratings = pl.read_csv(data_dir / "ratings.csv", schema_overrides={"userId": pl.Int32, "movieId": pl.Int32, "rating": pl.Float32, "timestamp": pl.Int64})
    ordered = ratings.sort(["userId", "timestamp", "movieId"], descending=[False, True, True])
    marked = ordered.with_columns([
        pl.len().over("userId").alias("user_rating_count"),
        pl.int_range(0, pl.len()).over("userId").alias("latest_rank"),
    ])
    holdout = marked.filter((pl.col("latest_rank") == 0) & (pl.col("user_rating_count") >= 6))
    training = marked.filter(~((pl.col("latest_rank") == 0) & (pl.col("user_rating_count") >= 6))).select("userId", "movieId", "rating", "timestamp")
    test = pl.read_csv(data_dir / "test_pairs.csv", schema_overrides={"userId": pl.Int32, "movieId": pl.Int32, "timestamp": pl.Int64}).with_row_index("row_id")
    return training, holdout.select("userId", "movieId", "rating", "timestamp"), test


def stream_predict(events: pl.DataFrame, queries: pl.DataFrame, max_user: int, max_movie: int) -> np.ndarray:
    events = events.sort("timestamp")
    queries = queries.with_row_index("query_order").sort(["timestamp", "query_order"])
    event_ts = events["timestamp"].to_numpy(); event_users = events["userId"].to_numpy(); event_movies = events["movieId"].to_numpy(); event_ratings = events["rating"].to_numpy()
    q_ts = queries["timestamp"].to_numpy(); q_users = queries["userId"].to_numpy(); q_movies = queries["movieId"].to_numpy()
    user_sum = np.zeros(max_user + 1, dtype=np.float64); user_count = np.zeros(max_user + 1, dtype=np.int32)
    movie_sum = np.zeros(max_movie + 1, dtype=np.float64); movie_count = np.zeros(max_movie + 1, dtype=np.int32)
    predictions = np.empty(len(queries), dtype=np.float64); pointer = 0; total_sum = 0.0; total_count = 0
    for i, cutoff in enumerate(q_ts):
        end = int(np.searchsorted(event_ts, cutoff, side="left"))
        while pointer < end:
            u = int(event_users[pointer]); m = int(event_movies[pointer]); r = float(event_ratings[pointer])
            user_sum[u] += r; user_count[u] += 1; movie_sum[m] += r; movie_count[m] += 1; total_sum += r; total_count += 1; pointer += 1
        global_mean = total_sum / total_count if total_count else 3.5
        u = int(q_users[i]); m = int(q_movies[i])
        user_mean = user_sum[u] / user_count[u] if user_count[u] else global_mean
        movie_mean = movie_sum[m] / movie_count[m] if movie_count[m] else global_mean
        predictions[i] = np.clip(0.5 * global_mean + 0.25 * user_mean + 0.25 * movie_mean, MIN_RATING, MAX_RATING)
    restored = np.empty(len(queries), dtype=np.float64); restored[queries["query_order"].to_numpy()] = predictions
    return restored


def rmse(pred: np.ndarray, labels: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - labels.astype(np.float64)) ** 2)))


def run(data_dir: Path, output: Path, metrics_path: Path) -> dict:
    started = time.perf_counter(); training, validation, test = prepare(data_dir)
    max_user = int(max(training["userId"].max(), validation["userId"].max(), test["userId"].max())); max_movie = int(max(training["movieId"].max(), validation["movieId"].max(), test["movieId"].max()))
    validation_predictions = stream_predict(training, validation.select("userId", "movieId", "timestamp"), max_user, max_movie)
    test_predictions = stream_predict(training, test.select("userId", "movieId", "timestamp"), max_user, max_movie)
    result = test.with_columns(pl.Series("predicted_rating", test_predictions)).select("userId", "movieId", "timestamp", "predicted_rating")
    output.parent.mkdir(parents=True, exist_ok=True); result.write_csv(output, float_scientific=False, float_precision=8)
    report = {"experiment": "hybrid-event-time-rating-regression-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": "hybrid-half-half-latest100k-v1", "training_rows": training.height, "validation": {"rows": validation.height, "rmse": rmse(validation_predictions, validation["rating"].to_numpy()), "algorithm": "event-time cumulative global/user/item mean weighted 0.5/0.25/0.25", "strictly_before_query_timestamp": True}, "test_predictions": {"rows": result.height, "output": output.name, "output_sha256": sha256(output), "test_rmse": None, "test_rmse_note": "test_pairs.csv has no rating labels"}, "runtime": {"elapsed_seconds": time.perf_counter() - started, "event_order": "timestamp ascending; equal timestamps retain source order", "source_data_copied_into_repository": False}}
    metrics_path.parent.mkdir(parents=True, exist_ok=True); metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2)); return report


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data-dir", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--metrics", type=Path, required=True); args = parser.parse_args(); run(args.data_dir.resolve(), args.output.resolve(), args.metrics.resolve())


if __name__ == "__main__":
    main()

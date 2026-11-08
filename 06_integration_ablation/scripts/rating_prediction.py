"""Train explicit Spark ALS on the hybrid ratings and predict test_pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.storagelevel import StorageLevel

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "06_integration_ablation"
SEED = 20261108
RATING_MIN = 0.5
RATING_MAX = 5.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rmse_report(predictions, mean_rating: float) -> dict:
    prediction = F.col("prediction")
    missing = prediction.isNull() | F.isnan(prediction)
    filled = F.when(missing, F.lit(mean_rating)).otherwise(prediction)
    clipped = F.least(F.lit(RATING_MAX), F.greatest(F.lit(RATING_MIN), filled))
    baseline = min(RATING_MAX, max(RATING_MIN, mean_rating))
    evaluated = (predictions.withColumn("prediction_filled", filled)
                 .withColumn("prediction_clipped", clipped)
                 .withColumn("baseline_prediction", F.lit(baseline)))

    def summarize(frame) -> dict:
        row = frame.agg(
            F.count(F.lit(1)).alias("rows"),
            F.sum(F.when(missing, F.lit(1)).otherwise(F.lit(0))).alias("fallback_rows"),
            F.sqrt(F.avg(F.pow(F.col("prediction_filled") - F.col("rating"), 2))).alias("rmse_raw"),
            F.sqrt(F.avg(F.pow(F.col("prediction_clipped") - F.col("rating"), 2))).alias("rmse_clipped"),
            F.sqrt(F.avg(F.pow(F.col("baseline_prediction") - F.col("rating"), 2))).alias("global_mean_rmse"),
        ).first()
        return {"rows": int(row["rows"]), "fallback_rows": int(row["fallback_rows"] or 0),
                "rmse_raw": float(row["rmse_raw"]), "rmse_clipped": float(row["rmse_clipped"]),
                "global_mean_rmse": float(row["global_mean_rmse"])}

    result = {"overall": summarize(evaluated)}
    for label, predicate in (("movielens32m", F.col("userId") <= 200948),
                             ("synthetic", F.col("userId") >= 200949)):
        result[label] = summarize(evaluated.where(predicate))
    return result


def prepare_test_frame(data_dir: Path):
    test = pl.read_csv(data_dir / "test_pairs.csv", infer_schema_length=1000)
    expected = {"userId", "movieId", "timestamp"}
    if set(test.columns) != expected:
        raise ValueError(f"test_pairs.csv columns must be {sorted(expected)}, got {test.columns}")
    if test.null_count().sum_horizontal().item() != 0:
        raise ValueError("test_pairs.csv contains null fields")
    test = test.with_row_index("row_id").with_columns(
        pl.col("row_id").cast(pl.Int64),
        pl.col("userId").cast(pl.Int32),
        pl.col("movieId").cast(pl.Int32),
        pl.col("timestamp").cast(pl.Int64),
    )
    return test


def factor_matrix(factors, max_id: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((max_id + 1, rank), dtype=np.float32)
    known = np.zeros(max_id + 1, dtype=np.bool_)
    for row in factors.select("id", "features").toLocalIterator():
        values[row.id] = row.features
        known[row.id] = True
    return values, known


def evaluate_trial(model, validation, global_mean: float) -> dict:
    report = rmse_report(model.transform(validation), global_mean)
    return report


def run(data_dir: Path, output_path: Path | None, driver_memory: str, partitions: int) -> dict:
    ratings_path = data_dir / "ratings.csv"
    test_path = data_dir / "test_pairs.csv"
    summary_path = data_dir / "summary.json"
    for path in (ratings_path, test_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset") != "hybrid-half-half-latest100k-v1":
        raise ValueError(f"Unexpected dataset summary: {summary.get('dataset')!r}")
    expected_rows = int(summary["training_ratings"])
    id_ranges = json.loads((data_dir / "id_ranges.json").read_text(encoding="utf-8"))
    movielens_user_max = int(id_ranges["movielens32m_users"][1])
    synthetic_user_min = int(id_ranges["synthetic_users"][0])

    jdk = ROOT.parent / ".venv" / "jdk11"
    if jdk.exists():
        os.environ.setdefault("JAVA_HOME", str(jdk))
        os.environ["PATH"] = str(jdk / "bin") + os.pathsep + os.environ.get("PATH", "")

    temp_dir = ROOT / ".temp" / "spark-phase6-rating-prediction"
    temp_dir.mkdir(parents=True, exist_ok=True)
    spark = (SparkSession.builder.appName("Hybrid-Rating-Prediction-Stage6")
             .master("local[*]")
             .config("spark.driver.memory", driver_memory)
             .config("spark.sql.shuffle.partitions", str(partitions))
             .config("spark.default.parallelism", str(partitions))
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.ui.enabled", "false")
             .config("spark.local.dir", str(temp_dir))
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    started = time.perf_counter()
    try:
        schema = "userId int, movieId int, rating float, timestamp long"
        ratings = (spark.read.schema(schema).option("header", True).csv(str(ratings_path))
                   .persist(StorageLevel.MEMORY_AND_DISK))
        actual_rows = ratings.count()
        if actual_rows != expected_rows:
            raise ValueError(f"ratings.csv has {actual_rows} rows; summary expects {expected_rows}")
        invalid = ratings.where(
            F.col("userId").isNull() | F.col("movieId").isNull() | F.col("rating").isNull() |
            F.col("timestamp").isNull() | (F.col("rating") < RATING_MIN) | (F.col("rating") > RATING_MAX)
        ).limit(1).count()
        if invalid:
            raise ValueError("ratings.csv contains nulls or ratings outside [0.5, 5.0]")

        test_original = prepare_test_frame(data_dir)
        test_count = test_original.height
        if test_count != int(summary["hidden_test_ratings"]):
            raise ValueError(f"test_pairs.csv has {test_count} rows; summary expects {summary['hidden_test_ratings']}")
        if "rating" in test_original.columns:
            raise ValueError("test_pairs.csv unexpectedly contains a rating label")

        per_user = Window.partitionBy("userId")
        latest_first = Window.partitionBy("userId").orderBy(F.desc("timestamp"), F.desc("movieId"))
        ranked = (ratings
                  .withColumn("user_rating_count", F.count(F.lit(1)).over(per_user))
                  .withColumn("latest_rank", F.row_number().over(latest_first))
                  .persist(StorageLevel.MEMORY_AND_DISK))
        ranked.count()
        validation = (ranked.where((F.col("latest_rank") == 1) & (F.col("user_rating_count") >= 6))
                      .select("userId", "movieId", "rating", "timestamp"))
        training = ranked.where(~((F.col("latest_rank") == 1) & (F.col("user_rating_count") >= 6)))
        training = training.select("userId", "movieId", "rating")
        validation_count = validation.count()
        train_count = training.count()
        if validation_count == 0:
            raise ValueError("Temporal validation split is empty")
        global_mean = float(training.agg(F.avg("rating")).first()[0])

        trials = []
        for rank in (16, 32):
            fit_start = time.perf_counter()
            model = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
                        rank=rank, maxIter=8, regParam=0.1,
                        implicitPrefs=False, nonnegative=False, coldStartStrategy="nan",
                        seed=SEED, numUserBlocks=10, numItemBlocks=10,
                        intermediateStorageLevel="MEMORY_AND_DISK",
                        finalStorageLevel="MEMORY_AND_DISK").fit(training)
            scores = evaluate_trial(model, validation, global_mean)
            trial = {"rank": rank, "iterations": 8, "reg_param": 0.1,
                     "fit_seconds": time.perf_counter() - fit_start,
                     "validation": scores}
            trials.append(trial)
            model.userFactors.unpersist()
            model.itemFactors.unpersist()
            print(json.dumps({"validation_trial": trial}, ensure_ascii=False), flush=True)
        best = min(trials, key=lambda result: result["validation"]["overall"]["rmse_clipped"])

        final_fit_start = time.perf_counter()
        final_model = ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
                          rank=best["rank"], maxIter=8,
                          regParam=0.1, implicitPrefs=False, nonnegative=False,
                          coldStartStrategy="nan", seed=SEED, numUserBlocks=10,
                          numItemBlocks=10, intermediateStorageLevel="MEMORY_AND_DISK",
                          finalStorageLevel="MEMORY_AND_DISK").fit(
                              ranked.select("userId", "movieId", "rating"))
        final_fit_seconds = time.perf_counter() - final_fit_start

        user_max = int(id_ranges["synthetic_users"][1])
        item_max = int(id_ranges["synthetic_movies"][1])
        rank = int(best["rank"])
        user_values, known_users = factor_matrix(final_model.userFactors, user_max, rank)
        item_values, known_items = factor_matrix(final_model.itemFactors, item_max, rank)
        user_ids = test_original["userId"].to_numpy().astype(np.int64, copy=False)
        item_ids = test_original["movieId"].to_numpy().astype(np.int64, copy=False)
        in_bounds = ((user_ids <= user_max) & (item_ids <= item_max) &
                     (user_ids >= 0) & (item_ids >= 0))
        known = np.zeros(test_count, dtype=np.bool_)
        valid_rows = np.flatnonzero(in_bounds)
        known[valid_rows] = known_users[user_ids[valid_rows]] & known_items[item_ids[valid_rows]]
        prediction_values = np.full(test_count, global_mean, dtype=np.float64)
        if np.any(known):
            prediction_values[known] = np.einsum(
                "ij,ij->i", user_values[user_ids[known]], item_values[item_ids[known]], optimize=True)
        fallback_count = int((~known).sum())
        prediction_values = np.clip(prediction_values, RATING_MIN, RATING_MAX)
        result = test_original.with_columns(
            pl.Series("predicted_rating", prediction_values, dtype=pl.Float64)
        ).select("userId", "movieId", "timestamp", "predicted_rating")
        output_path = output_path or data_dir / "test_pairs_predictions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_csv(output_path, float_scientific=False, float_precision=8)

        metrics = {
            "experiment": "hybrid-explicit-als-rating-regression-v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": summary["dataset"],
            "training_data": {
                "path_name": ratings_path.name,
                "rows": actual_rows,
                "sha256": sha256(ratings_path),
                "movielens32m_user_range": id_ranges["movielens32m_users"],
                "synthetic_user_range": id_ranges["synthetic_users"],
            },
            "validation": {
                "method": "latest timestamped rating per user with at least six total user ratings; ties broken by movieId descending",
                "rows": validation_count,
                "training_rows_after_holdout": train_count,
                "training_global_mean": global_mean,
                "global_mean_baseline_rmse": best["validation"]["overall"]["global_mean_rmse"],
                "model_trials": trials,
                "selected_rank": best["rank"],
                "selection_metric": "overall validation RMSE after clipping to [0.5, 5.0]",
                "label_scope": "held-out ratings from ratings.csv; not test_pairs ground truth",
            },
            "final_fit": {"training_rows": actual_rows, "rank": best["rank"],
                          "iterations": 8, "reg_param": 0.1, "fit_seconds": final_fit_seconds},
            "test_predictions": {
                "input_rows": test_count,
                "prediction_rows": result.height,
                "output": output_path.name,
                "output_location": "beside the supplied test_pairs.csv",
                "output_sha256": sha256(output_path),
                "global_mean_fallback_rows": fallback_count,
                "test_rmse": None,
                "test_rmse_note": "test_pairs.csv has no rating column; hidden test labels are unavailable in supplied files",
                "source_counts": {
                    "movielens32m": int(test_original.filter(pl.col("userId") <= movielens_user_max).height),
                    "synthetic": int(test_original.filter(pl.col("userId") >= synthetic_user_min).height),
                },
            },
            "runtime": {"python": platform.python_version(), "spark": spark.version,
                        "master": "local[*]", "driver_memory": driver_memory,
                        "shuffle_partitions": partitions,
                        "elapsed_seconds": time.perf_counter() - started,
                        "source_data_copied_into_repository": False},
        }
        log_dir = STAGE / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "rating_prediction_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        final_model.userFactors.unpersist()
        final_model.itemFactors.unpersist()
        ranked.unpersist()
        ratings.unpersist()
        return metrics
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None,
                        help="Prediction CSV path (defaults to test_pairs_predictions.csv beside input data)")
    parser.add_argument("--driver-memory", default="12g")
    parser.add_argument("--shuffle-partitions", type=int, default=80)
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.output.resolve() if args.output else None,
        args.driver_memory, args.shuffle_partitions)


if __name__ == "__main__":
    main()

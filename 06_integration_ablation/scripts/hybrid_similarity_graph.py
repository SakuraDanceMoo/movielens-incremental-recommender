"""Explicit-rating similarity and graph propagation models for the hybrid data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.storagelevel import StorageLevel

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "06_integration_ablation"
MIN_RATING, MAX_RATING = 0.5, 5.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(frame, prediction_col: str, global_mean: float) -> dict:
    prediction = F.col(prediction_col)
    filled = F.coalesce(prediction, F.lit(global_mean))
    clipped = F.least(F.lit(MAX_RATING), F.greatest(F.lit(MIN_RATING), filled))
    summary = frame.select(
        F.sqrt(F.avg(F.pow(clipped - F.col("rating"), 2))).alias("rmse"),
        F.sum(F.when(prediction.isNull(), 1).otherwise(0)).alias("fallback_rows"),
        F.count(F.lit(1)).alias("rows"),
    ).first()
    result = {"rows": int(summary["rows"]), "rmse": float(summary["rmse"]),
              "fallback_rows": int(summary["fallback_rows"] or 0)}
    for label, predicate in (("movielens32m", F.col("userId") <= 200948),
                             ("synthetic", F.col("userId") >= 200949)):
        row = frame.where(predicate).select(
            F.sqrt(F.avg(F.pow(clipped - F.col("rating"), 2))).alias("rmse"),
            F.count(F.lit(1)).alias("rows"),
        ).first()
        result[label] = {"rows": int(row["rows"]), "rmse": float(row["rmse"])}
    return result


def load_test(data_dir: Path) -> pl.DataFrame:
    test = pl.read_csv(data_dir / "test_pairs.csv", infer_schema_length=1000)
    if test.columns != ["userId", "movieId", "timestamp"] or test.height != 100000:
        raise ValueError("Unexpected test_pairs schema or row count")
    return test.with_row_index("row_id").with_columns(
        pl.col("row_id").cast(pl.Int64), pl.col("userId").cast(pl.Int32),
        pl.col("movieId").cast(pl.Int32), pl.col("timestamp").cast(pl.Int64),
    )


def write_test_predictions(test: pl.DataFrame, rows: list[tuple], output: Path) -> str:
    values = {int(row[0]): max(MIN_RATING, min(MAX_RATING, float(row[1]))) for row in rows}
    try:
        predictions = [values[int(row_id)] for row_id in test["row_id"]]
    except KeyError as exc:
        raise ValueError("Spark prediction rows do not cover the input test row ids") from exc
    result = test.with_columns(
        pl.Series("predicted_rating", predictions, dtype=pl.Float64)
    ).select("userId", "movieId", "timestamp", "predicted_rating")
    result.write_csv(output, float_scientific=False, float_precision=8)
    return sha256(output)


def run(mode: str, data_dir: Path, output: Path, metrics_path: Path, driver_memory: str, partitions: int) -> dict:
    ratings_path = data_dir / "ratings.csv"
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    test = load_test(data_dir)
    jdk = ROOT.parent / ".venv" / "jdk11"
    if jdk.exists():
        os.environ.setdefault("JAVA_HOME", str(jdk))
        os.environ["PATH"] = str(jdk / "bin") + os.pathsep + os.environ.get("PATH", "")
    spark = (SparkSession.builder.appName(f"Hybrid-{mode}-rating-regression")
             .master("local[*]").config("spark.driver.memory", driver_memory)
             .config("spark.sql.shuffle.partitions", str(partitions))
             .config("spark.default.parallelism", str(partitions))
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    started = time.perf_counter()
    try:
        schema = "userId int, movieId int, rating float, timestamp long"
        ratings = spark.read.schema(schema).option("header", True).csv(str(ratings_path)).persist(StorageLevel.MEMORY_AND_DISK)
        if ratings.count() != int(summary["training_ratings"]):
            raise ValueError("ratings.csv row count does not match summary.json")
        by_user = Window.partitionBy("userId")
        latest = Window.partitionBy("userId").orderBy(F.desc("timestamp"), F.desc("movieId"))
        ranked = (ratings.withColumn("user_rating_count", F.count(F.lit(1)).over(by_user))
                  .withColumn("latest_rank", F.row_number().over(latest)).persist(StorageLevel.MEMORY_AND_DISK))
        ranked.count()
        validation = ranked.where((F.col("latest_rank") == 1) & (F.col("user_rating_count") >= 6)).select("userId", "movieId", "rating", "timestamp")
        training = ranked.where(~((F.col("latest_rank") == 1) & (F.col("user_rating_count") >= 6))).select("userId", "movieId", "rating")
        global_mean = float(training.agg(F.avg("rating")).first()[0])

        movies_path = data_dir / "movies.csv"
        movies = (spark.read.option("header", True).option("multiLine", True)
                  .schema("movieId int, title string, genres string").csv(str(movies_path))
                  .select("movieId", F.explode(F.split("genres", "\\|")).alias("genre"))
                  .where(F.col("genre") != "(no genres listed)").dropDuplicates(["movieId", "genre"]))
        item_mean = training.groupBy("movieId").agg(F.avg("rating").alias("item_mean"))
        if mode == "similarity":
            user_genre = (training.join(movies, "movieId")
                          .groupBy("userId", "genre")
                          .agg(F.avg("rating").alias("genre_mean")))
            model_name = "genre_item_similarity_rating_v1"
        elif mode == "graph":
            item_degree = training.groupBy("movieId").agg(F.count(F.lit(1)).alias("item_degree"))
            weighted = (training.join(item_degree, "movieId")
                        .withColumn("degree_weight", F.lit(1.0) / F.sqrt(F.col("item_degree"))))
            user_genre = (weighted.join(movies, "movieId")
                          .groupBy("userId", "genre")
                          .agg((F.sum(F.col("rating") * F.col("degree_weight")) /
                                F.sum("degree_weight")).alias("genre_mean")))
            model_name = "degree_normalized_graph_propagation_rating_v1"
        else:
            raise ValueError(mode)

        def score_queries(queries):
            query = queries.withColumn("query_id", F.monotonically_increasing_id())
            target = query.join(movies, "movieId", "left")
            joined = target.join(user_genre, ["userId", "genre"], "left")
            agg = (joined.groupBy("query_id")
                   .agg(F.avg("genre_mean").alias("genre_prediction")))
            scored = (query.join(agg.select("query_id", "genre_prediction"), "query_id")
                      .join(item_mean, "movieId", "left")
                      .withColumn("prediction", F.coalesce("genre_prediction", "item_mean", F.lit(global_mean))))
            return scored

        validation_scored = score_queries(validation)
        validation_result = metric(validation_scored, "prediction", global_mean)
        # Native CSV avoids the Windows Python-worker callback problem; row_id is derived from input order.
        test_spark_ordered = (spark.read.schema("userId int, movieId int, timestamp long").option("header", True)
                             .csv(str(data_dir / "test_pairs.csv")).coalesce(1)
                             .withColumn("row_id", F.monotonically_increasing_id()))
        test_scored = score_queries(test_spark_ordered)
        prediction_rows = [(int(row["row_id"]), float(row["prediction"])) for row in
                           test_scored.select("row_id", "prediction").orderBy("row_id").toLocalIterator()]
        output.parent.mkdir(parents=True, exist_ok=True)
        output_sha = write_test_predictions(test, prediction_rows, output)
        report = {
            "experiment": model_name, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": summary["dataset"], "training_rows": int(training.count()),
            "validation": validation_result, "global_mean": global_mean,
            "test_predictions": {"rows": len(prediction_rows), "output": output.name,
                                  "output_sha256": output_sha, "test_rmse": None,
                                  "test_rmse_note": "test_pairs.csv has no rating labels"},
            "model": {"algorithm": "genre similarity" if mode == "similarity" else "degree-normalized two-hop user-movie graph propagation",
                      "fallback": "item mean then global mean", "data_scope": "all hybrid ratings except per-user temporal holdout"},
            "runtime": {"python": platform.python_version(), "spark": spark.version, "master": "local[*]",
                        "elapsed_seconds": time.perf_counter() - started, "partitions": partitions},
        }
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return report
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["similarity", "graph"], required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--driver-memory", default="12g")
    parser.add_argument("--shuffle-partitions", type=int, default=80)
    args = parser.parse_args()
    run(args.mode, args.data_dir.resolve(), args.output.resolve(), args.metrics.resolve(), args.driver_memory, args.shuffle_partitions)


if __name__ == "__main__":
    main()

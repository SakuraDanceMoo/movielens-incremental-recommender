"""Reproduce rank-16 ALS on the fixed holdout, export local factors for residuals."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "06_integration_ablation"
CACHE = STAGE / "data" / "processed" / "transformer"


def run(data_dir: Path, full: bool) -> None:
    jdk = ROOT.parent / ".venv" / "jdk11"
    os.environ.setdefault("JAVA_HOME", str(jdk))
    os.environ["PATH"] = str(jdk / "bin") + os.pathsep + os.environ["PATH"]
    from pyspark.ml.recommendation import ALS
    from pyspark.sql import SparkSession, Window, functions as F
    from pyspark.storagelevel import StorageLevel
    from rating_prediction import factor_matrix
    spark = (SparkSession.builder.appName("Hybrid-Transformer-ALS-Residual").master("local[*]")
             .config("spark.driver.memory", "10g").config("spark.sql.shuffle.partitions", "80")
             .config("spark.default.parallelism", "80").config("spark.ui.enabled", "false")
             .config("spark.local.dir", str(ROOT / ".temp" / "transformer-spark")).getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    start = time.perf_counter()
    try:
        data = spark.read.schema("userId int, movieId int, rating float, timestamp long").option("header", True).csv(str(data_dir / "ratings.csv"))
        if not full:
            ranked = (data.withColumn("count", F.count(F.lit(1)).over(Window.partitionBy("userId")))
                      .withColumn("rank", F.row_number().over(Window.partitionBy("userId").orderBy(F.desc("timestamp"), F.desc("movieId")))))
            data = ranked.where(~((F.col("rank") == 1) & (F.col("count") >= 6)))
        data = data.select("userId", "movieId", "rating").persist(StorageLevel.MEMORY_AND_DISK)
        rows = data.count()
        mean = float(data.agg(F.avg("rating")).first()[0])
        model = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", rank=16,
                    maxIter=8, regParam=0.1, implicitPrefs=False, nonnegative=False,
                    coldStartStrategy="nan", seed=20261108, numUserBlocks=10, numItemBlocks=10,
                    intermediateStorageLevel="MEMORY_AND_DISK", finalStorageLevel="MEMORY_AND_DISK").fit(data)
        u, ku = factor_matrix(model.userFactors, 401896, 16)
        i, ki = factor_matrix(model.itemFactors, 175170, 16)
        tag = "full" if full else "fit"
        np.savez(CACHE / f"als_{tag}.npz", user=u, item=i, known_user=ku, known_item=ki, mean=mean)
        report = {"split": tag, "rows": rows, "rank": 16, "iterations": 8,
                  "reg_param": 0.1, "seed": 20261108, "elapsed_seconds": time.perf_counter() - start}
        if not full:
            vu, vi, y = [np.load(CACHE / f"valid_{k}.npy") for k in ["u", "i", "y"]]
            pred = np.einsum("ij,ij->i", u[vu], i[vi])
            missing = ~(ku[vu] & ki[vi])
            pred[missing] = mean
            np.save(CACHE / "als_valid_predictions.npy", pred)
            report.update(validation_rmse=float(np.sqrt(np.mean((np.clip(pred, .5, 5).astype(float)-y)**2))), fallback_rows=int(missing.sum()))
        (STAGE / "logs" / "transformer" / f"als_{tag}_reproduction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    run(args.data_dir, args.full)

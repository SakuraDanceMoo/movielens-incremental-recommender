"""Run Spark ALS scale experiments and distributed ItemCF on MovieLens-32M."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psutil
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.storagelevel import StorageLevel

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "03_distributed_recommendation"
DATA = ROOT / "01_data_baseline/data/processed"
BASELINE = ROOT / "01_data_baseline/logs/baseline_metrics.json"
SEED = 20260907
K = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank(candidates: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.lexsort((candidates, -scores))[:K]


def metric(relevant: np.ndarray, total: int) -> tuple[float, float]:
    discounts = 1.0 / np.log2(np.arange(2, len(relevant) + 2))
    ideal = np.sum(1.0 / np.log2(np.arange(2, min(K, total) + 2)))
    hits = int(np.count_nonzero(relevant))
    return float(np.dot(relevant, discounts) / ideal), hits / int(total)


def load_protocol():
    arrays = np.load(DATA / "candidates.npz")
    values = {name: arrays[name] for name in
              ("user_ids", "positive_items", "positive_offsets", "negatives")}
    arrays.close()
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_path = DATA / "baseline_per_user.csv.gz"
    baseline_scores = {}
    with gzip.open(baseline_path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            baseline_scores[int(row["userId"])] = float(row["ndcg_at_10"])
    return values, baseline_scores, baseline


def evaluate_factors(model, protocol, baseline_scores, scale, write_top10):
    users, positives, offsets, negatives = (protocol[name] for name in
        ("user_ids", "positive_items", "positive_offsets", "negatives"))
    rank_size = len(model.userFactors.first().features)
    user_factors = np.zeros((int(users.max()) + 1, rank_size), dtype=np.float32)
    item_max = max(int(positives.max()), int(negatives.max()))
    item_factors = np.zeros((item_max + 1, rank_size), dtype=np.float32)
    for row in model.userFactors.select("id", "features").toLocalIterator():
        user_factors[row.id] = row.features
    for row in model.itemFactors.select("id", "features").toLocalIterator():
        if row.id <= item_max:
            item_factors[row.id] = row.features

    selected = np.flatnonzero(users % 100 < int(round(scale * 100)))
    ndcgs, recalls, top_rows = [], [], []
    per_user_path = STAGE / "data/processed" / f"als_{int(scale * 100):03d}_per_user.csv.gz"
    semantic = hashlib.sha256()
    started = time.perf_counter()
    with gzip.open(per_user_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["userId", "ndcg_at_10", "recall_at_10", "test_positives"])
        for index in selected:
            user = int(users[index])
            pos = positives[offsets[index]:offsets[index + 1]]
            candidates = np.concatenate((pos, negatives[index])).astype(np.int32, copy=False)
            scores = item_factors[candidates] @ user_factors[user]
            order = rank(candidates, scores)
            relevance = (order < len(pos)).astype(np.uint8)
            ndcg, recall = metric(relevance, len(pos))
            ndcgs.append(ndcg)
            recalls.append(recall)
            writer.writerow([user, f"{ndcg:.12g}", f"{recall:.12g}", len(pos)])
            semantic.update(np.asarray([user], dtype=np.int32).tobytes())
            semantic.update(candidates[order].tobytes())
            semantic.update(scores[order].tobytes())
            if write_top10:
                top_rows.extend((user, int(candidates[j]), r + 1, float(scores[j]))
                                for r, j in enumerate(order))
    score_seconds = time.perf_counter() - started
    if write_top10:
        output = STAGE / "data/processed/als_top10.csv.gz"
        with gzip.open(output, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["userId", "movieId", "rank", "score", "model_version"])
            writer.writerows((*row, "spark-implicit-als-v1") for row in top_rows)
    return {
        "scale": scale,
        "evaluated_users": len(selected),
        "ndcg_at_10": float(np.mean(ndcgs)),
        "recall_at_10": float(np.mean(recalls)),
        "matching_user_baseline_ndcg_at_10": float(np.mean([baseline_scores[int(users[i])] for i in selected])),
        "ndcg_relative_change_vs_matching_baseline": float(np.mean(ndcgs) /
            np.mean([baseline_scores[int(users[i])] for i in selected]) - 1),
        "candidate_scoring_seconds": score_seconds,
        "candidate_scoring_users_per_second": len(selected) / score_seconds,
        "per_user_file": per_user_path.relative_to(STAGE).as_posix(),
        "scientific_output_sha256": semantic.hexdigest(),
        "top10_file": "data/processed/als_top10.csv.gz" if write_top10 else None,
    }


def run_itemcf(spark, positive_rows, protocol, baseline_scores, cfg):
    users = protocol["user_ids"][:cfg["itemcf_users"]]
    scoped = (positive_rows.where(F.col("userId").isin([int(user) for user in users]))
              .select("userId", "movieId").dropDuplicates(["userId", "movieId"])
              .persist(StorageLevel.MEMORY_AND_DISK))
    interactions = scoped.count()
    t0 = time.perf_counter()
    left = scoped.select("userId", F.col("movieId").alias("src"))
    right = scoped.select("userId", F.col("movieId").alias("dst"))
    cooccurrence = (left.join(right, "userId").where(F.col("src") != F.col("dst"))
                    .groupBy("src", "dst").count())
    item_counts = scoped.groupBy(F.col("movieId").alias("src")).count()
    src_counts = item_counts.select("src", F.col("count").alias("src_count"))
    dst_counts = item_counts.select(F.col("src").alias("dst"), F.col("count").alias("dst_count"))
    scores = (cooccurrence.join(src_counts, "src").join(dst_counts, "dst")
              .select("src", "dst", (F.col("count") / F.sqrt(F.col("src_count") * F.col("dst_count"))).alias("score")))
    order = Window.partitionBy("src").orderBy(F.desc("score"), F.asc("dst"))
    neighbors = scores.withColumn("rn", F.row_number().over(order)).where(F.col("rn") <= cfg["itemcf_neighbors"])
    neighbor_rows = neighbors.select("src", "dst", "score").collect()
    neighbor_seconds = time.perf_counter() - t0
    neighbor_map: dict[int, list[tuple[int, float]]] = {}
    for row in neighbor_rows:
        neighbor_map.setdefault(row.src, []).append((row.dst, float(row.score)))
    histories = {}
    for row in scoped.groupBy("userId").agg(F.collect_list("movieId").alias("items")).collect():
        histories[int(row.userId)] = row.items

    positives, offsets, negatives = (protocol[name] for name in
        ("positive_items", "positive_offsets", "negatives"))
    ndcgs, recalls, output_rows = [], [], []
    for index, user in enumerate(users):
        scores_by_item: dict[int, float] = {}
        for history_item in histories.get(int(user), ()):
            for target, similarity in neighbor_map.get(int(history_item), ()):
                scores_by_item[target] = scores_by_item.get(target, 0.0) + similarity
        pos = positives[offsets[index]:offsets[index + 1]]
        candidates = np.concatenate((pos, negatives[index])).astype(np.int32, copy=False)
        candidate_scores = np.fromiter((scores_by_item.get(int(item), 0.0) for item in candidates),
                                       dtype=np.float64, count=len(candidates))
        ranked = rank(candidates, candidate_scores)
        relevant = (ranked < len(pos)).astype(np.uint8)
        ndcg, recall = metric(relevant, len(pos))
        ndcgs.append(ndcg)
        recalls.append(recall)
        output_rows.extend((int(user), int(candidates[j]), r + 1, float(candidate_scores[j]))
                           for r, j in enumerate(ranked))
    output = STAGE / "data/processed/itemcf_spark_top10.csv.gz"
    with gzip.open(output, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["userId", "movieId", "rank", "score", "model_version"])
        writer.writerows((*row, "spark-itemcf-cosine-v1") for row in output_rows)
    scoped.unpersist()
    return {
        "algorithm": "distributed item-item cosine co-occurrence",
        "workset_users": len(users),
        "positive_train_interactions": interactions,
        "item_pairs_scored": int(len(neighbor_rows)),
        "neighbors_per_item": cfg["itemcf_neighbors"],
        "neighbor_build_seconds": neighbor_seconds,
        "ndcg_at_10": float(np.mean(ndcgs)),
        "recall_at_10": float(np.mean(recalls)),
        "matching_user_baseline_ndcg_at_10": float(np.mean([baseline_scores[int(u)] for u in users])),
        "top10_file": output.relative_to(STAGE).as_posix(),
    }


def run(args):
    if not (DATA / "train.parquet").is_file() or not (DATA / "candidates.npz").is_file():
        raise FileNotFoundError("Stage-one train.parquet and candidates.npz are required; run stage one first.")
    protocol, baseline_scores, baseline_metrics = load_protocol()
    scales = [float(v) for v in args.scales.split(",")]
    if not scales or scales[-1] != 1.0 or any(not 0 < x <= 1 for x in scales):
        raise ValueError("Scales must be comma-separated fractions in (0,1] and end with 1.0.")

    os.environ["PYSPARK_PYTHON"] = sys.executable
    spark = (SparkSession.builder.appName("MovieLens-32M-Stage3")
             .master(args.master)
             .config("spark.driver.memory", args.driver_memory)
             .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
             .config("spark.default.parallelism", str(args.shuffle_partitions))
             .config("spark.sql.adaptive.enabled", "true")
             .config("spark.ui.enabled", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    STAGE.joinpath("logs").mkdir(parents=True, exist_ok=True)
    STAGE.joinpath("data/processed").mkdir(parents=True, exist_ok=True)
    STAGE.joinpath("curves").mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    ratings = (spark.read.parquet(str(DATA / "train.parquet"))
               .select("userId", "movieId", "rating")
               .persist(StorageLevel.MEMORY_AND_DISK))
    total_rows = ratings.count()
    positive = ratings.where(F.col("rating") >= 4.0).select("userId", "movieId")
    baseline = baseline_metrics["ndcg_at_10"]
    java_version = spark.sparkContext._jvm.java.lang.System.getProperty("java.version")
    results = []
    for scale in scales:
        threshold = int(round(scale * 100))
        sample = (positive.where(F.pmod(F.col("userId"), F.lit(100)) < threshold)
                  .select(F.col("userId").cast("int").alias("user"),
                          F.col("movieId").cast("int").alias("item"),
                          F.lit(1.0).alias("rating"))
                  .persist(StorageLevel.MEMORY_AND_DISK))
        rows = sample.count()
        fit_start = time.perf_counter()
        als = ALS(userCol="user", itemCol="item", ratingCol="rating", rank=args.rank,
                  maxIter=args.iterations, regParam=args.reg_param, alpha=args.alpha,
                  implicitPrefs=True, nonnegative=False, coldStartStrategy="nan",
                  seed=SEED, numUserBlocks=args.blocks, numItemBlocks=args.blocks,
                  intermediateStorageLevel="MEMORY_AND_DISK",
                  finalStorageLevel="MEMORY_AND_DISK")
        model = als.fit(sample)
        fit_seconds = time.perf_counter() - fit_start
        evaluated = evaluate_factors(model, protocol, baseline_scores, scale, scale == 1.0)
        model.userFactors.unpersist()
        model.itemFactors.unpersist()
        evaluated.update({
            "training_rows": rows,
            "training_rows_per_second": rows / fit_seconds,
            "als_fit_seconds": fit_seconds,
            "rank": args.rank,
            "iterations": args.iterations,
            "regularization": args.reg_param,
            "alpha": args.alpha,
            "matching_user_baseline_ndcg_at_10": evaluated["matching_user_baseline_ndcg_at_10"],
            "overall_stage_one_baseline_ndcg_at_10": baseline,
        })
        results.append(evaluated)
        sample.unpersist()
        print(json.dumps({"scale_result": evaluated}, ensure_ascii=False), flush=True)

    itemcf_positive = (ratings.where(F.col("rating") >= 4.0)
                       .select(F.col("userId").cast("int"), F.col("movieId").cast("int")))
    itemcf = run_itemcf(spark, itemcf_positive, protocol, baseline_scores, {
        "itemcf_users": args.itemcf_users, "itemcf_neighbors": args.itemcf_neighbors,
    })
    ratings.unpersist()
    spark.stop()

    report = {
        "experiment": "phase3-spark-als-itemcf-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MovieLens 32M",
        "training_input": "01_data_baseline/data/processed/train.parquet; rows with rating >= 4.0 become implicit positive interactions",
        "training_rows_in_parquet": total_rows,
        "candidate_protocol": "frozen stage-one positives plus 100 fixed negatives; seed 20260907; same user/item ID order; macro user metrics",
        "candidate_sha256": sha256(DATA / "candidates.npz"),
        "baseline_ndcg_at_10": baseline,
        "runtime": {
            "spark_version": spark.version,
            "java_version": java_version,
            "master": args.master,
            "deployment_scope": "single Windows host, local Spark mode; no remote worker nodes",
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "available_memory_gib_at_start": round(psutil.virtual_memory().available / 1024**3, 2),
            "driver_memory": args.driver_memory,
            "shuffle_partitions": args.shuffle_partitions,
            "elapsed_seconds_including_spark_startup": time.perf_counter() - started,
        },
        "als_scale_results": results,
        "distributed_itemcf": itemcf,
        "acceptance": {
            "als_ndcg_above_matching_user_baseline": all(
                r["ndcg_at_10"] > r["matching_user_baseline_ndcg_at_10"] for r in results),
            "full_population_ndcg_above_baseline": results[-1]["ndcg_at_10"] > baseline,
            "multi_node_cluster_verified": False,
        },
    }
    report_path = STAGE / "logs/spark_metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_curve(results)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def make_curve(results):
    x = [r["training_rows"] for r in results]
    throughput = [r["training_rows_per_second"] for r in results]
    ndcg = [r["ndcg_at_10"] for r in results]
    fig, left = plt.subplots(figsize=(8, 5), layout="constrained")
    line1 = left.plot(x, throughput, marker="o", color="#3568a4", label="ALS training throughput")
    left.set_xlabel("Positive training interactions")
    left.set_ylabel("Training interactions per second", color="#3568a4")
    left.tick_params(axis="y", labelcolor="#3568a4")
    right = left.twinx()
    line2 = right.plot(x, ndcg, marker="s", color="#438763", label="NDCG@10")
    right.axhline(json.loads(BASELINE.read_text(encoding="utf-8"))["ndcg_at_10"],
                  color="#bd5a48", linestyle="--", linewidth=1, label="Full baseline")
    right.set_ylabel("NDCG@10", color="#438763")
    right.tick_params(axis="y", labelcolor="#438763")
    lines = line1 + line2 + [right.lines[-1]]
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    fig.savefig(STAGE / "curves/als_scalability.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="local[*]")
    parser.add_argument("--driver-memory", default="8g")
    parser.add_argument("--shuffle-partitions", type=int, default=max(32, min(256, (os.cpu_count() or 4) * 2)))
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--reg-param", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--scales", default="0.25,0.50,1.00")
    parser.add_argument("--itemcf-users", type=int, default=5000)
    parser.add_argument("--itemcf-neighbors", type=int, default=50)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

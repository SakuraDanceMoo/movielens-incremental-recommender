"""Full-data integrity checks, chronological splits, fixed candidates and train-only fit."""
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import psutil

from core import sample_negatives
from download import STAGE, sha256


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def checks(frame, keys=None):
    return {"rows": frame.height, "schema": {k: str(v) for k, v in frame.schema.items()},
            "nulls": frame.null_count().row(0, named=True),
            "duplicate_rows": frame.height - frame.unique().height,
            "duplicate_primary_keys": None if keys is None else frame.height - frame.select(keys).unique().height}


def summary(values):
    return {"min": int(values.min()), "max": int(values.max()), "mean": float(values.mean()),
            "quantiles": dict(zip(["p25", "p50", "p75", "p90", "p99"],
                                  np.quantile(values, [.25, .5, .75, .9, .99]).tolist()))}


def fit(train, processed, threshold):
    # This function receives TRAIN only. No validation/test labels are read here.
    positive = train.filter(pl.col("rating") >= threshold)
    positive.select("movieId", "timestamp").sort(["timestamp", "movieId"]).write_parquet(
        processed / "train_positive_events.parquet", compression="zstd")
    positive.group_by("movieId").len(name="positive_count").sort(
        ["positive_count", "movieId"], descending=[True, False]).write_csv(processed / "popularity.csv")
    return positive.height


def main():
    started = time.perf_counter()
    cfg = json.loads((STAGE / "config.json").read_text(encoding="utf-8"))
    raw, processed, logs, curves = [STAGE / p for p in ["data/raw", "data/processed", "logs", "curves"]]
    for p in [processed, logs, curves]:
        p.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((STAGE / "data/raw_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if sha256(STAGE / item["path"]) != item["sha256"]:
            raise ValueError(f"Raw SHA-256 mismatch: {item['path']}")
    print("Raw hashes verified; reading all four tables", flush=True)
    numeric = {"userId": pl.Int32, "movieId": pl.Int32, "rating": pl.Float32,
               "timestamp": pl.Int64, "imdbId": pl.Int32, "tmdbId": pl.Int32}
    frames = {}
    for name in ["ratings", "movies", "tags", "links"]:
        header = (raw / f"{name}.csv").open(encoding="utf-8").readline().strip().split(",")
        frames[name] = pl.read_csv(raw / f"{name}.csv", schema_overrides={k: v for k, v in numeric.items() if k in header})
    ratings, movies, tags, links = [frames[n] for n in ["ratings", "movies", "tags", "links"]]
    quality = {n: checks(f, ["movieId"] if n in ["movies", "links"] else None) for n, f in frames.items()}
    catalog = movies["movieId"].sort().to_numpy()
    users = ratings["userId"].unique().sort()
    quality["ratings"]["duplicate_user_movie_pairs"] = ratings.height - ratings.select("userId", "movieId").unique().height
    quality["foreign_keys"] = {
        "ratings_movie_unknown": ratings.filter(~pl.col("movieId").is_in(movies["movieId"].implode())).height,
        "tags_movie_unknown": tags.filter(~pl.col("movieId").is_in(movies["movieId"].implode())).height,
        "tags_user_not_in_ratings": tags.filter(~pl.col("userId").is_in(users.implode())).height,
        "links_movie_unknown": links.filter(~pl.col("movieId").is_in(movies["movieId"].implode())).height,
        "movies_without_links": movies.filter(~pl.col("movieId").is_in(links["movieId"].implode())).height}
    quality["ratings"]["invalid_rating_values"] = ratings.filter(
        (~pl.col("rating").is_between(.5, 5)) | ((pl.col("rating") * 2) % 1 != 0)).height
    quality["ratings"]["nonpositive_ids_or_timestamp"] = ratings.filter(
        (pl.col("userId") <= 0) | (pl.col("movieId") <= 0) | (pl.col("timestamp") <= 0)).height
    quality["tags"]["empty_or_whitespace_tags"] = tags.filter(pl.col("tag").str.strip_chars().fill_null("") == "").height
    quality["notes"] = ["links.tmdbId is optional external metadata; missing values are reported and retained.",
                        "Tags are events, not a user/movie primary-key table. Full duplicate rows are reported without modifying raw data.",
                        "No user master table exists. Tag-only users are reported separately."]
    fatal = (any(quality["ratings"]["nulls"].values()) or any(quality["movies"]["nulls"].values())
             or quality["movies"]["duplicate_primary_keys"] or quality["links"]["duplicate_primary_keys"]
             or quality["ratings"]["duplicate_user_movie_pairs"] or quality["ratings"]["invalid_rating_values"]
             or quality["ratings"]["nonpositive_ids_or_timestamp"]
             or any(quality["foreign_keys"][k] for k in ["ratings_movie_unknown", "tags_movie_unknown", "links_movie_unknown", "movies_without_links"])
             or any(quality["tags"]["nulls"][k] for k in ["userId", "movieId", "timestamp"])
             or any(quality["links"]["nulls"][k] for k in ["movieId", "imdbId"]))
    quality["required_field_and_relation_checks_passed"] = not bool(fatal)
    save_json(logs / "data_quality.json", quality)
    print("Quality:", json.dumps(quality, ensure_ascii=False), flush=True)
    if fatal:
        raise ValueError("Required integrity checks failed; see data_quality.json")
    user_activity = ratings.group_by("userId").len().sort("userId")
    movie_heat = movies.select("movieId").join(ratings.group_by("movieId").len(), on="movieId", how="left").fill_null(0).sort("movieId")
    yearly = ratings.select(pl.from_epoch("timestamp", time_unit="s").dt.year().alias("year")).group_by("year").len().sort("year")
    tmin, tmax = int(ratings["timestamp"].min()), int(ratings["timestamp"].max())
    statistics = {"ratings": ratings.height, "users": len(users), "movies": len(catalog), "tags": tags.height,
                  "rating_density": ratings.height / (len(users) * len(catalog)),
                  "user_activity": summary(user_activity["len"].to_numpy()),
                  "movie_heat": summary(movie_heat["len"].to_numpy()),
                  "movies_without_ratings": movie_heat.filter(pl.col("len") == 0).height,
                  "rating_histogram": ratings.group_by("rating").len().sort("rating").to_dicts(),
                  "time_range_utc": [datetime.fromtimestamp(t, timezone.utc).isoformat() for t in [tmin, tmax]],
                  "year_counts": yearly.to_dicts()}
    save_json(logs / "descriptive_statistics.json", statistics)
    user_activity.write_csv(processed / "user_activity.csv")
    movie_heat.write_csv(processed / "movie_heat.csv")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, vals, label in [(axs[0], user_activity["len"].to_numpy(), "Ratings per user"),
                            (axs[1], movie_heat["len"].to_numpy(), "Ratings per movie")]:
        positive_values = vals[vals > 0]
        ax.hist(positive_values, bins=np.geomspace(positive_values.min(), positive_values.max() + 1, 45), color="#3568a4")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(label); ax.set_ylabel("Count")
    axs[2].bar(yearly["year"].to_numpy(), yearly["len"].to_numpy() / 1e6, color="#3568a4")
    axs[2].set_xlabel("Rating year (UTC)"); axs[2].set_ylabel("Ratings (millions)")
    fig.savefig(curves / "data_distributions.png", dpi=160); plt.close(fig)
    del frames, tags, links
    gc.collect()
    print("Sorting and splitting full ratings", flush=True)
    ratings = ratings.with_row_index("rowId").sort(["userId", "timestamp", "rowId"])
    uid = ratings["userId"].to_numpy()
    user_ids, starts, counts = np.unique(uid, return_index=True, return_counts=True)
    position = np.arange(len(uid), dtype=np.int32) - np.repeat(starts.astype(np.int32), counts)
    train_end, val_end = (counts * 8 // 10), (counts * 9 // 10)
    split = np.where(position < np.repeat(train_end, counts), 0,
                     np.where(position < np.repeat(val_end, counts), 1, 2)).astype(np.uint8)
    ratings = ratings.with_columns(pl.Series("split", split))
    split_counts, boundary_rows = {}, []
    for code, name in enumerate(["train", "validation", "test"]):
        subset = ratings.filter(pl.col("split") == code).drop("split")
        subset.write_parquet(processed / f"{name}.parquet", compression="zstd")
        split_counts[name] = subset.height
        if name == "train":
            train_positives = fit(subset, processed, cfg["positive_threshold"])
    times = ratings["timestamp"].to_numpy()
    for u, start, n, a, b in zip(user_ids, starts, counts, train_end, val_end):
        boundary_rows.append((int(u), int(n), int(a), int(b - a), int(n - b),
                              int(times[start + a - 1]), int(times[start + a]),
                              int(times[start + b - 1]), int(times[start + b])))
    boundaries = pl.DataFrame(boundary_rows, schema=["userId", "total", "train", "validation", "test",
                            "train_last_ts", "validation_first_ts", "validation_last_ts", "test_first_ts"], orient="row")
    boundaries.write_csv(processed / "split_boundaries.csv")
    boundary_report = {"counts": split_counts, "policy": "floor(0.8*n), floor(0.9*n), residual test",
                       "users": len(user_ids), "all_rows_partitioned_once": sum(split_counts.values()) == ratings.height,
                       "train_after_validation": boundaries.filter(pl.col("train_last_ts") > pl.col("validation_first_ts")).height,
                       "validation_after_test": boundaries.filter(pl.col("validation_last_ts") > pl.col("test_first_ts")).height,
                       "train_validation_equal_timestamp_users": boundaries.filter(pl.col("train_last_ts") == pl.col("validation_first_ts")).height,
                       "validation_test_equal_timestamp_users": boundaries.filter(pl.col("validation_last_ts") == pl.col("test_first_ts")).height,
                       "train_positive_events": train_positives,
                       "teacher_alignment": "pending; teacher scripts not supplied"}
    save_json(logs / "split_report.json", boundary_report)
    print("Generating fixed candidates for every user with test positives", flush=True)
    rng = np.random.default_rng(cfg["seed"])
    items = ratings["movieId"].to_numpy()
    scores = ratings["rating"].to_numpy()
    candidate_users, cutoffs, positives, negatives, positive_offsets = [], [], [], [], [0]
    # Sampling is performed by a separate preparation component, never by fitting.
    for user_index, (u, start, n, b) in enumerate(zip(user_ids, starts, counts, val_end)):
        if user_index % 10000 == 0:
            print(f"candidate preparation {user_index}/{len(user_ids)} users", flush=True)
        ts = slice(start + b, start + n)
        pos = np.unique(items[ts][scores[ts] >= cfg["positive_threshold"]])
        if len(pos) == 0:
            continue
        neg = sample_negatives(catalog, items[start:start + n], rng, cfg["negatives_per_user"])
        candidate_users.append(u); cutoffs.append(times[start + b]); positives.append(pos); negatives.append(neg)
        positive_offsets.append(positive_offsets[-1] + len(pos))
    np.savez_compressed(processed / "candidates.npz", user_ids=np.array(candidate_users, dtype=np.int32),
                        cutoffs=np.array(cutoffs, dtype=np.int64), positive_items=np.concatenate(positives).astype(np.int32),
                        positive_offsets=np.array(positive_offsets, dtype=np.int64), negatives=np.stack(negatives).astype(np.int32))
    report = {"seed": cfg["seed"], "rng": "numpy PCG64 default_rng; rejection sample, ordered first occurrences",
              "evaluated_users": len(candidate_users), "excluded_no_test_positive_users": len(user_ids) - len(candidate_users),
              "positive_items": int(positive_offsets[-1]), "negatives_per_user": cfg["negatives_per_user"],
              "positive_definition": "test rating >=4", "negative_exclusion": "all rated movies, regardless of rating or split",
              "recommendation_cutoff": "first timestamp of user's test block, including nonpositive rows",
              "candidate_sha256": sha256(processed / "candidates.npz")}
    save_json(logs / "candidate_report.json", report)
    save_json(logs / "preparation_environment.json", {"python": sys.version, "platform": platform.platform(),
               "processor": platform.processor(), "logical_cpus": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
               "versions": {"numpy": np.__version__, "polars": pl.__version__, "matplotlib": matplotlib.__version__, "psutil": psutil.__version__},
               "seconds": time.perf_counter() - started, "ratings_per_second_end_to_end": ratings.height / (time.perf_counter() - started),
               "process_rss_bytes_at_end": psutil.Process().memory_info().rss,
               "rss_note": "point-in-time RSS, not peak memory"})
    artifacts = [{"path": p.relative_to(STAGE).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
                 for p in sorted(processed.iterdir()) if p.is_file() and p.suffix in [".parquet", ".npz", ".csv"]]
    save_json(STAGE / "data/processed_manifest.json", {"files": artifacts, "generated_from_raw_manifest": "data/raw_manifest.json"})
    print("Preparation completed:", json.dumps({"split": boundary_report, "candidates": report,
          "seconds": time.perf_counter() - started}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

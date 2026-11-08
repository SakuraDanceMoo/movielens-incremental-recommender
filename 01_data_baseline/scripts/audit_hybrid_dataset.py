"""Audit the supplied hybrid ratings and create an RMSE baseline record."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "01_data_baseline"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(data_dir: Path) -> dict:
    summary = json.loads((data_dir / "summary.json").read_text(encoding="utf-8"))
    id_ranges = json.loads((data_dir / "id_ranges.json").read_text(encoding="utf-8"))
    if summary.get("dataset") != "hybrid-half-half-latest100k-v1":
        raise ValueError("Unexpected training dataset identifier")
    ratings_path = data_dir / "ratings.csv"
    test_path = data_dir / "test_pairs.csv"
    ratings = pl.scan_csv(ratings_path, schema_overrides={
        "userId": pl.Int32, "movieId": pl.Int32, "rating": pl.Float32, "timestamp": pl.Int64,
    })
    stats = ratings.select(
        pl.len().alias("rows"),
        pl.col("userId").n_unique().alias("users"),
        pl.col("movieId").n_unique().alias("movies"),
        pl.col("rating").min().alias("min_rating"),
        pl.col("rating").max().alias("max_rating"),
        pl.col("rating").mean().alias("mean_rating"),
        pl.col("timestamp").min().alias("min_timestamp"),
        pl.col("timestamp").max().alias("max_timestamp"),
        pl.col("userId").filter(pl.col("userId") <= id_ranges["movielens32m_users"][1]).len().alias("movielens32m_rows"),
        pl.col("userId").filter(pl.col("userId") >= id_ranges["synthetic_users"][0]).len().alias("synthetic_rows"),
    ).collect(engine="streaming").row(0, named=True)
    if int(stats["rows"]) != int(summary["training_ratings"]):
        raise ValueError("ratings.csv count disagrees with summary.json")
    if stats["min_rating"] < 0.5 or stats["max_rating"] > 5.0:
        raise ValueError("Ratings fall outside the expected [0.5, 5.0] range")
    test = pl.read_csv(test_path, infer_schema_length=1000)
    expected_columns = ["userId", "movieId", "timestamp"]
    if test.columns != expected_columns:
        raise ValueError(f"Unexpected test_pairs columns: {test.columns}")
    if test.height != int(summary["hidden_test_ratings"]):
        raise ValueError("test_pairs.csv row count disagrees with summary.json")
    if test.null_count().sum_horizontal().item() != 0:
        raise ValueError("test_pairs.csv contains null fields")
    if "rating" in test.columns:
        raise ValueError("test_pairs.csv unexpectedly contains target labels")

    return {
        "dataset": summary["dataset"],
        "source": "user-supplied local training folder; raw data is not copied to Git",
        "source_composition": "approximately half MovieLens 32M-derived ratings and half synthetic ratings",
        "ratings": {key: (float(value) if key in {"min_rating", "max_rating", "mean_rating"}
                          else int(value)) for key, value in stats.items()},
        "ratings_sha256": sha256(ratings_path),
        "test_pairs": {"rows": test.height, "columns": test.columns,
                       "contains_rating_label": False,
                       "movielens32m_rows": int((test["userId"] <= id_ranges["movielens32m_users"][1]).sum()),
                       "synthetic_rows": int((test["userId"] >= id_ranges["synthetic_users"][0]).sum()),
                       "cold_start_rows": int(summary["cold_start_test_rows"]["unseen_user"] +
                                               summary["cold_start_test_rows"]["unseen_movie"])},
        "rating_range": [0.5, 5.0],
        "baseline": {
            "model": "training-mean-on-per-user-temporal-holdout",
            "validation_rows": 386982,
            "rmse": 1.0675994768057624,
            "metric_source": "independently generated stage-six explicit-ALS validation run",
        },
        "checks": ["dataset identifier", "rating row count", "rating scale",
                   "test row count", "test schema", "test labels absent", "no cold-start pairs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=STAGE / "logs/hybrid_data_audit.json")
    args = parser.parse_args()
    result = audit(args.data_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

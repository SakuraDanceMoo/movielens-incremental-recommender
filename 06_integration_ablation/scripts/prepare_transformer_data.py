"""Immutable hybrid data intake and exactly the established latest-user holdout."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "06_integration_ablation"
CACHE = STAGE / "data" / "processed" / "transformer"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(data_dir: Path) -> None:
    started = time.perf_counter()
    CACHE.mkdir(parents=True, exist_ok=True)
    ratings_sha = sha256(data_dir / "ratings.csv")
    expected = "b1e1b5e22a177ee32a207e058fa22a3f3601cd77f68b4a091625237a43b96815"
    if ratings_sha != expected:
        raise ValueError("Input changed from the audited hybrid ratings; audit before reuse")
    schema = {"userId": pl.Int32, "movieId": pl.Int32,
              "rating": pl.Float32, "timestamp": pl.Int64}
    df = pl.read_csv(data_dir / "ratings.csv", schema_overrides=schema)
    df = df.with_row_index("row_id")
    # A deterministic row key avoids dropping multiple equal timestamp/movie records.
    latest = (df.sort(["userId", "timestamp", "movieId"], descending=[False, True, True])
              .group_by("userId", maintain_order=True)
              .agg(pl.first("row_id"), pl.len().alias("user_count"))
              .filter(pl.col("user_count") >= 6))
    indices = latest["row_id"].to_numpy()
    val_mask = np.zeros(df.height, dtype=bool)
    val_mask[indices] = True
    assert df.height == 31900204 and int(val_mask.sum()) == 386982
    for split, mask in [("fit", ~val_mask), ("valid", val_mask), ("full", None)]:
        for column, key in [("userId", "u"), ("movieId", "i"), ("rating", "y")]:
            values = df[column].to_numpy()
            np.save(CACHE / f"{split}_{key}.npy", values if mask is None else values[mask])
        if mask is not None:
            np.save(CACHE / f"{split}_t.npy", df["timestamp"].to_numpy()[mask])
    np.save(CACHE / "valid_row_ids.npy", np.flatnonzero(val_mask))
    train_mean = float(df.filter(pl.Series(~val_mask))["rating"].mean())
    actual_baseline = float(np.sqrt(np.mean((df["rating"].to_numpy()[val_mask].astype(float) - train_mean) ** 2)))
    del df, latest, val_mask
    gc.collect()
    test = pl.read_csv(data_dir / "test_pairs.csv", schema_overrides={
        "userId": pl.Int32, "movieId": pl.Int32, "timestamp": pl.Int64})
    for col, key in [("userId", "u"), ("movieId", "i"), ("timestamp", "t")]:
        np.save(CACHE / f"test_{key}.npy", test[col].to_numpy())
    movies = pl.read_csv(data_dir / "movies.csv")
    vocabulary = sorted({g for genres in movies["genres"].to_list() for g in genres.split("|")})
    genre_features = np.zeros((175171, len(vocabulary)), np.float32)
    years = np.zeros((175171, 1), np.float32)
    import re
    for mid, title, genres in movies.iter_rows():
        for genre in genres.split("|"):
            genre_features[mid, vocabulary.index(genre)] = 1
        match = re.search(r"\((\d{4})\)", title)
        years[mid] = (float(match[1]) - 1990) / 40 if match else 0
    np.save(CACHE / "genre_features.npy", genre_features)
    np.save(CACHE / "year_features.npy", years)
    report = {
        "dataset": "hybrid-half-half-latest100k-v1", "ratings_sha256": ratings_sha,
        "test_pairs_sha256": sha256(data_dir / "test_pairs.csv"),
        "movies_sha256": sha256(data_dir / "movies.csv"),
        "fit_rows": 31513222, "valid_rows": 386982, "full_rows": 31900204,
        "train_global_mean": train_mean, "global_mean_validation_rmse": actual_baseline,
        "validation_row_ids_sha256": sha256(CACHE / "valid_row_ids.npy"),
        "split": "one latest timestamp, movieId descending, per user with >=6 ratings",
        "genres": vocabulary, "source_threshold": 200948,
        "test_labels_available": False, "elapsed_seconds": time.perf_counter() - started,
    }
    log_dir = STAGE / "logs" / "transformer"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "data_protocol.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    prepare(parser.parse_args().data_dir)

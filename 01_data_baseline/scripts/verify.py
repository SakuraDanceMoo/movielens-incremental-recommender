"""Independent full-artifact checks, frozen sampling replay and Top-10 metric recomputation."""
import gzip
import json
import time

import numpy as np
import polars as pl

from core import sample_negatives
from download import STAGE, sha256
from prepare import save_json


def main():
    started = time.perf_counter()
    processed, logs = STAGE / "data/processed", STAGE / "logs"
    cfg = json.loads((STAGE / "config.json").read_text(encoding="utf-8"))
    raw_manifest = json.loads((STAGE / "data/raw_manifest.json").read_text(encoding="utf-8"))
    processed_manifest = json.loads((STAGE / "data/processed_manifest.json").read_text(encoding="utf-8"))
    raw_ok = all(sha256(STAGE / i["path"]) == i["sha256"] for i in raw_manifest["files"])
    processed_ok = all(sha256(STAGE / i["path"]) == i["sha256"] for i in processed_manifest["files"])
    frames = [pl.read_parquet(processed / f"{s}.parquet") for s in ["train", "validation", "test"]]
    row_ids = np.concatenate([f["rowId"].to_numpy() for f in frames])
    row_ids.sort()
    partition_ok = np.array_equal(row_ids, np.arange(len(row_ids), dtype=row_ids.dtype))
    del row_ids
    totals = pl.concat([f.select("userId") for f in frames]).group_by("userId").len().sort("userId")
    n = totals["len"].to_numpy().astype(np.int64)
    expected = [n * 8 // 10, n * 9 // 10 - n * 8 // 10, n - n * 9 // 10]
    ratios_ok = all(np.array_equal(f.group_by("userId").len().sort("userId")["len"].to_numpy(), e)
                    for f, e in zip(frames, expected))
    maxima = [f.group_by("userId").agg(pl.col("timestamp").max()).sort("userId")["timestamp"].to_numpy() for f in frames]
    minima = [f.group_by("userId").agg(pl.col("timestamp").min()).sort("userId")["timestamp"].to_numpy() for f in frames]
    chronological_ok = bool(np.all(maxima[0] <= minima[1]) and np.all(maxima[1] <= minima[2]))
    rated = pl.concat([f.select("userId", "movieId") for f in frames]).sort(["userId", "movieId"])
    uid = rated["userId"].to_numpy()
    user_ids, starts, lengths = np.unique(uid, return_index=True, return_counts=True)
    rated_items = rated["movieId"].to_numpy()
    catalog = pl.read_csv(STAGE / "data/raw/movies.csv")["movieId"].sort().to_numpy()
    test_positives = frames[2].filter(pl.col("rating") >= cfg["positive_threshold"]).select("userId", "movieId").sort(["userId", "movieId"])
    test_min = frames[2].group_by("userId").agg(pl.col("timestamp").min()).sort("userId")
    with np.load(processed / "candidates.npz") as z:
        candidates = {key: z[key] for key in z.files}
    cusers, negative = candidates["user_ids"], candidates["negatives"]
    expected_users = test_positives["userId"].unique().sort().to_numpy()
    candidate_users_ok = np.array_equal(cusers, expected_users)
    counts = test_positives.group_by("userId").len().sort("userId")["len"].to_numpy()
    positives_ok = np.array_equal(candidates["positive_items"], test_positives["movieId"].to_numpy()) and np.array_equal(np.diff(candidates["positive_offsets"]), counts)
    cutoffs_ok = np.array_equal(candidates["cutoffs"], test_min.filter(pl.col("userId").is_in(pl.Series(cusers).implode()))["timestamp"].to_numpy())
    negative_shape_ok = negative.shape == (len(cusers), cfg["negatives_per_user"])
    negatives_unique_ok = bool(np.all(np.diff(np.sort(negative, axis=1), axis=1) != 0))
    negative_catalog_ok = bool(np.isin(negative, catalog).all())
    rng = np.random.default_rng(cfg["seed"])
    replay_ok, exclusion_ok = True, True
    for index, u in enumerate(cusers):
        location = np.searchsorted(user_ids, u)
        items = rated_items[starts[location]:starts[location] + lengths[location]]
        exclusion_ok &= not bool(np.isin(negative[index], items).any())
        replay_ok &= np.array_equal(sample_negatives(catalog, items, rng, cfg["negatives_per_user"]), negative[index])
    top = pl.read_csv(processed / "baseline_top10.csv.gz").sort(["userId", "rank"])
    k = cfg["k"]
    top_ok = top.height == len(cusers) * k and np.array_equal(top["userId"].to_numpy(), np.repeat(cusers, k))
    ranks_ok = np.array_equal(top["rank"].to_numpy(), np.tile(np.arange(1, k + 1), len(cusers)))
    top_items = top["movieId"].to_numpy().reshape(-1, k)
    unique_top_ok = bool(np.all(np.diff(np.sort(top_items, axis=1), axis=1) != 0))
    scale = int(catalog.max()) + 1
    pos_keys = test_positives["userId"].to_numpy().astype(np.int64) * scale + test_positives["movieId"].to_numpy()
    top_keys = top["userId"].to_numpy().astype(np.int64) * scale + top["movieId"].to_numpy()
    relevance = np.isin(top_keys, pos_keys).reshape(-1, k)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    ideals = np.cumsum(discounts)[np.minimum(counts, k) - 1]
    independent_ndcg = float((relevance @ discounts / ideals).mean())
    independent_recall = float((relevance.sum(axis=1) / counts).mean())
    baseline = json.loads((logs / "baseline_metrics.json").read_text(encoding="utf-8"))
    repeat = json.loads((logs / "repeat_metrics.json").read_text(encoding="utf-8"))
    metric_ok = bool(np.isclose(independent_ndcg, baseline["ndcg_at_10"], atol=1e-12, rtol=0)
                     and np.isclose(independent_recall, baseline["recall_at_10"], atol=1e-12, rtol=0))
    # Independent count reconstruction at early, late and intermediate cutoffs.
    events = pl.read_parquet(processed / "train_positive_events.parquet")
    events_items, events_times = events["movieId"].to_numpy(), events["timestamp"].to_numpy()
    points = np.unique(np.argsort(candidates["cutoffs"])[np.linspace(0, len(cusers) - 1, 7).astype(int)])
    temporal_score_ok, sampled_ranking_ok = True, True
    for i in points:
        cutoff = candidates["cutoffs"][i]
        reference = np.bincount(events_items[events_times < cutoff], minlength=scale)
        temporal_score_ok &= np.array_equal(reference[top_items[i]], top["score"].to_numpy().reshape(-1, k)[i])
        possible = np.concatenate([candidates["positive_items"][candidates["positive_offsets"][i]:candidates["positive_offsets"][i + 1]], negative[i]])
        expected_top = sorted(possible.tolist(), key=lambda movie: (-reference[movie], movie))[:k]
        sampled_ranking_ok &= np.array_equal(top_items[i], expected_top)
    repeat_ok = baseline["scientific_output_sha256"] == repeat["scientific_output_sha256"]
    candidate_top_ok = True
    for i in range(len(cusers)):
        possible = np.concatenate([candidates["positive_items"][candidates["positive_offsets"][i]:candidates["positive_offsets"][i + 1]], negative[i]])
        candidate_top_ok &= bool(np.isin(top_items[i], possible).all())
    checks = {"raw_hashes": raw_ok, "processed_hashes": processed_ok, "partition_no_loss_or_overlap": partition_ok,
              "exact_per_user_floor_80_90_boundaries": ratios_ok, "chronological_order": chronological_ok,
              "candidate_user_population": candidate_users_ok, "exact_test_positives": positives_ok,
              "first_test_cutoffs": cutoffs_ok, "negative_shape": negative_shape_ok,
              "unique_negatives": negatives_unique_ok, "negative_catalog_membership": negative_catalog_ok,
              "negative_excludes_all_user_ratings": exclusion_ok, "seeded_sampling_replay": replay_ok,
              "top10_user_population": top_ok, "top10_ranks": ranks_ok, "unique_top10_items": unique_top_ok,
              "top10_from_frozen_candidates": candidate_top_ok, "independent_metrics": metric_ok,
              "independent_temporal_scores_seven_cutoffs": temporal_score_ok,
              "independent_ranking_seven_cutoffs": sampled_ranking_ok,
              "repeat_scientific_output_identical": repeat_ok}
    report = {"checks": {k: bool(v) for k, v in checks.items()}, "all_passed": bool(all(checks.values())),
              "independent_ndcg_at_10": independent_ndcg, "independent_recall_at_10": independent_recall,
              "verification_seconds": time.perf_counter() - started,
              "teacher_alignment": "unverified; teacher scripts not provided",
              "reproducibility_scope": "full negative sampling replay plus second full evaluation; timing and generated_at excluded"}
    save_json(logs / "verification.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

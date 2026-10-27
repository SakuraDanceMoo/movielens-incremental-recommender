"""Replay time-ordered positive ratings into incremental recommendation state."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import psutil

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "04_streaming_incremental"
BASELINE_STAGE = ROOT / "01_data_baseline"
SEED = 20260907
K = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank(candidates: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.lexsort((candidates, -scores))[:K]


def metrics(relevant: np.ndarray, positive_count: int) -> tuple[float, float]:
    discounts = 1.0 / np.log2(np.arange(2, len(relevant) + 2))
    ideal = np.sum(1.0 / np.log2(np.arange(2, min(K, positive_count) + 2)))
    return (float(np.dot(relevant, discounts) / ideal),
            int(np.count_nonzero(relevant)) / int(positive_count))


def load_genres(path: Path, max_item: int):
    movies = pl.read_csv(path, schema_overrides={"movieId": pl.Int32})
    genres = sorted({genre for value in movies["genres"].drop_nulls().to_list()
                     for genre in value.split("|") if genre != "(no genres listed)"})
    genre_id = {genre: index for index, genre in enumerate(genres)}
    item_genres = np.full((max_item + 1, max(1, len(genres))), -1, dtype=np.int16)
    for row in movies.select("movieId", "genres").iter_rows(named=True):
        item = int(row["movieId"])
        if item > max_item:
            continue
        indices = [genre_id[g] for g in row["genres"].split("|") if g in genre_id]
        item_genres[item, :len(indices)] = indices
    return genres, item_genres


def load_inputs():
    processed = BASELINE_STAGE / "data/processed"
    cfg = json.loads((STAGE / "config.json").read_text(encoding="utf-8"))
    candidates = np.load(processed / "candidates.npz")
    protocol = {key: candidates[key] for key in
                ("user_ids", "cutoffs", "positive_items", "positive_offsets", "negatives")}
    candidates.close()
    events = (pl.scan_parquet(processed / "train.parquet")
              .filter(pl.col("rating") >= cfg["positive_threshold"])
              .select(pl.col("rowId").cast(pl.Int64), pl.col("userId").cast(pl.Int32),
                      pl.col("movieId").cast(pl.Int32), pl.col("timestamp").cast(pl.Int64))
              .sort(["timestamp", "rowId"])
              .collect(engine="streaming"))
    arrays = {name: events[name].to_numpy() for name in ("rowId", "userId", "movieId", "timestamp")}
    if np.any(arrays["timestamp"][1:] < arrays["timestamp"][:-1]):
        raise AssertionError("Replay events are not ordered by event time")
    maximum_item = max(int(protocol["positive_items"].max()), int(protocol["negatives"].max()),
                       int(arrays["movieId"].max()))
    maximum_user = max(int(protocol["user_ids"].max()), int(arrays["userId"].max()))
    genres, item_genres = load_genres(BASELINE_STAGE / "data/raw/movies.csv", maximum_item)
    return cfg, protocol, arrays, genres, item_genres, maximum_user, maximum_item


class ReplayState:
    def __init__(self, events, item_genres, maximum_user, maximum_item, window_seconds):
        self.events = events
        self.item_genres = item_genres
        self.window_seconds = int(window_seconds)
        self.cursor = 0
        self.window_start = 0
        self.watermark = 0
        self.all_time_item_counts = np.zeros(maximum_item + 1, dtype=np.int32)
        self.window_item_counts = np.zeros(maximum_item + 1, dtype=np.int32)
        self.user_positive_counts = np.zeros(maximum_user + 1, dtype=np.int32)
        genre_count = int(item_genres.max()) + 1
        self.user_genre_counts = np.zeros((maximum_user + 1, genre_count), dtype=np.int32)
        self.late_events_dropped = 0

    def advance_before(self, cutoff: int, event_latencies_ns: np.ndarray) -> tuple[int, int]:
        target = int(np.searchsorted(self.events["timestamp"], cutoff, side="left"))
        if target < self.cursor:
            raise ValueError("Event-time query cutoff moved backwards")
        timestamps = self.events["timestamp"]
        users, items, row_ids = self.events["userId"], self.events["movieId"], self.events["rowId"]
        applied = 0
        for index in range(self.cursor, target):
            start_ns = time.perf_counter_ns()
            timestamp = int(timestamps[index])
            if timestamp < self.watermark:
                self.late_events_dropped += 1
                event_latencies_ns[index] = time.perf_counter_ns() - start_ns
                continue
            self.watermark = timestamp
            expiry = timestamp - self.window_seconds
            while self.window_start < index and int(timestamps[self.window_start]) < expiry:
                self.window_item_counts[int(items[self.window_start])] -= 1
                self.window_start += 1

            user, item = int(users[index]), int(items[index])
            if row_ids[index] < 0:
                raise ValueError("rowId must be a stable nonnegative source offset")
            self.all_time_item_counts[item] += 1
            self.window_item_counts[item] += 1
            self.user_positive_counts[user] += 1
            for genre in self.item_genres[item]:
                if genre >= 0:
                    self.user_genre_counts[user, int(genre)] += 1
            event_latencies_ns[index] = time.perf_counter_ns() - start_ns
            applied += 1
        self.cursor = target
        maintenance_started = time.perf_counter_ns()
        self.watermark = max(self.watermark, int(cutoff))
        expiry = self.watermark - self.window_seconds
        while self.window_start < self.cursor and int(timestamps[self.window_start]) < expiry:
            self.window_item_counts[int(items[self.window_start])] -= 1
            self.window_start += 1
        return applied, time.perf_counter_ns() - maintenance_started

    def digest(self) -> str:
        digest = hashlib.sha256()
        for array in (self.all_time_item_counts, self.window_item_counts,
                      self.user_positive_counts, self.user_genre_counts):
            digest.update(array.tobytes())
        digest.update(np.asarray([self.cursor, self.window_start, self.watermark], dtype=np.int64).tobytes())
        return digest.hexdigest()


def recommend(state: ReplayState, user: int, candidates: np.ndarray,
              window_weight: float, genre_weight: float) -> np.ndarray:
    all_time = state.all_time_item_counts[candidates].astype(np.float64)
    recent = state.window_item_counts[candidates].astype(np.float64)
    scores = all_time + window_weight * recent
    profile = state.user_genre_counts[user].astype(np.float64)
    profile_total = float(profile.sum())
    if profile_total:
        genres = state.item_genres[candidates]
        valid = genres >= 0
        values = profile[np.maximum(genres, 0)]
        affinity = np.divide((values * valid).sum(axis=1), valid.sum(axis=1),
                             out=np.zeros(len(candidates), dtype=np.float64), where=valid.sum(axis=1) > 0)
        scores += genre_weight * affinity / profile_total
    return scores


def write_gzip_csv(path: Path, header, rows):
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as binary:
        with io.TextIOWrapper(binary, encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)


def run():
    started = time.perf_counter()
    cfg, protocol, events, genres, item_genres, maximum_user, maximum_item = load_inputs()
    users, cutoffs = protocol["user_ids"], protocol["cutoffs"]
    positives, offsets, negatives = (protocol[name] for name in
                                     ("positive_items", "positive_offsets", "negatives"))
    baseline_path = BASELINE_STAGE / "data/processed/baseline_per_user.csv.gz"
    baseline_scores = {}
    with gzip.open(baseline_path, "rt", encoding="utf-8", newline="") as stream:
        baseline_scores = {int(row["userId"]): (float(row["ndcg_at_10"]), float(row["recall_at_10"]))
                           for row in csv.DictReader(stream)}

    source_event_count = len(events["timestamp"])
    latencies_ns = np.zeros(source_event_count, dtype=np.uint32)
    state = ReplayState(events, item_genres, maximum_user, maximum_item,
                        cfg["window_days"] * 24 * 60 * 60)
    ndcgs = np.zeros(len(users), dtype=np.float64)
    recalls = np.zeros(len(users), dtype=np.float64)
    candidate_latencies_ns = np.zeros(len(users), dtype=np.uint32)
    watermark_latencies_ns = np.zeros(len(users), dtype=np.uint32)
    rows = []
    top_rows = []
    snapshots = []
    order = np.argsort(cutoffs, kind="stable")
    started_replay = time.perf_counter()
    for position, index in enumerate(order):
        before = state.cursor
        _, watermark_latencies_ns[position] = state.advance_before(int(cutoffs[index]), latencies_ns)
        if position in {0, len(order) // 4, len(order) // 2, 3 * len(order) // 4, len(order) - 1}:
            snapshots.append({"userId": int(users[index]), "cutoff": int(cutoffs[index]),
                              "source_offset": state.cursor, "watermark": state.watermark,
                              "window_start_offset": state.window_start,
                              "window_positive_events": int(state.window_item_counts.sum()),
                              "active_window_items": int(np.count_nonzero(state.window_item_counts)),
                              "profiled_users": int(np.count_nonzero(state.user_positive_counts)),
                              "state_sha256": state.digest()})
        user = int(users[index])
        pos = positives[offsets[index]:offsets[index + 1]]
        candidates = np.concatenate((pos, negatives[index])).astype(np.int32, copy=False)
        start_ns = time.perf_counter_ns()
        scores = recommend(state, user, candidates, cfg["window_score_weight"], cfg["genre_affinity_weight"])
        ranked = rank(candidates, scores)
        relevant = (ranked < len(pos)).astype(np.uint8)
        ndcgs[index], recalls[index] = metrics(relevant, len(pos))
        candidate_latencies_ns[index] = time.perf_counter_ns() - start_ns
        rows.append((user, f"{ndcgs[index]:.12g}", f"{recalls[index]:.12g}", len(pos),
                     candidate_latencies_ns[index] / 1e6))
        top_rows.extend((user, int(candidates[j]), rank_index + 1, float(scores[j]), "streaming-hybrid-v1")
                        for rank_index, j in enumerate(ranked))
        if state.cursor < before:
            raise AssertionError("Source offset moved backwards")
    replay_seconds = time.perf_counter() - started_replay

    repeated_cursor, repeated_digest = state.cursor, state.digest()
    state.advance_before(int(cutoffs[order[-1]]), latencies_ns)
    idempotent_replay = state.cursor == repeated_cursor and state.digest() == repeated_digest
    if not idempotent_replay:
        raise AssertionError("Replaying an already-consumed source offset changed state")

    output = STAGE / "data/processed"
    logs = STAGE / "logs"
    curves = STAGE / "curves"
    for directory in (output, logs, curves):
        directory.mkdir(parents=True, exist_ok=True)
    per_user_path = output / "stream_per_user.csv.gz"
    write_gzip_csv(per_user_path, ["userId", "ndcg_at_10", "recall_at_10", "test_positives", "candidate_scoring_ms"], rows)
    top_path = output / "stream_top10.csv.gz"
    write_gzip_csv(top_path, ["userId", "movieId", "rank", "score", "model_version"], top_rows)

    processed_event_count = state.cursor
    latency_ms = latencies_ns[:processed_event_count].astype(np.float64) / 1e6
    candidate_ms = candidate_latencies_ns.astype(np.float64) / 1e6
    watermark_ms = watermark_latencies_ns.astype(np.float64) / 1e6
    baseline_ndcg = float(np.mean([baseline_scores[int(user)][0] for user in users]))
    baseline_recall = float(np.mean([baseline_scores[int(user)][1] for user in users]))
    generated = datetime.now(timezone.utc).isoformat()
    report = {
        "experiment": "phase4-event-time-incremental-replay-v1",
        "generated_at_utc": generated,
        "dataset": cfg["dataset"],
        "input": "01_data_baseline/data/processed/train.parquet positive training events; stable rowId is source offset",
        "candidate_protocol": "frozen stage-one test positives plus 100 fixed negatives; score before each user's first test timestamp",
        "candidate_sha256": sha256(BASELINE_STAGE / "data/processed/candidates.npz"),
        "source_positive_event_count": source_event_count,
        "processed_positive_event_count": processed_event_count,
        "users_evaluated": len(users),
        "state": {
            "window_days": cfg["window_days"],
            "late_event_policy": cfg["late_event_policy"],
            "late_events_dropped": state.late_events_dropped,
            "duplicate_policy": cfg["duplicate_policy"],
            "replayed_source_offset_idempotent": idempotent_replay,
            "final_source_offset": state.cursor,
            "final_state_sha256": state.digest(),
            "user_profile_genres": genres,
            "active_window_positive_items": int(np.count_nonzero(state.window_item_counts)),
            "profiled_users": int(np.count_nonzero(state.user_positive_counts)),
            "checkpoints": snapshots,
        },
        "ranking": {
            "model": "cumulative positive count + window_score_weight * 30-day positive count + genre_affinity_weight * mean genre count / user genre-event count",
            "window_score_weight": cfg["window_score_weight"],
            "genre_affinity_weight": cfg["genre_affinity_weight"],
            "tie_break": "score descending then movieId ascending",
            "ndcg_at_10": float(ndcgs.mean()),
            "recall_at_10": float(recalls.mean()),
            "matching_user_stage_one_baseline_ndcg_at_10": baseline_ndcg,
            "matching_user_stage_one_baseline_recall_at_10": baseline_recall,
            "ndcg_relative_change_vs_baseline": float(ndcgs.mean() / baseline_ndcg - 1.0),
            "recall_relative_change_vs_baseline": float(recalls.mean() / baseline_recall - 1.0),
            "ndcg_no_degradation_met": bool(ndcgs.mean() >= baseline_ndcg),
            "recall_no_degradation_met": bool(recalls.mean() >= baseline_recall),
        },
        "performance": {
            "replay_and_candidate_evaluation_seconds": replay_seconds,
            "input_load_and_total_elapsed_seconds": time.perf_counter() - started,
            "events_per_second_excluding_input_load": processed_event_count / replay_seconds,
            "event_update_latency_ms": {"mean": float(latency_ms.mean()), "p50": float(np.quantile(latency_ms, .5)),
                                        "p95": float(np.quantile(latency_ms, .95)), "p99": float(np.quantile(latency_ms, .99)),
                                        "max": float(latency_ms.max())},
            "candidate_scoring_latency_ms": {"mean": float(candidate_ms.mean()), "p50": float(np.quantile(candidate_ms, .5)),
                                             "p95": float(np.quantile(candidate_ms, .95)), "p99": float(np.quantile(candidate_ms, .99)),
                                             "max": float(candidate_ms.max())},
            "watermark_window_maintenance_latency_ms": {"mean": float(watermark_ms.mean()), "p50": float(np.quantile(watermark_ms, .5)),
                                                         "p95": float(np.quantile(watermark_ms, .95)), "p99": float(np.quantile(watermark_ms, .99)),
                                                         "max": float(watermark_ms.max())},
            "target_event_processing_p99_ms": cfg["target_event_processing_p99_ms"],
            "event_update_p99_below_target": bool(np.quantile(latency_ms, .99) < cfg["target_event_processing_p99_ms"]),
            "scope": "single-process in-memory state update; excludes storage, network, broker and checkpoint I/O",
            "python": platform.python_version(),
            "polars": pl.__version__,
            "cpu_count": os.cpu_count(),
            "available_memory_gib_at_start": round(psutil.virtual_memory().available / 1024**3, 2),
        },
        "outputs": {
            "per_user_metrics": per_user_path.relative_to(STAGE).as_posix(),
            "top10": top_path.relative_to(STAGE).as_posix(),
            "latency_curve": "curves/streaming_latency.png",
        },
    }
    report["outputs"]["per_user_metrics_sha256"] = sha256(per_user_path)
    report["outputs"]["top10_sha256"] = sha256(top_path)
    (logs / "stream_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    lower_bound = max(float(latency_ms[latency_ms > 0].min()), 1e-5)
    ax.hist(latency_ms, bins=np.geomspace(lower_bound, cfg["target_event_processing_p99_ms"], 90),
            color="#367a70", alpha=0.9)
    ax.axvline(cfg["target_event_processing_p99_ms"], color="#b85548", linestyle="--", label="100 ms target")
    ax.axvline(float(np.quantile(latency_ms, .99)), color="#315f9a", linestyle=":", label="Measured p99")
    ax.set_xscale("log")
    ax.set_xlim(lower_bound, cfg["target_event_processing_p99_ms"] * 1.1)
    ax.set_xlabel("Per-event state update latency (ms)")
    ax.set_ylabel("Training positive events")
    ax.set_title("Event-time replay update latency")
    ax.legend()
    fig.savefig(curves / "streaming_latency.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-days", type=int)
    args = parser.parse_args()
    if args.window_days is not None:
        cfg_path = STAGE / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["window_days"] = args.window_days
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    run()


if __name__ == "__main__":
    main()

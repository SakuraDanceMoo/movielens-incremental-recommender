"""Build a MovieLens user-movie graph and evaluate personalized PageRank."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "05_graph_recommendation"
BASE = ROOT / "01_data_baseline"
ALS_TOP10 = ROOT / "03_distributed_recommendation/data/processed/als_top10.csv.gz"
K = 10


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(relevance, positives):
    discounts = 1.0 / np.log2(np.arange(2, len(relevance) + 2))
    ideal = np.sum(1.0 / np.log2(np.arange(2, min(K, positives) + 2)))
    return float(np.dot(relevance, discounts) / ideal), int(np.count_nonzero(relevance)) / int(positives)


def load_inputs(cfg):
    path = BASE / "data/processed/candidates.npz"
    with np.load(path) as archive:
        count = min(cfg["users"], len(archive["user_ids"]))
        protocol = {key: archive[key] if key in {"positive_items", "positive_offsets"} else archive[key][:count]
                    for key in archive.files}
    users = protocol["user_ids"]
    user_frame = pl.DataFrame({"userId": users.astype(np.int32)})
    events = (pl.read_parquet(BASE / "data/processed/train.parquet", columns=["rowId", "userId", "movieId", "rating"])
              .filter(pl.col("rating") >= 4.0)
              .join(user_frame, on="userId", how="inner")
              .select("userId", "movieId").unique().sort(["userId", "movieId"]))
    return protocol, events, sha256(path)


def graph_matrix(users, events):
    user_index = {int(user): index for index, user in enumerate(users)}
    movie_ids = np.sort(events["movieId"].unique().to_numpy().astype(np.int32))
    movie_index = {int(movie): index for index, movie in enumerate(movie_ids)}
    user_rows = np.fromiter((user_index[int(user)] for user in events["userId"].to_numpy()), dtype=np.int32)
    movie_cols = np.fromiter((movie_index[int(movie)] for movie in events["movieId"].to_numpy()), dtype=np.int32)
    n_users, n_movies = len(users), len(movie_ids)
    n_nodes = n_users + n_movies
    left = user_rows
    right = n_users + movie_cols
    rows = np.concatenate((left, right))
    cols = np.concatenate((right, left))
    adjacency = sparse.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)),
                                  shape=(n_nodes, n_nodes))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    transition = (sparse.diags(np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)) @ adjacency).tocsr()
    return transition.transpose().tocsr(), movie_ids, user_index, n_nodes, degree


def ppr(query_node, transition_t, restart, max_iterations, tolerance):
    vector = np.zeros(transition_t.shape[0], dtype=np.float64)
    vector[query_node] = 1.0
    for iteration in range(1, max_iterations + 1):
        updated = (1.0 - restart) * (transition_t @ vector)
        updated[query_node] += restart
        delta = float(np.abs(updated - vector).sum())
        vector = updated
        if delta < tolerance:
            return vector, iteration, delta
    return vector, max_iterations, delta


def rank_ids(ids, scores):
    return np.lexsort((ids, -scores))[:K]


def read_top10(path):
    by_user = defaultdict(dict)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            by_user[int(row["userId"])][int(row["movieId"])] = int(row["rank"])
    return by_user


def reciprocal_rank_fusion(inputs, movie_ids):
    scores = np.zeros(len(movie_ids), dtype=np.float64)
    for ranking in inputs:
        for movie, rank in ranking.items():
            index = movie_ids.get(movie)
            if index is not None:
                scores[index] += 1.0 / rank
    return scores


def evaluate(protocol, users, recommendations, name):
    positives, offsets, negatives = (protocol[key] for key in
                                     ("positive_items", "positive_offsets", "negatives"))
    ndcgs, recalls, rows = [], [], []
    for index, user_value in enumerate(users):
        user = int(user_value)
        pos = positives[offsets[index]:offsets[index + 1]]
        candidate_ids = np.concatenate((pos, negatives[index])).astype(np.int32, copy=False)
        score_map = recommendations[index]
        candidate_scores = np.fromiter((score_map.get(int(movie), 0.0) for movie in candidate_ids),
                                       dtype=np.float64, count=len(candidate_ids))
        order = np.lexsort((candidate_ids, -candidate_scores))[:K]
        relevant = (order < len(pos)).astype(np.uint8)
        ndcg, recall = metric(relevant, len(pos))
        ndcgs.append(ndcg)
        recalls.append(recall)
        rows.extend((user, int(candidate_ids[j]), rank + 1, float(candidate_scores[j]), name)
                    for rank, j in enumerate(order))
    return {"ndcg_at_10": float(np.mean(ndcgs)), "recall_at_10": float(np.mean(recalls)),
            "evaluated_users": len(users), "top10_rows": rows}


def write_gzip(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["userId", "movieId", "rank", "score", "model_version"])
        writer.writerows(rows)


def main():
    started = time.perf_counter()
    cfg = json.loads((STAGE / "config.json").read_text(encoding="utf-8"))
    protocol, events, candidate_sha = load_inputs(cfg)
    users = protocol["user_ids"]
    transition_t, movie_ids, user_index, n_nodes, degree = graph_matrix(users, events)
    graph_movie_positions = {int(movie): i for i, movie in enumerate(movie_ids)}
    als = read_top10(ALS_TOP10)
    movies = pl.read_csv(BASE / "data/raw/movies.csv", schema_overrides={"movieId": pl.Int32})
    movie_genres = {int(row["movieId"]): row["genres"] for row in
                    movies.select("movieId", "genres").iter_rows(named=True)}
    user_genre_counts = defaultdict(lambda: defaultdict(int))
    for user_value, movie_value in events.select("userId", "movieId").iter_rows():
        for genre in movie_genres.get(int(movie_value), "").split("|"):
            if genre and genre != "(no genres listed)":
                user_genre_counts[int(user_value)][genre] += 1
    graph_rankings, fusion_rankings = [], []
    ppr_iterations, ppr_deltas, per_user, fused_per_user = [], [], [], []
    graph_rows, fusion_rows = [], []
    ppr_started = time.perf_counter()
    for index, user_value in enumerate(users):
        user = int(user_value)
        vector, iterations, delta = ppr(user_index[user], transition_t,
                                        cfg["ppr"]["restart_probability"],
                                        cfg["ppr"]["max_iterations"], cfg["ppr"]["l1_tolerance"])
        ppr_iterations.append(iterations)
        ppr_deltas.append(delta)
        pos = protocol["positive_items"][protocol["positive_offsets"][index]:protocol["positive_offsets"][index + 1]]
        candidate_ids = np.concatenate((pos, protocol["negatives"][index])).astype(np.int32, copy=False)
        candidate_graph_scores = np.fromiter(
            (vector[len(users) + graph_movie_positions[int(movie)]].item()
             if int(movie) in graph_movie_positions else 0.0 for movie in candidate_ids),
            dtype=np.float64, count=len(candidate_ids))
        graph_order = rank_ids(candidate_ids, candidate_graph_scores)
        graph_ordering = {int(candidate_ids[j]): rank + 1 for rank, j in enumerate(graph_order)}
        graph_rankings.append({int(candidate_ids[j]): float(candidate_graph_scores[j]) for j in graph_order})
        graph_result = evaluate({"user_ids": users[index:index+1], "positive_items": protocol["positive_items"],
                                 "positive_offsets": np.asarray([protocol["positive_offsets"][index], protocol["positive_offsets"][index+1]]),
                                 "negatives": protocol["negatives"][index:index+1]}, users[index:index+1],
                                [graph_rankings[-1]], "graph-ppr")
        per_user.append((user, graph_result["ndcg_at_10"], graph_result["recall_at_10"], iterations, delta))
        graph_rows.extend(graph_result["top10_rows"])

        als_ranks = als.get(user, {})
        fused = defaultdict(float)
        for movie, rank in graph_ordering.items():
            fused[movie] += 1.0 / rank
        for movie, rank in als_ranks.items():
            fused[movie] += 1.0 / rank
        fusion_rankings.append(dict(fused))
        fused_result = evaluate({"user_ids": users[index:index+1], "positive_items": protocol["positive_items"],
                                 "positive_offsets": np.asarray([protocol["positive_offsets"][index], protocol["positive_offsets"][index+1]]),
                                 "negatives": protocol["negatives"][index:index+1]}, users[index:index+1],
                                [fusion_rankings[-1]], "graph-als-rrf")
        fused_per_user.append((user, fused_result["ndcg_at_10"], fused_result["recall_at_10"]))
        fusion_rows.extend(fused_result["top10_rows"])
        if (index + 1) % 500 == 0:
            print(json.dumps({"users_completed": index + 1, "total": len(users)}, ensure_ascii=False), flush=True)
    ppr_seconds = time.perf_counter() - ppr_started

    als_ndcgs, als_recalls = [], []
    for index, user_value in enumerate(users):
        user = int(user_value)
        pos = protocol["positive_items"][protocol["positive_offsets"][index]:protocol["positive_offsets"][index + 1]]
        candidate_ids = np.concatenate((pos, protocol["negatives"][index])).astype(np.int32, copy=False)
        ranks = als.get(user, {})
        scores = np.fromiter((1.0 / ranks[movie] if int(movie) in ranks else 0.0 for movie in candidate_ids),
                             dtype=np.float64, count=len(candidate_ids))
        order = np.lexsort((candidate_ids, -scores))[:K]
        relevance = (order < len(pos)).astype(np.uint8)
        ndcg, recall = metric(relevance, len(pos))
        als_ndcgs.append(ndcg)
        als_recalls.append(recall)

    graph_metrics = {"ndcg_at_10": float(np.mean([row[1] for row in per_user])),
                     "recall_at_10": float(np.mean([row[2] for row in per_user])),
                     "evaluated_users": len(users), "top10_rows": graph_rows}
    fused_metrics = {"ndcg_at_10": float(np.mean([row[1] for row in fused_per_user])),
                     "recall_at_10": float(np.mean([row[2] for row in fused_per_user])),
                     "evaluated_users": len(users), "top10_rows": fusion_rows}
    als_mean = float(np.mean(als_ndcgs))
    output = STAGE / "data/processed"
    logs = STAGE / "logs"
    for directory in (output, logs):
        directory.mkdir(parents=True, exist_ok=True)
    graph_path = output / "graph_ppr_top10.csv.gz"
    fused_path = output / "graph_als_fused_top10.csv.gz"
    node_attributes_path = output / "graph_node_attributes.csv.gz"
    (STAGE / "curves").mkdir(parents=True, exist_ok=True)
    write_gzip(graph_path, graph_rows)
    write_gzip(fused_path, fusion_rows)
    with gzip.open(node_attributes_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_type", "node_id", "degree", "genres_or_profile"])
        for user, position in user_index.items():
            profile = ";".join(f"{genre}={count}" for genre, count in sorted(user_genre_counts[user].items()))
            writer.writerow(["user", user, int(degree[position]), profile])
        for movie, position in graph_movie_positions.items():
            writer.writerow(["movie", movie, int(degree[len(users) + position]), movie_genres.get(movie, "")])
    baseline_scores = {}
    with gzip.open(BASE / "data/processed/baseline_per_user.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            baseline_scores[int(row["userId"])] = float(row["ndcg_at_10"])
    matching_baseline = float(np.mean([baseline_scores[int(user)] for user in users]))
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    labels = ["Popular baseline", "Stage 3 ALS", "Graph PPR", "ALS + PPR"]
    ndcg_values = [matching_baseline, als_mean, graph_metrics["ndcg_at_10"], fused_metrics["ndcg_at_10"]]
    baseline_recalls = {}
    with gzip.open(BASE / "data/processed/baseline_per_user.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        baseline_recalls = {int(row["userId"]): float(row["recall_at_10"]) for row in csv.DictReader(stream)}
    recall_values = [float(np.mean([baseline_recalls[int(user)] for user in users])),
                     float(np.mean(als_recalls)), graph_metrics["recall_at_10"], fused_metrics["recall_at_10"]]
    x = np.arange(len(labels))
    ax.plot(x, ndcg_values, marker="o", color="#32679a", label="NDCG@10")
    ax.plot(x, recall_values, marker="s", color="#3b8264", label="Recall@10")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Macro metric")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Graph PPR and ALS on shared 5,000-user set")
    fig.savefig(STAGE / "curves/graph_model_comparison.png", dpi=160)
    plt.close(fig)
    report = {
        "experiment": "phase5-bipartite-ppr-v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": cfg["dataset"],
        "input_candidate_sha256": candidate_sha,
        "graph": {"users": len(users), "movie_nodes": len(movie_ids), "nodes": n_nodes,
                  "unique_positive_edges": int(events.height), "directed_adjacency_nnz": int(transition_t.nnz),
                  "degree_max": int(degree.max()), "degree_median": float(np.median(degree[degree > 0])),
                  "user_profile_attribute_dimension": len({genre for profile in user_genre_counts.values() for genre in profile}),
                  "movie_genre_attribute_rows": len(movie_ids), "node_attributes_file": node_attributes_path.relative_to(STAGE).as_posix()},
        "ppr": {"restart_probability": cfg["ppr"]["restart_probability"],
                "max_iterations": cfg["ppr"]["max_iterations"], "l1_tolerance": cfg["ppr"]["l1_tolerance"],
                "iterations_mean": float(np.mean(ppr_iterations)), "iterations_p95": float(np.quantile(ppr_iterations, .95)),
                "final_l1_delta_max": float(max(ppr_deltas)), "query_seconds": ppr_seconds,
                "users_per_second": len(users) / ppr_seconds,
                "ndcg_at_10": graph_metrics["ndcg_at_10"], "recall_at_10": graph_metrics["recall_at_10"],
                "matching_user_baseline_ndcg_at_10": matching_baseline,
                "same_workset_stage3_als_ndcg_at_10": als_mean,
                "same_workset_stage3_als_recall_at_10": float(np.mean(als_recalls)),
                "relative_ndcg_change_vs_matching_baseline": graph_metrics["ndcg_at_10"] / matching_baseline - 1.0,
                "relative_ndcg_change_vs_stage3_als": graph_metrics["ndcg_at_10"] / als_mean - 1.0,
                "top10_file": graph_path.relative_to(STAGE).as_posix()},
        "graph_als_rank_fusion": {"fusion": cfg["fusion"], "ndcg_at_10": fused_metrics["ndcg_at_10"],
                                  "recall_at_10": fused_metrics["recall_at_10"],
                                  "relative_ndcg_change_vs_matching_baseline": fused_metrics["ndcg_at_10"] / matching_baseline - 1.0,
                                  "top10_file": fused_path.relative_to(STAGE).as_posix()},
        "runtime": {"deployment_scope": "single Windows host, scipy sparse matrix PPR; no distributed graph engine",
                    "graph_build_seconds_included_in_total": True,
                    "elapsed_seconds": time.perf_counter() - started,
                    "python": __import__("platform").python_version(), "numpy": np.__version__,
                    "scipy": __import__("scipy").__version__},
        "acceptance": {"reasonable_negative_result_documented": True,
                       "ndcg_above_stage3_als_on_same_workset": graph_metrics["ndcg_at_10"] > als_mean,
                       "ppr_residual_within_tolerance": max(ppr_deltas) <= cfg["ppr"]["l1_tolerance"],
                       "multi_node_graph_runtime_verified": False},
    }
    report["outputs"] = {"graph_top10_sha256": sha256(graph_path), "fused_top10_sha256": sha256(fused_path),
                          "node_attributes_sha256": sha256(node_attributes_path)}
    (logs / "graph_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with gzip.open(output / "graph_per_user.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["userId", "ndcg_at_10", "recall_at_10", "iterations", "final_l1_delta"])
        writer.writerows(per_user)
    with gzip.open(output / "graph_als_fused_per_user.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["userId", "ndcg_at_10", "recall_at_10"])
        writer.writerows(fused_per_user)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

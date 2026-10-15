"""Run a deterministic sparse ItemCF versus Random Projection LSH benchmark.

The benchmark deliberately uses a documented 5,000-user working set because a
full 87,585 x 87,585 exact similarity matrix is not a safe single-node artifact.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import platform
import time
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import psutil
import scipy
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE = SCRIPT_DIR.parent
ROOT = STAGE.parent
sys_path = str(STAGE / "../01_data_baseline/scripts")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
from core import metrics, rank_items
from download import sha256
from prepare import save_json


def load_config():
    return json.loads((STAGE / "config.json").read_text(encoding="utf-8"))


def item_ids():
    return pl.read_csv(STAGE / "../01_data_baseline/data/raw/movies.csv", columns=["movieId"],
                       schema_overrides={"movieId": pl.Int32})["movieId"].sort().to_numpy()


def benchmark_users_and_candidates(cfg):
    candidate_path = STAGE / "../01_data_baseline/data/processed/candidates.npz"
    with np.load(candidate_path) as z:
        all_users = z["user_ids"]
        take = min(cfg["benchmark_users"], len(all_users))
        return {key: z[key][:take] if key not in {"positive_items", "positive_offsets"} else z[key]
                for key in z.files} | {"candidate_path": candidate_path, "take": take}


def load_positive_train(user_ids):
    train_path = STAGE / "../01_data_baseline/data/processed/train.parquet"
    user_frame = pl.DataFrame({"userId": np.asarray(user_ids, dtype=np.int32)})
    frame = (pl.read_parquet(train_path, columns=["userId", "movieId", "rating"], row_index_name=None)
             .filter(pl.col("rating") >= 4.0)
             .join(user_frame, on="userId", how="inner"))
    return frame.sort(["userId", "movieId"])


def build_matrix(frame, user_ids, catalog):
    user_pos = {int(u): i for i, u in enumerate(user_ids.tolist())}
    item_pos = {int(m): i for i, m in enumerate(catalog.tolist())}
    rows = np.fromiter((user_pos[int(x)] for x in frame["userId"].to_numpy()), dtype=np.int32)
    cols = np.fromiter((item_pos[int(x)] for x in frame["movieId"].to_numpy()), dtype=np.int32)
    data = np.ones(len(rows), dtype=np.float32)
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(catalog)), dtype=np.float32)
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    return matrix, user_pos, item_pos


def exact_similarity(matrix):
    started = time.perf_counter()
    cooc = (matrix.T @ matrix).tocsr()
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    inv_norm = np.zeros_like(counts, dtype=np.float64)
    valid = counts > 0
    inv_norm[valid] = 1.0 / np.sqrt(counts[valid])
    sim = cooc.astype(np.float32).multiply(inv_norm[:, None]).multiply(inv_norm[None, :]).tocsr()
    sim.eliminate_zeros()
    return sim, {"seconds": time.perf_counter() - started, "nnz": int(sim.nnz),
                 "matrix_bytes_estimate": int(sim.data.nbytes + sim.indices.nbytes + sim.indptr.nbytes),
                 "positive_item_count": int(valid.sum())}


def user_similarity(matrix):
    """Exact cosine UserCF on the same positive binary training matrix."""
    started = time.perf_counter()
    cooc = (matrix @ matrix.T).tocsr()
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    counts = np.asarray(matrix.sum(axis=1)).ravel()
    inv_norm = np.zeros_like(counts, dtype=np.float64)
    valid = counts > 0
    inv_norm[valid] = 1.0 / np.sqrt(counts[valid])
    sim = cooc.astype(np.float32).multiply(inv_norm[:, None]).multiply(inv_norm[None, :]).tocsr()
    sim.eliminate_zeros()
    return sim, {"seconds": time.perf_counter() - started, "nnz": int(sim.nnz),
                 "matrix_bytes_estimate": int(sim.data.nbytes + sim.indices.nbytes + sim.indptr.nbytes),
                 "active_user_count": int(valid.sum())}


def top_neighbors(row, k):
    if row.nnz == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    ids, values = row.indices, row.data
    order = np.lexsort((ids, -values))[:k]
    return ids[order].astype(np.int32), values[order].astype(np.float32)


def build_lsh(matrix, cfg):
    started = time.perf_counter()
    tables, planes, seed = cfg["lsh"]["tables"], cfg["lsh"]["planes"], cfg["lsh"]["seed"]
    n_users, n_items = matrix.shape
    weights = np.random.default_rng(seed).choice(np.array([-1, 1], dtype=np.int8),
                                                  size=(tables, n_users, planes))
    buckets = []
    signatures = []
    active_items = matrix.getnnz(axis=0) > 0
    bit_weights = (1 << np.arange(planes, dtype=np.uint32))
    for table in range(tables):
        projection = matrix.T @ weights[table].astype(np.float32)
        bits = (np.asarray(projection) >= 0).astype(np.uint32) @ bit_weights
        signatures.append(bits)
        table_buckets = defaultdict(list)
        for item, key in enumerate(bits.tolist()):
            if active_items[item]:
                table_buckets[int(key)].append(item)
        buckets.append(table_buckets)
    pool_sizes = []
    neighbor_rows, neighbor_cols, neighbor_values = [], [], []
    exact_sim = _CURRENT_SIM
    k = cfg["neighbor_k"]
    for item in range(n_items):
        pool = set()
        for table, table_signatures in zip(buckets, signatures):
            pool.update(table[int(table_signatures[item])])
        pool.discard(item)
        pool_sizes.append(len(pool))
        if pool:
            candidates = np.fromiter(pool, dtype=np.int32)
            row = exact_sim[item, candidates]
            values = np.asarray(row.toarray()).ravel()
            keep = values > 0
            candidates, values = candidates[keep], values[keep]
            if len(candidates):
                order = np.lexsort((candidates, -values))[:k]
                neighbor_rows.extend([item] * len(order)); neighbor_cols.extend(candidates[order].tolist()); neighbor_values.extend(values[order].tolist())
    approx = sparse.csr_matrix((np.asarray(neighbor_values, dtype=np.float32),
                                (np.asarray(neighbor_rows, dtype=np.int32), np.asarray(neighbor_cols, dtype=np.int32))),
                               shape=(n_items, n_items), dtype=np.float32)
    return approx, buckets, {"seconds": time.perf_counter() - started, "tables": tables, "planes": planes,
                             "mean_candidate_pool": float(np.mean(pool_sizes)), "p50_candidate_pool": float(np.quantile(pool_sizes, .5)),
                             "p95_candidate_pool": float(np.quantile(pool_sizes, .95)), "max_candidate_pool": int(max(pool_sizes)),
                             "neighbor_nnz": int(approx.nnz)}


def evaluate_recommender(sim, user_frame, candidates, user_ids, catalog, cfg, model_name):
    item_pos = {int(m): i for i, m in enumerate(catalog.tolist())}
    history = defaultdict(list)
    for u, m in zip(user_frame["userId"].to_numpy(), user_frame["movieId"].to_numpy()):
        history[int(u)].append(item_pos[int(m)])
    pos_items, pos_offsets, negatives = candidates["positive_items"], candidates["positive_offsets"], candidates["negatives"]
    ndcgs, recalls, latencies, output_rows = [], [], [], []
    candidate_by_user = {int(u): i for i, u in enumerate(candidates["user_ids"])}
    started = time.perf_counter()
    for i, user in enumerate(user_ids):
        candidate_index = candidate_by_user[int(user)]
        pos = pos_items[pos_offsets[candidate_index]:pos_offsets[candidate_index + 1]]
        cand_movie = np.concatenate([pos, negatives[candidate_index]]).astype(np.int32)
        cand_index = np.fromiter((item_pos[int(m)] for m in cand_movie), dtype=np.int32)
        rank_start = time.perf_counter()
        if history[int(user)]:
            scores = np.asarray(sim[history[int(user)]].sum(axis=0)).ravel()[cand_index]
        else:
            scores = np.zeros(len(cand_index), dtype=np.float32)
        ranked = rank_items(cand_movie, scores, cfg["k"])
        relevance = (ranked < len(pos)).astype(np.uint8)
        ndcg, recall = metrics(relevance, len(pos), cfg["k"])
        ndcgs.append(ndcg); recalls.append(recall); latencies.append((time.perf_counter() - rank_start) * 1000)
        for rank, idx in enumerate(ranked, 1):
            output_rows.append((int(user), int(cand_movie[idx]), rank, float(scores[idx]), model_name))
    elapsed = time.perf_counter() - started
    return {"model": model_name, "ndcg_at_10": float(np.mean(ndcgs)), "recall_at_10": float(np.mean(recalls)),
            "evaluated_users": len(ndcgs), "candidate_mean": float(np.mean([len(pos_items[pos_offsets[i]:pos_offsets[i+1]]) + len(negatives[i]) for i in range(len(user_ids))])),
            "query_seconds": elapsed, "users_per_second": len(ndcgs) / elapsed,
            "query_latency_ms": {"mean": float(np.mean(latencies)), "p50": float(np.quantile(latencies, .5)), "p95": float(np.quantile(latencies, .95)), "p99": float(np.quantile(latencies, .99))},
            "top10_rows": output_rows, "ndcgs": np.asarray(ndcgs), "recalls": np.asarray(recalls)}


def evaluate_usercf(sim, matrix, candidates, user_ids, catalog, cfg):
    """Score each frozen candidate set from the top similar training users."""
    pos_items, pos_offsets, negatives = candidates["positive_items"], candidates["positive_offsets"], candidates["negatives"]
    ndcgs, recalls, latencies, output_rows = [], [], [], []
    candidate_by_user = {int(u): i for i, u in enumerate(candidates["user_ids"])}
    started = time.perf_counter()
    for user_index, user in enumerate(user_ids):
        candidate_index = candidate_by_user[int(user)]
        pos = pos_items[pos_offsets[candidate_index]:pos_offsets[candidate_index + 1]]
        cand_movie = np.concatenate([pos, negatives[candidate_index]]).astype(np.int32)
        cand_index = np.searchsorted(catalog, cand_movie).astype(np.int32)
        rank_start = time.perf_counter()
        neighbor_ids, neighbor_weights = top_neighbors(sim.getrow(user_index), cfg["neighbor_k"])
        if len(neighbor_ids):
            # Only materialize the small candidate slice, avoiding a dense user x item score vector.
            scores = np.asarray(matrix[neighbor_ids][:, cand_index].T @ neighbor_weights).ravel()
        else:
            scores = np.zeros(len(cand_index), dtype=np.float32)
        ranked = rank_items(cand_movie, scores, cfg["k"])
        relevance = (ranked < len(pos)).astype(np.uint8)
        ndcg, recall = metrics(relevance, len(pos), cfg["k"])
        ndcgs.append(ndcg); recalls.append(recall); latencies.append((time.perf_counter() - rank_start) * 1000)
        for rank, idx in enumerate(ranked, 1):
            output_rows.append((int(user), int(cand_movie[idx]), rank, float(scores[idx]), "usercf_exact"))
    elapsed = time.perf_counter() - started
    return {"model": "usercf_exact", "ndcg_at_10": float(np.mean(ndcgs)), "recall_at_10": float(np.mean(recalls)),
            "evaluated_users": len(ndcgs), "candidate_mean": float(np.mean([len(pos_items[pos_offsets[i]:pos_offsets[i+1]]) + len(negatives[i]) for i in range(len(user_ids))])),
            "query_seconds": elapsed, "users_per_second": len(ndcgs) / elapsed,
            "query_latency_ms": {"mean": float(np.mean(latencies)), "p50": float(np.quantile(latencies, .5)), "p95": float(np.quantile(latencies, .95)), "p99": float(np.quantile(latencies, .99))},
            "top10_rows": output_rows, "ndcgs": np.asarray(ndcgs), "recalls": np.asarray(recalls)}


def write_top10(rows, path):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["userId", "movieId", "rank", "score", "model"]); writer.writerows(rows)


def neighbor_overlap(exact_sim, approx_sim, catalog, k):
    recalls, jaccards = [], []
    for item in range(len(catalog)):
        exact_ids, _ = top_neighbors(exact_sim.getrow(item), k)
        approx_ids, _ = top_neighbors(approx_sim.getrow(item), k)
        if len(exact_ids):
            exact_set, approx_set = set(exact_ids.tolist()), set(approx_ids.tolist())
            inter = len(exact_set & approx_set)
            union = len(exact_set | approx_set)
            recalls.append(inter / len(exact_set)); jaccards.append(inter / union if union else 1.0)
    return float(np.mean(recalls)), float(np.mean(jaccards)), len(recalls)


def main(run_sweep=False):
    global _CURRENT_SIM
    cfg = load_config(); started = time.perf_counter()
    processed = STAGE / "data/processed"; logs = STAGE / "logs"; curves = STAGE / "curves"
    processed.mkdir(exist_ok=True); logs.mkdir(exist_ok=True); curves.mkdir(exist_ok=True)
    candidates = benchmark_users_and_candidates(cfg)
    user_ids = candidates["user_ids"]
    catalog = item_ids()
    train = load_positive_train(user_ids)
    matrix, user_pos, item_pos = build_matrix(train, user_ids, catalog)
    _CURRENT_SIM, exact_build = exact_similarity(matrix)
    exact_eval = evaluate_recommender(_CURRENT_SIM, train, candidates, user_ids, catalog, cfg, "itemcf_exact")
    user_sim, user_build = user_similarity(matrix)
    user_eval = evaluate_usercf(user_sim, matrix, candidates, user_ids, catalog, cfg)
    lsh_sim, buckets, lsh_build = build_lsh(matrix, cfg); del buckets
    lsh_eval = evaluate_recommender(lsh_sim, train, candidates, user_ids, catalog, cfg, "itemcf_lsh")
    neighbor_recall, neighbor_jaccard, neighbor_items = neighbor_overlap(_CURRENT_SIM, lsh_sim, catalog, cfg["neighbor_k"])
    exact_metrics = {k: v for k, v in exact_eval.items() if k not in {"top10_rows", "ndcgs", "recalls"}}
    user_metrics = {k: v for k, v in user_eval.items() if k not in {"top10_rows", "ndcgs", "recalls"}}
    lsh_metrics = {k: v for k, v in lsh_eval.items() if k not in {"top10_rows", "ndcgs", "recalls"}}
    for result in [exact_metrics, user_metrics, lsh_metrics]:
        result["benchmark_users"] = len(user_ids); result["positive_train_interactions"] = int(matrix.nnz)
        result["stage_one_candidates_sha256"] = sha256(candidates["candidate_path"])
        result["selection_note"] = cfg["benchmark_user_selection"]
    comparison = {"exact_build": exact_build, "usercf_build": user_build, "lsh_build": lsh_build,
                  "neighbor_recall_at_50": neighbor_recall, "neighbor_jaccard_at_50": neighbor_jaccard,
                  "neighbor_items_compared": neighbor_items,
                  "lsh_query_speedup_vs_exact": exact_build["seconds"] / lsh_build["seconds"],
                  "recommendation_query_speedup_lsh_vs_exact": exact_eval["query_seconds"] / lsh_eval["query_seconds"],
                  "lsh_ndcg_relative_change": lsh_eval["ndcg_at_10"] / exact_eval["ndcg_at_10"] - 1,
                  "lsh_recall_relative_change": lsh_eval["recall_at_10"] / exact_eval["recall_at_10"] - 1,
                  "neighbor_target": cfg["neighbor_k"], "lsh_acceptance_rule": "LSH neighbor recall >= 0.90 and query speedup >= 10x are required by Word; these are measured, not assumed.",
                  "scope_note": cfg["full_population_note"]}
    sweep = []
    if run_sweep:
        for tables, planes in cfg.get("lsh_grid", [[4, 16], [8, 16], [16, 12], [32, 10]]):
            sweep_cfg = dict(cfg)
            sweep_cfg["lsh"] = dict(cfg["lsh"], tables=tables, planes=planes)
            if tables == cfg["lsh"]["tables"] and planes == cfg["lsh"]["planes"]:
                sim, build, evaluation = lsh_sim, lsh_build, lsh_eval
                nr, nj, ni = neighbor_recall, neighbor_jaccard, neighbor_items
            else:
                sim, _, build = build_lsh(matrix, sweep_cfg)
                evaluation = evaluate_recommender(sim, train, candidates, user_ids, catalog, sweep_cfg, f"itemcf_lsh_{tables}x{planes}")
                nr, nj, ni = neighbor_overlap(_CURRENT_SIM, sim, catalog, cfg["neighbor_k"])
            sweep.append({"tables": tables, "planes": planes, "build": build, "neighbor_recall_at_50": nr,
                          "neighbor_jaccard_at_50": nj, "neighbor_items_compared": ni,
                          "ndcg_at_10": evaluation["ndcg_at_10"], "recall_at_10": evaluation["recall_at_10"],
                          "query_latency_ms": evaluation["query_latency_ms"], "query_seconds": evaluation["query_seconds"],
                          "recommendation_speedup_vs_exact": exact_eval["query_seconds"] / evaluation["query_seconds"],
                          "target_normalized_score": 0.5 * min(nr / 0.90, 1.0) + 0.5 * min((exact_eval["query_seconds"] / evaluation["query_seconds"]) / 10.0, 1.0)})
        if sweep:
            selected = max(sweep, key=lambda x: (x["target_normalized_score"], x["neighbor_recall_at_50"], -x["build"]["mean_candidate_pool"]))
            comparison["parameter_selection"] = {
                "criterion": "0.5*min(neighbor_recall_at_50/0.90,1) + 0.5*min(recommendation_speedup_vs_exact/10,1); ties prefer higher recall then smaller candidate pool",
                "recommended_tables": selected["tables"],
                "recommended_planes": selected["planes"],
                "recommended_target_normalized_score": selected["target_normalized_score"],
                "recommended_mean_candidate_pool": selected["build"]["mean_candidate_pool"],
                "recommended_neighbor_recall_at_50": selected["neighbor_recall_at_50"],
                "recommended_speedup": selected["recommendation_speedup_vs_exact"],
                "recommended_status": "best tested tradeoff; Word thresholds still unmet",
                "efficiency_first_config": {"tables": 4, "planes": 16, "mean_candidate_pool": sweep[0]["build"]["mean_candidate_pool"], "neighbor_recall_at_50": sweep[0]["neighbor_recall_at_50"], "speedup": sweep[0]["recommendation_speedup_vs_exact"]}
            }
    comparison["lsh_parameter_sweep"] = sweep
    if sweep:
        fig2, ax2 = plt.subplots(figsize=(7, 5), layout="constrained")
        x = np.asarray([row["recommendation_speedup_vs_exact"] for row in sweep], dtype=float)
        y = np.asarray([row["neighbor_recall_at_50"] for row in sweep], dtype=float)
        sizes = np.asarray([row["build"]["mean_candidate_pool"] for row in sweep], dtype=float)
        sizes = 80 + 320 * (sizes - sizes.min()) / max(float(sizes.max() - sizes.min()), 1.0)
        ax2.plot(x, y, color="#9aa7b8", linewidth=1.5, zorder=1)
        ax2.scatter(x, y, s=sizes, c=np.arange(len(sweep)), cmap="viridis", edgecolor="white", linewidth=1.0, zorder=2)
        for row in sweep:
            ax2.annotate(f"{row['tables']}×{row['planes']}", (row["recommendation_speedup_vs_exact"], row["neighbor_recall_at_50"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
        ax2.axhline(0.90, color="#c44e52", linestyle="--", linewidth=1.0, label="Word recall target 0.90")
        ax2.axvline(10.0, color="#4c72b0", linestyle="--", linewidth=1.0, label="Word speed target 10×")
        ax2.set_xlabel("Recommendation query speedup vs exact ItemCF (×; higher is faster)")
        ax2.set_ylabel("Neighbor Recall@50")
        ax2.set_title("LSH precision efficiency tradeoff")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="lower right", fontsize=8)
        fig2.savefig(curves / "lsh_precision_efficiency.png", dpi=180)
        plt.close(fig2)
    write_top10(exact_eval["top10_rows"], processed / "itemcf_exact_top10.csv.gz")
    write_top10(user_eval["top10_rows"], processed / "usercf_exact_top10.csv.gz")
    write_top10(lsh_eval["top10_rows"], processed / "itemcf_lsh_top10.csv.gz")
    save_json(logs / "itemcf_metrics.json", exact_metrics)
    save_json(logs / "usercf_metrics.json", user_metrics)
    save_json(logs / "lsh_metrics.json", lsh_metrics)
    save_json(logs / "similarity_recall_metrics.json", comparison)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    axes[0].bar(["Most Popular\n(stage 1)", "ItemCF exact", "UserCF exact", "ItemCF LSH"], [cfg["stage_one_baseline_ndcg_at_10"], exact_eval["ndcg_at_10"], user_eval["ndcg_at_10"], lsh_eval["ndcg_at_10"]], color=["#8b9db6", "#3568a4", "#8063aa", "#438763"]); axes[0].set_ylim(0, 1); axes[0].set_ylabel("NDCG@10")
    axes[1].bar(["ItemCF", "UserCF", "LSH"], [exact_eval["query_latency_ms"]["mean"], user_eval["query_latency_ms"]["mean"], lsh_eval["query_latency_ms"]["mean"]], color=["#3568a4", "#8063aa", "#438763"]); axes[1].set_ylabel("Mean query ms")
    axes[2].bar(["Recall@50", "Jaccard@50"], [comparison["neighbor_recall_at_50"], comparison["neighbor_jaccard_at_50"]], color=["#3568a4", "#8063aa"]); axes[2].set_ylim(0, 1)
    fig.savefig(curves / "phase2_metrics.png", dpi=160); plt.close(fig)
    save_json(logs / "phase2_environment.json", {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "polars": pl.__version__, "scipy": scipy.__version__, "logical_cpus": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total, "seconds_total": time.perf_counter() - started, "rss_bytes_end": psutil.Process().memory_info().rss, "scope": cfg["full_population_note"]})
    verification = {"exact_similarity_sparse": bool(_CURRENT_SIM.nnz > 0), "user_similarity_sparse": bool(user_sim.nnz > 0), "lsh_similarity_sparse": bool(lsh_sim.nnz > 0), "all_top10_rows_exact": len(exact_eval["top10_rows"]) == len(user_ids) * 10, "all_top10_rows_usercf": len(user_eval["top10_rows"]) == len(user_ids) * 10, "all_top10_rows_lsh": len(lsh_eval["top10_rows"]) == len(user_ids) * 10, "same_candidate_user_population": exact_eval["evaluated_users"] == lsh_eval["evaluated_users"] == user_eval["evaluated_users"], "no_test_labels_in_fit": True, "raw_data_unchanged": True, "teacher_script_alignment": "unverified; teacher scripts not supplied"}
    save_json(logs / "verification.json", verification)
    print(json.dumps({"itemcf": exact_metrics, "usercf": user_metrics, "lsh": lsh_metrics, "comparison": comparison, "verification": verification}, ensure_ascii=False, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x), flush=True)
    if not all(verification.values()):
        raise SystemExit("Stage 2 verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="also evaluate the documented LSH parameter grid")
    main(run_sweep=parser.parse_args().sweep)

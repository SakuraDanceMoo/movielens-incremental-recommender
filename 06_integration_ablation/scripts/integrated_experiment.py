"""Evaluate an integrated multi-source recommender and component ablations."""
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

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "06_integration_ablation"
BASE = ROOT / "01_data_baseline"
SOURCES = {
    "lsh": ROOT / "02_similarity_recall/data/processed/itemcf_lsh_top10.csv.gz",
    "als": ROOT / "03_distributed_recommendation/data/processed/als_top10.csv.gz",
    "graph": ROOT / "05_graph_recommendation/data/processed/graph_ppr_top10.csv.gz",
    "incremental": ROOT / "04_streaming_incremental/data/processed/stream_top10.csv.gz",
}
VARIANTS = {
    "full": ("lsh", "als", "graph", "incremental"),
    "without_incremental": ("lsh", "als", "graph"),
    "without_graph": ("lsh", "als", "incremental"),
    "without_lsh": ("als", "graph", "incremental"),
    "als_only": ("als",),
}
K = 10


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_top10(path, users_set):
    result = defaultdict(dict)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            user = int(row["userId"])
            if user in users_set:
                result[user][int(row["movieId"])] = int(row["rank"])
    return result


def metric(relevance, positives):
    discounts = 1.0 / np.log2(np.arange(2, len(relevance) + 2))
    ideal = np.sum(1.0 / np.log2(np.arange(2, min(K, positives) + 2)))
    return float(np.dot(relevance, discounts) / ideal), int(np.count_nonzero(relevance)) / int(positives)


def main():
    started = time.perf_counter()
    cfg = json.loads((STAGE / "config.json").read_text(encoding="utf-8"))
    candidate_path = BASE / "data/processed/candidates.npz"
    with np.load(candidate_path) as archive:
        count = min(cfg["users"], len(archive["user_ids"]))
        users = archive["user_ids"][:count]
        positives, offsets = archive["positive_items"], archive["positive_offsets"]
        negatives = archive["negatives"][:count]
    users_set = set(map(int, users))
    source_rankings = {name: read_top10(path, users_set) for name, path in SOURCES.items()}

    baseline = {}
    with gzip.open(BASE / "data/processed/baseline_per_user.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            user = int(row["userId"])
            if user in users_set:
                baseline[user] = (float(row["ndcg_at_10"]), float(row["recall_at_10"]))
    baseline_ndcg = float(np.mean([baseline[int(user)][0] for user in users]))
    baseline_recall = float(np.mean([baseline[int(user)][1] for user in users]))

    all_results = {}
    detailed_rows = []
    all_top_rows = []
    full_latencies = []
    full_candidate_counts = []
    full_candidate_recalls = []
    full_output = []
    for variant, active_sources in VARIANTS.items():
        ndcgs, recalls = [], []
        user_rows = []
        latencies = []
        candidate_counts = []
        candidate_recalls = []
        top_rows = []
        for index, user_value in enumerate(users):
            user = int(user_value)
            candidate_items = set()
            for source in active_sources:
                candidate_items.update(source_rankings[source].get(user, {}).keys())
            candidate_ids = np.asarray(sorted(candidate_items), dtype=np.int32)
            started_rank = time.perf_counter_ns()
            fused_scores = np.zeros(len(candidate_ids), dtype=np.float64)
            for source in active_sources:
                ranks = source_rankings[source].get(user, {})
                fused_scores += np.fromiter((1.0 / ranks[item] if int(item) in ranks else 0.0
                                             for item in candidate_ids), dtype=np.float64,
                                            count=len(candidate_ids))
            order = np.lexsort((candidate_ids, -fused_scores))[:K]
            latency_ms = (time.perf_counter_ns() - started_rank) / 1e6
            pos = positives[offsets[index]:offsets[index + 1]]
            pos_set = set(map(int, pos))
            relevance = np.asarray([int(int(candidate_ids[j]) in pos_set) for j in order], dtype=np.uint8)
            if len(relevance) < K:
                relevance = np.pad(relevance, (0, K - len(relevance)))
            ndcg, recall = metric(relevance, len(pos))
            candidate_recall = len(candidate_items & pos_set) / len(pos)
            ndcgs.append(ndcg)
            recalls.append(recall)
            user_rows.append((user, ndcg, recall, len(candidate_items), candidate_recall, latency_ms))
            latencies.append(latency_ms)
            candidate_counts.append(len(candidate_items))
            candidate_recalls.append(candidate_recall)
            top_rows.extend((user, int(candidate_ids[j]), rank + 1, float(fused_scores[j]), variant)
                            for rank, j in enumerate(order))
        result = {
            "active_sources": list(active_sources),
            "evaluated_users": len(users),
            "candidate_pool_mean": float(np.mean(candidate_counts)),
            "candidate_pool_p95": float(np.quantile(candidate_counts, .95)),
            "candidate_positive_recall": float(np.mean(candidate_recalls)),
            "ndcg_at_10": float(np.mean(ndcgs)),
            "recall_at_10": float(np.mean(recalls)),
            "ndcg_relative_change_vs_matching_baseline": float(np.mean(ndcgs) / baseline_ndcg - 1.0),
            "recall_relative_change_vs_matching_baseline": float(np.mean(recalls) / baseline_recall - 1.0),
            "rerank_latency_ms": {"mean": float(np.mean(latencies)), "p50": float(np.quantile(latencies, .5)),
                                  "p95": float(np.quantile(latencies, .95)), "p99": float(np.quantile(latencies, .99)),
                                  "max": float(np.max(latencies))},
            "rerank_users_per_second": float(len(users) / (sum(latencies) / 1000.0)),
        }
        all_results[variant] = result
        detailed_rows.extend((variant, *row) for row in user_rows)
        all_top_rows.extend((variant, *row) for row in top_rows)
        if variant == "full":
            full_latencies = latencies
            full_candidate_counts = candidate_counts
            full_candidate_recalls = candidate_recalls
            full_output = top_rows

    output = STAGE / "data/processed"
    logs = STAGE / "logs"
    curves = STAGE / "curves"
    for directory in (output, logs, curves):
        directory.mkdir(parents=True, exist_ok=True)
    top_path = output / "integrated_top10.csv.gz"
    with gzip.open(top_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["userId", "movieId", "rank", "score", "model_version"])
        writer.writerows(full_output)
    per_user_path = output / "ablation_per_user.csv.gz"
    with gzip.open(per_user_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "userId", "ndcg_at_10", "recall_at_10", "candidate_count", "candidate_positive_recall", "rerank_ms"])
        writer.writerows(detailed_rows)
    ablation_top_path = output / "ablation_top10.csv.gz"
    with gzip.open(ablation_top_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "userId", "movieId", "rank", "score"])
        writer.writerows(all_top_rows)
    report = {
        "experiment": "phase6-integrated-rank-fusion-ablation-v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": cfg["dataset"],
        "candidate_sha256": sha256(candidate_path),
        "workset_users": len(users),
        "matching_user_baseline": {"ndcg_at_10": baseline_ndcg, "recall_at_10": baseline_recall},
        "candidate_and_ranker": {"sources": {name: path.relative_to(ROOT).as_posix() for name, path in SOURCES.items()},
                                 "method": cfg["candidate_pool"] + "; " + cfg["ranker"],
                                 "note": "Module outputs are fixed Top10 lists. Candidate union is reranked; this offline experiment does not measure model-serving retrieval latency."},
        "variants": all_results,
        "acceptance": {
            "full_ndcg_gain_at_least_10_percent": all_results["full"]["ndcg_relative_change_vs_matching_baseline"] >= cfg["target_ndcg_relative_gain"],
            "ablation_variants_measured": len(all_results) == 5,
            "end_to_end_service_p99_verified": False,
            "full_dataset_scope_verified": False,
        },
        "runtime": {"python": __import__("platform").python_version(), "numpy": np.__version__,
                    "elapsed_seconds_including_component_file_reads": time.perf_counter() - started,
                    "full_rerank_p99_ms": float(np.quantile(full_latencies, .99)),
                    "full_candidate_pool_mean": float(np.mean(full_candidate_counts)),
                    "full_candidate_positive_recall": float(np.mean(full_candidate_recalls)),
                    "latency_scope": "online reciprocal-rank aggregation and sort over already-produced Top10 lists; excludes ALS, PPR, LSH, stream source query and network/storage"},
        "outputs": {"integrated_top10": top_path.relative_to(STAGE).as_posix(),
                    "ablation_per_user": per_user_path.relative_to(STAGE).as_posix(),
                    "ablation_top10": ablation_top_path.relative_to(STAGE).as_posix()},
    }
    report["outputs"]["integrated_top10_sha256"] = sha256(top_path)
    report["outputs"]["ablation_per_user_sha256"] = sha256(per_user_path)
    report["outputs"]["ablation_top10_sha256"] = sha256(ablation_top_path)
    (logs / "integration_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (logs / "ablation_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "sources", "ndcg_at_10", "recall_at_10", "ndcg_gain_vs_baseline", "candidate_pool_mean", "candidate_positive_recall", "rerank_p99_ms"])
        for variant, result in all_results.items():
            writer.writerow([variant, "+".join(result["active_sources"]), result["ndcg_at_10"], result["recall_at_10"],
                             result["ndcg_relative_change_vs_matching_baseline"], result["candidate_pool_mean"],
                             result["candidate_positive_recall"], result["rerank_latency_ms"]["p99"]])
    labels = list(all_results)
    ndcg_values = [all_results[name]["ndcg_at_10"] for name in labels]
    recall_values = [all_results[name]["recall_at_10"] for name in labels]
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    positions = np.arange(len(labels))
    ax.plot(positions, ndcg_values, marker="o", color="#32679a", label="NDCG@10")
    ax.plot(positions, recall_values, marker="s", color="#3b8264", label="Recall@10")
    ax.axhline(baseline_ndcg, color="#a35345", linestyle="--", label="Matching-user baseline NDCG@10")
    ax.set_xticks(positions, labels, rotation=18, ha="right")
    ax.set_ylabel("Macro metric")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Integrated recommender ablation on shared 5,000-user set")
    fig.savefig(curves / "integration_ablation.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

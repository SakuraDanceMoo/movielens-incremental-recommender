"""Prepare explicit derived tag/link metadata, preserving immutable raw CSV bytes."""
import json
import polars as pl
from download import STAGE, sha256
from prepare import save_json


def main():
    raw, processed = STAGE / "data/raw", STAGE / "data/processed"
    tags = pl.read_csv(raw / "tags.csv", schema_overrides={"userId": pl.Int32, "movieId": pl.Int32, "timestamp": pl.Int64})
    clean = tags.with_columns(pl.col("tag").str.strip_chars()).filter(pl.col("tag").fill_null("") != "")
    clean.write_parquet(processed / "tags_clean.parquet", compression="zstd")
    links = pl.read_csv(raw / "links.csv", schema_overrides={"movieId": pl.Int32, "imdbId": pl.Int32, "tmdbId": pl.Int32})
    links.write_parquet(processed / "links_checked.parquet", compression="zstd")
    result = {"raw_tags": tags.height, "clean_tags": clean.height, "removed_blank_tag_events": tags.height - clean.height,
              "trimmed_tag_events": tags.filter(pl.col("tag") != pl.col("tag").str.strip_chars()).height,
              "raw_links": links.height, "missing_optional_tmdb_ids_retained": links["tmdbId"].null_count(),
              "raw_files_modified": False, "baseline_uses_tags_or_links": False,
              "warning": "Tag events span all times. Future stages must filter timestamp < recommendation cutoff before feature construction.",
              "input_sha256": {name: sha256(raw / name) for name in ["tags.csv", "links.csv"]}}
    save_json(STAGE / "logs/metadata_cleanup.json", result)
    manifest_path = STAGE / "data/processed_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    new_paths = {"data/processed/tags_clean.parquet", "data/processed/links_checked.parquet"}
    manifest["files"] = [i for i in manifest["files"] if i["path"] not in new_paths]
    for name in sorted(new_paths):
        p = STAGE / name
        manifest["files"].append({"path": name, "bytes": p.stat().st_size, "sha256": sha256(p)})
    manifest["files"].sort(key=lambda x: x["path"])
    save_json(manifest_path, manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

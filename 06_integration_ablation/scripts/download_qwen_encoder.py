"""Pin and download the official Qwen3 embedding encoder with byte checks."""
from __future__ import annotations

import hashlib
import json
import ssl
from pathlib import Path

import httpx

MODEL = "Qwen/Qwen3-Embedding-0.6B"
ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT.parent / ".venv" / "models" / "Qwen3-Embedding-0.6B"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    log = ROOT / "06_integration_ablation" / "logs" / "transformer" / "encoder_manifest.json"
    existing_revision = json.loads(log.read_text()).get("revision") if log.exists() else None
    with httpx.Client(verify=ssl.create_default_context(), follow_redirects=True, timeout=120) as client:
        api_path=f"https://huggingface.co/api/models/{MODEL}"
        if existing_revision:api_path += "/revision/" + existing_revision
        response = client.get(api_path, params={"blobs": "true"})
        response.raise_for_status()
        info = response.json()
        revision = info["sha"]
        upstream = {entry["rfilename"]: entry for entry in info["siblings"]}
        files = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
        manifest = {"model_id": MODEL, "revision": revision, "files": {}}
        for name in files:
            dest = DEST / name
            if not dest.exists():
                partial = DEST / (name + ".partial")
                with client.stream("GET", f"https://huggingface.co/{MODEL}/resolve/{revision}/{name}") as stream:
                    stream.raise_for_status()
                    size = int(stream.headers.get("Content-Length", "0"))
                    last = 0
                    with partial.open("wb") as output:
                        for block in stream.iter_bytes(8 * 1024 * 1024):
                            output.write(block)
                            if output.tell() - last > 100 * 1024 * 1024:
                                last = output.tell()
                                print(f"{name}: {last / 1024**2:.0f} MiB downloaded", flush=True)
                    if size and partial.stat().st_size != size:
                        raise ValueError("Truncated model download")
                partial.rename(dest)
            digest = hashlib.sha256()
            with dest.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            expected_sha = upstream.get(name, {}).get("lfs", {}).get("sha256")
            if expected_sha and digest.hexdigest() != expected_sha:
                raise ValueError("Official LFS SHA-256 mismatch: " + name)
            manifest["files"][name] = {"bytes": dest.stat().st_size, "sha256": digest.hexdigest(),
                                       "upstream_lfs_sha256": expected_sha}
            print(f"verified {name}: {dest.stat().st_size} bytes", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"model_id": MODEL, "revision": revision, "destination": str(DEST)}), flush=True)


if __name__ == "__main__":
    main()

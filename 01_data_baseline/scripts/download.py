"""Download immutable official MovieLens bytes; do not redistribute raw data."""
import hashlib
import json
import re
import time
import ssl
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import httpx

STAGE = Path(__file__).resolve().parents[1]
URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    raw = STAGE / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "ml-32m.zip"
    started = time.perf_counter()
    if not archive.exists():
        temp = raw / "ml-32m.zip.part"
        # Range requests bound each transfer and validate offsets before append.
        # Raw completed files are immutable; only a .part file may be resumed.
        offset = temp.stat().st_size if temp.exists() else 0
        total = None
        proxy = urllib.request.getproxies().get("https")
        with httpx.Client(proxy=proxy, verify=ssl.create_default_context(),
                          timeout=httpx.Timeout(30, connect=10), follow_redirects=True) as client, temp.open("ab") as f:
            while total is None or offset < total:
                end = offset + 8 * 1024 * 1024 - 1
                for attempt in range(6):
                    try:
                        response = client.get(URL, headers={"Range": f"bytes={offset}-{end}"})
                        content_range = response.headers.get("Content-Range", "")
                        if response.status_code != 206 or not content_range.startswith(f"bytes {offset}-"):
                            raise ValueError("Server did not honor the requested byte range")
                        bounds, total_text = content_range.split(" ")[1].split("/")
                        lo, hi = map(int, bounds.split("-"))
                        block = response.content
                        total = int(total_text)
                        if lo != offset or len(block) != hi - lo + 1:
                            raise ValueError("Incomplete or wrong byte range")
                        break
                    except (OSError, ValueError, httpx.HTTPError) as e:
                        print(f"range {offset}-{end} attempt {attempt + 1}: {type(e).__name__}: {e}", flush=True)
                        if attempt == 5:
                            raise
                f.write(block); f.flush(); offset += len(block)
                print(f"downloaded {offset / 1024**2:.1f}/{total / 1024**2:.1f} MiB", flush=True)
        temp.rename(archive)
    with zipfile.ZipFile(archive) as z:
        for name in ["ratings.csv", "movies.csv", "tags.csv", "links.csv", "README.txt"]:
            content = z.read("ml-32m/" + name)  # ZipFile also verifies CRC.
            out = raw / name
            if out.exists():
                if sha256(out) != hashlib.sha256(content).hexdigest():
                    raise ValueError(f"Existing raw bytes differ from archive: {name}")
            else:
                out.write_bytes(content)
    official = re.findall(r"\| ([0-9a-f]{32}) \| ([a-z]+\.csv) \|", (raw / "README.txt").read_text(encoding="utf-8"))
    md5_checks = []
    for expected, name in official:
        h = hashlib.md5()
        with (raw / name).open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        md5_checks.append({"file": name, "publisher_md5": expected, "actual_md5": h.hexdigest(), "matches": h.hexdigest() == expected})
    if len(md5_checks) != 4 or not all(i["matches"] for i in md5_checks):
        raise ValueError("Official README MD5 checks failed")
    (STAGE / "logs").mkdir(parents=True, exist_ok=True)
    (STAGE / "logs/publisher_checksums.json").write_text(json.dumps({"source": "official archive README.txt",
                 "all_passed": True, "checks": md5_checks}, indent=2) + "\n", encoding="utf-8")
    files = [{"path": p.relative_to(STAGE).as_posix(), "bytes": p.stat().st_size,
              "sha256": sha256(p), "encoding": "binary ZIP" if p.suffix == ".zip" else "UTF-8"}
             for p in sorted(raw.iterdir()) if p.name in
             {"ml-32m.zip", "ratings.csv", "movies.csv", "tags.csv", "links.csv", "README.txt"}]
    manifest = {"dataset": "MovieLens 32M", "source": URL,
                "retrieved_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                "license": "See local data/raw/README.txt and https://grouplens.org/datasets/movielens/",
                "redistributed": False, "hash_provenance": "locally computed, not a publisher authenticity checksum",
                "files": files}
    target = STAGE / "data/raw_manifest.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old["files"] != files:
            raise ValueError("Previously registered raw bytes changed")
    else:
        target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"download_and_extract_seconds": time.perf_counter() - started,
                      "files": files}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

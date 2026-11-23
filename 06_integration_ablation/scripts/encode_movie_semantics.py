"""Frozen Qwen3-Embedding movie semantics; no rating or user text enters encoder."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT.parent / ".venv"
os.environ.setdefault("HF_HOME", str(ENV / "cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    torch.set_num_threads(6)
    stage = ROOT / "06_integration_ablation"
    cache = stage / "data" / "processed" / "transformer"
    movies = pl.read_csv(ROOT / "data" / "raw" / "hybrid" / "movies.csv")
    ids = movies["movieId"].to_numpy()
    cleaned = [(re.sub(r"^SYN\d+:\s*", "", title), genres.replace("|", ", "))
               for _, title, genres in movies.iter_rows()]
    texts = [f"Movie: {title}. Genres: {genres}." for title, genres in cleaned]
    model_dir = ENV / "models" / "Qwen3-Embedding-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left", local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa", local_files_only=True).cuda().eval()
    embeddings = np.zeros((175171, 128), dtype=np.float32)
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(texts), 128):
            tokens = tokenizer(texts[begin:begin+128], padding=True, truncation=True,
                               max_length=64, return_tensors="pt").to("cuda")
            hidden = model(**tokens, use_cache=False).last_hidden_state[:, -1, :128]
            values = F.normalize(hidden.float(), p=2, dim=-1).cpu().numpy()
            embeddings[ids[begin:begin+128]] = values
            if begin % (128*50) == 0:
                print(json.dumps({"encoded_movies": min(begin+128,len(texts)), "total":len(texts),
                                  "seconds":round(time.perf_counter()-start,2)}), flush=True)
    np.save(cache / "qwen_movie_embeddings.npy", embeddings)
    result = {"model":"Qwen/Qwen3-Embedding-0.6B", "rows":len(texts),
              "feature_dimension":128, "max_tokens":64, "batch_size":128,
              "pooling":"last token, first128 matryoshka dimensions, L2 normalize",
              "input":"movie title (synthetic numeric prefix removed) and genres only",
              "fine_tuned":False,"ratings_used":False,"elapsed_seconds":time.perf_counter()-start}
    (stage / "logs" / "transformer" / "semantic_encoding.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()

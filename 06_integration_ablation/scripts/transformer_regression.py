"""Qwen semantic feature-token Transformer for explicit rating regression.

Not an HLLM/HSTU reproduction: a small supervised feature Transformer uses a
frozen official Qwen3 encoder and ALS warm start under the existing RMSE task.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "06_integration_ablation"
CACHE = STAGE / "data" / "processed" / "transformer"
LOGS = STAGE / "logs" / "transformer"
MODELS = STAGE / "models"
VARIANTS = ("full", "no_semantics", "no_attention", "no_als_init", "mf_only",
            "random_no_semantics", "random_no_attention", "random_mf_only")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class FeatureBlock(nn.Module):
    """Pre-normalized SDPA self-attention and SwiGLU feed-forward residuals."""
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.gate_up = nn.Linear(dim, 4 * dim)
        self.down = nn.Linear(2 * dim, dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, length, dim = tokens.shape
        q, k, v = self.qkv(self.norm1(tokens)).chunk(3, dim=-1)
        q, k, v = [t.view(b, length, self.heads, dim // self.heads).transpose(1, 2) for t in (q, k, v)]
        attended = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, length, dim)
        x = tokens + self.proj(attended)
        gate, up = self.gate_up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * up)


class RatingTransformer(nn.Module):
    def __init__(self, als: dict, semantic: np.ndarray, meta: np.ndarray,
                 variant: str, dim: int = 32):
        super().__init__()
        self.variant, self.dim = variant, dim
        nusers, nitems = als["user"].shape[0], als["item"].shape[0]
        self.user = nn.Embedding(nusers, dim)
        self.item = nn.Embedding(nitems, dim)
        nn.init.normal_(self.user.weight, std=.03)
        nn.init.normal_(self.item.weight, std=.03)
        self.user_bias = nn.Embedding(nusers, 1)
        self.item_bias = nn.Embedding(nitems, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        self.source_bias = nn.Embedding(2, 1)
        nn.init.zeros_(self.source_bias.weight)
        self.mean = float(als["mean"])
        self.random_init = variant == "no_als_init" or variant.startswith("random_")
        self.mf_only = variant in ("mf_only", "random_mf_only")
        self.no_semantics = "no_semantics" in variant
        self.no_attention = "no_attention" in variant
        if not self.random_init:
            with torch.no_grad():
                self.user.weight[:, :16].copy_(torch.from_numpy(als["user"]))
                self.item.weight[:, :16].copy_(torch.from_numpy(als["item"]))
        self.register_buffer("known_user", torch.from_numpy(als["known_user"]))
        self.register_buffer("known_item", torch.from_numpy(als["known_item"]))
        self.register_buffer("semantics", torch.from_numpy(semantic))
        self.register_buffer("metadata", torch.from_numpy(meta))
        if not self.mf_only:
            self.user_proj = nn.Linear(dim, dim)
            self.item_proj = nn.Linear(dim, dim)
            self.semantic_proj = nn.Linear(semantic.shape[1], dim)
            self.metadata_proj = nn.Linear(meta.shape[1], dim)
            self.type_embedding = nn.Parameter(torch.randn(4, dim) * .02)
            self.block = (nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2*dim), nn.SiLU(), nn.Linear(2*dim, dim))
                          if self.no_attention else FeatureBlock(dim))
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = self.user(u), self.item(i)
        known = self.known_user[u] & self.known_item[i]
        base = (a*b).sum(-1)
        if self.random_init:
            base = base + self.mean
        else:
            base = torch.where(known, base, self.mean)
        bias = self.user_bias(u).squeeze(-1) + self.item_bias(i).squeeze(-1)
        prediction = base + bias + self.source_bias((u > 200948).long()).squeeze(-1)
        if not self.mf_only:
            semantic = self.semantics[i]
            if self.no_semantics:
                semantic = torch.zeros_like(semantic)
            tokens = torch.stack([self.user_proj(a), self.item_proj(b),
                                  self.semantic_proj(semantic), self.metadata_proj(self.metadata[i])], dim=1)
            tokens = tokens + self.type_embedding
            encoded = self.norm(self.block(tokens)).mean(1)
            prediction = prediction + .5 * torch.tanh(self.head(encoded).squeeze(-1))
        regularizer = a.square().mean() + b.square().mean() + .1*bias.square().mean()
        return prediction.float(), regularizer


def load_inputs(full: bool = False):
    tag = "full" if full else "fit"
    als = dict(np.load(CACHE / f"als_{tag}.npz"))
    semantic_path = CACHE / "qwen_movie_embeddings.npy"
    semantics = np.load(semantic_path) if semantic_path.exists() else np.zeros((175171,128), np.float32)
    meta = np.concatenate([np.load(CACHE / "genre_features.npy"), np.load(CACHE / "year_features.npy")], axis=1)
    return als, semantics, meta


@torch.inference_mode()
def predict(model, u, i, batch: int = 32768) -> np.ndarray:
    model.eval()
    result = []
    for offset in range(0,len(u),batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            values, _ = model(u[offset:offset+batch], i[offset:offset+batch])
        result.append(values.float().cpu().numpy())
    return np.concatenate(result)


def scores(y: np.ndarray, pred: np.ndarray, u: np.ndarray) -> dict:
    clipped = np.clip(pred.astype(np.float64), .5, 5)
    error = (clipped - y.astype(np.float64)) ** 2
    return {"rmse":float(np.sqrt(error.mean())),
            "movielens_rmse":float(np.sqrt(error[u<=200948].mean())),
            "synthetic_rmse":float(np.sqrt(error[u>200948].mean()))}


def blend_selection(y: np.ndarray, baseline: np.ndarray, neural: np.ndarray) -> tuple[float, np.ndarray]:
    """Validation-selected convex ensemble; explicitly not an unseen-test score."""
    candidates = np.linspace(0., 1., 21)
    losses = [np.mean((np.clip((1-w)*baseline+w*neural,.5,5).astype(np.float64)-y)**2) for w in candidates]
    weight = float(candidates[int(np.argmin(losses))])
    return weight, (1-weight)*baseline+weight*neural


def train(variant: str, epochs: int, seed: int, batch: int, full_refit: bool = False,
          checkpoint_path: Path | None = None, output: Path | None = None) -> dict:
    seed_all(seed)
    LOGS.mkdir(parents=True,exist_ok=True)
    MODELS.mkdir(parents=True,exist_ok=True)
    als, semantics, meta = load_inputs(full_refit)
    if variant not in ["mf_only", "no_semantics", "random_no_semantics", "random_mf_only"] and not (CACHE / "qwen_movie_embeddings.npy").exists():
        raise FileNotFoundError("Run encode_movie_semantics.py first; semantic experiments cannot use placeholders")
    model = RatingTransformer(als, semantics, meta, variant).cuda()
    tag = "full" if full_refit else "fit"
    train_u, train_i, train_y = [torch.from_numpy(np.load(CACHE/f"{tag}_{key}.npy")).cuda() for key in ["u","i","y"]]
    valid_u_np, valid_i_np, valid_y = [np.load(CACHE/f"valid_{key}.npy") for key in ["u","i","y"]]
    vu,vi = torch.from_numpy(valid_u_np).cuda(),torch.from_numpy(valid_i_np).cuda()
    # Same seed, batches, optimizer and budget for each one-component ablation.
    factor_parameters = [model.user.weight,model.item.weight]
    factor_ids = {id(p) for p in factor_parameters}
    other = [p for p in model.parameters() if id(p) not in factor_ids]
    optimizer = torch.optim.AdamW([{"params":factor_parameters,"lr":.001},
                                  {"params":other,"lr":.002}],weight_decay=0, fused=True)
    checkpoint_tag="full_refit_" if full_refit else ""
    path = checkpoint_path or MODELS / f"transformer_{variant}_{checkpoint_tag}seed{seed}.pt"
    history, best, best_epoch, best_weight = [], math.inf, 0, 1.
    als_valid = np.load(CACHE / "als_valid_predictions.npy") if not full_refit else None
    start=time.perf_counter()
    n=len(train_u)
    order_generator=torch.Generator(device="cuda").manual_seed(seed+777)
    for epoch in range(1,epochs+1):
        model.train()
        ep_start=time.perf_counter()
        order=torch.randperm(n, device="cuda",dtype=torch.int32,generator=order_generator)
        running=torch.zeros((),device="cuda")
        # Fixed staircase decay, independent of validation results.
        decay=.7**max(0,epoch-2)
        optimizer.param_groups[0]["lr"] = .001*decay
        optimizer.param_groups[1]["lr"] = .002*decay
        for step,begin in enumerate(range(0,n,batch)):
            index=order[begin:begin+batch]
            u,i,y=train_u[index],train_i[index],train_y[index]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                pred,regularizer=model(u,i)
                mse=F.mse_loss(pred,y)
                loss=mse + .01*regularizer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            running += mse.detach()*len(index)
            if step and step % 400==0:
                print(json.dumps({"variant":variant,"epoch":epoch,"step":step,
                                  "steps":math.ceil(n/batch),"seconds":round(time.perf_counter()-ep_start,1)}),flush=True)
        torch.cuda.synchronize()
        record={"epoch":epoch,"train_mse":float(running.item()/n),"epoch_seconds":time.perf_counter()-ep_start}
        if not full_refit:
            val_pred=predict(model,vu,vi)
            raw_scores = scores(valid_y,val_pred,valid_u_np)
            weight, blended = blend_selection(valid_y, als_valid, val_pred)
            record.update(scores(valid_y,blended,valid_u_np))
            record.update(raw_neural_rmse=raw_scores["rmse"], neural_weight=weight)
            if record["rmse"] < best:
                best,best_epoch,best_weight=record["rmse"],epoch,weight
                torch.save({"variant":variant,"seed":seed,"epoch":epoch,"neural_weight":weight,"state_dict":model.state_dict(),"dim":32},path)
                np.save(CACHE/f"valid_{variant}_seed{seed}.npy",blended)
                np.save(CACHE/f"valid_neural_{variant}_seed{seed}.npy",val_pred)
        else:
            best_epoch=epoch
        history.append(record)
        (LOGS/f"history_{variant}_{'refit' if full_refit else 'validation'}_seed{seed}.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        print(json.dumps({"variant":variant,"seed":seed,"full_refit":full_refit,**record}),flush=True)
    if full_refit:
        # Full-refit epoch count/weight are selected exclusively on validation.
        previous=json.loads((LOGS/f"result_{variant}_validation_seed{seed}.json").read_text())
        best_weight=previous["neural_weight"]
        torch.save({"variant":variant,"seed":seed,"epoch":epochs,"neural_weight":best_weight,"state_dict":model.state_dict(),"dim":32},path)
        test_u,test_i=[torch.from_numpy(np.load(CACHE/f"test_{k}.npy")).cuda() for k in ['u','i']]
        raw_neural=predict(model,test_u,test_i)
        tu,ti=test_u.cpu().numpy(),test_i.cpu().numpy()
        baseline=np.einsum('ij,ij->i',als['user'][tu],als['item'][ti])
        baseline[~(als['known_user'][tu]&als['known_item'][ti])]=float(als['mean'])
        values=np.clip((1-best_weight)*baseline+best_weight*raw_neural,.5,5.)
        import polars as pl
        source=pl.read_csv(ROOT/'data/raw/hybrid/test_pairs.csv')
        result=source.with_columns(pl.Series('predicted_rating',values.astype(float)))
        raw_test_path=CACHE/f"test_neural_{variant}_seed{seed}.npy"
        np.save(raw_test_path,raw_neural.astype(np.float32))
        output=output or ROOT/'data/raw/hybrid'/f"test_pairs_predictions_{variant}.csv"
        result.write_csv(output,float_precision=8)
        outcome={"variant":variant,"seed":seed,"epochs":epochs,"neural_weight":best_weight,"rows":result.height,
                 "training_rows":n,"checkpoint_file":path.name,
                 "null_predictions":int(result['predicted_rating'].null_count()),
                 "min_prediction":float(values.min()),"max_prediction":float(values.max()),
                 "elapsed_seconds":time.perf_counter()-start,"output_file":output.name,"test_rmse":None}
    else:
        saved=np.load(CACHE/f"valid_{variant}_seed{seed}.npy")
        outcome={"variant":variant,"seed":seed,"epochs":epochs,"best_epoch":best_epoch,"neural_weight":best_weight,
                 "train_rows":n,"valid_rows":len(valid_y),"parameters":sum(p.numel() for p in model.parameters()),
                 "elapsed_seconds":time.perf_counter()-start,"batch_size":batch,
                 "best_validation":scores(valid_y,saved,valid_u_np)}
        outcome["selection_scope"]="epoch and convex ensemble weight selected on the same labeled validation; not unbiased test performance"
    (LOGS/f"result_{variant}_{'refit' if full_refit else 'validation'}_seed{seed}.json").write_text(json.dumps(outcome,indent=2),encoding="utf-8")
    print(json.dumps(outcome),flush=True)
    return outcome


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--variant',choices=VARIANTS,default='full')
    parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--seed',type=int,default=20261123)
    parser.add_argument('--batch-size',type=int,default=16384)
    parser.add_argument('--full-refit',action='store_true')
    args=parser.parse_args()
    train(args.variant,args.epochs,args.seed,args.batch_size,args.full_refit)

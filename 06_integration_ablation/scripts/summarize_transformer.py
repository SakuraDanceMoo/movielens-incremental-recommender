"""Summarize actual two-seed ablations and select a deployable Transformer."""
from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/'06_integration_ablation'
CACHE=STAGE/'data/processed/transformer'
LOGS=STAGE/'logs/transformer'
VARIANTS=['full','no_semantics','no_attention','no_als_init','mf_only']
EXTRA=['random_no_semantics','random_no_attention','random_mf_only']


def main() -> None:
    variants=VARIANTS+EXTRA if all((LOGS/f'result_{v}_validation_seed{s}.json').exists() for v in EXTRA for s in [20261123,20261124]) else VARIANTS
    records=[]
    for seed in [20261123,20261124]:
        for variant in variants:
            r=json.loads((LOGS/f'result_{variant}_validation_seed{seed}.json').read_text())
            records.append(r)
    rows=[]
    for variant in variants:
        selected=[r for r in records if r['variant']==variant]
        values=np.array([r['best_validation']['rmse'] for r in selected])
        rows.append({'variant':variant,'seeds':len(values),'mean_rmse':float(values.mean()),
                     'std_rmse':float(values.std(ddof=1)),
                     'rmse_seed20261123':float(values[0]),'rmse_seed20261124':float(values[1]),
                     'mean_seconds':float(np.mean([r['elapsed_seconds'] for r in selected])),
                     'parameters':selected[0]['parameters']})
    with (LOGS/'ablation.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Keep an actual Transformer in the final selection. The no-attention and
    # MF controls remain reported even if either is better.
    eligible=[r for r in rows if r['variant'] in ['full','no_als_init']]
    selected_variant=min(eligible,key=lambda r:r['mean_rmse'])['variant']
    selected=min((r for r in records if r['variant']==selected_variant),key=lambda r:r['best_validation']['rmse'])
    baseline=np.load(CACHE/'als_valid_predictions.npy').astype(float)
    prediction=np.load(CACHE/f"valid_{selected_variant}_seed{selected['seed']}.npy").astype(float)
    y=np.load(CACHE/'valid_y.npy').astype(float)
    a=(np.clip(baseline,.5,5)-y)**2;b=(np.clip(prediction,.5,5)-y)**2
    # One row per held-out user. Paired conditional bootstrap does not account
    # for model-selection optimism or unavailable hidden-test labels.
    rng=np.random.default_rng(20261123)
    bootstrap=[]
    for _ in range(400):
        idx=rng.integers(0,len(y),len(y))
        bootstrap.append(float(np.sqrt(a[idx].mean())-np.sqrt(b[idx].mean())))
    baseline_rmse=float(np.sqrt(a.mean()))
    rmse=float(np.sqrt(b.mean()))
    selection={"variant":selected_variant,"seed":selected['seed'],
               "best_epoch":selected['best_epoch'],"neural_weight":selected['neural_weight'],
               "validation_rmse":rmse,"baseline_rmse":baseline_rmse,
               "absolute_reduction":baseline_rmse-rmse,
               "relative_reduction_percent":100*(1-rmse/baseline_rmse),
               "conditional_paired_bootstrap_delta_95ci":np.quantile(bootstrap,[.025,.975]).tolist(),
               "bootstrap_repeats":400,"validation_rows":len(y),
               "selection_rule":"lowest two-seed mean among Qwen-semantic Transformer variants (ALS or random init); then lowest validation seed",
               "unrestricted_best_variant":min(rows,key=lambda r:r['mean_rmse'])['variant'],
               "all_variant_means":{r['variant']:r['mean_rmse'] for r in rows},
               "random_start_structures":[r for r in rows if r['variant'].startswith('random_') or r['variant']=='no_als_init'],
               "caveat":"same validation is used for epoch, fusion weight, variant and seed selection; CI conditional on selection, not hidden-test generalization",
               "all_runs":records,"ablation_summary":rows}
    (LOGS/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    import importlib.metadata as md
    environment={"python":platform.python_version(),"torch":torch.__version__,
                 "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),
                 "packages":{p:md.version(p) for p in ['transformers','polars','numpy','matplotlib','python-docx']},
                 "seeds":[20261123,20261124],"device":"single GPU", "batch_size":16384,
                 "training_rows_per_epoch":31513222,"epochs_per_run":4,
                 "frozen_encoder":"Qwen/Qwen3-Embedding-0.6B","trainable_dim":32,
                 "heads":4,"blocks":1,"tokens":["user factors","movie factors","Qwen semantics","genres and year"],
                 "optimizer":"AdamW fused, factor lr0.001, other lr0.002, no weight_decay",
                 "schedule":"lr multiplied by 0.7^(epoch-2) after epoch2",
                 "loss":"MSE + 0.01 * (mean(user_factor^2)+mean(item_factor^2)+0.1*mean(bias^2))",
                 "batch_order":"separate CUDA Generator seed+777, identical per variant",
                 "candidate_neural_weights":[round(float(x),2) for x in np.linspace(0,1,21)]}
    (LOGS/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in selection.items() if k!='all_runs'},indent=2),flush=True)


if __name__=='__main__':main()

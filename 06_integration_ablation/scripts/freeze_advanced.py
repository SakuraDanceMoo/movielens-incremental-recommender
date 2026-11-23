"""Freeze development-only model choices, then open the audit metrics once."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import polars as pl
from advanced_rating import development_mask

ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT/'06_integration_ablation'
CACHE=STAGE/'data/processed/advanced';OLD=STAGE/'data/processed/transformer';LOGS=STAGE/'logs/advanced'


def rmse(y,p,mask):
    return float(np.sqrt(np.mean((np.clip(p[mask],.5,5)-y[mask])**2)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidates',nargs='+',required=True)
    ap.add_argument('--ablations',nargs='*',default=[]);a=ap.parse_args()
    frozen=LOGS/'frozen_selection.json'
    if frozen.exists():raise FileExistsError('Selection already frozen; refusing to reopen audit for model selection')
    y=np.load(OLD/'valid_y.npy').astype(float);u=np.load(OLD/'valid_u.npy');dev=development_mask(u)
    scores={};predictions={}
    for name in a.candidates:
        predictions[name]=np.load(CACHE/f'valid_{name}.npy').astype(float)
        scores[name]=rmse(y,predictions[name],dev)
    best=min(scores,key=scores.get);weights={best:1.0}
    # No audit labels or audit metric are evaluated before this file is written.
    selection={'selected':best,'weights':weights,'development_candidates':scores,
               'development_rows':int(dev.sum()),'audit_rows':int((~dev).sum()),
               'rule':'lowest development OOF RMSE; fixed before opening current-round audit',
               'prediction_sha256':hashlib.sha256((CACHE/f'valid_{best}.npy').read_bytes()).hexdigest(),
               'audit_historical_caveat':'earlier project models used the whole outer validation for tuning; audit is reserved only for the new calibration round, not historically untouched test data',
               'test_rmse':None}
    frozen.write_text(json.dumps(selection,indent=2),encoding='utf-8')
    np.save(CACHE/'valid_advanced_final.npy',np.clip(predictions[best],.5,5))
    compare={'original_als':np.load(OLD/'als_valid_predictions.npy').astype(float),
             'previous_transformer_ensemble':np.load(OLD/'valid_final_ensemble.npy').astype(float)}
    names=list(dict.fromkeys(a.candidates+a.ablations))
    for name in names:compare[name]=np.load(CACHE/f'valid_{name}.npy').astype(float)
    rows=[]
    for name,p in compare.items():
        row={'name':name,'development_rmse':rmse(y,p,dev),'audit_rmse':rmse(y,p,~dev),
             'overall_rmse':rmse(y,p,np.ones(len(y),bool)),
             'audit_movielens_rmse':rmse(y,p,(~dev)&(u<=200948)),
             'audit_synthetic_rmse':rmse(y,p,(~dev)&(u>200948))}
        rows.append(row)
    final=predictions[best];old=compare['previous_transformer_ensemble']
    # Paired bootstrap over audit users; each user has exactly one held-out row.
    eold=(np.clip(old[~dev],.5,5)-y[~dev])**2;enew=(np.clip(final[~dev],.5,5)-y[~dev])**2
    rng=np.random.default_rng(20261123);deltas=[]
    for _ in range(1000):
        index=rng.integers(0,len(enew),len(enew))
        deltas.append(float(np.sqrt(eold[index].mean())-np.sqrt(enew[index].mean())))
    report={'selected':best,'rows':rows,'audit_paired_old_minus_new_rmse_ci95':np.quantile(deltas,[.025,.975]).tolist(),
            'bootstrap_replicates':1000,'bootstrap_seed':20261123,'test_rmse':None,
            'overall_protocol':'development rows use calibration OOF predictions; audit rows use five-fold model average; base fitting split is unchanged',
            'selection':selection}
    (LOGS/'audit_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pl.DataFrame(rows).write_csv(LOGS/'advanced_comparison.csv')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

"""Independently verify logged errors, paired data protocol and delivered CSV."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/'06_integration_ablation'
CACHE=STAGE/'data/processed/transformer'
LOGS=STAGE/'logs/transformer'


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    checks=[]
    def check(condition,name):
        if not condition:raise AssertionError(name)
        checks.append(name)
    manifest=json.loads((ROOT/'data/hybrid_manifest.json').read_text())
    for record in manifest['files']:
        p=ROOT/'data/raw/hybrid'/record['file']
        check(p.stat().st_size==record['bytes'] and sha256(p)==record['sha256'],f"migrated bytes unchanged: {p.name}")
    y=np.load(CACHE/'valid_y.npy').astype(float)
    baseline=np.load(CACHE/'als_valid_predictions.npy').astype(float)
    baseline_rmse=float(np.sqrt(np.mean((np.clip(baseline,.5,5)-y)**2)))
    check(abs(baseline_rmse-0.8974749137472182)<1e-7,'reproduced original ALS on identical holdout')
    check(len(y)==386982 and len(np.load(CACHE/'fit_y.npy',mmap_mode='r'))==31513222,'exact fitting/validation row counts')
    for seed in [20261123,20261124]:
        for variant in ['full','no_semantics','no_attention','no_als_init','mf_only','random_no_semantics','random_no_attention','random_mf_only']:
            pred=np.load(CACHE/f'valid_{variant}_seed{seed}.npy').astype(float)
            report=json.loads((LOGS/f'result_{variant}_validation_seed{seed}.json').read_text())
            value=float(np.sqrt(np.mean((np.clip(pred,.5,5)-y)**2)))
            check(len(pred)==len(y) and np.isfinite(pred).all(),f'{variant} seed{seed}: all validation predictions finite')
            check(abs(value-report['best_validation']['rmse'])<1e-10,f'{variant} seed{seed}: RMSE independently recomputed')
    selection=json.loads((LOGS/'ensemble_selection.json').read_text())
    final_values=np.load(CACHE/'valid_final_ensemble.npy').astype(float)
    check(abs(float(np.sqrt(np.mean((np.clip(final_values,.5,5)-y)**2)))-selection['validation_rmse'])<1e-10,'final simplex ensemble independently recomputed')
    weights=selection['branch_weights'];seed=selection['seed']
    check(abs(sum(weights.values())-1)<1e-12 and min(weights.values())>=-1e-12,'three branch weights form convex combination')
    reconstructed=(weights['random_init_qwen_transformer']*np.load(CACHE/f'valid_neural_no_als_init_seed{seed}.npy').astype(float)
                   +weights['random_init_mf']*np.load(CACHE/f'valid_neural_random_mf_only_seed{seed}.npy').astype(float)
                   +weights['original_als']*baseline)
    check(np.allclose(np.clip(reconstructed,.5,5),final_values,rtol=0,atol=1e-12),'validation ensemble uses raw neural branches, exactly matching test fusion')
    check(selection['validation_rmse']<baseline_rmse,'selected Transformer lowers baseline validation RMSE')
    original=pl.read_csv(ROOT/'data/raw/hybrid/test_pairs.csv')
    output=ROOT/'data/raw/hybrid/test_pairs_predictions_transformer.csv'
    predictions=pl.read_csv(output)
    check(predictions.columns==['userId','movieId','timestamp','predicted_rating'],'final CSV column order')
    check(original.equals(predictions.select(original.columns)),'final CSV preserves every test key and row order')
    values=predictions['predicted_rating'].to_numpy()
    check(len(values)==100000 and np.isfinite(values).all(),'100000 finite final predictions')
    check(values.min()>=.5 and values.max()<=5,'predictions bounded in allowed rating range')
    check('rating' not in original.columns,'test labels unavailable; no test RMSE claim')
    result={'passed':True,'checks':checks,'check_count':len(checks),
            'selected_validation_rmse':selection['validation_rmse'],'original_als_validation_rmse':baseline_rmse,
            'prediction_file':output.name,'prediction_sha256':sha256(output),
            'prediction_rows':100000,'min_prediction':float(values.min()),'max_prediction':float(values.max()),
            'columns':predictions.columns,'test_rmse':None,'test_rmse_note':'hidden test labels not supplied'}
    (LOGS/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

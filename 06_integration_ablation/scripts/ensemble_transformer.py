"""Validation-only simplex selection among independently trained model branches."""
from __future__ import annotations

import hashlib
import json
import csv
from pathlib import Path

import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/'06_integration_ablation';CACHE=STAGE/'data/processed/transformer';LOGS=STAGE/'logs/transformer'


def main():
    y=np.load(CACHE/'valid_y.npy').astype(np.float64)
    als=np.load(CACHE/'als_valid_predictions.npy').astype(np.float64)
    seed_results=[]
    choices=[]
    ablation_runs=[]
    for seed in [20261123,20261124]:
        neural=np.load(CACHE/f'valid_neural_no_als_init_seed{seed}.npy').astype(np.float64)
        mf=np.load(CACHE/f'valid_neural_random_mf_only_seed{seed}.npy').astype(np.float64)
        best=(float('inf'),None,None)
        removed={key:(float('inf'),None) for key in ['without_qwen','without_mf','without_als']}
        for q_index in range(21):
            for m_index in range(21-q_index):
                tw,mw=q_index/20,m_index/20
                aw=1-tw-mw
                p=np.clip(tw*neural+mw*mf+aw*als,.5,5)
                score=float(np.sqrt(np.mean((p-y)**2)))
                if score<best[0]:best=(score,(tw,mw,aw),p)
                for key,condition in [('without_qwen',q_index==0),('without_mf',m_index==0),('without_als',q_index+m_index==20)]:
                    if condition and score<removed[key][0]:removed[key]=(score,(tw,mw,aw))
        seed_results.append({'seed':seed,'rmse':best[0],'weights':best[1]})
        choices.append(best)
        ablation_runs.append({'seed':seed,'full':best[0],**{key:value[0] for key,value in removed.items()}})
    selected_index=int(np.argmin([r['rmse'] for r in seed_results]))
    seed=seed_results[selected_index]['seed']
    rmse,weights,pred=choices[selected_index]
    np.save(CACHE/'valid_final_ensemble.npy',pred)
    users=np.load(CACHE/'valid_u.npy')
    err=(np.clip(pred,.5,5)-y)**2
    result={'validation_rows':len(y),'seed':seed,'branch_weights':{'random_init_qwen_transformer':weights[0],'random_init_mf':weights[1],'original_als':weights[2]},
            'validation_rmse':rmse,'movielens_rmse':float(np.sqrt(err[users<=200948].mean())),
            'synthetic_rmse':float(np.sqrt(err[users>200948].mean())),
            'two_seed_mean_rmse':float(np.mean([r['rmse'] for r in seed_results])),
            'two_seed_std_rmse':float(np.std([r['rmse'] for r in seed_results],ddof=1)),
            'seed_comparisons':seed_results,
            'qwen_als_fused_control_rmse':json.loads((LOGS/f'result_no_als_init_validation_seed{seed}.json').read_text())['best_validation']['rmse'],
            'mf_als_fused_control_rmse':json.loads((LOGS/f'result_random_mf_only_validation_seed{seed}.json').read_text())['best_validation']['rmse'],
            'als_rmse':float(np.sqrt(np.mean((np.clip(als,.5,5)-y)**2))),
            'selection':'231 simplex candidates in step0.05 per seed, two seeds; selected and evaluated on same validation; not an independent test estimate',
            'output':'data/raw/hybrid/test_pairs_predictions_transformer.csv','test_rmse':None}
    ablation_rows=[]
    for key in ['full','without_qwen','without_mf','without_als']:
        values=[r[key] for r in ablation_runs]
        ablation_rows.append({'variant':key,'mean_rmse':float(np.mean(values)),
                              'std_rmse':float(np.std(values,ddof=1)),
                              'seed20261123':values[0],'seed20261124':values[1]})
    with (LOGS/'ensemble_ablation.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(ablation_rows[0]));writer.writeheader();writer.writerows(ablation_rows)
    result['branch_ablation']=ablation_rows
    (LOGS/'ensemble_selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # Selection runs before final refitting, then this same script deterministically
    # reproduces it and writes test output once both matching refits exist.
    if not all((CACHE/f'test_neural_{v}_seed{seed}.npy').exists() for v in ['no_als_init','random_mf_only']):
        print(json.dumps(result,indent=2),flush=True)
        return
    test=pl.read_csv(ROOT/'data/raw/hybrid/test_pairs.csv')
    alsf=dict(np.load(CACHE/'als_full.npz'))
    tu,ti=[test[k].to_numpy().astype(np.int64) for k in ['userId','movieId']]
    als_test=np.einsum('ij,ij->i',alsf['user'][tu],alsf['item'][ti])
    als_test[~(alsf['known_user'][tu]&alsf['known_item'][ti])]=float(alsf['mean'])
    qwen=np.load(CACHE/f'test_neural_no_als_init_seed{seed}.npy').astype(float)
    mf_test=np.load(CACHE/f'test_neural_random_mf_only_seed{seed}.npy').astype(float)
    values=np.clip(weights[0]*qwen+weights[1]*mf_test+weights[2]*als_test,.5,5)
    out=ROOT/'data/raw/hybrid/test_pairs_predictions_transformer.csv'
    test.with_columns(pl.Series('predicted_rating',values)).write_csv(out,float_precision=8)
    h=hashlib.sha256(out.read_bytes()).hexdigest()
    result.update({'prediction_rows':len(values),'prediction_sha256':h,'min_prediction':float(values.min()),'max_prediction':float(values.max()),
                   'validation_relative_reduction_vs_als_percent':100*(1-rmse/result['als_rmse'])})
    (LOGS/'ensemble_selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

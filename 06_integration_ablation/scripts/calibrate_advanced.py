"""Out-of-fold residual calibration; audit labels are never used for fitting."""
from __future__ import annotations
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
from advanced_rating import development_mask

ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT/'06_integration_ablation'
CACHE=STAGE/'data/processed/advanced';OLD=STAGE/'data/processed/transformer';LOGS=STAGE/'logs/advanced';MODELS=STAGE/'models'


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='calibration_full');p.add_argument('--without-history',action='store_true')
    p.add_argument('--without-neural',action='store_true');p.add_argument('--without-semantics',action='store_true')
    p.add_argument('--depth',type=int,default=6);p.add_argument('--iterations',type=int,default=700)
    p.add_argument('--seed',type=int,default=20261123);a=p.parse_args()
    MODELS.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True)
    data=np.load(CACHE/'valid_features.npy');meta=json.loads((LOGS/'valid_features.json').read_text());names=meta['names']
    u=np.load(OLD/'valid_u.npy');y=np.load(OLD/'valid_y.npy').astype(float);base=np.load(OLD/'valid_final_ensemble.npy').astype(float)
    dev=development_mask(u);folds=(u.astype(np.int64)*2246822519%5).astype(int)
    # This multiplier mod5 is nonzero and differs from the dev-mask multiplier.
    # Use a mixed hash so folds remain populated inside the development cohort.
    mixed=u.astype(np.uint64);mixed=(mixed^(mixed>>np.uint64(16)))*np.uint64(0x45d9f3b);mixed=mixed^(mixed>>np.uint64(16))
    folds=(mixed%np.uint64(5)).astype(int)
    # Raw latent coordinates have arbitrary rotations after a full-data refit.
    # Keep invariant norms/biases/cosines, never stack coordinate axes.
    keep=np.array([not (n.startswith(('latent_user_','latent_item_')) and n.rsplit('_',1)[-1].isdigit()) for n in names])
    if a.without_history:keep &= np.array([not n.startswith(('history','recent','memory','svdpp','qwen_history')) for n in names])
    if a.without_neural:keep &= np.array([not n.startswith(('mf64','mf128','memory','svdpp','latent_','history_mf')) for n in names])
    if a.without_semantics:keep &= np.array([not n.startswith('qwen_') for n in names])
    data=data[:,keep];selected=[n for n,k in zip(names,keep) if k]
    oof=np.empty(len(y),np.float64);audit_predictions=[];fold_scores=[];importance=[];start=time.perf_counter()
    params=dict(iterations=a.iterations,depth=a.depth,learning_rate=.035,l2_leaf_reg=25,loss_function='RMSE',
                random_seed=a.seed,thread_count=8,task_type='CPU',verbose=False,allow_writing_files=False,
                bootstrap_type='Bernoulli',subsample=.8)
    for fold in range(5):
        training=dev&(folds!=fold);held=dev&(folds==fold)
        assert training.sum()>0 and held.sum()>0 and not np.any(training&held)
        model=CatBoostRegressor(**params)
        model.fit(data[training],y[training]-base[training])
        values=base[held]+model.predict(data[held]);oof[held]=values
        audit_predictions.append(base[~dev]+model.predict(data[~dev]))
        fold_scores.append(float(np.sqrt(np.mean((np.clip(values,.5,5)-y[held])**2))))
        importance.append(model.feature_importances_)
        model.save_model(str(MODELS/f'advanced_{a.name}_fold{fold}.cbm'))
        print(json.dumps({'calibration':a.name,'fold':fold,'training_rows':int(training.sum()),'heldout_rows':int(held.sum()),'development_oof_rmse':fold_scores[-1],'seconds':time.perf_counter()-start}),flush=True)
    oof[~dev]=np.mean(audit_predictions,axis=0)
    np.save(CACHE/f'valid_{a.name}.npy',oof)
    # Do not expose any audit or overall value until all model/feature variants
    # have been frozen in freeze_advanced.py.
    result={'name':a.name,'params':params,'features':selected,'feature_count':len(selected),
            'feature_matrix_sha256':hashlib.sha256((CACHE/'valid_features.npy').read_bytes()).hexdigest(),
            'development_oof_rmse':float(np.sqrt(np.mean((np.clip(oof[dev],.5,5)-y[dev])**2))),
            'fold_development_rmse':fold_scores,'development_rows':int(dev.sum()),'audit_rows':int((~dev).sum()),
            'feature_importance':dict(sorted(zip(selected,np.mean(importance,axis=0).tolist()),key=lambda x:-x[1])),
            'seconds':time.perf_counter()-start,
            'protocol':'five-fold OOF only among development users; no audit label enters calibration fitting, early stopping, or hyperparameter tuning',
            'caveat':'development labels from the outer holdout train calibration for other development users; OOF prevents self-label fitting; audit is the stricter current-round comparison'}
    (LOGS/f'result_{a.name}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['feature_importance','features']}),flush=True)


if __name__=='__main__':main()

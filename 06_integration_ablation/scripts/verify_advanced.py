"""Independent artifact checks, including fold reconstruction and raw-byte hashes."""
from __future__ import annotations
import hashlib,json,platform,sys
from importlib.metadata import version
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from advanced_rating import development_mask

ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT/'06_integration_ablation'
CACHE=STAGE/'data/processed/advanced';OLD=STAGE/'data/processed/transformer';LOGS=STAGE/'logs/advanced'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    checks=[]
    def check(condition,name):
        assert condition,name
        checks.append(name)
    for rec in json.loads((ROOT/'data/hybrid_manifest.json').read_text())['files']:
        path=ROOT/'data/raw/hybrid'/rec['file']
        check(path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256'],'original bytes preserved: '+rec['file'])
    y=np.load(OLD/'valid_y.npy').astype(float);u=np.load(OLD/'valid_u.npy');dev=development_mask(u)
    check(len(y)==386982 and dev.sum()==309513 and (~dev).sum()==77469,'outer/development/audit row counts')
    check(len(np.unique(u))==len(u),'audit and OOF unit is one distinct user per heldout row')
    x=np.load(CACHE/'valid_features.npy');meta=json.loads((LOGS/'valid_features.json').read_text());names=meta['names']
    check(sha(CACHE/'valid_features.npy')==meta['feature_matrix_sha256'],'validation feature matrix fingerprint')
    check(x.shape==(386982,len(names)) and np.isfinite(x).all(),'complete finite validation feature matrix')
    for name,expected in meta['neural_prediction_hashes'].items():
        check(sha(CACHE/f'valid_{name}.npy')==expected and np.array_equal(x[:,names.index(name)],np.load(CACHE/f'valid_{name}.npy')),'fixed final checkpoint predictions: '+name)
    for tag in ['fit','full']:
        report=json.loads((LOGS/f'temporal_history_{tag}.json').read_text())
        for key in [tag,'test' if tag=='full' else 'valid']:
            check(report[key]['strict_timestamp_assertions_passed'],'all-token strict timestamp assertions recorded: '+key)
    h=u.astype(np.uint64);h=(h^(h>>np.uint64(16)))*np.uint64(0x45d9f3b);h=h^(h>>np.uint64(16));folds=(h%np.uint64(5)).astype(int)
    selection=json.loads((LOGS/'frozen_selection.json').read_text());name=selection['selected']
    result=json.loads((LOGS/f'result_{name}.json').read_text());cols=[names.index(n) for n in result['features']]
    check(not any(n.startswith(('latent_user_','latent_item_')) and n.rsplit('_',1)[-1].isdigit() for n in result['features']),'arbitrary latent coordinate axes excluded from deployed features')
    check(result['feature_matrix_sha256']==meta['feature_matrix_sha256'],'selected calibration used final feature snapshot')
    p=np.load(CACHE/f'valid_{name}.npy');base=np.load(OLD/'valid_final_ensemble.npy');reconstructed=np.empty(len(y));audit=[]
    for fold in range(5):
        held=dev&(folds==fold);model=CatBoostRegressor();model.load_model(str(STAGE/f'models/advanced_{name}_fold{fold}.cbm'))
        reconstructed[held]=base[held]+model.predict(x[held][:,cols]);audit.append(base[~dev]+model.predict(x[~dev][:,cols]))
    reconstructed[~dev]=np.mean(audit,axis=0)
    check(np.allclose(reconstructed,p,rtol=0,atol=1e-12),'OOF development and five-model audit predictions independently reconstructed')
    recomputed=float(np.sqrt(np.mean((np.clip(p[dev],.5,5)-y[dev])**2)))
    check(abs(recomputed-result['development_oof_rmse'])<1e-12,'development RMSE independently recomputed')
    evaluation=json.loads((LOGS/'audit_evaluation.json').read_text());selected=next(r for r in evaluation['rows'] if r['name']==name)
    for field,mask in [('audit_rmse',~dev),('overall_rmse',np.ones(len(y),bool))]:
        value=float(np.sqrt(np.mean((np.clip(p[mask],.5,5)-y[mask])**2)))
        check(abs(value-selected[field])<1e-12,field+' independently recomputed')
    original=pl.read_csv(ROOT/'data/raw/hybrid/test_pairs.csv');out=ROOT/'data/raw/hybrid/test_pairs_predictions_advanced.csv';prediction=pl.read_csv(out)
    check(prediction.columns==original.columns+['predicted_rating'],'CSV schema')
    check(prediction.select(original.columns).equals(original),'all test keys and row order preserved')
    values=prediction['predicted_rating'].to_numpy()
    check(len(values)==100000 and np.isfinite(values).all() and values.min()>=.5 and values.max()<=5,'100000 finite bounded predictions')
    check(np.allclose(values,np.load(CACHE/'test_advanced_final.npy'),rtol=0,atol=5.01e-9),'CSV rounding agrees with cached scores')
    check('rating' not in original.columns and evaluation['test_rmse'] is None,'no test RMSE without labels')
    prediction_manifest=json.loads((LOGS/'prediction_manifest.json').read_text())
    check(sha(out)==prediction_manifest['sha256'],'final CSV SHA256')
    testmeta=json.loads((LOGS/'test_features.json').read_text());testx=np.load(CACHE/'test_features.npy')
    check(np.isfinite(testx).all() and all(n in testmeta['names'] for n in result['features']),'full-refit test features align by name')
    for neural in meta['neural_prediction_hashes']:
        refit=json.loads((LOGS/f'result_{neural}_refit.json').read_text())
        validation=json.loads((LOGS/f'result_{neural}_validation.json').read_text())
        check(refit['fit_rows']==31900204 and refit['best_epoch']==validation['best_epoch'],'full-data refit uses frozen epochs: '+neural)
    report={'passed':True,'check_count':len(checks),'checks':checks,'selected':name,**selected,
            'prediction_sha256':sha(out),'prediction_rows':len(values),'test_rmse':None,
            'python':sys.executable,'platform':platform.platform(),
            'packages':{n:version(n) for n in ['catboost','numpy','torch','polars','matplotlib']}}
    (LOGS/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

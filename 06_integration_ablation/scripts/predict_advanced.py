"""Apply frozen fold calibrators to full-refit features, preserving test row order."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import polars as pl
from catboost import CatBoostRegressor

ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT/'06_integration_ablation'
CACHE=STAGE/'data/processed/advanced';LOGS=STAGE/'logs/advanced';MODELS=STAGE/'models'


def main():
    selection=json.loads((LOGS/'frozen_selection.json').read_text())
    names=json.loads((LOGS/'test_features.json').read_text())['names']
    data=np.load(CACHE/'test_features.npy');base=data[:,names.index('old_ensemble')].astype(float)
    residual=np.zeros(len(base));used={}
    for name,weight in selection['weights'].items():
        result=json.loads((LOGS/f'result_{name}.json').read_text());features=result['features']
        indices=[names.index(n) for n in features];values=[]
        for fold in range(5):
            model=CatBoostRegressor();model.load_model(str(MODELS/f'advanced_{name}_fold{fold}.cbm'))
            assert len(model.feature_names_)==len(features)
            values.append(model.predict(data[:,indices]))
        residual+=weight*np.mean(values,axis=0);used[name]={'folds':5,'features':features}
    pred=np.clip(base+residual,.5,5);assert np.isfinite(pred).all()
    np.save(CACHE/'test_advanced_final.npy',pred)
    raw=pl.read_csv(ROOT/'data/raw/hybrid/test_pairs.csv')
    output=ROOT/'data/raw/hybrid/test_pairs_predictions_advanced.csv'
    if output.exists():raise FileExistsError('Preserve existing predictions; choose a new output filename')
    raw.with_columns(pl.Series('predicted_rating',pred)).write_csv(output,float_precision=8)
    report={'output':output.name,'rows':len(pred),'columns':raw.columns+['predicted_rating'],
            'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'minimum':float(pred.min()),'maximum':float(pred.max()),
            'calibration_models':used,'refit_base_rows':31900204,'calibration_training_rows_per_fold':'development rows excluding each fold',
            'note':'base networks refit on full ratings; frozen calibrators remain the five development-fold models; no test labels supplied',
            'test_rmse':None}
    (LOGS/'prediction_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

"""Observed-rating statistics and residual-calibration features, labels excluded."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import hashlib
ROOT=Path(__file__).resolve().parents[2];STAGE=ROOT/'06_integration_ablation'
OLD=STAGE/'data/processed/transformer';CACHE=STAGE/'data/processed/advanced';LOGS=STAGE/'logs/advanced'


def features(full=False):
    start=time.perf_counter();tag='full' if full else 'fit';query='test' if full else 'valid'
    u,i,y=[np.load(OLD/f'{tag}_{k}.npy',mmap_mode='r') for k in ['u','i','y']]
    qu,qi,qt=[np.load(OLD/f'{query}_{k}.npy') for k in ['u','i','t']]
    mean=float(y.mean());columns=[];names=[]
    def add(name,value):names.append(name);columns.append(np.asarray(value,dtype=np.float32))
    add('source',qu>200948)
    for axis,ids,query_ids,size in [('user',u,qu,401897),('item',i,qi,175171)]:
        count=np.bincount(ids,minlength=size).astype(float)
        sums=np.bincount(ids,weights=y,minlength=size)
        squares=np.bincount(ids,weights=np.asarray(y,dtype=float)**2,minlength=size)
        avg=sums/np.maximum(count,1)
        std=np.sqrt(np.maximum(squares/np.maximum(count,1)-avg**2,0))
        add(axis+'_log_count',np.log1p(count[query_ids]))
        add(axis+'_mean',(sums[query_ids]+10*mean)/(count[query_ids]+10))
        add(axis+'_std',std[query_ids])
    add('query_year',(qt-946684800)/31557600)
    add('query_month',((qt//2629800)%12))
    genre=np.load(OLD/'genre_features.npy')
    add('movie_year',np.load(OLD/'year_features.npy')[qi,0])
    # Per-user genre statistics only from fitting labels, never validation.
    user_genre_sum=np.zeros((401897,genre.shape[1]),np.float64)
    user_genre_count=np.zeros_like(user_genre_sum)
    for g in range(genre.shape[1]):
        active=genre[np.asarray(i),g]>.5
        user_genre_sum[:,g]=np.bincount(u[active],weights=y[active],minlength=401897)
        user_genre_count[:,g]=np.bincount(u[active],minlength=401897)
    target_genres=genre[qi]
    preferences=(user_genre_sum[qu]+5*mean)/(user_genre_count[qu]+5)
    add('user_genre_mean',(preferences*target_genres).sum(-1)/np.maximum(target_genres.sum(-1),1))
    for g in range(genre.shape[1]):add('genre_'+str(g),genre[qi,g])
    hi=np.load(CACHE/f'{query}_hi.npy');hr=np.load(CACHE/f'{query}_hr.npy').astype(float)*.5
    valid=hi>0
    add('history_count',valid.sum(-1))
    add('recent_mean',(hr*valid).sum(-1)/np.maximum(valid.sum(-1),1))
    add('recent_last_rating',hr[:,0])
    add('recent_std',np.sqrt(np.maximum((hr**2*valid).sum(-1)/np.maximum(valid.sum(-1),1)-((hr*valid).sum(-1)/np.maximum(valid.sum(-1),1))**2,0)))
    for horizon in [3,5,10]:
        add(f'recent{horizon}_mean',(hr[:,:horizon]*valid[:,:horizon]).sum(-1)/np.maximum(valid[:,:horizon].sum(-1),1))
    semantics=np.load(OLD/'qwen_movie_embeddings.npy')
    for d in range(16):add(f'qwen_{d}',semantics[qi,d])
    # Exact old-round raw predictions, identical full-refit combination.
    if full:
        als=dict(np.load(OLD/'als_full.npz'))
        als_p=np.einsum('ij,ij->i',als['user'][qu],als['item'][qi])
        als_p[~(als['known_user'][qu]&als['known_item'][qi])]=float(als['mean'])
        qwen=np.load(OLD/'test_neural_no_als_init_seed20261123.npy')
        mf=np.load(OLD/'test_neural_random_mf_only_seed20261123.npy')
        previous=np.clip(.25*qwen+.55*mf+.2*als_p,.5,5)
    else:
        als_p=np.load(OLD/'als_valid_predictions.npy');qwen=np.load(OLD/'valid_neural_no_als_init_seed20261123.npy')
        mf=np.load(OLD/'valid_neural_random_mf_only_seed20261123.npy');previous=np.load(OLD/'valid_final_ensemble.npy')
    for name,val in [('old_als',als_p),('old_qwen',qwen),('old_mf',mf),('old_ensemble',previous)]:add(name,val)
    for name in ['mf64_r003','mf64_r03','mf64_r08','mf128_r04','svdpp32_r03','memory32_time_r03','memory32_notime_r03','mf64_time_r03']:
        path=CACHE/f'{query}_{name}.npy'
        if path.exists():add(name,np.load(path))
    # Candidate-conditioned memory statistics read only already observed prefix
    # labels. User/item vectors are trained on the fitting split, not the query.
    checkpoint=STAGE/'models'/f'advanced_mf64_r03_{"refit" if full else "validation"}.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']
        user=state['user.weight'].numpy();item=state['item.weight'].numpy()
        ub=state['ub.weight'].numpy()[:,0];ib=state['ib.weight'].numpy()[:,0]
        add('latent_user_norm',np.linalg.norm(user[qu],axis=-1))
        add('latent_item_norm',np.linalg.norm(item[qi],axis=-1))
        add('latent_user_bias',ub[qu]);add('latent_item_bias',ib[qi])
        for d in range(16):
            add(f'latent_user_{d}',user[qu,d]);add(f'latent_item_{d}',item[qi,d])
        aggregate={k:np.zeros(len(qu),np.float32) for k in ['history_mf_cosine_max','history_mf_weighted_rating','history_mf_weighted_residual','qwen_history_cosine_max','qwen_history_weighted_rating']}
        for begin in range(0,len(qu),4096):
            end=min(begin+4096,len(qu));hids=hi[begin:end];mask=hids>0
            target=item[qi[begin:end]];past=item[hids]
            cosine=np.einsum('bd,bld->bl',target,past)/(np.maximum(np.linalg.norm(target,axis=-1)[:,None]*np.linalg.norm(past,axis=-1),1e-6))
            logits=np.where(mask,cosine/.2,-30);weight=np.exp(logits-logits.max(-1,keepdims=True))*mask;weight/=np.maximum(weight.sum(-1,keepdims=True),1e-8)
            past_pred=mean+ub[qu[begin:end],None]+ib[hids]+np.einsum('bd,bld->bl',user[qu[begin:end]],past)
            aggregate['history_mf_cosine_max'][begin:end]=np.where(mask,cosine,-1).max(-1)
            aggregate['history_mf_weighted_rating'][begin:end]=(weight*hr[begin:end]).sum(-1)
            aggregate['history_mf_weighted_residual'][begin:end]=(weight*(hr[begin:end]-past_pred)).sum(-1)
            scos=np.einsum('bd,bld->bl',semantics[qi[begin:end]],semantics[hids])
            sw=np.exp(np.where(mask,scos/.2,-30)-np.where(mask,scos/.2,-30).max(-1,keepdims=True))*mask;sw/=np.maximum(sw.sum(-1,keepdims=True),1e-8)
            aggregate['qwen_history_cosine_max'][begin:end]=np.where(mask,scos,-1).max(-1)
            aggregate['qwen_history_weighted_rating'][begin:end]=(sw*hr[begin:end]).sum(-1)
        for name,values in aggregate.items():add(name,values)
    matrix=np.stack(columns,axis=1)
    assert np.isfinite(matrix).all()
    np.save(CACHE/f'{query}_features.npy',matrix)
    source_hashes={}
    for name in names:
        source=CACHE/f'{query}_{name}.npy'
        if source.exists():source_hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    (LOGS/f'{query}_features.json').write_text(json.dumps({'names':names,'rows':len(qu),'columns':len(names),'fit_rows':len(u),'seconds':time.perf_counter()-start,'labels_used':'fitting ratings only; query labels excluded','neural_prediction_hashes':source_hashes,'feature_matrix_sha256':hashlib.sha256((CACHE/f'{query}_features.npy').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    print(json.dumps({'query':query,'rows':len(qu),'features':len(names),'seconds':time.perf_counter()-start}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');features(p.parse_args().full)

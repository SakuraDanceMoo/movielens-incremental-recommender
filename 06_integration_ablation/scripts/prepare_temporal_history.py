"""Strict timestamp-prefix memory for every supervised training/query pair."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/'06_integration_ablation';OLD=STAGE/'data/processed/transformer';CACHE=STAGE/'data/processed/advanced'


def pack_key(users,timestamps):
    return (users.astype(np.uint64)<<np.uint64(32))+timestamps.astype(np.uint64)


def write_history(name,users,times,keys,ordered_users,items,ratings,length=16):
    hi=np.lib.format.open_memmap(CACHE/f'{name}_hi.npy',mode='w+',dtype=np.int32,shape=(len(users),length))
    hr=np.lib.format.open_memmap(CACHE/f'{name}_hr.npy',mode='w+',dtype=np.uint8,shape=(len(users),length))
    nonzero=0;checked=0
    for start in range(0,len(users),250000):
        stop=min(start+250000,len(users))
        qkey=pack_key(users[start:stop],times[start:stop])
        end=np.searchsorted(keys,qkey,side='left')
        index=end[:,None]-np.arange(1,length+1)[None,:]
        clipped=np.maximum(index,0)
        mask=(index>=0)&(ordered_users[clipped]==users[start:stop,None])
        # Independent strictness assertion is evaluated on every nonpadding token.
        assert np.all((keys[clipped]<qkey[:,None])|~mask)
        hi[start:stop]=np.where(mask,items[clipped],0)
        hr[start:stop]=np.where(mask,ratings[clipped],0)
        nonzero+=int(mask.sum());checked+=mask.size
        if start and start%5000000==0:print(json.dumps({'split':name,'history_rows':stop,'total':len(users)}),flush=True)
    hi.flush();hr.flush()
    return {'rows':len(users),'tokens':length,'nonpadding_tokens':nonzero,'checked_tokens':checked,
            'strict_timestamp_assertions_passed':True,'average_history':nonzero/len(users)}


def main(full=False):
    CACHE.mkdir(parents=True,exist_ok=True)
    LOGS=STAGE/'logs/advanced';LOGS.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();tag='full' if full else 'fit'
    u,i,y=[np.load(OLD/f'{tag}_{k}.npy') for k in ['u','i','y']]
    if full:
        t=pl.read_csv(ROOT/'data/raw/hybrid/ratings.csv',columns=['timestamp'],schema_overrides={'timestamp':pl.Int64})['timestamp'].to_numpy()
    else:t=np.load(OLD/'fit_t.npy')
    key=pack_key(u,t)
    order=np.argsort(key,kind='stable')
    keys=key[order];ou=u[order];oi=i[order];oy=y[order];ot=t[order]
    del key,order,u,i,y,t
    for k,v in [('u',ou),('i',oi),('y',oy),('t',ot)]:np.save(CACHE/f'{tag}_{k}.npy',v)
    rating_tokens=np.rint(oy*2).astype(np.uint8)
    report={tag:write_history(tag,ou,ot,keys,ou,oi,rating_tokens)}
    query='test' if full else 'valid'
    qu,qi,qt=[np.load(OLD/f'{query}_{k}.npy') for k in ['u','i','t']]
    for k,v in [('u',qu),('i',qi),('t',qt)]:np.save(CACHE/f'{query}_{k}.npy',v)
    if not full:np.save(CACHE/'valid_y.npy',np.load(OLD/'valid_y.npy'))
    report[query]=write_history(query,qu,qt,keys,ou,oi,rating_tokens)
    report.update({'elapsed_seconds':time.perf_counter()-start,
                   'protocol':'last16 fitting ratings strictly earlier than query timestamp; same-timestamp events excluded; no query labels in history',
                   'fit_rows':len(ou),'query_rows':len(qu)})
    (LOGS/f'temporal_history_{tag}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');a=p.parse_args();main(a.full)

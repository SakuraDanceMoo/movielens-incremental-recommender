"""Temporal history-conditioned explicit recommendation and strong MF controls."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from transformer_regression import seed_all

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/'06_integration_ablation'
OLD=STAGE/'data/processed/transformer'
CACHE=STAGE/'data/processed/advanced'
LOGS=STAGE/'logs/advanced'
MODELS=STAGE/'models'


def development_mask(users: np.ndarray) -> np.ndarray:
    # Held-out user label selection is separated for this round. Earlier project
    # models have already used the original validation labels; not a new test set.
    return (users.astype(np.int64)*2654435761 % 5)!=0


class HistoryRecommender(nn.Module):
    def __init__(self,dim=64,mode='mf',mean=3.5391497099,length=16,temporal=False,no_ratings=False):
        super().__init__()
        self.dim,self.mode,self.mean,self.length=dim,mode,mean,length
        self.temporal,self.no_ratings=temporal,no_ratings
        self.user=nn.Embedding(401897,dim,padding_idx=0)
        self.item=nn.Embedding(175171,dim,padding_idx=0)
        nn.init.normal_(self.user.weight,std=.04)
        nn.init.normal_(self.item.weight,std=.04)
        self.ub=nn.Embedding(401897,1,padding_idx=0)
        self.ib=nn.Embedding(175171,1,padding_idx=0)
        nn.init.zeros_(self.ub.weight);nn.init.zeros_(self.ib.weight)
        self.domain=nn.Embedding(2,1);nn.init.zeros_(self.domain.weight)
        with torch.no_grad():self.item.weight[0].zero_();self.user.weight[0].zero_()
        if temporal:
            self.source_time=nn.Embedding(82,1);nn.init.zeros_(self.source_time.weight)
            self.item_time=nn.Embedding(175171*9,1);nn.init.zeros_(self.item_time.weight)
        if mode!='mf':
            self.history=nn.Embedding(175171,dim,padding_idx=0)
            nn.init.normal_(self.history.weight,std=.025)
            with torch.no_grad():self.history.weight[0].zero_()
            self.rating=nn.Embedding(11,dim,padding_idx=0)
            nn.init.normal_(self.rating.weight,std=.025)
            self.query=nn.Linear(2*dim,dim)
            self.key=nn.Linear(dim,dim)
            self.value=nn.Linear(dim,dim)
            self.norm=nn.LayerNorm(dim)
            self.ff=nn.Sequential(nn.Linear(dim,2*dim),nn.SiLU(),nn.Linear(2*dim,dim))
            self.mix=nn.Linear(2*dim,dim)
            self.position=nn.Parameter(torch.randn(length,dim)*.01)
            self.history_scale=nn.Parameter(torch.tensor(.2))
            self.residual_scale=nn.Embedding(2,1);nn.init.constant_(self.residual_scale.weight,.2)

    def forward(self,u,i,hi=None,hr=None,t=None):
        a,b=self.user(u),self.item(i)
        recent_residual=None
        if self.mode!='mf':
            h=self.history(hi)
            valid=hi!=0
            implicit=(h*valid.unsqueeze(-1)).sum(1)/valid.sum(1).clamp_min(1).sqrt().unsqueeze(-1)
            if self.mode=='svdpp':
                a=a+self.history_scale*implicit
            else:
                h=h+(0 if self.no_ratings else self.rating(hr.long()))+self.position
                h=self.norm(h)
                q=self.query(torch.cat([a,b],-1)).unsqueeze(1)
                # Every token is strictly earlier than the query; no query label
                # or future event can enter keys/values.
                batch,lenh,_=h.shape
                q=q.view(batch,1,4,self.dim//4).transpose(1,2)
                k=self.key(h).view(batch,lenh,4,self.dim//4).transpose(1,2)
                v=self.value(h).view(batch,lenh,4,self.dim//4).transpose(1,2)
                memory=F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:]).transpose(1,2).reshape(batch,self.dim)
                memory=memory+self.ff(memory)
                memory=memory*valid.any(-1,keepdim=True)
                gate=torch.sigmoid(self.mix(torch.cat([a,b],-1)))
                a=a+self.history_scale*(gate*memory+(1-gate)*implicit)
                if not self.no_ratings:
                    past_prediction=self.mean+self.ub(u)+self.ib(hi).squeeze(-1)+(self.user(u).unsqueeze(1)*self.item(hi)).sum(-1)
                    residual=(hr.float()*.5-past_prediction)*valid
                    recent_residual=residual.sum(1)/(valid.sum(1)+5.)
        pred=self.mean+(a*b).sum(-1)+self.ub(u).squeeze(-1)+self.ib(i).squeeze(-1)+self.domain((u>200948).long()).squeeze(-1)
        if recent_residual is not None:pred=pred+self.residual_scale((u>200948).long()).squeeze(-1)*recent_residual
        if self.temporal:
            year=(t//31557600-20).clamp(0,40).long()
            pred=pred+self.source_time((u>200948).long()*41+year).squeeze(-1)+self.item_time(i.long()*9+(year//5)).squeeze(-1)
        reg=(self.user(u).square().sum(-1).mean()+b.square().sum(-1).mean())
        if self.mode!='mf':reg=reg+.1*self.history(hi).square().sum(-1).mean()
        if self.temporal:reg=reg+.3*self.item_time(i.long()*9+(year//5)).square().mean()
        return pred.float(),reg


@torch.inference_mode()
def prediction(model,arrays,batch=16384):
    model.eval();result=[]
    for start in range(0,len(arrays['u']),batch):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            args=[arrays[k][start:start+batch] for k in ['u','i']]
            if model.mode!='mf':args += [arrays[k][start:start+batch] for k in ['hi','hr']]
            p,_=model(*args,t=arrays['t'][start:start+batch] if model.temporal else None)
        result.append(p.cpu().numpy())
    return np.concatenate(result)


def evaluate(y,p,users,mask):
    error=(np.clip(p.astype(float),.5,5)-y.astype(float))**2
    return {'rmse':float(np.sqrt(error[mask].mean())),
            'movielens_rmse':float(np.sqrt(error[mask&(users<=200948)].mean())),
            'synthetic_rmse':float(np.sqrt(error[mask&(users>200948)].mean()))}


def train(args):
    seed_all(args.seed);CACHE.mkdir(parents=True,exist_ok=True);LOGS.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    full=args.refit
    data=CACHE if args.mode!='mf' or args.temporal else OLD
    split='full' if full else 'fit'
    arrays={k:torch.from_numpy(np.load(data/f'{split}_{k}.npy')).cuda() for k in ['u','i','y']}
    if args.temporal:arrays['t']=torch.from_numpy(np.load(data/f'{split}_t.npy')).cuda()
    if args.mode!='mf':
        for k in ['hi','hr']:arrays[k]=torch.from_numpy(np.load(CACHE/f'{split}_{k}.npy')).cuda()
    valid_np={k:np.load(data/f'valid_{k}.npy') for k in ['u','i','y']}
    valid={k:torch.from_numpy(valid_np[k]).cuda() for k in ['u','i']}
    if args.temporal:valid['t']=torch.from_numpy(np.load(data/'valid_t.npy')).cuda()
    if args.mode!='mf':
        for k in ['hi','hr']:valid[k]=torch.from_numpy(np.load(CACHE/f'valid_{k}.npy')).cuda()
    dev=development_mask(valid_np['u'])
    mean=float(np.load(OLD/f'als_{split}.npz')['mean'])
    model=HistoryRecommender(args.dim,args.mode,mean,temporal=args.temporal,no_ratings=args.no_ratings).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0,fused=True)
    generator=torch.Generator(device='cuda').manual_seed(args.seed+777)
    best=math.inf;best_epoch=0;history=[];start=time.perf_counter()
    path=MODELS/f'advanced_{args.name}_{"refit" if full else "validation"}.pt'
    n=len(arrays['u']);patience=0
    for epoch in range(1,args.epochs+1):
        model.train();epstart=time.perf_counter();order=torch.randperm(n,device='cuda',dtype=torch.int32,generator=generator)
        total=torch.zeros((),device='cuda')
        # Longer budget and slower annealing than the first-round four epochs.
        optimizer.param_groups[0]['lr']=args.lr*(.85**max(0,epoch-3))
        for step,begin in enumerate(range(0,n,args.batch)):
            index=order[begin:begin+args.batch];optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                x=[arrays[k][index] for k in ['u','i']]
                if args.mode!='mf':x += [arrays[k][index] for k in ['hi','hr']]
                pred,reg=model(*x,t=arrays['t'][index] if args.temporal else None)
                mse=F.mse_loss(pred,arrays['y'][index]);loss=mse+args.reg*reg
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();total+=mse.detach()*len(index)
            if step and step%600==0:print(json.dumps({'name':args.name,'epoch':epoch,'step':step,'seconds':round(time.perf_counter()-epstart,1)}),flush=True)
        torch.cuda.synchronize()
        row={'epoch':epoch,'train_mse':float(total.item()/n),'seconds':time.perf_counter()-epstart}
        if not full:
            values=prediction(model,valid,args.batch)
            metrics=evaluate(valid_np['y'],values,valid_np['u'],dev)
            row.update(development=metrics)
            if metrics['rmse']<best:
                best=metrics['rmse'];best_epoch=epoch;patience=0
                np.save(CACHE/f'valid_{args.name}.npy',values)
                torch.save({'state_dict':model.state_dict(),'config':vars(args),'mean':mean,'best_epoch':epoch},path)
            else:patience+=1
        history.append(row)
        (LOGS/f'history_{args.name}_{"refit" if full else "validation"}.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(json.dumps({'name':args.name,**row}),flush=True)
        if not full and patience>=3:
            print(json.dumps({'name':args.name,'early_stop_epoch':epoch,'development_patience':3}),flush=True)
            break
    if full:
        torch.save({'state_dict':model.state_dict(),'config':vars(args),'mean':mean,'best_epoch':args.epochs},path)
        test={k:torch.from_numpy(np.load(data/f'test_{k}.npy')).cuda() for k in ['u','i']}
        if args.temporal:test['t']=torch.from_numpy(np.load(data/'test_t.npy')).cuda()
        if args.mode!='mf':
            for k in ['hi','hr']:test[k]=torch.from_numpy(np.load(CACHE/f'test_{k}.npy')).cuda()
        np.save(CACHE/f'test_{args.name}.npy',prediction(model,test,args.batch))
    result={'config':vars(args),'best_epoch':best_epoch if not full else args.epochs,
            'development_rmse':best if not full else None,'seconds':time.perf_counter()-start,
            'fit_rows':n,'development_rows':int(dev.sum()),'audit_rows':int((~dev).sum()),
            'parameters':sum(p.numel() for p in model.parameters()),'history':history,
            'selection':'development users only; audit values withheld until this round is frozen'}
    (LOGS/f'result_{args.name}_{"refit" if full else "validation"}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--mode',choices=['mf','svdpp','attention'],default='mf')
    p.add_argument('--dim',type=int,default=64);p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--batch',type=int,default=16384);p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--reg',type=float,default=.003);p.add_argument('--seed',type=int,default=20261123)
    p.add_argument('--refit',action='store_true')
    p.add_argument('--temporal',action='store_true');p.add_argument('--no-ratings',action='store_true')
    train(p.parse_args())

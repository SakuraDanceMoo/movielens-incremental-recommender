"""Protocol behavior tests: time ties, user isolation, masks and empty memory."""
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
import prepare_temporal_history as temporal
from advanced_rating import HistoryRecommender,development_mask


def main():
    checks=[]
    def check(value,name):
        assert value,name
        checks.append(name)
    users=np.array([1,1,1,2,2]);times=np.array([10,20,20,5,25])
    keys=temporal.pack_key(users,times);items=np.array([11,12,13,21,22]);ratings=np.array([2,4,6,8,10],np.uint8)
    old=temporal.CACHE
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3]/'.venv/tmp',prefix='history-test-') as tmp:
        temporal.CACHE=Path(tmp)
        report=temporal.write_history('query',np.array([1,1,2,3,1]),np.array([20,21,5,100,1]),keys,users,items,ratings,length=4)
        hi=np.load(Path(tmp)/'query_hi.npy');hr=np.load(Path(tmp)/'query_hr.npy')
        check(hi[0].tolist()==[11,0,0,0],'same-timestamp records excluded')
        check(hi[1].tolist()==[13,12,11,0],'strictly earlier history in reverse chronological order')
        check(not hi[2:].any(),'no previous-user leakage; unseen users and empty histories padded')
        check(hr[1].tolist()==[6,4,2,0],'rating tokens remain aligned with historical items')
        check(report['nonpadding_tokens']==4,'history accounting agrees with constructed tokens')
    temporal.CACHE=old
    u=np.arange(1,10001);dev=development_mask(u)
    h=u.astype(np.uint64);h=(h^(h>>np.uint64(16)))*np.uint64(0x45d9f3b);h=h^(h>>np.uint64(16));fold=(h%np.uint64(5)).astype(int)
    seen=np.zeros(len(u),int)
    for f in range(5):
        train=dev&(fold!=f);held=dev&(fold==f)
        check(not np.any(train&(~dev)) and not np.any(held&train) and train.any() and held.any(),f'fold{f}: training/held/audit disjoint')
        seen+=held
    check(np.all(seen[dev]==1) and np.all(seen[~dev]==0),'each development user predicted out of fold exactly once')
    torch.manual_seed(42)
    model=HistoryRecommender(8,'attention',temporal=True,no_ratings=True).eval()
    qu=torch.tensor([1,2]);qi=torch.tensor([11,21]);hi=torch.zeros((2,16),dtype=torch.long);hr=torch.zeros_like(hi);t=torch.tensor([1000000000,1700000000])
    with torch.inference_mode():
        a,_=model(qu,qi,hi,hr,t=t)
        b,_=model(qu,qi,hi,torch.ones_like(hr)*10,t=t)
        check(bool(torch.isfinite(a).all()),'all-padding attention returns finite scores')
        check(torch.equal(a,b),'no-ratings branch ignores rating token values')
        hi[:,0]=3
        c,_=model(qu,qi,hi,hr,t=t);d,_=model(qu,qi,hi,torch.ones_like(hr)*10,t=t)
        check(torch.equal(c,d),'no-ratings branch ignores ratings with nonempty history')
    report={'passed':True,'check_count':len(checks),'checks':checks}
    output=Path(__file__).resolve().parents[1]/'logs/advanced/protocol_tests.json'
    output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()

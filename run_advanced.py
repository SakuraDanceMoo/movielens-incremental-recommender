"""Stage-six advanced history and capacity experiments, shared D-drive runtime."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ENV=ROOT.parent/'.venv'
SCRIPT=ROOT/'06_integration_ablation/scripts/advanced_rating.py'
TRIALS=[
 ['--name','mf64_r003','--mode','mf','--dim','64','--reg','.003','--epochs','8'],
 ['--name','mf64_r03','--mode','mf','--dim','64','--reg','.03','--epochs','10'],
 ['--name','memory32_time_r03','--mode','attention','--dim','32','--reg','.03','--epochs','6','--temporal'],
 ['--name','mf64_r08','--mode','mf','--dim','64','--reg','.08','--epochs','10'],
 ['--name','mf128_r04','--mode','mf','--dim','128','--reg','.04','--epochs','10'],
 ['--name','svdpp32_r03','--mode','svdpp','--dim','32','--reg','.03','--epochs','8'],
 ['--name','memory32_notime_r03','--mode','attention','--dim','32','--reg','.03','--epochs','8'],
 ['--name','mf64_time_r03','--mode','mf','--dim','64','--reg','.03','--epochs','8','--temporal'],
]


def main():
    p=argparse.ArgumentParser();p.add_argument('--trials',action='store_true');p.add_argument('--refit',nargs='*')
    p.add_argument('--calibration-ablations',action='store_true')
    args=p.parse_args()
    for key,path in {'TEMP':ENV/'tmp','TMP':ENV/'tmp','MPLCONFIGDIR':ENV/'cache/matplotlib','UV_CACHE_DIR':ENV/'cache/uv','HF_HOME':ENV/'cache/huggingface'}.items():
        path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    if args.trials:
        for trial in TRIALS:
            print('EXPERIMENT',' '.join(trial),flush=True)
            subprocess.run([sys.executable,'-u',str(SCRIPT),*trial],check=True,cwd=ROOT)
    if args.calibration_ablations:
        for suffix,flag in [('no_semantics','--without-semantics'),('no_history','--without-history'),('no_neural','--without-neural')]:
            subprocess.run([sys.executable,'-u',str(SCRIPT.with_name('calibrate_advanced.py')),
                            '--name','calibration_v2_d8_'+suffix,'--depth','8','--iterations','1200',flag],check=True,cwd=ROOT)
        subprocess.run([sys.executable,'-u',str(SCRIPT.with_name('calibrate_advanced.py')),
                        '--name','calibration_v2_d8_seed2','--depth','8','--iterations','1200','--seed','20261124'],check=True,cwd=ROOT)
    if args.refit:
        for name in args.refit:
            r=json.loads((ROOT/f'06_integration_ablation/logs/advanced/result_{name}_validation.json').read_text())
            c=r['config']
            cmd=[sys.executable,'-u',str(SCRIPT),'--name',name,'--mode',c['mode'],'--dim',str(c['dim']),
                 '--reg',str(c['reg']),'--lr',str(c['lr']),'--epochs',str(r['best_epoch']),'--seed',str(c['seed']),'--refit']
            if c.get('temporal'):cmd+=['--temporal']
            if c.get('no_ratings'):cmd+=['--no-ratings']
            subprocess.run(cmd,check=True,cwd=ROOT)


if __name__=='__main__':main()

"""Reproduce the stage-six Qwen3/Transformer rating-regression experiments."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = ROOT.parent / ".venv"
STAGE = ROOT / "06_integration_ablation"
SCRIPTS = STAGE / "scripts"


def execute(script: str, *args: str) -> None:
    command=[sys.executable,"-u",str(SCRIPTS/script),*args]
    print("RUN",script," ".join(args),flush=True)
    subprocess.run(command,check=True,cwd=ROOT)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/raw/hybrid')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--encode',action='store_true')
    parser.add_argument('--experiments',action='store_true')
    parser.add_argument('--extra-ablations',action='store_true')
    parser.add_argument('--refit',action='store_true')
    args=parser.parse_args()
    for key,value in {"HF_HOME":ENV/'cache/huggingface',"UV_CACHE_DIR":ENV/'cache/uv',
                      "TEMP":ENV/'tmp',"TMP":ENV/'tmp',"MPLCONFIGDIR":ENV/'cache/matplotlib'}.items():
        value.mkdir(parents=True,exist_ok=True)
        os.environ[key]=str(value)
    if args.prepare:
        execute('prepare_transformer_data.py','--data-dir',str(args.data_dir))
        execute('cache_als_factors.py','--data-dir',str(args.data_dir))
    if args.encode:
        execute('download_qwen_encoder.py')
        execute('encode_movie_semantics.py')
    if args.experiments:
        for seed in [20261123,20261124]:
            for variant in ['full','no_semantics','no_attention','no_als_init','mf_only']:
                execute('transformer_regression.py','--variant',variant,'--seed',str(seed),'--epochs','4')
        execute('summarize_transformer.py')
    if args.extra_ablations:
        # Match the best random-initialization Transformer with its own
        # component removals; warm-start controls alone cannot establish this.
        for seed in [20261123,20261124]:
            for variant in ['random_no_semantics','random_no_attention','random_mf_only']:
                execute('transformer_regression.py','--variant',variant,'--seed',str(seed),'--epochs','4')
        execute('summarize_transformer.py')
    if args.refit:
        execute('ensemble_transformer.py')
        selection=json.loads((STAGE/'logs/transformer/ensemble_selection.json').read_text())
        if not (STAGE/'data/processed/transformer/als_full.npz').exists():
            execute('cache_als_factors.py','--data-dir',str(args.data_dir),'--full')
        candidates=[('no_als_init',selection['seed']),('random_mf_only',selection['seed'])]
        for variant,seed in candidates:
            result=json.loads((STAGE/'logs/transformer'/f'result_{variant}_validation_seed{seed}.json').read_text())
            execute('transformer_regression.py','--variant',variant,'--seed',str(seed),
                    '--epochs',str(result['best_epoch']),'--full-refit')
        execute('ensemble_transformer.py')
    if not any([args.prepare,args.encode,args.experiments,args.extra_ablations,args.refit]):
        parser.error('Choose --prepare, --encode, --experiments and/or --refit')


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""20-environment benchmark orchestrator for identical TensorLy/TensorKitchen inputs.

Each shard covers five of 20 reward realizations and ranks 3, 5, 8.
The existing benchmark.py is used unchanged for simulation, TensorLy and
common downstream evaluation. The official TensorKitchen code executes
in benchmark_full.jl. Python custom CP-ALS historical numbers are NOT
silently labeled TensorLy.

Usage:
    python full_suite.py generate --start 0 --end 5
    python full_suite.py tensorly --start 0 --end 5
    # julia benchmark_full.jl ...
    python full_suite.py evaluate --start 0 --end 5

See README_FULL.md for run configuration and limitations.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

import benchmark as old

HERE = Path(__file__).resolve().parent
BASE = HERE / "results" / "full"
RANKS = (3, 5, 8)
RESTARTS = 3
MAXITER = 240

def folders(start, end):
    for env in range(start, end):
        for rank in RANKS:
            yield env, rank, BASE / f"env_{env:02d}" / f"rank_{rank}"

def make_one(env, rank, folder):
    folder.mkdir(parents=True, exist_ok=True)
    # The same RW agent random streams are reused across reward environments,
    # but the potential reward table varies by environment.
    original = old.generate_env
    def varying_environment(seed, T, block=30):
        return original(seed + 979 * env, T, block)
    old.generate_env = varying_environment
    try:
        old.generate("reference", folder)
    finally:
        old.generate_env = original
    metadata_path = folder / "settings.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        env_index=env,
        env_seed=14131 + 979 * env,
        rank=rank,
        restarts=RESTARTS,
        maxiter=MAXITER,
        experiment="full TensorLy vs TensorKitchen matched inputs",
    )
    metadata_path.write_text(json.dumps(metadata, indent=2))
    # Only one source for every method: these Float64 column-major binary arrays.
    assert all((folder / f"{task}_{part}.f64").exists()
               for task in ("phen", "cont") for part in ("train", "test"))

def warm_python():
    import numpy as np
    from tensorly.decomposition import parafac
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        parafac(np.random.default_rng(1).random((12, 15, 4)), rank=3,
                n_iter_max=3, init="random", random_state=1)

def metrics_summary(rows):
    import numpy as np
    from collections import defaultdict
    grouping=defaultdict(list)
    keys=('train_rel_error','test_rel_error','fit_seconds','accuracy',
          'r2_alpha','r2_beta','mae_alpha','mae_beta')
    for row in rows:
        grouping[(row['task'], row['rank'], row['method'])].append(row)
    outputs=[]
    for (task,rank,method),group in sorted(grouping.items()):
        d={'task':task,'rank':rank,'method':method,'n_environment':len(group)}
        for key in keys:
            vals=np.array([float(g[key]) for g in group
                           if g.get(key) is not None and str(g[key]).strip() not in ('','nan','NaN')])
            if len(vals):
                sd=float(np.std(vals,ddof=1)) if len(vals)>1 else float('nan')
                half=1.96*sd/np.sqrt(len(vals)) if len(vals)>1 else float('nan')
                d.update({f'{key}_mean':float(vals.mean()),f'{key}_sd':sd,
                          f'{key}_ci95_low':float(vals.mean()-half),
                          f'{key}_ci95_high':float(vals.mean()+half)})
        outputs.append(d)
    return outputs

def paired_summary(rows):
    from collections import defaultdict
    import numpy as np
    compare=('train_rel_error','test_rel_error','fit_seconds','accuracy',
             'r2_alpha','r2_beta')
    grp=defaultdict(dict)
    for r in rows:
        grp[(int(r['env_index']),int(r['rank']),r['task'])][r['method']]=r
    entries=[]
    for method in ('TensorKitchen','TensorKitchen_NN'):
        for rank in RANKS:
            for task in ('phen','cont'):
                for key in compare:
                    deltas=[]
                    for (env,rr,tt),g in grp.items():
                        if rr!=rank or tt!=task or method not in g or 'TensorLy' not in g:
                            continue
                        if key not in g[method] or key not in g['TensorLy']:
                            continue
                        try:
                            a=float(g[method][key]);b=float(g['TensorLy'][key])
                        except (ValueError,TypeError):continue
                        if np.isfinite(a) and np.isfinite(b):deltas.append(a-b)
                    if not deltas:continue
                    a=np.array(deltas)
                    sd=float(a.std(ddof=1)) if len(a)>1 else float('nan')
                    half=1.96*sd/np.sqrt(len(a)) if len(a)>1 else float('nan')
                    entries.append(dict(task=task,rank=rank,method=method,
                        baseline='TensorLy',metric=key,n=len(a),mean_difference=float(a.mean()),
                        ci95_low=float(a.mean()-half),ci95_high=float(a.mean()+half)))
    return entries

def write_csv(path, rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase', choices=['generate','tensorly','evaluate'])
    p.add_argument('--start',type=int,default=int(os.getenv('RW_ENV_START','0')))
    p.add_argument('--end',type=int,default=int(os.getenv('RW_ENV_END','5')))
    a=p.parse_args()
    if not (0<=a.start<a.end<=20):raise ValueError('environment range must be within 0..20')
    first=time.perf_counter()
    if a.phase=='tensorly': warm_python()
    rows=[];fits=[]
    for env,rank,folder in folders(a.start,a.end):
        if a.phase=='generate':
            make_one(env,rank,folder)
        elif a.phase=='tensorly':
            old.fit_tensorly(folder)
        else:
            old.evaluate(folder)
            import pandas as pd
            rows.extend(pd.read_csv(folder/'comparison_metrics.csv').to_dict('records'))
            for f in ('tensorly_fits.csv','tensorkitchen_fits.csv'):
                for x in pd.read_csv(folder/f).to_dict('records'):
                    fits.append(dict(env_index=env,rank=rank,**x))
        print(f'COMPLETED phase={a.phase} env={env} rank={rank} elapsed_seconds={time.perf_counter()-first:.1f}',flush=True)
    if a.phase=='evaluate':
        shard=BASE/f"shard_{a.start:02d}_{a.end:02d}"
        shard.mkdir(parents=True,exist_ok=True)
        write_csv(shard/'metrics.csv',rows)
        write_csv(shard/'all_starts.csv',fits)
        write_csv(shard/'summary.csv',metrics_summary(rows))
        write_csv(shard/'paired.csv',paired_summary(rows))
        print(f'FINISHED {len(rows)} head-to-head score rows, {len(fits)} fits; shard={shard}',flush=True)
if __name__=='__main__':main()

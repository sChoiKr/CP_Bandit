#!/usr/bin/env python3
"""Strict matched-start RW benchmark: same X, initial CP factors, rank, budget and decoder.
Run phases: generate, tensorly, evaluate. Four jobs cover the 20 environments.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import csv
import numpy as np
import benchmark as base
import full_suite as full

ROOT = Path(__file__).resolve().parent / "results" / "matched"
RANKS = (3,5,8)
STARTS = 3
ITER = 240

def iterator(lo,hi):
    for e in range(lo,hi):
        for r in RANKS:
            yield e,r,ROOT / f"env_{e:02d}" / f"rank_{r}"

def initial_factors(folder, task, rank, start, nonnegative, dims, env):
    # Generated only once in Python, subsequently read verbatim by both solvers.
    typ = "nn" if nonnegative else "cp"
    rng=np.random.default_rng(940039 + 100_003*env + 101*rank + 23*start +
                              (1 if task=="cont" else 0) + (5 if nonnegative else 0))
    factors=[]
    for dim in dims:
        f = rng.random((dim,rank)) + .05 if nonnegative else rng.standard_normal((dim,rank))
        f /= np.maximum(np.linalg.norm(f,axis=0,keepdims=True),1e-12)
        factors.append(f)
    # Scale rank-one components to a reproducible fixed start.
    factors[0] *= np.sqrt(dims[0] * dims[1]) / max(1.,np.sqrt(rank))
    for mode,f in enumerate(factors,1):
        np.savetxt(folder/f"{task}_init_{typ}_s{start}_factor{mode}.csv",
                   f, delimiter=",",fmt="%.18e")
    return factors

def factor_files(folder,task,rank,start,nonnegative,dims):
    typ="nn" if nonnegative else "cp"
    arrays=[np.loadtxt(folder/f"{task}_init_{typ}_s{start}_factor{k}.csv",
                        delimiter=",",ndmin=2) for k in (1,2,3)]
    assert [a.shape for a in arrays]==[(d,rank) for d in dims]
    assert all(np.isfinite(a).all() for a in arrays)
    if nonnegative:
        assert all(np.min(a)>0 for a in arrays)
    return arrays

def generate(lo,hi):
    for env,rank,folder in iterator(lo,hi):
        folder.mkdir(parents=True,exist_ok=True)
        # Existing full suite data generation, identical to completed prior 20-env test.
        original=full.BASE
        full.BASE=ROOT
        try:
            full.make_one(env,rank,folder)
        finally:
            full.BASE=original
        mpath=folder/"settings.json"
        meta=json.loads(mpath.read_text())
        meta.update(rank=rank,restarts=STARTS,maxiter=ITER,tol=0.,
                    env_index=env,comparison="shared explicit initial factors / 240 sweeps")
        mpath.write_text(json.dumps(meta,indent=2))
        hashes={}
        for task in ("phen","cont"):
            dims=(meta[f"{task}_train"],meta["n_trials"],4)
            for start in range(STARTS):
                initial_factors(folder,task,rank,start,False,dims,env)
                initial_factors(folder,task,rank,start,True,dims,env)
            for part in ("train","test"):
                f=folder/f"{task}_{part}.f64"
                hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
        (folder/"tensor_sha256.json").write_text(json.dumps(hashes,indent=2))
        print(f"GENERATED env={env} rank={rank}",flush=True)

def fit_python(lo,hi):
    import tensorly as tl
    from tensorly.decomposition import parafac, non_negative_parafac_hals
    # outside timing
    warm=np.random.default_rng(3).random((12,15,4))
    parafac(warm,rank=3,n_iter_max=2,init="random",random_state=4)
    non_negative_parafac_hals(warm,rank=3,n_iter_max=2,init="random",random_state=4)
    for env,rank,folder in iterator(lo,hi):
        meta=json.loads((folder/"settings.json").read_text())
        rows=[]
        for task in ("phen","cont"):
            dims=(meta[f"{task}_train"],meta["n_trials"],4)
            X=base.read_array(folder/f"{task}_train.f64",dims)
            for method,nn in (("TensorLy",False),("TensorLy_NN",True)):
                for start in range(STARTS):
                    initial=factor_files(folder,task,rank,start,nn,dims)
                    init=(np.ones(rank,dtype=np.float64),initial)
                    t0=time.perf_counter()
                    if nn:
                        res,errs=non_negative_parafac_hals(
                            X,rank,n_iter_max=ITER,init=init,tol=0.,
                            normalize_factors=False,exact=False,return_errors=True,
                            cvg_criterion="abs_rec_error")
                    else:
                        res,errs=parafac(
                            X,rank,n_iter_max=ITER,init=init,tol=0.,
                            normalize_factors=False,return_errors=True,
                            l2_reg=0.,linesearch=False,cvg_criterion="abs_rec_error")
                    seconds=time.perf_counter()-t0
                    weight,U=res
                    err=float(np.linalg.norm(X-tl.cp_to_tensor(res))/np.linalg.norm(X))
                    prefix=folder/f"{task}_{method}_s{start}"
                    np.savetxt(str(prefix)+"_weights.csv",np.asarray(weight),delimiter=",",fmt="%.18e")
                    for k,fac in enumerate(U,1):
                        np.savetxt(str(prefix)+f"_factor{k}.csv",fac,delimiter=",",fmt="%.18e")
                    rows.append(dict(task=task,method=method,start=start,
                        train_rel_error=err,iterations=len(errs),fit_seconds=seconds,
                        converged=False))
                    print(f"FITTED env={env} rank={rank} task={task} method={method} "
                          f"start={start} error={err:.7g} seconds={seconds:.4f}",flush=True)
        full.write_csv(folder/"tensorly_fits.csv",rows)

def evaluate(lo,hi):
    rows=[];fits=[]
    for env,rank,folder in iterator(lo,hi):
        base.evaluate(folder)
        import pandas as pd
        rows.extend(dict(row,env_index=env,rank=rank)
            for row in pd.read_csv(folder/"comparison_metrics.csv").to_dict("records"))
        for file in ("tensorly_fits.csv","tensorkitchen_fits.csv"):
            fits.extend(dict(row,env_index=env,rank=rank) for row in
                pd.read_csv(folder/file).to_dict("records"))
        print(f"EVALUATED env={env} rank={rank}",flush=True)
    shard=ROOT/f"shard_{lo:02d}_{hi:02d}"
    shard.mkdir(parents=True,exist_ok=True)
    full.write_csv(shard/"metrics.csv",rows)
    full.write_csv(shard/"all_starts.csv",fits)
    full.write_csv(shard/"summary.csv",full.metrics_summary(rows))
    # Paired by identical environment, rank, task. Never pair NNCPD with unconstrained CPD.
    pairs=[]
    lookup={(int(row["env_index"]),int(row["rank"]),row["task"],row["method"]):row
            for row in rows}
    for rank in RANKS:
        for task in ("phen","cont"):
            for tested,baseline in (("TensorKitchen","TensorLy"),
                                    ("TensorKitchen_NN","TensorLy_NN")):
                for metric in ("train_rel_error","test_rel_error","fit_seconds","accuracy",
                               "r2_alpha","r2_beta"):
                    differences=[]
                    for env in range(lo,hi):
                        a=lookup.get((env,rank,task,tested))
                        b=lookup.get((env,rank,task,baseline))
                        if a is None or b is None:continue
                        try:
                            diff=float(a[metric])-float(b[metric])
                            if np.isfinite(diff):differences.append(diff)
                        except (ValueError,KeyError,TypeError):pass
                    if not differences:continue
                    d=np.asarray(differences)
                    sd=d.std(ddof=1) if len(d)>1 else np.nan
                    pairs.append(dict(rank=rank,task=task,tested=tested,baseline=baseline,
                        metric=metric,n=len(d),mean_difference=d.mean(),
                        std_difference=sd,ci95_low=d.mean()-1.96*sd/np.sqrt(len(d)),
                        ci95_high=d.mean()+1.96*sd/np.sqrt(len(d))))
    full.write_csv(shard/"paired.csv",pairs)
    print(f"SUCCESS matched shard {lo}-{hi}: {len(rows)} observations "
          f"{len(fits)} starts, {len(pairs)} paired rows",flush=True)

if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("phase",choices=["generate","tensorly","evaluate"])
    pa.add_argument("--start",type=int,default=int(os.getenv("RW_ENV_START","0")))
    pa.add_argument("--end",type=int,default=int(os.getenv("RW_ENV_END","5")))
    args=pa.parse_args()
    assert 0<=args.start<args.end<=20
    {"generate":generate,"tensorly":fit_python,"evaluate":evaluate}[args.phase](
        args.start,args.end)

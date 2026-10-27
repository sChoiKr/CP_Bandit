#!/usr/bin/env python3
"""CP rank and local parameter identifiability under RW behavior.
Same behavioral tensor/dataset across all ranks. Mean-response map estimated
using CRN Monte Carlo; Jacobian is a finite difference of mean observations.
Report full-space geometry vs CP-observation-subspace geometry.
"""
import argparse, csv, json, os
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tensorly_tensorkitchen"))
import benchmark as rw

BASE=Path(__file__).resolve().parent/"results"/"rank_geometry"
RANKS=(2,3,4,5,6,8)
T=150
NTRAIN=400
NTEST=100
H_ALPHA=.04
H_BETA=.35
N_MC=int(os.getenv("RW_GEOM_MC","256"))
GRID=[(a,b) for a in (.2,.5,.8) for b in (1.,4.,7.)]

def write_csv(path,rows):
    if not rows:return
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def env_rewards(env):
    return rw.generate_env(14131+979*env,T)

def generate(start,end):
    for env in range(start,end):
        folder=BASE/f"env_{env:02d}"
        folder.mkdir(parents=True,exist_ok=True)
        U=env_rewards(env)
        rng=np.random.default_rng(916522)
        aa=rng.uniform(.05,.95,NTRAIN+NTEST)
        bb=rng.uniform(.5,8.,NTRAIN+NTEST)
        X=rw.continuous_trajectories(aa,bb,U,909991)
        rw.write_array(folder/"train.f64",X[:NTRAIN])
        rw.write_array(folder/"test.f64",X[NTRAIN:])
        np.savetxt(folder/"train_y.csv",np.column_stack((aa[:NTRAIN],bb[:NTRAIN])),delimiter=",")
        np.savetxt(folder/"test_y.csv",np.column_stack((aa[NTRAIN:],bb[NTRAIN:])),delimiter=",")
        json.dump({"env":env,"rank_list":RANKS,"trials":T,"n_train":NTRAIN,"n_test":NTEST,
                   "mc":N_MC,"h_alpha":H_ALPHA,"h_beta":H_BETA,
                   "alpha_grid":[.2,.5,.8],"beta_grid":[1.,4.,7.]},
                  (folder/"settings.json").open("w"),indent=2)
        print("GENERATED env",env,flush=True)

def simulate_mean(alpha,beta,U,uniforms):
    """Vectorized Monte Carlo expected behavioral observation; shared uniform draws."""
    n,t=uniforms.shape
    q=np.full((n,2),.5)
    out=np.zeros((t,4),float)
    rows=np.arange(n)
    for i in range(t):
        z=np.clip(beta*(q[:,0]-q[:,1]),-40,40)
        choice=(uniforms[:,i]>=1/(1+np.exp(-z))).astype(int)
        rew=U[i,choice]
        counts=np.bincount(choice,minlength=2)/n
        rewards=np.bincount(choice,weights=rew,minlength=2)/n
        out[i,:2]=counts
        out[i,2:]=rewards
        q[rows,choice]+=alpha*(rew-q[rows,choice])
    # consistent flattening: vec trial x feature with trial fastest.
    return out.reshape(-1,order="F")

def subspace(folder,rank):
    prefix=folder/f"cp_rank_{rank}"
    w=np.atleast_1d(np.loadtxt(str(prefix)+"_weights.csv",delimiter=","))
    B=np.loadtxt(str(prefix)+"_factor2.csv",delimiter=",",ndmin=2)
    C=np.loadtxt(str(prefix)+"_factor3.csv",delimiter=",",ndmin=2)
    K=np.column_stack([w[k]*np.kron(C[:,k],B[:,k]) for k in range(rank)])
    Q,s,_=np.linalg.svd(K,full_matrices=False)
    threshold=max(K.shape)*np.finfo(float).eps*s[0]
    keep=s>threshold
    return Q[:,keep],s

def geom(J):
    ja,jb=J[:,0],J[:,1]
    sa=np.linalg.norm(ja);sb=np.linalg.norm(jb)
    cosine=float(np.dot(ja,jb)/(sa*sb)) if sa*sb>0 else np.nan
    conditional=float(np.linalg.norm(ja-jb*np.dot(ja,jb)/np.dot(jb,jb))) if sb>0 else sa
    sing=np.linalg.svd(J,compute_uv=False)
    return dict(sigma_min=float(sing[-1]),sigma_max=float(sing[0]),
                alpha_sensitivity=float(sa),beta_sensitivity=float(sb),
                alpha_conditional_sensitivity=conditional,
                alpha_beta_cosine=cosine,
                condition=float(sing[0]/sing[-1]) if sing[-1]>1e-14 else float("inf"))

def evaluate(start,end):
    for env in range(start,end):
        folder=BASE/f"env_{env:02d}"
        U=env_rewards(env)
        rows=[]
        rng=np.random.default_rng(100200+env)
        # Crucial: all ranks share the same estimates of the full-observation Jacobian.
        for a,b in GRID:
            uniforms=rng.random((N_MC,T))
            Ja=(simulate_mean(a+H_ALPHA,b,U,uniforms)-
                simulate_mean(a-H_ALPHA,b,U,uniforms))/(2*H_ALPHA)
            Jb=(simulate_mean(a,b+H_BETA,U,uniforms)-
                simulate_mean(a,b-H_BETA,U,uniforms))/(2*H_BETA)
            J=np.column_stack((Ja,Jb))
            full=geom(J)
            rows.append(dict(env=env,rank=0,alpha=a,beta=b,**full,
                             dictionary_condition=np.nan,dictionary_rank=600))
            for rank in RANKS:
                Q,sv=subspace(folder,rank)
                met=geom(Q.T@J)
                rows.append(dict(env=env,rank=rank,alpha=a,beta=b,**met,
                                 dictionary_condition=float(sv[0]/sv[-1]),
                                 dictionary_rank=Q.shape[1]))
        write_csv(folder/"geometry.csv",rows)
        print("GEOMETRY env",env,"rows",len(rows),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["generate","geometry"])
    p.add_argument("--start",type=int,default=int(os.getenv("RW_ENV_START","0")))
    p.add_argument("--end",type=int,default=int(os.getenv("RW_ENV_END","5")))
    a=p.parse_args()
    (generate if a.phase=="generate" else evaluate)(a.start,a.end)

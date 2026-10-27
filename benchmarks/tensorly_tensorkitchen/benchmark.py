#!/usr/bin/env python3
"""Paired TensorLy vs TensorKitchen benchmark, sharing bitwise-identical RW arrays.
Run: python benchmark.py generate|tensorly|evaluate [--mode smoke|reference].
Historical Python CP baseline used a custom ALS, not TensorLy.
"""
import argparse, csv, json, os, time, warnings
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
GROUPS = [(0.8,4.0),(0.2,4.0),(0.5,0.75),(0.5,7.0)]

def generate_env(seed,T,block=30):
    rng = np.random.default_rng(seed)
    p = np.array([(.8,.2) if (t//block)%2 == 0 else (.2,.8) for t in range(T)])
    return (rng.random((T,2)) < p).astype(np.float64)

# Same RW equations, agent ordering and seeds as original Python benchmark.
# Use NumPy's legacy RandomState; original simulator called np.random.seed under Numba.
def discrete_trajectories(alpha,beta,U,n,seed):
    rng = np.random.RandomState(seed)
    X = np.zeros((n,len(U),4),np.float64)
    for i in range(n):
        q0=q1=0.5
        for t in range(len(U)):
            z=beta*(q0-q1)
            p=1.0/(1.0+np.exp(-z)) if z>=0 else np.exp(z)/(1.0+np.exp(z))
            a=0 if rng.random_sample() < p else 1
            reward=U[t,a]
            X[i,t,a]=1.
            X[i,t,2+a]=reward
            if a==0: q0+=alpha*(reward-q0)
            else: q1+=alpha*(reward-q1)
    return X

def continuous_trajectories(alpha,beta,U,seed):
    rng=np.random.RandomState(seed)
    X=np.zeros((len(alpha),len(U),4),np.float64)
    for i in range(len(alpha)):
        q0=q1=0.5
        for t in range(len(U)):
            z=beta[i]*(q0-q1)
            p=1./(1.+np.exp(-z)) if z>=0 else np.exp(z)/(1.+np.exp(z))
            a=0 if rng.random_sample()<p else 1
            reward=U[t,a]
            X[i,t,a]=1.
            X[i,t,2+a]=reward
            if a==0: q0+=alpha[i]*(reward-q0)
            else: q1+=alpha[i]*(reward-q1)
    return X

def write_array(path,X):
    np.asarray(X,dtype='<f8').ravel(order='F').tofile(path)

def read_array(path,shape):
    return np.fromfile(path,dtype='<f8').reshape(shape,order='F')

def cfg_for(mode):
    if mode=='smoke': return dict(T=60,phen_train=16,phen_test=4,cont_train=80,cont_test=20,R=3,restarts=1,maxiter=35)
    return dict(T=150,phen_train=80,phen_test=20,cont_train=400,cont_test=100,R=5,restarts=3,maxiter=240)

def generate(mode,folder):
    cfg=cfg_for(mode)
    U=generate_env(14131,cfg['T'])
    parts=[]
    for g,(a,b) in enumerate(GROUPS):
        X=discrete_trajectories(a,b,U,cfg['phen_train']+cfg['phen_test'],850001+g*101)
        parts.append(X)
    Xtr=np.concatenate([x[:cfg['phen_train']] for x in parts])
    Xte=np.concatenate([x[cfg['phen_train']:] for x in parts])
    write_array(folder/'phen_train.f64',Xtr)
    write_array(folder/'phen_test.f64',Xte)
    ytr=np.repeat(np.arange(4),cfg['phen_train'])
    yte=np.repeat(np.arange(4),cfg['phen_test'])
    np.savetxt(folder/'phen_train_y.csv',ytr,fmt='%d',delimiter=',')
    np.savetxt(folder/'phen_test_y.csv',yte,fmt='%d',delimiter=',')
    rng=np.random.default_rng(916522)
    n=cfg['cont_train']+cfg['cont_test']
    alpha=rng.uniform(.05,.95,n)
    beta=rng.uniform(.5,8.,n)
    X=continuous_trajectories(alpha,beta,U,909991)
    write_array(folder/'cont_train.f64',X[:cfg['cont_train']])
    write_array(folder/'cont_test.f64',X[cfg['cont_train']:])
    np.savetxt(folder/'cont_train_y.csv',np.column_stack([alpha[:cfg['cont_train']],beta[:cfg['cont_train']]]),delimiter=',',fmt='%.18e')
    np.savetxt(folder/'cont_test_y.csv',np.column_stack([alpha[cfg['cont_train']:],beta[cfg['cont_train']:]]),delimiter=',',fmt='%.18e')
    meta={'mode':mode,'env_index':0,'env_seed':14131,'n_trials':cfg['T'],'rank':cfg['R'],'restarts':cfg['restarts'],'maxiter':cfg['maxiter'],'tol':1e-7,
          'phen_train':cfg['phen_train']*4,'phen_test':cfg['phen_test']*4,'cont_train':cfg['cont_train'],'cont_test':cfg['cont_test'],
          'feature_count':4,'randomness':'NumPy MT19937 agent choices; potential shared; exact shared array files','dtype':'little-endian Float64, Fortran order'}
    (folder/'settings.json').write_text(json.dumps(meta,indent=2))
    print('GENERATED '+json.dumps(meta),flush=True)

def fit_tensorly(folder):
    import tensorly as tl
    from tensorly.decomposition import parafac
    meta=json.loads((folder/'settings.json').read_text())
    R=meta['rank']; restarts=meta['restarts']; maxiter=meta['maxiter']
    out=[]
    for task in ('phen','cont'):
        X=read_array(folder/(task+'_train.f64'),(meta[task+'_train'],meta['n_trials'],4))
        for start in range(restarts):
            seed=31001+R*991+199*start if task=='phen' else 12701+R*911+199*start
            t0=time.perf_counter()
            res,errors=parafac(X,rank=R,init='random',random_state=seed,normalize_factors=False,
                                n_iter_max=maxiter,tol=meta['tol'],return_errors=True,
                                l2_reg=0.0,linesearch=False)
            seconds=time.perf_counter()-t0
            weights,fac=res
            yhat=tl.cp_to_tensor(res)
            err=float(np.linalg.norm(X-yhat)/np.linalg.norm(X))
            prefix=folder/(task+'_TensorLy_s'+str(start))
            np.savetxt(str(prefix)+'_weights.csv',np.asarray(weights).reshape(-1),delimiter=',',fmt='%.18e')
            for k,A in enumerate(fac):
                np.savetxt(str(prefix)+f'_factor{k+1}.csv',A,delimiter=',',fmt='%.18e')
            out.append((task,'TensorLy',start,err,len(errors),seconds,'unknown'))
            print(f'TensorLy {task} start {start}: relerr={err:.6f} iters={len(errors)} seconds={seconds:.3f}',flush=True)
    with (folder/'tensorly_fits.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['task','method','start','train_rel_error','iterations','fit_seconds','converged']);w.writerows(out)

def evaluate(folder):
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression,RidgeCV
    from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error,f1_score
    m=json.loads((folder/'settings.json').read_text())
    rows=[]; fits=[]
    for task in ('phen','cont'):
        Xtr=read_array(folder/(task+'_train.f64'),(m[task+'_train'],m['n_trials'],4))
        Xte=read_array(folder/(task+'_test.f64'),(m[task+'_test'],m['n_trials'],4))
        ytr=np.loadtxt(folder/(task+'_train_y.csv'),delimiter=',')
        yte=np.loadtxt(folder/(task+'_test_y.csv'),delimiter=',')
        for method,record_name in [('TensorLy','tensorly_fits.csv'),('TensorLy_NN','tensorly_fits.csv'),('TensorKitchen','tensorkitchen_fits.csv'),('TensorKitchen_NN','tensorkitchen_fits.csv')]:
            table=pd.read_csv(folder/record_name)
            entries=table[(table.task==task)&(table.method==method)]
            if entries.empty: continue
            best=entries.loc[entries.train_rel_error.idxmin()]
            start=int(best.start)
            prefix=folder/(task+'_'+method+'_s'+str(start))
            w=np.atleast_1d(np.loadtxt(str(prefix)+'_weights.csv',delimiter=','))
            B=np.atleast_2d(np.loadtxt(str(prefix)+'_factor2.csv',delimiter=','))
            C=np.atleast_2d(np.loadtxt(str(prefix)+'_factor3.csv',delimiter=','))
            K=np.column_stack([w[r]*np.kron(C[:,r],B[:,r]) for r in range(m['rank'])])
            # Same LS coordinates for both ordinary CP methods. NNCPD uses NNLS.
            def project(X):
                V=X.transpose(0,2,1).reshape(len(X),-1)
                if method.endswith('_NN'):
                    from scipy.optimize import nnls
                    z=np.stack([nnls(K,y,maxiter=1000)[0] for y in V])
                else:
                    z=V@np.linalg.pinv(K).T
                yp=z@K.T
                err=np.linalg.norm(V-yp,axis=1)/np.maximum(np.linalg.norm(V,axis=1),1e-12)
                return z,float(err.mean())
            ztr,_=project(Xtr);zte,e=project(Xte)
            row=dict(mode=m['mode'],env_index=0,task=task,method=method,rank=m['rank'],starts=m['restarts'],
                     train_rel_error=best.train_rel_error,test_rel_error=e,
                     fit_seconds=float(entries.fit_seconds.sum()),best_fit_seconds=float(best.fit_seconds),
                     min_iterations=int(best.iterations),converged=str(best.converged))
            if task=='phen':
                clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=700,random_state=0))
                clf.fit(ztr,ytr.astype(int))
                pred=clf.predict(zte)
                row.update(accuracy=float(accuracy_score(yte,pred)),macro_f1=float(f1_score(yte,pred,average='macro')))
            else:
                scaler=StandardScaler();aa=scaler.fit_transform(ztr);bb=scaler.transform(zte)
                for j,par in enumerate(('alpha','beta')):
                    reg=RidgeCV(alphas=[.1,1.,10.,100.,1000.],cv=3)
                    reg.fit(aa,ytr[:,j]);p=reg.predict(bb)
                    row['r2_'+par]=float(r2_score(yte[:,j],p))
                    row['mae_'+par]=float(mean_absolute_error(yte[:,j],p))
            rows.append(row)
            print(json.dumps(row),flush=True)
    pd.DataFrame(rows).to_csv(folder/'comparison_metrics.csv',index=False)
    print('WROTE '+str(folder/'comparison_metrics.csv'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['generate','tensorly','evaluate'])
    p.add_argument('--mode',choices=['smoke','reference'],default='smoke')
    p.add_argument('--folder',default=None)
    x=p.parse_args()
    folder=Path(x.folder) if x.folder else ROOT/'results'/x.mode
    folder.mkdir(parents=True,exist_ok=True)
    {'generate':generate,'tensorly':fit_tensorly,'evaluate':evaluate}[x.phase](x.mode,folder) if x.phase=='generate' else {'tensorly':fit_tensorly,'evaluate':evaluate}[x.phase](folder)

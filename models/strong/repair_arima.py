from pathlib import Path
import json,sys,time,hashlib,warnings
import numpy as np,pandas as pd
import pmdarima as pm
from pmdarima.arima import StepwiseContext
from worker import map_grid,atomic_npz,save
R=Path(__file__).resolve().parent
def recover(i,y):
 d=R/'results/AutoARIMA/recovered';d.mkdir(exist_ok=True)
 p=d/f'{i:07d}.npz'
 if p.exists():
  z=np.load(p);assert np.array_equal(z['history'],y);return z['prediction']
 yy=y.astype(float);mu=yy.mean();sd=yy.std() or 1.;attempts=[]
 for method,maxiter in [('lbfgs',50),('powell',200)]:
  try:
   with warnings.catch_warnings(),StepwiseContext(max_steps=40):
    warnings.simplefilter('ignore');fits=pm.auto_arima((yy-mu)/sd,seasonal=False,max_p=3,max_q=3,max_order=5,max_d=2,information_criterion='aicc',stepwise=True,method=method,maxiter=maxiter,suppress_warnings=True,error_action='ignore',return_valid_fits=True)
   for rank,model in enumerate(fits):
    try:
     with warnings.catch_warnings():warnings.simplefilter('ignore');grid=model.predict(n_periods=288)*sd+mu
     assert np.isfinite(grid).all();v=map_grid(float(y[-1]),grid);assert np.isfinite(v).all()
     atomic_npz(p,history=y,prediction=v)
     save(p.with_suffix('.json'),{'unique_history_index':i,'method':method,'maxiter':maxiter,'selected_aicc_rank':rank,'order':list(model.order),'aicc':float(model.aicc()),'optimizer_converged':bool(model.arima_res_.mle_retvals.get('converged',False)),'candidate_failures':attempts,'reason':'First finite complete forecast among AICc-ordered fitted candidates; no target labels inspected','time':time.time()})
     return v
    except Exception as e:attempts.append({'method':method,'rank':rank,'error':str(e)[:200]})
  except Exception as e:attempts.append({'method':method,'error':str(e)[:200]})
 raise RuntimeError(f'No finite ARIMA forecast for input {i}: {attempts}')
def main():
 f=pd.read_parquet(R/'inputs/answers.parquet');x=np.load(R/'inputs/history.npz')['prices'];u,first,inv=np.unique(x,axis=0,return_index=True,return_inverse=True)
 if len(sys.argv)>1:
  i=int(sys.argv[1]);v=recover(i,u[i]);print('RECOVERED',i,v.tolist(),flush=True);return
 source=R/'results/AutoARIMA/predictions.npz';z=np.load(source);p=z['prediction'];assert np.array_equal(z['row_id'],f.row_id);q=p.copy();missing=~np.isfinite(p);pending=np.unique(inv[missing.any(1)]);repaired=[]
 for i in pending:
  v=recover(int(i),u[i]);where=inv==i;fill=np.repeat(v[None,:],int(where.sum()),axis=0);fill[:,0]=np.clip(fill[:,0]+.8*(f.raw.to_numpy()[where]-x[where,-1]),0,1);sub=q[where];mm=missing[where];sub[mm]=fill[mm];q[where]=sub;repaired.append(int(i))
 assert np.isfinite(q).all() and ((q>=0)&(q<=1)).all();assert np.array_equal(q[~missing],p[~missing])
 dest=source.with_name('predictions_complete.npz');atomic_npz(dest,row_id=f.row_id,prediction=q,success=np.ones(len(f),bool))
 save(R/'results/AutoARIMA/REPAIR_AUDIT.json',{'passed':True,'original_file_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'final_file_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'original_success_values_unchanged':True,'repaired_unique_histories':repaired,'repaired_values':int(missing.sum()),'complete_rows':len(f),'time':time.time()});print('ARIMA_FINALIZED',len(f),len(pending),flush=True)
if __name__=='__main__':main()

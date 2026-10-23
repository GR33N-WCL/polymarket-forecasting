import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time,hashlib,traceback,warnings
import numpy as np,pandas as pd
from numerics import weights
R=Path(__file__).resolve().parent;H=[1,5,15,30,60,180,360,1440]
def save(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');t.replace(p)
def progress(model,**kw):
 x={'model':model,'pid':os.getpid(),'time':time.time(),**kw};save(R/'results'/model/'STATUS.json',x);print(json.dumps(x),flush=True)
def load():
 f=pd.read_parquet(R/'inputs/answers.parquet');z=np.load(R/'inputs/history.npz');assert np.array_equal(z['row_id'],f.row_id)
 return f,z['prices']
def atomic_npz(path,**kw):
 p=Path(path);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_name(p.name+'.tmp.npz');np.savez_compressed(t,**kw);t.replace(p)
def map_grid(p0,a):
 q=np.asarray(a)[...,[0,0,2,5,11,35,71,287]].copy();q[...,0]=p0+.2*(q[...,0]-p0)
 return np.clip(q,0,1)
def arima_chunk(a):
 start,x,config=a
 import pmdarima as pm
 from pmdarima.arima import StepwiseContext
 result=np.full((len(x),8),np.nan);receipts=[]
 for i,y in enumerate(x):
  try:
   history=y.astype(float);center=float(history.mean());scale=float(history.std());scale=scale if scale>0 else 1.0
   with warnings.catch_warnings(),StepwiseContext(max_steps=40):
    warnings.simplefilter('ignore')
    model=pm.auto_arima((history-center)/scale,seasonal=False,max_p=3,max_q=3,max_order=5,max_d=2,information_criterion='aicc',stepwise=True,method='lbfgs',maxiter=50,suppress_warnings=True,error_action='ignore')
    r=model.predict(n_periods=288)*scale+center
   assert np.isfinite(r).all()
   result[i]=map_grid(float(y[-1]),r)
   receipts.append({'i':start+i,'order':list(model.order),'aicc':float(model.aicc()),'optimizer_converged':bool(model.arima_res_.mle_retvals.get('converged',False))})
  except Exception as e:receipts.append({'i':start+i,'error':type(e).__name__+': '+str(e)[:300]})
 return start,result,receipts

def autoarima():
 from concurrent.futures import ProcessPoolExecutor,as_completed
 f,x=load();dest=R/'results/AutoARIMA';dest.mkdir(parents=True,exist_ok=True)
 u,first,inv=np.unique(x,axis=0,return_index=True,return_inverse=True)
 c=json.loads((R/'PROTOCOL_LOCK.json').read_text(encoding='utf8'))['autoarima'];config={k:v for k,v in c.items() if k not in ['library','fit']}
 total=len(u);step=64;todo=[];q=np.full((total,8),np.nan);done=0
 for start in range(0,total,step):
  p=dest/'chunks'/f'{start:07d}.npz'
  if p.exists():
   z=np.load(p);assert np.array_equal(z['history'],u[start:start+step]);q[start:start+len(z['prediction'])]=z['prediction'];done+=len(z['prediction'])
  else:todo.append((start,u[start:start+step],config))
 progress('AutoARIMA',stage='fitting',unique_histories=total,done=done,total_rows=len(f))
 workers=int(os.environ.get('BASELINE_CPU_WORKERS','6'));started=time.time()
 with ProcessPoolExecutor(max_workers=workers,mp_context=__import__('multiprocessing').get_context('spawn')) as pool:
  fs=[pool.submit(arima_chunk,a) for a in todo]
  for future in as_completed(fs):
   start,p,receipt=future.result();q[start:start+len(p)]=p;done+=len(p)
   atomic_npz(dest/'chunks'/f'{start:07d}.npz',prediction=p,history=u[start:start+len(p)])
   save(dest/'chunks'/f'{start:07d}.json',receipt)
   progress('AutoARIMA',stage='fitting',unique_histories=total,done=done,failed=done-int(np.isfinite(q).all(axis=1).sum()),seconds=time.time()-started)
 raw=q[inv];raw[:,0]=np.clip(raw[:,0]+.8*(f.raw.to_numpy()-x[:,-1]),0,1);success=np.isfinite(raw).all(axis=1)
 atomic_npz(dest/'predictions.npz',row_id=f.row_id,prediction=raw,success=success)
 progress('AutoARIMA',stage='complete',generated=int(success.sum()),failed=int((~success).sum()),rows=len(f),seconds=time.time()-started)
def features(x,p0):
 dx=np.diff(x,axis=1);cols=[x-p0[:,None],dx,p0[:,None]]
 for k in [3,12,36,72]:
  v=x[:,-k:];cols.extend([(v.mean(1)-p0)[:,None],v.std(1)[:,None],(v.min(1)-p0)[:,None],(v.max(1)-p0)[:,None]])
 return np.ascontiguousarray(np.column_stack(cols),dtype=np.float32)
def lightgbm():
 import lightgbm as lgb
 from lightgbm import LGBMRegressor
 f,x=load();dest=R/'results/LightGBM';dest.mkdir(parents=True,exist_ok=True)
 t=pd.read_parquet(R/'inputs/training_observations.parquet');t['row_id']=np.arange(len(t));z=np.load(R/'inputs/training_history.npz');eligible=z['eligible']
 tr=t.origin.to_numpy();p0=t.raw.to_numpy();xt=features(z['prices'],p0);xe=features(x,f.raw.to_numpy())
 y=t[[f'target_{h}' for h in H]].to_numpy();valid=t[[f'complete_{h}' for h in H]].to_numpy()&np.isfinite(y)
 months=pd.to_datetime(f.origin,unit='s',utc=True).dt.strftime('%Y-%m').to_numpy();allmonths=sorted(set(months))
 c=json.loads((R/'PROTOCOL_LOCK.json').read_text(encoding='utf8'))['lightgbm'];p=np.full((len(f),8),np.nan)
 started=time.time()
 for month in allmonths:
  use=months==month;ids=np.flatnonzero(use);file=dest/'monthly'/f'{month}.npz'
  if file.exists():
   a=np.load(file);assert np.array_equal(a['row_id'],f.row_id.to_numpy()[use]);p[use]=a['prediction'];continue
  bound=int(pd.Timestamp(month+'-01',tz='UTC').timestamp());split=bound-30*86400;lo=bound-90*86400
  rows=[]
  for j,h in enumerate(H):
   fit=eligible&valid[:,j]&(tr>=lo)&(tr<split)&(tr+h*60+60<split)
   val=eligible&valid[:,j]&(tr>=split)&(tr<bound)&(tr+h*60+60<bound)
   item={'month':month,'horizon':h,'fit_n':int(fit.sum()),'validation_n':int(val.sum()),'train_label_max_available':int((tr[fit]+h*60+60).max()) if fit.any() else None,'validation_label_max_available':int((tr[val]+h*60+60).max()) if val.any() else None,'evaluation_start':bound}
   if fit.sum()<c['minimum_train_rows'] or val.sum()<c['minimum_validation_rows']:
    item['status']='insufficient_past_data_missing';rows.append(item);continue
   ww=weights(t[fit])*fit.sum();wv=weights(t[val]);dy=y[:,j]-p0;best=float('inf');bestmodel=None;trials=[]
   for k,spec in enumerate(c['grid']):
    model=LGBMRegressor(n_estimators=c['max_estimators'],learning_rate=c['learning_rate'],reg_lambda=c['reg_lambda'],random_state=13,n_jobs=int(os.environ.get('BASELINE_LGB_THREADS','4')),verbosity=-1,deterministic=True,force_col_wise=True,**spec)
    model.fit(xt[fit],dy[fit],sample_weight=ww,eval_set=[(xt[val],dy[val])],eval_sample_weight=[wv*val.sum()],callbacks=[lgb.early_stopping(c['early_stopping_rounds'],verbose=False)])
    vp=np.clip(p0[val]+model.predict(xt[val]),0,1);score=float(wv@((vp-y[val,j])**2));trials.append({'spec':spec,'best_iteration':int(model.best_iteration_),'validation_mse':score})
    if score<best:best=score;bestmodel=model;choice=k
   p[use,j]=np.clip(f.raw.to_numpy()[use]+bestmodel.predict(xe[use]),0,1)
   modelpath=dest/'models'/f'{month}_{h}.txt';modelpath.parent.mkdir(exist_ok=True);bestmodel.booster_.save_model(str(modelpath))
   trainpred=np.clip(p0[fit]+bestmodel.predict(xt[fit]),0,1)
   item.update(status='generated',selected_trial=choice,trials=trials,train_mse=float((ww/ww.sum())@((trainpred-y[fit,j])**2)),validation_mse=best)
   rows.append(item);progress('LightGBM',stage='training',month=month,horizon=h,validation_mse=best,generated_values=int(np.isfinite(p).sum()),seconds=time.time()-started)
  atomic_npz(file,row_id=f.row_id.to_numpy()[use],prediction=p[use]);save(dest/'monthly'/f'{month}.json',rows)
 success=np.isfinite(p).all(axis=1)
 atomic_npz(dest/'predictions.npz',row_id=f.row_id,prediction=p,success=np.isfinite(p))
 progress('LightGBM',stage='complete',generated=int(success.sum()),generated_values=int(np.isfinite(p).sum()),rows=len(f),seconds=time.time()-started)
def chronos():
 import torch
 from chronos import Chronos2Pipeline
 f,x=load();dest=R/'results/Chronos-2';dest.mkdir(parents=True,exist_ok=True)
 u,first,inv=np.unique(x,axis=0,return_index=True,return_inverse=True);total=len(u)
 device='cuda:0' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
 if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.04)
 pipe=Chronos2Pipeline.from_pretrained(str(R/'models/chronos-2'),device_map=device,dtype=torch.float32,local_files_only=True)
 step=int(os.environ.get('CHRONOS_BATCH','16'));q=np.full((total,8),np.nan);done=0;started=time.time()
 progress('Chronos-2',stage='loaded',device=device,parameters=sum(p.numel() for p in pipe.model.parameters()),unique_histories=total,batch_size=step)
 for start in range(0,total,step):
  batch=u[start:start+step];file=dest/'chunks'/f'{start:07d}.npz'
  if file.exists():
   z=np.load(file);assert np.array_equal(z['history'],batch);q[start:start+len(batch)]=z['prediction'];done+=len(batch);continue
  while device.startswith('cuda') and torch.cuda.mem_get_info()[0]<int(2.0*2**30):time.sleep(5)
  # Every contract is a separate univariate task. No cross-origin/task attention.
  with torch.inference_mode():
   preds=pipe.predict(batch[:,None,:],prediction_length=288,context_length=72,batch_size=step,cross_learning=False)
  grid=np.stack([v[0,pipe.quantiles.index(.5)].detach().float().cpu().numpy() for v in preds])
  assert grid.shape==(len(batch),288) and np.isfinite(grid).all()
  pp=map_grid(batch[:,-1],grid);q[start:start+len(batch)]=pp;done+=len(batch)
  atomic_npz(file,history=batch,prediction=pp)
  if (start//step)%8==0 or done==total:progress('Chronos-2',stage='predicting',device=device,done=done,total=total,seconds=time.time()-started,gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20 if device.startswith('cuda') else None)
 raw=q[inv];raw[:,0]=np.clip(raw[:,0]+.8*(f.raw.to_numpy()-x[:,-1]),0,1);assert np.isfinite(raw).all();atomic_npz(dest/'predictions.npz',row_id=f.row_id,prediction=raw,success=np.ones(len(f),bool))
 progress('Chronos-2',stage='complete',generated=len(f),rows=len(f),seconds=time.time()-started)
if __name__=='__main__':
 model=sys.argv[1]
 try:{'AutoARIMA':autoarima,'LightGBM':lightgbm,'Chronos-2':chronos}[model]()
 except Exception as e:
  progress(model,stage='failed',error=type(e).__name__+': '+str(e),traceback=traceback.format_exc());raise




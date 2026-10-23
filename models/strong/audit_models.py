from pathlib import Path
import os,json,hashlib,time
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd
from worker import save,map_grid,features
R=Path(__file__).resolve().parent
def run():
 import lightgbm as lgb
 f=pd.read_parquet(R/'inputs/answers.parquet');x=np.load(R/'inputs/history.npz')['prices'];p0=f.raw.to_numpy();ix=np.random.default_rng(13).choice(len(f),40,replace=False)
 independently=np.asarray([list(x[i].astype(float)-p0[i])+list(np.diff(x[i]))+[p0[i]]+[v for k in [3,12,36,72] for v in [float(x[i,-k:].mean())-p0[i],float(x[i,-k:].std()),float(x[i,-k:].min())-p0[i],float(x[i,-k:].max())-p0[i]]] for i in ix],dtype=np.float32)
 assert np.allclose(independently,features(x[ix],p0[ix]),atol=1e-7,rtol=0)
 pred=np.load(R/'results/LightGBM/predictions.npz')['prediction'];months=pd.to_datetime(f.origin,unit='s',utc=True).dt.strftime('%Y-%m').to_numpy();replay=[];receipts=0
 H=[1,5,15,30,60,180,360,1440]
 for file in sorted((R/'results/LightGBM/monthly').glob('*.json')):
  for entry in json.loads(file.read_text()):
   if entry['status']!='generated':continue
   receipts+=1;bound=entry['evaluation_start'];assert entry['train_label_max_available']<bound-30*86400 and entry['validation_label_max_available']<bound
   scores=[v['validation_mse'] for v in entry['trials']];assert entry['selected_trial']==int(np.argmin(scores))
   if entry['month'] in ['2024-11','2025-06','2026-08'] and entry['horizon'] in [5,15,1440]:
    row=np.flatnonzero(months==entry['month'])[0];booster=lgb.Booster(model_file=str(R/'results/LightGBM/models'/f"{entry['month']}_{entry['horizon']}.txt"));pp=float(np.clip(p0[row]+booster.predict(features(x[row:row+1],p0[row:row+1]),num_threads=1)[0],0,1));err=abs(pp-pred[row,H.index(entry['horizon'])]);assert err<1e-12;replay.append(err)
 # Predict isolated univariate contexts again to verify that batch companions do not contribute information.
 import torch
 from chronos import Chronos2Pipeline
 torch.set_num_threads(2);pipe=Chronos2Pipeline.from_pretrained(str(R/'models/chronos-2'),device_map='cpu',dtype=torch.float32,local_files_only=True)
 c=np.load(R/'results/Chronos-2/predictions.npz')['prediction'];sel=ix[:6]
 with torch.inference_mode():out=pipe.predict(x[sel,None,:],prediction_length=288,batch_size=3,context_length=72,cross_learning=False)
 grid=np.stack([a[0,pipe.quantiles.index(.5)].detach().numpy() for a in out]);check=map_grid(p0[sel],grid)
 # CUDA FP32 and CPU FP32 can differ slightly; this is a numerical replay, not bitwise identity.
 ce=float(abs(check-c[sel]).max());assert ce<5e-5,ce
 counts={}
 for key in ['AutoARIMA','LightGBM','Chronos-2']:
  path=R/'results'/key/'predictions.npz'
  if key=='AutoARIMA':path=path.with_name('predictions_complete.npz')
  z=np.load(path);assert np.array_equal(z['row_id'],f.row_id);pp=z['prediction'];finite=np.isfinite(pp);assert ((pp[finite]>=0)&(pp[finite]<=1)).all();counts[key]={'input_rows':len(f),'full_eight_output_rows':int(finite.all(1).sum()),'finite_values':int(finite.sum()),'missing_values':int((~finite).sum())}
 # Independently reconstruct four selected ARIMA forecasts from the original windows.
 import pmdarima as pm,warnings
 from pmdarima.arima import StepwiseContext
 ar=np.load(R/'results/AutoARIMA/predictions_complete.npz')['prediction'];ar_errors=[]
 for i in ix[:4]:
  yy=x[i].astype(float);mu=yy.mean();sd=yy.std() or 1.
  with warnings.catch_warnings(),StepwiseContext(max_steps=40):
   warnings.simplefilter('ignore');model=pm.auto_arima((yy-mu)/sd,seasonal=False,max_p=3,max_q=3,max_order=5,max_d=2,information_criterion='aicc',stepwise=True,method='lbfgs',maxiter=50,suppress_warnings=True,error_action='ignore');future=model.predict(n_periods=288)*sd+mu
  v=np.clip(future[[0,0,2,5,11,35,71,287]],0,1);v[0]=np.clip(.8*p0[i]+.2*future[0],0,1);ar_errors.append(float(abs(v-ar[i]).max()))
 assert max(ar_errors)<1e-6,ar_errors
 converged=not_converged=fit_failed=0
 for file in (R/'results/AutoARIMA/chunks').glob('*.json'):
  for item in json.loads(file.read_text()):
   if 'error' in item:fit_failed+=1;continue
   if item.get('optimizer_converged'):converged+=1
   else:not_converged+=1
 recovery=json.loads((R/'results/AutoARIMA/REPAIR_AUDIT.json').read_text());assert recovery['passed'] and recovery['original_success_values_unchanged']
 a={'passed':True,'feature_independent_rows':len(ix),'causal_training_receipts':receipts,'selected_trial_is_validation_minimum':True,'lightgbm_saved_model_replays':len(replay),'lightgbm_max_error':max(replay),'chronos_cpu_vs_gpu_replay_rows':len(sel),'chronos_max_abs_error':ce,'arima_replay_rows':len(ar_errors),'arima_replay_max_error':max(ar_errors),'arima_primary_optimizer_converged':converged,'arima_primary_optimizer_not_converged':not_converged,'arima_primary_prediction_failures':fit_failed,'arima_recovery':recovery,'counts':counts,'time':time.time()};save(R/'evaluation/MODEL_REPLAY_AUDIT.json',a);print(json.dumps(a),flush=True)
if __name__=='__main__':run()

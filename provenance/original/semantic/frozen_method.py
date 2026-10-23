"""Frozen signed-neighbor experiment. No inference calls, no frozen-source writes."""
from pathlib import Path
import sys,json,hashlib,time,importlib.util,collections
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
SRC=Path(CFG['prediction_source']);DATA=Path(CFG['data_source'])
sys.path[:0]=[str(SRC/'code'),str(SRC/'orchestration')]
from common import H,MODELS,read,sha,requests
from numerics import weights
spec=importlib.util.spec_from_file_location('frozen_evaluator',SRC/'orchestration/evaluate_v2.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)

def save(name,obj):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def progress(stage,**kw):save('STATUS.json',dict(stage=stage,time=time.time(),**kw));print(stage,kw,flush=True)
def asof(hist,t):
 a,last,p=hist;i=np.searchsorted(a,t,side='right')-1;s=np.maximum(i,0)
 valid=(i>=0)&(last[s]<=t)&((t-last[s])<=CFG['history_freshness_seconds'])
 return np.where(valid,p[s],np.nan)
def relation_stats(x,y):
 mask=np.isfinite(x)&np.isfinite(y);x=x[mask];y=y[mask];n=len(x)
 out={'n':n,'accepted':False,'reason':'insufficient_paired_changes'}
 if n<CFG['minimum_paired_changes']:return out
 if min(np.count_nonzero(abs(x)>1e-6),np.count_nonzero(abs(y)>1e-6))<CFG['minimum_nonzero_changes_each_contract']:
  out['reason']='insufficient_moving_changes';return out
 sx=x.std(ddof=1);sy=y.std(ddof=1)
 if min(sx,sy)<1e-8:out['reason']='flat_history';return out
 rho=float(np.corrcoef(x,y)[0,1]);out.update(rho=rho,sd_i=float(sx),sd_j=float(sy))
 if abs(rho)<CFG['minimum_abs_correlation']:out['reason']='weak_correlation';return out
 halves=[]
 for a,b in [(x[:n//2],y[:n//2]),(x[n//2:],y[n//2:])]:
  if len(a)<CFG['minimum_half_changes'] or min(a.std(),b.std())<1e-8:
   out['reason']='unstable_or_flat_half';return out
  halves.append(float(np.corrcoef(a,b)[0,1]))
 out['rho_halves']=halves
 if any(not np.isfinite(c) or c*rho<=0 for c in halves):out['reason']='sign_unstable';return out
 lo,hi=CFG['amplitude_ratio_clip']
 out.update(accepted=True,reason='accepted',sign=1 if rho>0 else -1,weight=rho*rho*n/(n+100),amplitude_i=float(np.clip(sx/sy,lo,hi)),amplitude_j=float(np.clip(sy/sx,lo,hi)))
 return out

from async_features import build_edges, messages, choose_lambdas

def main():
 lock=ROOT/'PROTOCOL_LOCK.json';identity={'protocol_sha256':sha(ROOT/'protocol.json'),'script_sha256':sha(Path(__file__)),'feature_script_sha256':sha(ROOT/'async_features.py'),'inputs':{p.name:sha(p) for p in [SRC/'inputs/answers.parquet',SRC/'inputs/systems.json',SRC/'inputs/CONTEXT_LOCK.json',DATA/'graph/event_edge_roles.csv']}}
 if lock.exists():assert read(lock)==identity,'Protocol/source changed; use a new directory'
 else:save('PROTOCOL_LOCK.json',identity)
 frame=pd.read_parquet(SRC/'inputs/answers.parquet').reset_index(drop=True);systems=read(SRC/'inputs/systems.json');ctx=read(SRC/'inputs/CONTEXT_LOCK.json')
 edges=build_edges(frame,globals());progress('edges_ready',candidate=len(edges),accepted=sum(e['accepted'] for e in edges))
 common_ids={i for req in requests() if req['id'] in ctx['whole_eligible'] for i in req['row_ids']};common=frame.row_id.isin(common_ids).to_numpy();position={int(v):i for i,v in enumerate(frame.row_id)}
 p0=frame.raw.to_numpy();metrics=[];parameters=[];coverage=[];sign_audits=[];outcomes=[];constraints=[]
 for key in MODELS:
  files=sorted((SRC/'results'/key/'accepted_responses').glob('*.json'));allcache={p.stem:sha(p) for p in files};save(f'cache_hashes/{key}.json',allcache)
  for mode in ['individual','whole_graph']:
   z=np.tile(p0[:,None],(1,8));present=np.zeros(len(frame),bool);nreq=0
   for p in files:
    a=read(p)
    if a['mode']!=mode:continue
    assert a['status']=='ok';ix=np.array([position[v] for v in a['row_ids']]);assert not present[ix].any();z[ix]=a['predictions'];present[ix]=True;nreq+=1
   ready=present.copy()
   for s in systems:
    for rr in s['row_ids']:
     ix=np.array([position[v] for v in rr])
     if not present[ix].all():ready[ix]=False
   expected=8566 if mode=='individual' else len(common_ids)
   coverage.append(dict(model=key,mode=mode,requests=nreq,contract_records=int(present.sum()),expected=expected,complete=int(present.sum())==expected))
   if not present.any():continue
   logic,solver,diag=ev.project_all(z,frame,systems,present)
   assert not (solver=='solver_failed_raw_fallback').any()
   def record(name,v,seed=-1,diags=None):
    if diags:
     constraints.extend(dict(model=key,mode=mode,variant=name,seed=seed,**x) for x in diags)
    for hi,h in enumerate(H):
     for panel in ['all','common']:
      mask=ready&frame[f'complete_{h}'].to_numpy()&np.isfinite(frame[f'target_{h}'].to_numpy())
      if panel=='common':mask &=common
      if not mask.any():continue
      ff=frame[mask];y=ff[f'target_{h}'].to_numpy();w=weights(ff);pred=v[mask,hi];raw=z[mask,hi];lp=logic[mask,hi];cur=p0[mask];se=(pred-y)**2;rawse=(raw-y)**2;ae=abs(pred-y);ra=abs(raw-y);changed=abs(pred-raw)>1e-10
      sign=lambda a:np.where(abs(a)<=1e-6,0,np.sign(a));truth=sign(y-cur);dg=sign(pred-cur)==truth;rdg=sign(raw-cur)==truth;moving=truth!=0
      mse=float(w@se);mae=float(w@ae);base=float(w@rawse);msemove=None
      vals=dict(model=key,mode=mode,variant=name,seed=seed,horizon=h,panel=panel,n=len(ff),mse=mse,mae=mae,mse_gain_percent=100*(base-mse)/base if base else None,mae_gain_percent=100*(float(w@ra)-mae)/float(w@ra) if float(w@ra) else None,delta_vs_logic=float(w@((lp-y)**2-se)),dg=float(w@dg),dg_gain_pp=100*float(w@(dg.astype(float)-rdg)),dg_moving=float(weights(ff[moving])@dg[moving]) if moving.any() else None,moving_count=int(moving.sum()),moving_fraction=float(moving.mean()),changed=int(changed.sum()),improved_changed=int(((se<rawse)&changed).sum()),harmed_changed=int(((se>rawse)&changed).sum()),improved_fraction_all=float(((se<rawse)&changed).mean()),improved_fraction_changed=float((se<rawse)[changed].mean()) if changed.any() else None)
      # Independent score path, not the weights helper.
      check=pd.DataFrame({'component':ff.component,'vid':ff.vid,'se':se,'ae':ae}).groupby(['component','vid'])[['se','ae']].mean().groupby('component').mean().mean()
      assert abs(check.se-mse)<1e-12 and abs(check.ae-mae)<1e-12
      if name in ['Raw','Logic','SignedSemantic','SignedSemantic+Logic']:
       vals['paired_mse_delta_ci']=ev.bootstrap(ff,rawse,se,w);vals['paired_mse_vs_logic_ci']=ev.bootstrap(ff,(lp-y)**2,se,w)
      metrics.append(vals)
    if seed==-1:
     for hi,h in enumerate(H):
      ix=np.flatnonzero(present)
      outcomes.append(pd.DataFrame({'model':key,'mode':mode,'variant':name,'row_id':frame.row_id.to_numpy()[ix],'horizon':h,'prediction':v[ix,hi],'ready':ready[ix]}))
   record('Raw',z);record('Logic',logic,diags=diag)
   for variant,seeds in [('SignedSemantic',[-1]),('AllPositive',[-1]),('ZeroNeighbor',[-1]),('PermutedSign',CFG['permutation_seeds'])]:
    for seed in seeds:
     message,support,info=messages(z,p0,present,edges,variant,seed)
     sem,params=choose_lambdas(frame,z,message,support,ready,key,mode,variant,seed);parameters.extend(params)
     assert np.array_equal(sem[~support],z[~support])
     sign_audits.append(dict(model=key,mode=mode,variant=variant,seed=seed,supported_rows=int(support.any(axis=1).sum()),supported_row_horizons=int(support.sum()),**info))
     if np.array_equal(sem,z):both=logic.copy();ds=[]
     else:
      both,ss,ds=ev.project_all(sem,frame,systems,present);assert not (ss=='solver_failed_raw_fallback').any()
     record(variant,sem,seed);record(variant+'+Logic',both,seed,ds)
   progress('evaluated',model=key,mode=mode,requests=nreq)
   pd.DataFrame(metrics).to_csv(ROOT/'metrics.csv',index=False);pd.DataFrame(parameters).to_csv(ROOT/'selected_parameters.csv',index=False)
   pd.DataFrame(coverage).to_csv(ROOT/'coverage.csv',index=False);pd.DataFrame(sign_audits).to_csv(ROOT/'sign_coverage.csv',index=False)
 pd.concat(outcomes,ignore_index=True).to_parquet(ROOT/'paired_predictions.parquet',index=False);pd.DataFrame(constraints).to_parquet(ROOT/'constraint_residuals.parquet',index=False)
 unchanged=all(sha(SRC/'results'/k/'accepted_responses'/f'{name}.json')==h for k in MODELS for name,h in read(ROOT/f'cache_hashes/{k}.json').items());assert unchanged
 save('AUDIT.json',{'score_replay_passed':True,'input_hashes_unchanged':all(sha(p)==identity['inputs'][p.name] for p in [SRC/'inputs/answers.parquet',SRC/'inputs/systems.json',SRC/'inputs/CONTEXT_LOCK.json',DATA/'graph/event_edge_roles.csv']),'accepted_predictions_unchanged':unchanged,'time':time.time()})
 progress('complete_for_snapshot',complete_models=[k for k in MODELS if all(c['complete'] for c in coverage if c['model']==k)])
if __name__=='__main__':main()

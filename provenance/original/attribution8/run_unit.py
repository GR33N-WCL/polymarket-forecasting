import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,sys,hashlib
import numpy as np,pandas as pd
from numerics import weights,project_batch
from graph_ops import canonical_relations,relation_residuals,permuted_columns,smooth_batch
import controls
O=Path(__file__).resolve().parents[1];I=Path('D:/polymarket_v2_llm_history303_20261016/inputs')
H=[1,5,15,30,60,180,360,1440]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def run(index):
 protocol=json.loads((O/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'));item=protocol['units'][index];model,seed=item['model'],item['seed'];dest=O/'results'/f'{model}_{seed}';dest.mkdir(parents=True,exist_ok=True)
 if (dest/'DONE.json').exists():
  d=json.loads((dest/'DONE.json').read_text());assert d['protocol_sha256']==sha(O/'PROTOCOL_LOCK.json');print('verified completed',model,seed,flush=True);return
 assert sha(item['file'])==item['sha256'];started=time.time()
 for n,h in protocol['helper_hashes'].items():assert sha(O/'code'/n)==h
 def state(stage):save(dest/'STATUS.json',{'model':model,'seed':seed,'stage':stage,'time':time.time()})
 f=pd.read_parquet(I/'answers.parquet');ids=f.row_id.to_numpy();pos={int(r):i for i,r in enumerate(ids)}
 with np.load(O/'source/COMMON_SUPPORT.npz') as s:
  universal=s['universal_success'];mask=s['score_mask'];legal=s['calibration_legal']
 with np.load(item['file']) as s:raw=s['Raw'].copy();own=s['success']
 raw[~own]=np.nan
 y=f[[f'target_{h}' for h in H]].to_numpy();p0=f.raw.to_numpy();times=f.origin.to_numpy()
 d=np.load(I/'neighbor_messages.npz');assert np.array_equal(d['row_id'],ids);msg=d['message'];support=d['support']
 systems=[]
 for s in json.loads((I/'systems.json').read_text()):
  ix=np.array([[pos[r] for r in row] for row in s['row_ids']]);good=universal[ix].all(1)
  systems.append((s['system_id'],np.array(s['A']),np.array(s['b']),ix,good))
 def mix(z,message):
  q=z.copy();use=support&universal;v=np.clip(p0[:,None]+.75*(z-p0[:,None])+.25*message,0,1);q[use]=v[use];return q
 def project(z,perm=None,smoothing=None):
  q=z.copy()
  for sid,a,b,ix,good in systems:
   if perm is not None:
    ps=int.from_bytes(hashlib.sha256(f'{perm}:{sid}'.encode()).digest()[:8],'little');a,_,_=permuted_columns(a,b,ps)
   v=z[ix].transpose(0,2,1);shape=v.shape;flat=v.reshape(-1,shape[-1]).copy();keep=good.ravel()
   assert np.isfinite(flat[keep]).all()
   if smoothing is None:
    pp=project_batch(flat[keep],a,b);assert np.max(pp@a.T-b,initial=0)<=1e-8
   else:pp=smooth_batch(flat[keep],a,b,smoothing)
   flat[keep]=pp;q[ix]=flat.reshape(shape).transpose(0,2,1)
  return q
 def records(predictions,select=mask,loss_arrays=None):
  out=[]
  for j,h in enumerate(H):
   use=select[:,j]
   if not use.any():continue
   ff=f[use];w=weights(ff);yy=y[use,j];zz=raw[use,j]
   for name,p in predictions.items():
    q=p[use,j];assert np.isfinite(q).all()
    delta=(zz-yy)**2-(q-yy)**2;moved=np.abs(q-zz)>1e-10
    out.append({'model':model,'seed':seed,'horizon':h,'version':name,'n':int(use.sum()),'events':int(ff.event_id.nunique()),'contracts':int(ff.market_id.nunique()),'mse':float(w@((q-yy)**2)),'mae':float(w@np.abs(q-yy)),'modified':int(moved.sum()),'improved':int((moved&(delta>0)).sum()),'harmed':int((moved&(delta<0)).sum()),'unchanged':int((~moved).sum())})
   for name,(se,ae) in (loss_arrays or {}).items():
    assert np.isfinite(se[use,j]).all() and np.isfinite(ae[use,j]).all()
    out.append({'model':model,'seed':seed,'horizon':h,'version':name,'n':int(use.sum()),'mse':float(w@se[use,j]),'mae':float(w@ae[use,j])})
  return out
 state('main')
 sem=mix(raw,msg);zero=mix(raw,np.zeros_like(msg));full=project(sem)
 variants={'Raw':raw,'Semantic':sem,'Logic':project(raw),'Full':full,'ZeroNeighbor':zero,'ZeroNeighbor+Logic':project(zero)}
 pd.DataFrame(records(variants)).to_csv(dest/'main.csv',index=False)
 np.savez_compressed(dest/'predictions.npz',row_id=ids,**variants)
 state('consistency')
 cons=[];raw_node=np.zeros_like(raw);origins=[];history_viol=[];history_good=[]
 for sid,a,b,ix,good in systems:
  scored=good&mask[ix].any(1);vv=raw[ix].transpose(0,2,1);flat=vv.reshape(-1,vv.shape[-1]);keep=good.ravel()
  rr=np.full((flat.shape[0],len(canonical_relations(a,b))),np.nan)
  rr[keep]=relation_residuals(flat[keep],a,b)
  rr=rr.reshape(vv.shape[:2]+(rr.shape[-1],));raw_node[ix]=np.nanmax(rr,axis=2)[:,None,:]
  origins.extend(times[ix[:,0]].tolist());history_viol.append(np.nanmax(rr,axis=2));history_good.append(good)
 for name,z in variants.items():
  rrall=[];useall=[]
  for sid,a,b,ix,good in systems:
   scored=good&mask[ix].any(1);v=z[ix].transpose(0,2,1);flat=v.reshape(-1,v.shape[-1]);active=scored.ravel();rr=relation_residuals(flat[active],a,b)
   ctx=np.flatnonzero(active)
   for j,h in enumerate(H):
    take=(ctx%8)==j
    if take.any():rrall.append((j,rr[take]))
  for j,h in enumerate(H):
   arrs=[a for jj,a in rrall if jj==j]
   if arrs:
    # Each context is equally weighted; each context's own relations are equally weighted.
    rates=np.concatenate([(a>1e-8).mean(1) for a in arrs]);mags=np.concatenate([a.mean(1) for a in arrs])
    cons.append({'model':model,'seed':seed,'horizon':h,'version':name,'contexts':len(rates),'rate':float(rates.mean()),'magnitude':float(mags.mean()),'max_residual':max(float(a.max(initial=0)) for a in arrs)})
 pd.DataFrame(cons).to_csv(dest/'consistency.csv',index=False)
 state('shared_constraints')
 partialmask=np.zeros_like(mask);losses={k:[np.full_like(raw,np.nan),np.full_like(raw,np.nan)] for k in ['SingleRelation','LeaveOneOut']};counts=[];cases=[];case_count=0
 for sid,a,b,ix,good in systems:
  rels=canonical_relations(a,b);members=[set(np.flatnonzero(a[r[0]])) for r in rels];neighbors={i:[j for j in range(len(rels)) if j!=i and members[i]&members[j]] for i in range(len(rels))};choices=[i for i,n in neighbors.items() if n]
  if not choices:continue
  active=good&mask[ix].any(1)
  if not active.any():continue
  partialmask[ix]=mask[ix]
  val=sem[ix].transpose(0,2,1);shape=val.shape;flat=val.reshape(-1,shape[-1]);keep=active.ravel();idx=np.flatnonzero(keep);before=relation_residuals(flat[keep],a,b)
  for mode in losses:
   se=np.zeros_like(val);ae=np.zeros_like(val)
   for choice in choices:
    kept=[choice] if mode=='SingleRelation' else [j for j in range(len(rels)) if j!=choice];omit=neighbors[choice] if mode=='SingleRelation' else [choice];rows=[r for k in kept for r in rels[k]]
    pp=flat.copy();pp[keep]=project_batch(flat[keep],a[rows],b[rows]);after=relation_residuals(pp[keep],a,b);eligible=before[:,omit]<=1e-8;new=eligible&(after[:,omit]>1e-8)
    q=pp.reshape(shape);se+=(q-y[ix].transpose(0,2,1))**2/len(choices);ae+=np.abs(q-y[ix].transpose(0,2,1))/len(choices)
    for j,h in enumerate(H):
     take=idx%8==j
     counts.append({'model':model,'seed':seed,'system':sid,'mode':mode,'choice':choice,'horizon':h,'contexts':int(take.sum()),'eligible_relations':int(eligible[take].sum()),'new_violations':int(new[take].sum()),'new_violating_contexts':int(new[take].any(1).sum())})
    bad=np.flatnonzero(new.any(1));case_count+=len(bad)
    for k in bad[:max(0,50-len(cases))]:
     at=idx[k];inst,j=divmod(int(at),8);ii=ix[inst]
     cases.append({'model':model,'seed':seed,'system':sid,'mode':mode,'choice':choice,'horizon':H[j],'row_ids':ids[ii].tolist(),'origin':int(times[ii[0]]),'semantic':val[inst,j].tolist(),'partial':q[inst,j].tolist(),'joint':full[ii,j].tolist()})
   losses[mode][0][ix]=se.transpose(0,2,1);losses[mode][1][ix]=ae.transpose(0,2,1)
 pd.DataFrame(records({'Semantic':sem,'Full':full},partialmask,losses)).to_csv(dest/'completeness.csv',index=False)
 pd.DataFrame(counts).to_csv(dest/'completeness_counts.csv',index=False);save(dest/'natural_cases.json',{'total_new_violating_context_choices':case_count,'examples_cap':50,'examples':cases})
 state('relation_and_direction_controls')
 perms=[];perm_se=np.zeros_like(raw);perm_ae=np.zeros_like(raw)
 for s in protocol['permutation_seeds']:
  q=project(sem,perm=s);perm_se+=(q-y)**2/10;perm_ae+=np.abs(q-y)/10
  for r in records({'PermutedLogic':q}):r['permutation_seed']=s;perms.append(r)
 pd.DataFrame(perms).to_csv(dest/'permutations.csv',index=False)
 reverse=project(mix(raw,-msg));pd.DataFrame(records({'ReversedMessage+Logic':reverse})).to_csv(dest/'semantic_direction.csv',index=False)
 state('conflict_bands')
 origins=np.array(origins);hv=np.concatenate(history_viol);hg=np.concatenate(history_good);months=pd.to_datetime(times,unit='s',utc=True).strftime('%Y-%m').to_numpy();bands=np.full(raw.shape,'none',object);thresholds=[]
 for month in sorted(set(months)):
  t=int(pd.Timestamp(month+'-01',tz='UTC').timestamp());test=months==month;past=(origins>=t-30*86400)&(origins<t)
  for j,h in enumerate(H):
   v=hv[past,j];ok=hg[past,j]&(v>1e-8);v=v[ok];cut=np.quantile(v,[1/3,2/3]) if len(v)>=30 else np.array([.001,.01]);cur=raw_node[test,j];bands[test,j]=np.where(cur<=1e-8,'none',np.where(cur<=cut[0],'low',np.where(cur<=cut[1],'medium','high')));thresholds.append({'month':month,'horizon':h,'positive_past_contexts':len(v),'cutoffs':cut.tolist()})
 bandrows=[]
 for band in ['none','low','medium','high']:
  for r in records({k:variants[k] for k in ['Raw','Semantic','Full']},mask&(bands==band)):r['band']=band;bandrows.append(r)
 pd.DataFrame(bandrows).to_csv(dest/'bands.csv',index=False);save(dest/'conflict_thresholds.json',thresholds)
 state('past_only_controls')
 smooths={0.:sem}
 for lam in [.01,.1,1.,10.,100.]:smooths[lam]=project(sem,smoothing=lam)
 smooth=sem.copy();control_out={k:raw.copy() for k in controls.METHODS};fitmask=np.zeros_like(mask);params=[]
 for month in sorted(set(months)):
  t=int(pd.Timestamp(month+'-01',tz='UTC').timestamp());test=months==month;lo=t-30*86400;split=t-10*86400
  for j,h in enumerate(H):
   fit=legal[:,j]&(times>=lo)&(times+h*60+60<split);sel=legal[:,j]&(times>=split)&(times+h*60+60<t);allval=legal[:,j]&(times>=lo)&(times+h*60+60<t)
   r={'month':month,'horizon':h,'fit':int(fit.sum()),'select':int(sel.sum()),'validation':int(allval.sum()),'selected':{}}
   lam=0.
   if allval.sum()>=20:
    w=weights(f[allval]);lam=min(smooths,key=lambda k:float(w@((smooths[k][allval,j]-y[allval,j])**2))) if all(np.isfinite(smooths[k][allval,j]).all() for k in smooths) else 0.
   smooth[test,j]=smooths[lam][test,j];r['smoothing_lambda']=lam
   if fit.sum()>=20 and sel.sum()>=20:
    fitmask[test,j]=True
    def data(ix):return raw[ix,j],p0[ix],y[ix,j],weights(f[ix])
    for name in controls.METHODS:
     spec,loss,candidates=controls.choose(name,data(fit),data(sel));fitted=controls.fit(spec,*data(allval));control_out[name][test,j]=controls.predict(fitted,raw[test,j],p0[test]);r['selected'][name]={'spec':spec,'fitted':fitted,'selection_mse':loss,'candidates':candidates}
   params.append(r)
 pd.DataFrame(records({'Semantic+Smooth':smooth})).to_csv(dest/'smooth.csv',index=False)
 composition=[]
 for name,v in control_out.items():
  q=project(mix(v,msg))
  for r in records({'Control':v,'Control+Full':q},mask&fitmask):r['control']=name;composition.append(r)
 pd.DataFrame(composition).to_csv(dest/'composition.csv',index=False);save(dest/'parameters.json',params);np.save(dest/'fit_mask.npy',fitmask)
 np.savez_compressed(dest/'diagnostic_losses.npz',PermutedLogic_mse=perm_se,PermutedLogic_mae=perm_ae,SingleRelation_mse=losses['SingleRelation'][0],SingleRelation_mae=losses['SingleRelation'][1],LeaveOneOut_mse=losses['LeaveOneOut'][0],LeaveOneOut_mae=losses['LeaveOneOut'][1],shared_mask=partialmask)
 np.savez_compressed(dest/'diagnostic_predictions.npz',ReversedMessageLogic=reverse,SemanticSmooth=smooth)
 save(dest/'DONE.json',{'model':model,'seed':seed,'seconds':time.time()-started,'protocol_sha256':sha(O/'PROTOCOL_LOCK.json'),'raw_source_sha256':item['sha256'],'counts':mask.sum(0).tolist(),'control_counts':(mask&fitmask).sum(0).tolist(),'shared_counts':partialmask.sum(0).tolist(),'fit_mask_sha256':sha(dest/'fit_mask.npy'),'main_predictions_sha256':sha(dest/'predictions.npz'),'no_base_model_calls':True,'time':time.time()})
 state('complete');print('DONE',model,seed,round(time.time()-started,1),flush=True)
if __name__=='__main__':run(int(sys.argv[1]))


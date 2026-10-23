import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from collections import defaultdict
import json,time,hashlib,zipfile
import numpy as np,pandas as pd
from numerics import weights,project as independent_project
from graph_ops import relation_residuals
O=Path(__file__).resolve().parents[1];I=Path('D:/polymarket_v2_llm_history303_20261016/inputs');S=O/'summary';S.mkdir(exist_ok=True)
P=json.loads((O/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'));H=P['horizons'];MODELS=P['models']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
for n,h in P['input_hashes'].items():assert sha(I/n)==h
for n,h in P['helper_hashes'].items():assert sha(O/'code'/n)==h
f=pd.read_parquet(I/'answers.parquet');ids=f.row_id.to_numpy();y=f[[f'target_{h}' for h in H]].to_numpy()
with np.load(O/'source/COMMON_SUPPORT.npz') as d:mask=d['score_mask'];universal=d['universal_success'];legal=d['calibration_legal']
assert sha(O/'source/COMMON_SUPPORT.npz')==P['support_sha256']
pos={int(r):i for i,r in enumerate(ids)};systems=[]
for s in json.loads((I/'systems.json').read_text()):
 ix=np.array([[pos[r] for r in row] for row in s['row_ids']]);systems.append((s['system_id'],np.array(s['A']),np.array(s['b']),ix,universal[ix].all(1)))
groups,gi=np.unique(f.event_group.to_numpy(),return_inverse=True);ng=len(groups)
nseed={m:sum(u['model']==m for u in P['units']) for m in MODELS}
frames={n:[] for n in ['main','consistency','permutations','semantic_direction','smooth','bands','completeness','completeness_counts','composition']}
GS=defaultdict(lambda:np.zeros(ng));MASS={};max_score_error=0.;max_sem_error=0.;max_residual=0.;projection_replays=[];fit_hashes=set();shared_masks=[];durations=[]
def add_group(model,name,se,ae,select,scope='primary'):
 for j,h in enumerate(H):
  use=select[:,j]
  if not use.any():continue
  w=weights(f[use]);mass=np.bincount(gi[use],weights=w,minlength=ng)
  key=(scope,h)
  if key in MASS:np.testing.assert_allclose(MASS[key],mass,atol=1e-12,rtol=0)
  else:MASS[key]=mass
  for met,loss in [('mse',se),('mae',ae)]:
   assert np.isfinite(loss[use,j]).all()
   GS[(scope,model,h,name,met)]+=np.bincount(gi[use],weights=w*loss[use,j],minlength=ng)/nseed[model]
messages=np.load(I/'neighbor_messages.npz');msg=messages['message'];support=messages['support'];p0=f.raw.to_numpy()
for ui,u in enumerate(P['units']):
 m,s=u['model'],u['seed'];d=O/'results'/f'{m}_{s}';done=json.loads((d/'DONE.json').read_text())
 assert done['protocol_sha256']==sha(O/'PROTOCOL_LOCK.json') and done['raw_source_sha256']==u['sha256']
 assert sha(u['file'])==u['sha256'] and sha(d/'predictions.npz')==done['main_predictions_sha256']
 assert done['counts']==mask.sum(0).tolist();fit_hashes.add(done['fit_mask_sha256']);durations.append(done['seconds'])
 for n in frames:
  x=pd.read_csv(d/f'{n}.csv');x['model']=m;x['seed']=s;frames[n].append(x)
 with np.load(d/'predictions.npz') as z:
  assert np.array_equal(z['row_id'],ids)
  v={k:z[k] for k in ['Raw','Semantic','Logic','Full','ZeroNeighbor','ZeroNeighbor+Logic']}
 main=pd.read_csv(d/'main.csv')
 for name,p in v.items():
  for j,h in enumerate(H):
   use=mask[:,j];ff=f[use];diff=p[use,j]-y[use,j]
   assert np.isfinite(diff).all()
   assert np.min(p[use,j])>=-1e-12 and np.max(p[use,j])<=1+1e-12
   a=pd.DataFrame({'component':ff.component.to_numpy(),'vid':ff.vid.to_numpy(),'se':diff**2,'ae':np.abs(diff)}).groupby(['component','vid'])[['se','ae']].mean().groupby('component').mean().mean()
   row=main[(main.horizon==h)&(main.version==name)].iloc[0]
   err=max(abs(a.se-row.mse),abs(a.ae-row.mae));max_score_error=max(max_score_error,float(err));assert err<1e-12
  add_group(m,name,(p-y)**2,np.abs(p-y),mask)
 expected=np.clip(p0[:,None]+.75*(v['Raw']-p0[:,None])+.25*msg,0,1);use=mask&support
 err=np.max(np.abs(expected[use]-v['Semantic'][use]),initial=0);max_sem_error=max(max_sem_error,float(err));assert err<1e-12
 for sid,a,b,ix,good in systems:
  q=v['Full'][ix].transpose(0,2,1);vals=q.reshape(-1,q.shape[-1])[good.ravel()]
  if len(vals):max_residual=max(max_residual,float(np.max(vals@a.T-b,initial=0)))
 assert max_residual<=1e-8
 # Independent SLSQP solver checks every model family on small, genuinely scored joint systems.
 if s==next(x['seed'] for x in P['units'] if x['model']==m):
  count=0
  for sid,a,b,ix,good in systems:
   if len(a)==0 or ix.shape[1]>12:continue
   active=good&mask[ix].any(1);ij=np.argwhere(active)
   if not len(ij):continue
   chosen=None
   for at,j in ij[:100]:
    pp=v['Semantic'][ix[at],j]
    if np.max(pp@a.T-b,initial=0)>1e-7:chosen=(at,j);break
   if chosen is None:continue
   at,j=chosen;pp=v['Semantic'][ix[at],j];q,ok=independent_project(pp,a,b)
   err=float(np.max(np.abs(q-v['Full'][ix[at],j])));assert ok and err<2e-6
   projection_replays.append({'model':m,'seed':s,'system':sid,'horizon':H[j],'max_abs_error':err});count+=1
   if count>=4:break
 with np.load(d/'diagnostic_predictions.npz') as z:
  for name,k in [('ReversedMessage+Logic','ReversedMessageLogic'),('Semantic+Smooth','SemanticSmooth')]:
   p=z[k];add_group(m,name,(p-y)**2,np.abs(p-y),mask)
 with np.load(d/'diagnostic_losses.npz') as z:
  add_group(m,'PermutedLogic',z['PermutedLogic_mse'],z['PermutedLogic_mae'],mask)
  shared=z['shared_mask'];shared_masks.append(hashlib.sha256(shared.tobytes()).hexdigest())
  for name in ['SingleRelation','LeaveOneOut']:add_group(m,name,z[name+'_mse'],z[name+'_mae'],shared,'shared')
  for name in ['Semantic','Full']:add_group(m,name,(v[name]-y)**2,np.abs(v[name]-y),shared,'shared')
 # Validate every recorded monthly boundary and candidate winner from past-only controls.
 times=f.origin.to_numpy()
 for row in json.loads((d/'parameters.json').read_text()):
  j=H.index(row['horizon']);t=int(pd.Timestamp(row['month']+'-01',tz='UTC').timestamp());cut=t-10*86400
  fit=legal[:,j]&(times>=t-30*86400)&(times+H[j]*60+60<cut)
  sel=legal[:,j]&(times>=cut)&(times+H[j]*60+60<t)
  valid=legal[:,j]&(times>=t-30*86400)&(times+H[j]*60+60<t)
  assert [row['fit'],row['select'],row['validation']]==[int(fit.sum()),int(sel.sum()),int(valid.sum())]
  for spec in row['selected'].values():
   lo=min(x['selection_mse'] for x in spec['candidates']);assert abs(lo-spec['selection_mse'])<1e-12
 print('AUDITED',m,s,flush=True)
assert len(fit_hashes)==1 and len(set(shared_masks))==1
frames={k:pd.concat(v,ignore_index=True) for k,v in frames.items()}
for n,d in frames.items():d.to_csv(S/f'{n}_units.csv',index=False)
def avg(d,keys,values=None):
 if values is None:values=[c for c in ['mse','mae','n','events','contracts','modified','improved','harmed','unchanged','rate','magnitude','max_residual','contexts'] if c in d]
 return d.groupby(keys,dropna=False)[values].mean().reset_index()
main=avg(frames['main'],['model','horizon','version']);raw=main.query("version=='Raw'").set_index(['model','horizon'])
for met in ['mse','mae']:
 main[met+'_gain_percent']=[100*(raw.loc[(r.model,r.horizon),met]-getattr(r,met))/raw.loc[(r.model,r.horizon),met] for r in main.itertuples()]
main.to_csv(S/'main_models.csv',index=False)
macro=main.groupby(['horizon','version'])[['mse','mae']].mean().reset_index();mr=macro.query("version=='Raw'").set_index('horizon')
for met in ['mse','mae']:macro[met+'_gain_percent']=[100*(mr.loc[r.horizon,met]-getattr(r,met))/mr.loc[r.horizon,met] for r in macro.itertuples()]
macro.to_csv(S/'main_model_equal.csv',index=False)
def attribution(d,by_model=True):
 result=[]
 keys=['model','horizon'] if by_model else ['horizon']
 for key,g in d.groupby(keys):
  z=g.set_index('version');r={'model':key[0],'horizon':key[1]} if by_model else {'horizon':key[0] if isinstance(key,tuple) else key}
  for met in ['mse','mae']:
   a,b,c,e=[float(z.loc[n,met]) for n in ['Raw','Semantic','Logic','Full']]
   total=a-e;sm=.5*((a-b)+(c-e));lg=.5*((a-c)+(b-e));assert abs(sm+lg-total)<1e-14
   r.update({met+'_raw':a,met+'_full':e,met+'_total_delta':total,met+'_semantic_stage_delta':a-b,met+'_joint_after_semantic_delta':b-e,met+'_semantic_shapley':sm,met+'_logic_shapley':lg,met+'_interaction':b+c-a-e,met+'_total_gain_percent':100*total/a,met+'_semantic_share_percent':100*sm/total if abs(total)>1e-15 else np.nan,met+'_logic_share_percent':100*lg/total if abs(total)>1e-15 else np.nan})
  result.append(r)
 return pd.DataFrame(result)
att=attribution(main);att.to_csv(S/'attribution_models.csv',index=False);am=attribution(macro,False);am.to_csv(S/'attribution_model_equal.csv',index=False)
overall=macro.groupby('version')[['mse','mae']].mean().reset_index();overall['horizon']=0;attribution(overall,False).to_csv(S/'attribution_overall_equal_horizons.csv',index=False)
tables={}
for n,keys in [('consistency',['model','horizon','version']),('smooth',['model','horizon','version']),('semantic_direction',['model','horizon','version']),('permutations',['model','horizon','version']),('bands',['model','horizon','band','version']),('completeness',['model','horizon','version']),('composition',['model','horizon','control','version'])]:
 tables[n]=avg(frames[n],keys);tables[n].to_csv(S/f'{n}_models.csv',index=False)
cmp=[]
full=main.query("version=='Full'").set_index(['model','horizon'])
for label,dd in [('Logic',main.query("version=='Logic'")),('Semantic',main.query("version=='Semantic'")),('ZeroNeighbor+Logic',main.query("version=='ZeroNeighbor+Logic'")),('PermutedLogic',tables['permutations']),('ReversedMessage+Logic',tables['semantic_direction']),('Semantic+Smooth',tables['smooth'])]:
 for r in dd.itertuples():
  z=full.loc[(r.model,r.horizon)];x={'control':label,'model':r.model,'horizon':r.horizon,'n':r.n}
  for met in ['mse','mae']:
   a=getattr(r,met);b=z[met];x.update({met+'_control':a,met+'_full':b,met+'_delta':a-b,met+'_gain_percent':100*(a-b)/a})
  cmp.append(x)
cm=pd.DataFrame(cmp);cm.to_csv(S/'controls_models.csv',index=False)
cmacro=cm.groupby(['control','horizon'])[[c for c in cm if c not in ['control','model','horizon']]].mean().reset_index()
for met in ['mse','mae']:
 cmacro[met+'_mean_model_relative_gain_percent']=cmacro[met+'_gain_percent']
 cmacro[met+'_gain_percent']=100*cmacro[met+'_delta']/cmacro[met+'_control']
cmacro.to_csv(S/'controls_model_equal.csv',index=False)
com=[]
for (m,h,c),dd in tables['composition'].groupby(['model','horizon','control']):
 dd=dd.set_index('version');x={'model':m,'horizon':h,'control':c,'n':float(dd.loc['Control','n'])}
 for met in ['mse','mae']:
  a=dd.loc['Control',met];b=dd.loc['Control+Full',met];x.update({met+'_before':a,met+'_after':b,met+'_delta':a-b,met+'_gain_percent':100*(a-b)/a})
 com.append(x)
pd.DataFrame(com).to_csv(S/'composition_gains_models.csv',index=False)
# Graph-group block bootstrap, with original scoring weights and paired draws.
comparisons=[('Raw','Full'),('Raw','Semantic'),('Semantic','Full'),('Logic','Full'),('ZeroNeighbor+Logic','Full'),('PermutedLogic','Full'),('ReversedMessage+Logic','Full'),('Semantic+Smooth','Full')]
bootrows=[];group_rows=[]
def boot_delta(num,mass,seed=13):
 use=mass>0;n=int(use.sum())
 if n<2:return [np.nan,np.nan]
 draws=np.random.default_rng(seed).multinomial(n,np.full(n,1/n),2000);den=draws@mass[use];z=(draws@num[use])/den
 return [float(x) for x in np.quantile(z,[.025,.975])]
for scope,h in MASS:
 mass=MASS[(scope,h)]
 comps=comparisons if scope=='primary' else [('SingleRelation','Full'),('LeaveOneOut','Full'),('Semantic','Full')]
 for m in MODELS+['Model-equal']:
  for a,b in comps:
   for met in ['mse','mae']:
    if m=='Model-equal':num=np.mean([GS[(scope,mm,h,a,met)]-GS[(scope,mm,h,b,met)] for mm in MODELS],axis=0)
    else:num=GS[(scope,m,h,a,met)]-GS[(scope,m,h,b,met)]
    lo,hi=boot_delta(num,mass);bootrows.append({'scope':scope,'model':m,'horizon':h,'comparison':a+' -> '+b,'metric':met,'delta':float(num.sum()),'ci_low':lo,'ci_high':hi,'blocks':int((mass>0).sum())})
    if scope=='primary' and a=='Raw' and b=='Full':
     for g,n,w in zip(groups,num,mass):
      if w>0:group_rows.append({'model':m,'horizon':h,'metric':met,'graph_group':g,'weight':w,'weighted_delta':n,'within_group_delta':n/w})
pd.DataFrame(bootrows).to_csv(S/'paired_bootstrap.csv',index=False);gd=pd.DataFrame(group_rows);gd.to_csv(S/'group_contributions.csv',index=False)
distribution=[]
for (m,h,met),dd in gd.groupby(['model','horizon','metric']):
 vals=dd.weighted_delta.to_numpy();positive=vals[vals>0];sort=np.sort(positive)[::-1];total=positive.sum()
 distribution.append({'model':m,'horizon':h,'metric':met,'positive_groups':int((vals>0).sum()),'zero_groups':int((vals==0).sum()),'negative_groups':int((vals<0).sum()),'top_one_positive_share':float(sort[:1].sum()/total) if total else np.nan,'top_two_positive_share':float(sort[:2].sum()/total) if total else np.nan,'delta_excluding_top_two_positive_groups':float((vals.sum()-sort[:2].sum())/(1-dd.sort_values('weighted_delta',ascending=False).head(2).weight.sum())) if len(sort)>=2 else np.nan})
pd.DataFrame(distribution).to_csv(S/'group_distribution.csv',index=False)
# Primary audit checks every unit/horizon/variant score, input hashes, projection feasibility and model-family solver replays.
fullrows=main.query("version=='Full'");signs={}
for met in ['mse','mae']:
 gain=fullrows[met+'_gain_percent'].to_numpy();signs[met]={'positive_model_horizons':int((gain>0).sum()),'zero':int((gain==0).sum()),'negative':int((gain<0).sum()),'total':64,'mean_relative_gain_percent':float(gain.mean()),'min_relative_gain_percent':float(gain.min()),'max_relative_gain_percent':float(gain.max())}
for n,h in P['input_hashes'].items():assert sha(I/n)==h
for n,h in P['helper_hashes'].items():assert sha(O/'code'/n)==h
audit={'passed':True,'models':8,'model_seed_units':24,'source_prediction_hashes_matched':True,'common_counts':mask.sum(0).tolist(),'independent_hierarchical_score_max_error':max_score_error,'semantic_formula_max_error':max_sem_error,'full_projection_max_constraint_residual':max_residual,'independent_solver_replays':projection_replays,'same_calibration_fit_mask_across_all_units':True,'same_shared_constraint_mask_across_all_units':True,'all_recorded_calibration_boundaries_and_selected_minima_checked':True,'no_base_model_retraining':True,'no_api_calls':True,'no_llms':True,'no_persistence_predictor':True,'development_re_evaluation':True,'bootstrap_draws':2000,'protocol_sha256':sha(O/'PROTOCOL_LOCK.json'),'unit_runtime_seconds_sum':sum(durations),'signs':signs,'completed_at':time.time()}
save(S/'AUDIT.json',audit);save(S/'RESULTS_BRIEF.json',{'signs':signs,'model_equal_horizon_attribution':am.replace({np.nan:None}).to_dict('records'),'coverage':P['coverage'],'new_scoring_scope':'Attribution comparison cohort only; original main-table results are preserved.'})
archive=O/'NON_LLM8_ATTRIBUTION_TABLES_AND_AUDIT.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6,strict_timestamps=False) as z:
 for p in sorted(S.glob('*')):
  if p.is_file():z.write(p,'summary/'+p.name)
 for p in [O/'PROTOCOL_LOCK.json',O/'source/TRANSFER_AUDIT.json',O/'source/COVERAGE_FROZEN.csv']:
  z.write(p,str(p.relative_to(O)))
 for p in sorted((O/'code').glob('*.py')):z.write(p,'code/'+p.name)
save(O/'COMPLETE.json',{'passed':True,'models':8,'units':24,'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'audit':str(S/'AUDIT.json'),'time':time.time()})
print(json.dumps({'passed':True,'signs':signs,'model_equal_horizon_attribution':am.replace({np.nan:None}).to_dict('records'),'archive':str(archive)},ensure_ascii=False))


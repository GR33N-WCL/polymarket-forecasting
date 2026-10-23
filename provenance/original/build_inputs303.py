"""User-fixed 303-event scope; exact frozen neural history, no forecast calls."""
from pathlib import Path
import json,hashlib,time,shutil,sys
import numpy as np,pandas as pd
O=Path(__file__).resolve().parent;I=O/'inputs';I.mkdir(parents=True,exist_ok=True)
R=Path('D:/polymarket_semantic520_unified_20261016');P=R/'recovered/F_output/datasets/polymarket_v2_374_20261005'
F=Path('F:/steemit/polymarket');PLAN=F/'outputs/plan1_cross_model_20261010'
OBS=R/'recovered/F_outputs/dataset_v2_20261005/transaction_first_restart/candidate374_development/multihorizon_report_20261005/observations.parquet'
H=[1,5,15,30,60,180,360,1440];GROUPS=['core:26','core:12','core:1','core:14'];LABELS=['G20','G07','G04','G09']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(2**20):h.update(b)
    return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
assert not (O/'RUN_LOCK.json').exists(),'Already sealed'
full=pd.read_parquet(OBS).reset_index(drop=True);full['row_id']=np.arange(len(full));keep=~full.event_group.isin(GROUPS).to_numpy();ids=np.flatnonzero(keep);f=full.loc[keep].reset_index(drop=True)
assert len(f)==183077 and f.event_id.nunique()==303
histpath=PLAN/'inputs/history.npz';hlock=json.loads((PLAN/'inputs/INPUT_LOCK.json').read_text(encoding='utf-8'));assert sha(histpath)==hlock['history_sha256']
with np.load(histpath) as z:
    assert z['prices'].shape==(198449,72)
    history=z['prices'][ids].copy();observed=z['observed'][ids];stale=z['stale'][ids];eligible=z['eligible'][ids];age=z['age'][ids]
assert np.max(abs(history[:,-1]-f.raw.to_numpy()))<3.1e-8
np.savez_compressed(I/'history.npz',row_id=ids,prices=history,observed=observed,stale=stale,eligible=eligible,age=age)
meta=F/'output/datasets/polymarket_v2_374_20261005/metadata/market_metadata.json';metadata=json.loads(meta.read_text(encoding='utf-8'))
missing=[];snapshots={}
for mid in f.market_id.astype(str).unique():
    m=metadata[mid];outs=m.get('outcomes',[])
    if isinstance(outs,str):outs=json.loads(outs)
    s=dict(question=m.get('question',''),outcome=outs[0] if outs else '',settlement_rules=m.get('description',''))
    if not all(s.values()):missing.append(mid)
    snapshots[mid]=s
assert not missing,('Missing full settlement text',missing[:20])
old=Path('D:/polymarket_old_llm_return_20261015/snapshot/inputs/requests.jsonl')
technical={r['row_id'] for line in old.read_text(encoding='utf-8').splitlines() if (r:=json.loads(line))['technical_pilot']}
with (I/'requests.jsonl').open('w',encoding='utf-8',newline='\n') as dst:
    for i,r in enumerate(f.itertuples()):
        s={**snapshots[str(r.market_id)],'prediction_time_utc':pd.Timestamp(r.origin,unit='s',tz='UTC').isoformat(),'current_price':float(r.raw),'forecast_horizons_minutes':H,'history_prices_5min_oldest_to_current':[float(v) for v in history[i]]}
        request=dict(row_id=int(r.row_id),market_id=str(r.market_id),bundle_id=f'{r.event_group}@{r.origin}',origin=int(r.origin),snapshot=s,technical_pilot=int(r.row_id) in technical)
        dst.write(json.dumps(request,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
        if i%25000==0:print('requests',i,len(f),flush=True)
# Frozen labels/masks and their actual trade timestamps: past histories remain isolated from prompts.
maxerr=0.;anomalies=0
for h in [0]+H:f[f'last_trade_{h}']=np.nan
for eid,rows in f.groupby('event_id',sort=False):
    path=P/'trades/minute_history'/f'{eid}.parquet';d=pd.read_parquet(path);d.market_id=d.market_id.astype(str);valid=(d.last_trade_timestamp<=d.available_at)&np.isfinite(d.price);anomalies+=int((~valid).sum());d=d[valid]
    for mid,jj in rows.groupby('market_id',sort=False).groups.items():
        jj=np.array(list(jj));g=d[d.market_id==str(mid)].sort_values('available_at');a=g.available_at.to_numpy();t=g.last_trade_timestamp.to_numpy();p=g.price.to_numpy()
        for h in [0]+H:
            q=f.origin.to_numpy()[jj]+h*60;ix=np.searchsorted(a,q,side='right')-1;safe=np.maximum(ix,0);valid=(ix>=0)&(t[safe]<=q)&(q-t[safe]<=60)
            f.loc[jj,f'last_trade_{h}']=np.where(valid,t[safe],np.nan)
            target=f.raw.to_numpy()[jj] if h==0 else f[f'target_{h}'].to_numpy()[jj]
            check=np.ones(len(jj),bool) if h==0 else f[f'complete_{h}'].to_numpy()[jj]&np.isfinite(target)
            assert valid[check].all(),(eid,mid,h)
            err=float(np.max(abs(p[safe[check]]-target[check]),initial=0));maxerr=max(maxerr,err);assert err<1e-9
f['bundle_id']=f.event_group.astype(str)+'@'+f.origin.astype(str);f.to_parquet(I/'answers.parquet',index=False)
systems=[]
with np.load(P/'graph/projection_systems.npz') as z:
    for k in sorted(v[:-2] for v in z.files if v.endswith('_A')):
        rr=z[k+'_row_indices'];membership=keep[rr];assert np.all(membership.all(1)|(~membership.any(1)))
        rr=rr[membership.all(1)]
        if len(rr):systems.append(dict(system_id=k,A=z[k+'_A'].tolist(),b=z[k+'_b'].tolist(),row_ids=rr.tolist()))
save(I/'systems.json',systems)
with np.load(R/'lambda_sensitivity_20261016/MESSAGES.npz') as z:np.savez_compressed(I/'neighbor_messages.npz',row_id=ids,message=z['message'][ids],support=z['support'][ids])
mask=np.load(R/'full/common_success_mask.npy')[ids];np.save(I/'baseline_common_success_mask.npy',mask)
graph=json.loads((P/'graph/event_graph.json').read_text());nodes=pd.DataFrame(graph['nodes']);retained=nodes[~nodes.event_group.isin(GROUPS)];roster=set(retained.event_id.astype(str));cross=[]
for e in graph['edges']:
    if (str(e['source']) in roster)!=(str(e['target']) in roster):cross.append(e)
assert not cross
retained.to_csv(I/'retained_events.csv',index=False,encoding='utf-8-sig');nodes[nodes.event_group.isin(GROUPS)].to_csv(I/'excluded_events.csv',index=False,encoding='utf-8-sig')
catalog=pd.read_csv(P/'graph/contract_catalog.csv',dtype=str);catalog[catalog.event_id.isin(roster)].to_csv(I/'retained_contracts.csv',index=False)
audit=dict(passed=True,date='2026-10-16',user_confirmed_groups=LABELS,group_ids=GROUPS,selection='Fixed before this new LLM experiment: aggregate prior four-neural-model eight-horizon relative MSE contribution; historical development selection disclosed',events=303,groups=int(f.event_group.nunique()),rows=len(f),contracts=int(f.market_id.nunique()),catalog_contracts=int(catalog.event_id.isin(roster).sum()),bundles=int(f.bundle_id.nunique()),scored_contracts_any_horizon=int(f.loc[f[[f'complete_{h}' for h in H]].any(axis=1),'market_id'].nunique()),score_counts={str(h):int(f[f'complete_{h}'].sum()) for h in H},baseline_common_success_counts=dict(zip(map(str,H),mask.sum(0).astype(int).tolist())),history_points=72,history_step_seconds=300,history_source_sha256=sha(histpath),history_values_identical_to_frozen_neural_input=True,current_alignment_float32_max_abs=float(np.max(abs(history[:,-1]-f.raw.to_numpy()))),history_missing_fill='Original frozen neural preprocessing: 0.5 before first observation, last known prices thereafter; no new null/fill policy',history_missing_points=int((~observed).sum()),history_eligibility_rows=int(eligible.sum()),source_anomalies_excluded=anomalies,target_replay_max_abs=maxerr,whole_group_removal_no_crossing_edges=True,projection_systems=len(systems),projection_instances=sum(len(s['row_ids']) for s in systems),semantic_messages='Exact original signed amplitude-scaled historical messages at lambda .25; complete components removed so no retained neighbor across exclusions',metadata_time_status='Retrospective frozen settlement metadata, no historical versions established')
save(I/'SCOPE_AND_INPUT_AUDIT.json',audit)
save(I/'INPUT_LOCK.json',dict(version='v2_history303_fixed4_20261016',created_at=time.time(),rows=len(f),bundles=int(f.bundle_id.nunique()),events=303,excluded_groups=GROUPS,history={'points':72,'step_minutes':5,'source':'same frozen history.npz as original neural models'},source_sha256={str(OBS):sha(OBS),str(histpath):sha(histpath),str(meta):sha(meta)},input_sha256={p.name:sha(p) for p in sorted(I.iterdir()) if p.is_file() and p.name!='INPUT_LOCK.json'},terminal=False))
print(json.dumps(audit,ensure_ascii=False),flush=True)

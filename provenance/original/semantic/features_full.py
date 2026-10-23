from pathlib import Path
import os,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import json,time,collections,concurrent.futures
import numpy as np,pandas as pd
from evaluate import ROOT,OLD,H,CFG,save,sha,paths,observations
from async_features import causal_sample
from frozen_method import relation_stats

def event_features(arg):
 eid,rows,data=arg;data=Path(data);out=ROOT/'feature_chunks'/f'{eid}.json'
 if out.exists():return eid,len(json.loads(out.read_text())['accepted'])
 roles=pd.read_csv(data/'graph/event_edge_roles.csv',dtype={'source':str,'target':str});roles=roles[roles.feature_source_to_target|roles.feature_target_to_source];assert len(roles)==520
 neighbors=sorted(set(roles.loc[roles.source==eid,'target'])|set(roles.loc[roles.target==eid,'source']))
 hist={};markets={};hashes={};badrows=0
 for event in [eid]+neighbors:
  p=data/'trades/minute_history'/f'{event}.parquet'
  if not p.exists():markets[event]=[];continue
  hashes[str(p)]=sha(p);d=pd.read_parquet(p);bad=(d.last_trade_timestamp>d.available_at)|(~np.isfinite(d.price));badrows+=int(bad.sum());d=d[~bad];markets[event]=[]
  for mid,g in d.groupby('market_id',sort=False):
   mid=str(mid);g=g.sort_values('available_at');hist[mid]=tuple(g[c].to_numpy() for c in ['available_at','last_trade_timestamp','price']);markets[event].append(mid)
 result=[];counts=collections.Counter();edgecounts=collections.defaultdict(collections.Counter);step=CFG['history_grid_seconds'];hh=np.array(H)
 for r in rows:
  mid=str(r['market_id']);origin=int(r['origin']);ih=hist.get(mid)
  if ih is None:continue
  end=((origin-1)//step)*step;start=max(end-CFG['history_days']*86400,int(ih[0][0]));start=((start+step-1)//step)*step;t=np.arange(start,end+1,step,dtype=np.int64)
  own=np.diff(causal_sample(ih,t,CFG['history_freshness_seconds'])[0]) if len(t)>1 else np.array([])
  for other in neighbors:
   for jm in sorted(markets[other]):
    jh=hist[jm];now,age=causal_sample(jh,[origin-1],CFG['history_freshness_seconds'])
    if not np.isfinite(now[0]):reason='no_recent_causal_neighbor_price';stats=None
    else:
     prev,_=causal_sample(jh,origin-1-hh*60,CFG['history_freshness_seconds']);valid=np.isfinite(prev)
     if not valid.any():reason='no_past_signal_window';stats=None
     else:
      stats=relation_stats(own,np.diff(causal_sample(jh,t,CFG['history_freshness_seconds'])[0])) if len(t)>1 else dict(n=0,accepted=False,reason='insufficient_paired_changes');reason=stats['reason']
    counts[reason]+=1;edgecounts[other][reason]+=1
    if stats and stats['accepted']:
     signal=now[0]-prev
     result.append(dict(bundle=r['bundle_id'],i=int(r['row_id']),row_i=int(r['row_id']),market_i=mid,market_j=jm,event_i=eid,event_j=other,origin=origin,**stats,history_end=end,signal_end=origin-1,neighbor_trade_age_seconds=int(age[0]),signal_valid=valid.tolist(),signal=[float(v) if np.isfinite(v) else None for v in signal],signal_decay=np.exp(-age[0]/np.maximum(300,hh*60)).tolist()))
 save(out,dict(event=eid,rows=len(rows),accepted=result,reasons=dict(counts),edge_reasons={k:dict(v) for k,v in edgecounts.items()},history_hashes=hashes,excluded_invalid_history_rows=badrows));return eid,len(result)

def main():
 _,data,source=paths();f=observations()[['row_id','event_id','market_id','origin','bundle_id']];(ROOT/'feature_chunks').mkdir(exist_ok=True);f.to_parquet(ROOT/'feature_inputs.parquet',index=False)
 tasks=[(str(e),g.to_dict('records'),str(data)) for e,g in f.groupby('event_id',sort=True)]
 done=0;accepted=0;start=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
  for event,n in pool.map(event_features,tasks,chunksize=1):
   done+=1;accepted+=n;save(ROOT/'FEATURE_STATUS.json',{'stage':'building','event_groups_done':done,'events_total':len(tasks),'accepted_instances_so_far':accepted,'elapsed':time.time()-start,'time':time.time()});print('features',done,len(tasks),event,n,flush=True)
 # Exact common-subset feature replay against the previously independent-audited method.
 old=[e for e in json.loads((OLD/'edge_instances.json').read_text()) if e['accepted']];ids=set(pd.read_parquet('D:/polymarket_old_llm_return_20261015/snapshot/inputs/answers.parquet').row_id);current={};counts=collections.Counter();hh={};allaccepted=[]
 for p in sorted((ROOT/'feature_chunks').glob('*.json')):
  d=json.loads(p.read_text());counts.update(d['reasons']);hh.update(d['history_hashes']);allaccepted+=d['accepted']
  for e in d['accepted']:
   if e['row_i'] in ids:current[(e['row_i'],e['market_j'])]=e
 assert len(current)==len(old),(len(current),len(old))
 for e in old:
  q=current[(e['row_i'],e['market_j'])]
  for k in ['sign','n','signal_valid','origin','history_end','signal_end']:assert q[k]==e[k],k
  for k in ['rho','sd_i','sd_j','weight','amplitude_i','neighbor_trade_age_seconds','signal','signal_decay','rho_halves']:assert np.allclose(np.array(q[k],float),np.array(e[k],float),atol=1e-12,rtol=1e-12,equal_nan=True),k
 oldhash=json.loads((OLD/'HISTORY_LOAD.json').read_text())['history_hashes'];byname={Path(k).name:v for k,v in oldhash.items()}
 for k,v in hh.items():
  if Path(k).name in byname:assert v==byname[Path(k).name],k
 assert all(e['history_end']<e['origin'] and e['signal_end']<e['origin'] for e in allaccepted)
 save(ROOT/'FEATURE_AUDIT.json',dict(passed=True,source=source,accepted_common_replayed=len(old),accepted_all=len(allaccepted),candidate_pair_instances=sum(counts.values()),reasons=dict(counts),supported_records=len({e['i'] for e in allaccepted}),accepted_event_edges=len({tuple(sorted((e['event_i'],e['event_j']))) for e in allaccepted}),history_hashes=hh,no_future_endpoints=True))
 save(ROOT/'FEATURE_STATUS.json',{'stage':'complete','events':len(tasks),'accepted':len(allaccepted),'time':time.time()})
if __name__=='__main__':main()


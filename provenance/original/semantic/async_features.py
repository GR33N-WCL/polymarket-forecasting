"""Causal historical-neighbor features, independent of LLM neighbor coverage."""
from pathlib import Path
import json,hashlib,collections,time
import numpy as np
import pandas as pd

ENV={}

def causal_sample(hist,t,max_age=86400):
    t=np.asarray(t,dtype=np.int64)
    a,last,p=hist
    ix=np.searchsorted(a,t,side='right')-1
    pos=np.maximum(ix,0)
    age=t-last[pos]
    ok=(ix>=0)&(age>=0)&(age<=max_age)
    return np.where(ok,p[pos],np.nan),np.where(ok,age,np.nan)

def build_edges(frame,env):
    ENV.update(env)
    root,data,cfg=env['ROOT'],env['DATA'],env['CFG']
    save,sha=env['save'],env['sha']
    if (root/'edge_instances.json').exists():
        return env['read'](root/'edge_instances.json')
    roles=pd.read_csv(data/'graph/event_edge_roles.csv',dtype={'source':str,'target':str})
    roles=roles[roles.feature_source_to_target|roles.feature_target_to_source]
    assert len(roles)==520
    adj=collections.defaultdict(set)
    for a,b in zip(roles.source,roles.target):
        adj[a].add(b);adj[b].add(a)
    target_events=set(frame.event_id.astype(str))
    needed=target_events|{b for a in target_events for b in adj[a]}
    histories={};markets=collections.defaultdict(list);hashes={};removed=0;missing=[]
    for eid in sorted(needed):
        path=data/'trades/minute_history'/f'{eid}.parquet'
        if not path.exists():missing.append(eid);continue
        hashes[str(path)]=sha(path)
        d=pd.read_parquet(path)
        bad=(d.last_trade_timestamp>d.available_at)|(~np.isfinite(d.price))
        removed+=int(bad.sum());d=d[~bad]
        for mid,g in d.groupby('market_id',sort=False):
            mid=str(mid);g=g.sort_values('available_at')
            histories[mid]=tuple(g[c].to_numpy() for c in ['available_at','last_trade_timestamp','price'])
            markets[eid].append(mid)
    save('HISTORY_LOAD.json',dict(event_edges=520,event_nodes=len(adj),loaded_events=len(hashes),contracts=len(histories),missing_history_events=missing,excluded_invalid_rows=removed,history_hashes=hashes))
    result=[];reasons=collections.Counter();step=cfg['history_grid_seconds'];H=np.array(env['H']);started=time.time()
    for row in frame.itertuples():
        eid=str(row.event_id);mid=str(row.market_id);origin=int(row.origin)
        ih=histories.get(mid)
        if ih is None:continue
        end=((origin-1)//step)*step
        start=max(end-cfg['history_days']*86400,int(ih[0][0]));start=((start+step-1)//step)*step
        t=np.arange(start,end+1,step,dtype=np.int64)
        own=np.diff(causal_sample(ih,t,cfg['history_freshness_seconds'])[0]) if len(t)>1 else np.array([])
        for other in sorted(adj[eid]):
            for jm in sorted(markets[other]):
                jh=histories[jm]
                c=dict(bundle=row.bundle_id,i=int(row.Index),row_i=int(row.row_id),market_i=mid,market_j=jm,event_i=eid,event_j=other,origin=origin)
                # The feature request is strictly earlier than the forecast origin.
                now,age=causal_sample(jh,[origin-1],cfg['history_freshness_seconds'])
                if not np.isfinite(now[0]):
                    reason='no_recent_causal_neighbor_price'
                    result.append(dict(**c,accepted=False,reason=reason,n=0));reasons[reason]+=1;continue
                prev,_=causal_sample(jh,origin-1-H*60,cfg['history_freshness_seconds'])
                valid=np.isfinite(prev)
                if not valid.any():
                    reason='no_past_signal_window';result.append(dict(**c,accepted=False,reason=reason,n=0));reasons[reason]+=1;continue
                signal=now[0]-prev
                if len(t)<2:stats=dict(n=0,accepted=False,reason='insufficient_paired_changes')
                else:stats=env['relation_stats'](own,np.diff(causal_sample(jh,t,cfg['history_freshness_seconds'])[0]))
                reasons[stats['reason']]+=1
                record=dict(**c,**stats,history_end=end,signal_end=origin-1,neighbor_trade_age_seconds=int(age[0]),signal_valid=valid.tolist())
                if stats['accepted']:
                    record['signal']=[float(v) if np.isfinite(v) else None for v in signal]
                    record['signal_decay']=np.exp(-age[0]/np.maximum(300,H*60)).tolist()
                result.append(record)
        if (row.Index+1)%200==0:
            env['progress']('building_async_edges',rows=int(row.Index+1),total=len(frame),candidate_instances=len(result),accepted=reasons['accepted'],elapsed_seconds=round(time.time()-started))
    good=[e for e in result if e['accepted']]
    save('edge_instances.json',result)
    save('EDGE_HISTORY_AUDIT.json',dict(event_roster_edges=520,event_roster_nodes=len(adj),candidate_pair_instances=len(result),accepted_pair_instances=len(good),reasons=dict(reasons),candidate_event_edges=len({tuple(sorted((e['event_i'],e['event_j']))) for e in result}),accepted_event_edges=len({tuple(sorted((e['event_i'],e['event_j']))) for e in good}),accepted_contract_pairs=len({tuple(sorted((e['market_i'],e['market_j']))) for e in good}),target_records=len({e['i'] for e in good}),target_contracts=len({e['market_i'] for e in good}),neighbor_contracts=len({e['market_j'] for e in good}),positive_instances=sum(e['sign']>0 for e in good),negative_instances=sum(e['sign']<0 for e in good),no_future_endpoints=all(e.get('history_end',-1)<e['origin'] and e.get('signal_end',-1)<e['origin'] for e in result)))
    return result

def messages(z,p0,present,edges,variant,seed=13):
    num=np.zeros_like(z);den=np.zeros_like(z)
    active=[e for e in edges if e['accepted'] and present[e['i']]]
    groups=collections.defaultdict(list)
    for e in active:groups[e['bundle']].append(e)
    changed=0
    for bundle,es in sorted(groups.items()):
        signs=np.array([e['sign'] for e in es])
        if variant=='AllPositive':used=np.ones(len(es))
        elif variant=='PermutedSign':
            key=int(hashlib.sha256(f'{seed}|{bundle}'.encode()).hexdigest()[:16],16)
            used=np.random.default_rng(key).permutation(signs);changed+=int((used!=signs).sum())
        else:used=signs
        for e,s in zip(es,used):
            i,w=e['i'],e['weight'];valid=np.array(e['signal_valid'])
            delta=np.array([v if v is not None else 0 for v in e['signal']])
            num[i]+=w*s*e['amplitude_i']*delta*np.array(e['signal_decay'])*valid
            den[i]+=w*valid
    support=den>0
    m=np.divide(num,den,out=np.zeros_like(num),where=support)
    if variant=='ZeroNeighbor':m[:]=0
    return m,support,dict(active_edges=len(active),positive=sum(e['sign']>0 for e in active),negative=sum(e['sign']<0 for e in active),changed_permuted_signs=changed,nonzero_message_cells=int((abs(m)>1e-12).sum()))

def choose_lambdas(frame,z,m,support,ready,key,mode,variant,seed):
    lam=ENV['CFG']['fixed_lambda'];out=z.copy();p0=frame.raw.to_numpy()
    active=support&ready[:,None]
    candidate=np.clip(p0[:,None]+(1-lam)*(z-p0[:,None])+lam*m,0,1)
    out[active]=candidate[active]
    params=[dict(model=key,mode=mode,variant=variant,seed=seed,horizon=h,lambda_value=lam,reason='preregistered_fixed_strength_no_target_selection',applied_rows=int(active[:,j].sum())) for j,h in enumerate(ENV['H'])]
    return out,params

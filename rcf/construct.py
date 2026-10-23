"""Minute aggregation, causal lookup, and replay checks for frozen dataset targets."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
H=[1,5,15,30,60,180,360,1440]

def minute_prices(normalized_fills):
    """Input price must already refer to the catalogued contract outcome.

    Equal timestamps retain input order as a deterministic tie-breaker. Trade IDs
    and deduplication must be established by the source-specific ingestion step.
    """
    f=normalized_fills.copy()
    required={'market_id','timestamp','price'}
    if not required<=set(f):raise ValueError('Expected market_id, timestamp (UTC epoch seconds), price')
    f=f[np.isfinite(f.timestamp)&np.isfinite(f.price)&f.price.between(0,1)].copy()
    f['minute']=np.floor(f.timestamp/60).astype('int64')*60
    f=f.sort_values('timestamp',kind='stable').groupby(['market_id','minute'],sort=True).tail(1)
    f['available_at']=f.minute+60;f['last_trade_timestamp']=f.timestamp
    return f[['market_id','available_at','last_trade_timestamp','price']].sort_values(['market_id','available_at']).reset_index(drop=True)

def causal_lookup(history,query,max_age):
    a,last,p=(history[k].to_numpy() for k in ['available_at','last_trade_timestamp','price'])
    q=np.asarray(query);ix=np.searchsorted(a,q,side='right')-1;safe=np.maximum(ix,0)
    if len(a)==0:return np.full(q.shape,np.nan),np.zeros(q.shape,bool)
    valid=(ix>=0)&(last[safe]<=q)&(q-last[safe]<=max_age)
    return np.where(valid,p[safe],np.nan),valid

def replay(data,output):
    data=Path(data);f=pd.read_parquet(data/'core/answers.parquet');maximum=0.;checked=0;badrows=0
    for eid,rows in f.groupby('event_id',sort=True):
        hist=pd.read_parquet(data/'history'/f'{eid}.parquet');hist.market_id=hist.market_id.astype(str)
        bad=(hist.last_trade_timestamp>hist.available_at)|~np.isfinite(hist.price);badrows+=int(bad.sum());hist=hist[~bad]
        for mid,rr in rows.groupby('market_id',sort=False):
            g=hist[hist.market_id==str(mid)].sort_values('available_at')
            for h in [0]+H:
                pred,ok=causal_lookup(g,rr.origin.to_numpy()+h*60,60)
                y=rr.raw.to_numpy() if h==0 else rr[f'target_{h}'].to_numpy()
                take=np.ones(len(rr),bool) if h==0 else rr[f'complete_{h}'].to_numpy()&np.isfinite(y)
                assert ok[take].all(),(eid,mid,h)
                err=float(np.max(abs(pred[take]-y[take]),initial=0));assert err<1e-9,(eid,mid,h,err)
                maximum=max(maximum,err);checked+=int(take.sum())
    e=np.load(data/'core/semantic_edge_arrays.npz');num=np.zeros_like(e['denominator'])
    np.add.at(num,e['target_index'],(e['weight']*e['signed_amplitude'])[:,None]*e['source_signal']*e['valid'])
    msg=np.divide(num,e['denominator'],out=np.zeros_like(num),where=e['denominator']>0)
    saved=np.load(data/'core/neighbor_messages.npz');assert np.array_equal(saved['support'],e['denominator']>0)
    semantic_error=float(np.max(abs(msg-saved['message']),initial=0));assert semantic_error<1e-12
    report={'status':'passed','checked_current_and_valid_targets':checked,'max_price_error':maximum,
            'invalid_history_rows_excluded':badrows,'semantic_slots':len(e['target_index']),
            'semantic_message_max_error':semantic_error,'scope':'replay at frozen origins; does not rerun historical event discovery or select new origins'}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--output',required=True)
    a=ap.parse_args();replay(a.data,a.output)

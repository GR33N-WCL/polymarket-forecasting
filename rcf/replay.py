"""Recompute the eight-model attribution panel from frozen predictions."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from .numerics import project_batch, weights

H=[1,5,15,30,60,180,360,1440]

def replay(data, output, model=None, seed=None, save_predictions=False):
    data, output=Path(data),Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory; previous results are preserved')
    output.mkdir(parents=True,exist_ok=True)
    f=pd.read_parquet(data/'core/answers.parquet');ids=f.row_id.to_numpy()
    pos={int(r):i for i,r in enumerate(ids)}
    z=np.load(data/'core/neighbor_messages.npz',allow_pickle=False)
    assert np.array_equal(z['row_id'],ids)
    message,support=z['message'],z['support']
    s=np.load(data/'benchmark/COMMON_SUPPORT.npz',allow_pickle=False)
    universal,mask=s['universal_success'],s['score_mask']
    assert not (mask&~universal).any()
    y=f[[f'target_{h}' for h in H]].to_numpy();p0=f.raw.to_numpy()
    systems=[]
    for s in json.loads((data/'core/systems.json').read_text()):
        ix=np.asarray([[pos[r] for r in rr] for rr in s['row_ids']])
        systems.append((np.array(s['A']),np.array(s['b']),ix,universal[ix].all(1)))
    def mix(raw,msg):
        out=raw.copy();use=support&universal
        v=np.clip(p0[:,None]+.75*(raw-p0[:,None])+.25*msg,0,1)
        out[use]=v[use];return out
    def project(raw):
        out=raw.copy()
        for a,b,ix,good in systems:
            v=raw[ix].transpose(0,2,1);shape=v.shape
            flat=v.reshape(-1,shape[-1]).copy();keep=good.ravel()
            assert np.isfinite(flat[keep]).all()
            pp=project_batch(flat[keep],a,b)
            assert np.max(pp@a.T-b,initial=0)<=1e-8
            flat[keep]=pp;out[ix]=flat.reshape(shape).transpose(0,2,1)
        return out
    cfg=json.loads((Path(__file__).parents[1]/'configs/benchmark.json').read_text())
    audits=[];tables=[]
    for u in cfg['units']:
        if model and u['model']!=model:continue
        if seed is not None and u['seed']!=seed:continue
        name=f"{u['model']}_{u['seed']}"
        cache=np.load(data/'benchmark'/u['file'],allow_pickle=False)
        assert np.array_equal(cache['row_id'],ids)
        raw=cache['Raw'].copy();raw[~cache['success']]=np.nan
        sem=mix(raw,message);zero=mix(raw,np.zeros_like(message))
        preds={'Raw':raw,'Semantic':sem,'Logic':project(raw),'Full':project(sem),
               'ZeroNeighbor':zero,'ZeroNeighbor+Logic':project(zero)}
        rows=[]
        for j,h in enumerate(H):
            use=mask[:,j];ff=f[use];w=weights(ff);yy=y[use,j];zz=raw[use,j]
            for variant,p in preds.items():
                q=p[use,j];assert np.isfinite(q).all()
                delta=(zz-yy)**2-(q-yy)**2;moved=np.abs(q-zz)>1e-10
                rows.append(dict(model=u['model'],seed=u['seed'],horizon=h,version=variant,n=int(use.sum()),
                    events=int(ff.event_id.nunique()),contracts=int(ff.market_id.nunique()),
                    mse=float(w@((q-yy)**2)),mae=float(w@abs(q-yy)),modified=int(moved.sum()),
                    improved=int((moved&(delta>0)).sum()),harmed=int((moved&(delta<0)).sum()),unchanged=int((~moved).sum())))
        table=pd.DataFrame(rows);table.to_csv(output/(name+'.csv'),index=False);tables.append(table)
        expected=pd.read_csv(data/'benchmark/expected'/f'{name}.csv')
        match=table.merge(expected,on=['model','seed','horizon','version'],suffixes=('','_expected'),validate='one_to_one')
        assert len(match)==len(table)==48
        err=max(float(abs(match[c]-match[c+'_expected']).max()) for c in ['mse','mae'])
        assert err<1e-10,(name,err)
        for c in ['n','modified','improved','harmed','unchanged']:assert (match[c]==match[c+'_expected']).all(),(name,c)
        if save_predictions:np.savez_compressed(output/(name+'.npz'),row_id=ids,**preds)
        audits.append(dict(unit=name,metric_rows=len(table),max_metric_error=err,status='passed'))
        print(name,'passed',err,flush=True)
    if not tables:raise ValueError('No matching unit')
    table=pd.concat(tables,ignore_index=True);table.to_csv(output/'per_seed.csv',index=False)
    table.groupby(['model','horizon','version'])[['mse','mae']].mean().reset_index().to_csv(output/'model_means.csv',index=False)
    report=dict(status='passed',units=len(audits),audit=audits,scope='eight non-LLM common-support attribution panel; not S1/S2/S3 main-table replication')
    (output/'REPLAY_AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');ap.add_argument('--output',required=True)
    ap.add_argument('--model');ap.add_argument('--seed',type=int);ap.add_argument('--save-predictions',action='store_true')
    a=ap.parse_args();replay(a.data,a.output,a.model,a.seed,a.save_predictions)

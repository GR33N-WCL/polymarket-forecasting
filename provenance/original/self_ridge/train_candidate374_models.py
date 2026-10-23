"""Complete four-method 5/15-minute comparison on the frozen 374-event cohort.

No event/origin selection, target changes or held-out performance tuning.
Single-contract daily fit caches are reusable because their training is local;
monthly hyperparameter selection and graph fits are repeated on all v2 rows.
"""
from pathlib import Path
import sys, json, time, argparse
sys.path.insert(0, str(Path.home()/'.cache/polymarket-experiment-deps'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from expanded_guarded_core import (Projector, asof, own_features, weights, fit_ridge,
    predict_ridge, month_windows, select_candidate, PENALTIES, ALPHAS, sha, guard_disk)
from evaluate_frozen_full_v7 import cutoff
from evaluate_final288_multihorizon import dump, FULL
from analyze_final288_multihorizon import score, paired
from run_performance_core_v3 import constraint_features
from connect_performance_candidate_v3 import append_messages

ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'outputs/dataset_v2_20261005/transaction_first_restart/candidate374_development'
OLD=ROOT/'outputs/final288_multihorizon_20261005'
POOL=C.parent/'certified_price_pool'
OUT=C/'four_method_comparison'
H=[5,15]
METHODS=['Market Persistence','Self-Ridge','Fixed Projection',FULL]

def source_paths():
    return [C/x for x in ['MANIFEST.json','events.csv','event_edges.csv','added_constraints.json','added_observations.parquet','metrics.csv','predictions_5.parquet','predictions_15.parquet']]+[
        OLD/x for x in ['observations.parquet','constraints.json','own_features.npy','geometry_features.npy','graph_features.npy','self_5.npz','self_15.npz','self_fit_audit_1_5_15_30_60_180_360_1440.parquet','active_message_roster.csv']]+[ROOT/'FINAL_DATASET.json']

def protocol():
    return dict(dataset='candidate374 retained-weather development cohort',counts=[374,550,5609,47],horizons=H,
        origins='Exactly all 198449 rows from candidate predictions; identical per-horizon targets, masks and original scoring identities',
        comparison='All four methods scored on exactly the same existing complete-vector masks',
        roster_selection='Frozen; no additions, removals, resampling or outcome-dependent decisions in this run',
        self_training='Per contract per UTC day, trailing 30 days, latest 5000 eligible history rows, at least 50; label query strictly before day start; predict clipped price change within +/-0.1',
        self_cache='Revalidate and reuse existing per-contract daily-fit grids for 87851 core rows; fit daily grids for every added contract; rerun monthly penalty selection on full v2 validation rows',
        self_selection='Choose penalty on previous 30-day validation MSE using existing macro weights; fixed penalty10 if fewer than50 validation rows; ties prefer larger penalty',
        graph_training='At each UTC month boundary, train on preceding days90..30, validate on preceding30 days, predict next calendar month; training and validation labels mature before their respective boundaries',
        features='11 own-history +12 current constraint geometry +9 neighbor-activity terms; core frozen feature roster retained, all74 additional event edges offered bidirectionally, only causal as-of messages used',
        neighbor_values='Mean five-minute signed and absolute contract changes, log1p(active contracts), fraction with lag; source trade age<=300s at query, no future values, no new semantic hard rules',
        selection='Same existing penalty/alpha grid; validation MSE and MAE each <=1.01 times fixed projection; minMSE then smalleralpha then largerpenalty; alpha0 fallback always available',
        insufficient_history='Self predicts persistence when <50 prior contract examples; graph predicts fixed projection when training or validation has<50 rows; fallback rows retained and reported',
        penalties=PENALTIES,alphas=ALPHAS,
        aggregation='Equal contract rows within vid, equal vids within component, equal scoring components; event graph groups are not scoring components',
        prediction_bounds=[0,1],projection_tolerance=1e-8,
        evaluation_status='Development-selected cohort with chronological model training; not an untouched independent test',
        fit_scope='Only two user-requested horizons; no model tuning after scored results are inspected')

def lock():
    OUT.mkdir(exist_ok=True);guard_disk(OUT)
    data=dict(protocol=protocol(),source_sha256={str(p):sha(p) for p in source_paths()},
              numerical_code_sha256={n:sha(ROOT/'scripts'/n) for n in ['expanded_guarded_core.py','expanded_guarded_fit.py','run_performance_core_v3.py','connect_performance_candidate_v3.py']})
    dest=OUT/'PROTOCOL_LOCK.json'
    if dest.exists():
        previous=json.loads(dest.read_text(encoding='utf-8'))
        assert all(previous[k]==v for k,v in data.items()),'Input or protocol changed'
    else:dump(dest,dict(**data,locked_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),entrypoint_sha256=sha(__file__)))

def new_histories(extra):
    meta=json.loads((C/'cap100_control/semantics/market_metadata.json').read_text(encoding='utf-8'))
    hist={};audits=[]
    for eid in sorted(extra.event_id.unique()):
        folder=POOL/'trades/events'/eid
        status=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        digest=sha(folder/'history.parquet');assert status['history_sha256']==digest
        z=pd.read_parquet(folder/'history.parquet')
        for mid,g in z.groupby('market_id',sort=False):
            if mid not in meta:continue
            g=g.sort_values('available_at')
            assert not g.available_at.duplicated().any()
            assert (g.last_trade_timestamp<=g.available_at).all()
            hist[mid]=(g.available_at.to_numpy(np.int64),g.last_trade_timestamp.to_numpy(np.int64),g.price.to_numpy(float))
        audits.append(dict(event_id=eid,history=str(folder/'history.parquet'),sha256=digest,rows=len(z)))
    assert set(extra.market_id)<=set(hist)
    return hist,meta,audits

def prepare():
    if (OUT/'PREPARATION.json').exists():return
    started=time.perf_counter()
    base=pd.read_parquet(C/'predictions_5.parquet')
    old=pd.read_parquet(OLD/'observations.parquet')
    extra=pd.read_parquet(C/'added_observations.parquet')
    nold=len(old);assert nold==87851 and len(base)==198449
    ids=['vid','component','event_component','event_id','market_id','origin','raw']
    pd.testing.assert_frame_equal(base.iloc[:nold][ids].reset_index(drop=True),old[ids],check_dtype=False)
    pd.testing.assert_frame_equal(base.iloc[nold:][ids].reset_index(drop=True),extra[ids],check_dtype=False)
    d=base[ids+['event_group','topic','source','bootstrap_cluster']].copy()
    for h in H:
        z=pd.read_parquet(C/f'predictions_{h}.parquet')
        pd.testing.assert_frame_equal(z[ids],d[ids],check_dtype=False)
        np.testing.assert_allclose(z['Fixed Projection'],base['Fixed Projection'],atol=0,rtol=0)
        d[f'target_{h}']=z.target;d[f'complete_{h}']=z.scored
        assert d.groupby('vid')[f'complete_{h}'].nunique().eq(1).all()
    print('Preparing causal histories and features for 86 additional events',flush=True)
    hist,meta,history_audit=new_histories(extra)
    ownnew=np.zeros((len(extra),11));future_replay={h:0. for h in H};current_replay=0.
    for mid,ii in extra.groupby('market_id',sort=False).indices.items():
        record=hist[mid];tt=extra.origin.to_numpy()[ii];end=cutoff(meta[mid])
        assert np.isfinite(end) and (tt<end).all()
        now=asof(record,tt)[0]
        np.testing.assert_allclose(now,extra.raw.to_numpy()[ii],atol=1e-10,rtol=0)
        current_replay=max(current_replay,float(np.max(abs(now-extra.raw.to_numpy()[ii]))))
        ownnew[ii]=own_features(record,tt)
        for h in H:
            yy=np.where(tt+60*h<end,asof(record,tt+60*h)[0],np.nan)
            oldy=extra[f'target_{h}'].to_numpy()[ii]
            np.testing.assert_allclose(yy,oldy,atol=1e-10,rtol=0,equal_nan=True)
            if np.isfinite(yy).any():future_replay[h]=max(future_replay[h],float(np.nanmax(abs(yy-oldy))))
    own=np.vstack([np.load(OLD/'own_features.npy'),ownnew])
    rules=json.loads((OLD/'constraints.json').read_text(encoding='utf-8'))+json.loads((C/'added_constraints.json').read_text(encoding='utf-8'))
    projector=Projector(d,rules,global_constraints=True)
    fixed=base['Fixed Projection'].to_numpy();replay=projector.apply(d.raw.to_numpy())
    np.testing.assert_allclose(replay,fixed,atol=1e-10,rtol=0)
    geometry=constraint_features(projector,d.raw.to_numpy(),fixed,own)
    np.testing.assert_allclose(geometry[:nold],np.load(OLD/'geometry_features.npy'),atol=1e-9,rtol=0)
    # All additional event edges are pre-declared; no edge selection using losses.
    nodes=pd.read_csv(C/'events.csv',dtype={'event_id':str}).set_index('event_id')
    catalog=pd.read_csv(C/'contract_catalog.csv',dtype=str).set_index('market_id')
    parts=[]
    for mid,record in hist.items():
        if mid not in catalog.index:continue
        eid=catalog.loc[mid,'event_id'];tt,last,price=record
        lag=asof(record,tt-300,300)[0]
        use=(tt<cutoff(meta[mid])) & (tt-last>=0) & (tt-last<=60)
        if use.any():parts.append(pd.DataFrame(dict(event_id=eid,t=tt[use],last=last[use],
            signed=np.nan_to_num((price-lag)[use]),absolute=np.nan_to_num(abs(price-lag)[use]),has_lag=np.isfinite(lag[use]).astype(float))))
    activity=pd.concat(parts,ignore_index=True).groupby(['event_id','t'],as_index=False).agg(signed=('signed','mean'),absolute=('absolute','mean'),active=('signed','size'),has_lag=('has_lag','mean'),last=('last','max'))
    records={eid:g.sort_values('t') for eid,g in activity.groupby('event_id')}
    edges=pd.read_csv(C/'event_edges.csv',dtype={'source':str,'target':str})
    edges=edges[edges.source.isin(extra.event_id)&edges.target.isin(extra.event_id)]
    assert len(edges)==74
    byevent=extra.groupby('event_id').indices;times=extra.origin.to_numpy()
    total=np.zeros((len(extra),4));count=np.zeros(len(extra),int);edge_audit=[]
    for e in edges.itertuples():
        for source,target in [(e.source,e.target),(e.target,e.source)]:
            ids0=byevent[target];g=records.get(source);good=np.zeros(len(ids0),bool)
            if g is not None and len(g):
                ix=np.searchsorted(g.t.to_numpy(),times[ids0],side='right')-1;safe=np.maximum(ix,0)
                age=times[ids0]-g['last'].to_numpy()[safe]
                good=(ix>=0)&(age>=0)&(age<=300)
                assert (g.t.to_numpy()[safe[good]]<=times[ids0][good]).all()
                ids=ids0[good]
                values=np.column_stack([g.signed.to_numpy()[safe[good]],g.absolute.to_numpy()[safe[good]],np.log1p(g.active.to_numpy()[safe[good]]),g.has_lag.to_numpy()[safe[good]]])
                total[ids]+=values;count[ids]+=1
            edge_audit.append(dict(source=source,target=target,rows=int(good.sum()),source_topic=nodes.loc[source,'topic'],max_allowed_trade_age_seconds=300))
    total/=np.maximum(count[:,None],1)
    graph=np.vstack([np.load(OLD/'graph_features.npy'),append_messages(geometry[nold:],total,count,extra.raw.to_numpy())])
    assert graph.shape==(len(d),32) and np.isfinite(graph).all() and np.isfinite(own).all()
    d.to_parquet(OUT/'observations.parquet',index=False)
    for name,array in [('own_features',own),('geometry_features',geometry),('graph_features',graph),('fixed_projection',fixed)]:np.save(OUT/f'{name}.npy',array)
    dump(OUT/'constraints.json',rules)
    pd.DataFrame(edge_audit).to_csv(OUT/'additional_neighbor_support.csv',index=False)
    dump(OUT/'history_audit.json',history_audit)
    # Core self forecasts are contract-local daily fits, not old pooled selections.
    newfolder=OUT/'new_contract_self';newfolder.mkdir(exist_ok=True)
    extra.to_parquet(newfolder/'observations.parquet',index=False);np.save(newfolder/'own_features.npy',ownnew)
    import expanded_guarded_fit as fitter
    old_histories,old_metadata=fitter.histories,fitter.metadata
    try:
        fitter.histories=lambda folder=None:hist
        fitter.metadata=lambda folder=None:({},meta,{})
        fitter.baseline(OUT,'new_contract_self',H)
    finally:fitter.histories, fitter.metadata=old_histories,old_metadata
    audits=pd.read_parquet(OLD/'self_fit_audit_1_5_15_30_60_180_360_1440.parquet')
    audits=audits[audits.horizon.isin(H)].copy();audits['fit_source']='reused_contract_local_grid'
    add_audit=pd.read_parquet(newfolder/'self_fit_audit_5_15.parquet');add_audit['fit_source']='new_contract_local_grid'
    audits=pd.concat([audits,add_audit],ignore_index=True)
    assert (audits.max_label_query<audits.day).all() and (audits.day<=audits.first_prediction).all()
    audits.to_parquet(OUT/'self_fit_audit.parquet',index=False)
    for h in H:
        a=np.load(OLD/f'self_{h}.npz');b=np.load(newfolder/f'self_{h}.npz')
        pp=np.vstack([a['forecasts'],b['forecasts']]);cc=np.concatenate([a['counts'],b['counts']])
        assert pp.shape==(len(d),5) and np.isfinite(pp).all()
        np.testing.assert_allclose(pp[cc<50],np.repeat(d.raw.to_numpy()[cc<50,None],5,axis=1),atol=0,rtol=0)
        np.savez_compressed(OUT/f'self_{h}.npz',forecasts=pp,counts=cc)
    dump(OUT/'PREPARATION.json',dict(rows=len(d),events=int(d.event_id.nunique()),contracts_with_origins=int(d.market_id.nunique()),
        core_rows=nold,additional_rows=len(extra),causal_own_features=True,causal_neighbor_features=True,
        additional_event_edges=74,additional_directed_edges_with_support=sum(r['rows']>0 for r in edge_audit),additional_rows_with_neighbor=int((count>0).sum()),
        current_price_replay_max_diff=current_replay,future_price_replay_max_diff=future_replay,
        fixed_projection_replay_max_diff=float(np.max(abs(replay-fixed))),projection_residual=projector.residual(fixed),
        self_fit_time_audit_passed=True,full_v2_monthly_selection_pending=True,seconds=time.perf_counter()-started))
    print('PREPARATION COMPLETE',round(time.perf_counter()-started,1),'seconds',flush=True)

def train(h):
    if (OUT/f'TRAINING_{h}.json').exists():return
    started=time.perf_counter();d=pd.read_parquet(OUT/'observations.parquet')
    x=np.load(OUT/'graph_features.npy');fixed=np.load(OUT/'fixed_projection.npy')
    pack=np.load(OUT/f'self_{h}.npz');self_preds,counts=pack['forecasts'],pack['counts']
    raw,times,y=d.raw.to_numpy(),d.origin.to_numpy(),d[f'target_{h}'].to_numpy();valid=d[f'complete_{h}'].to_numpy()
    projector=Projector(d,json.loads((OUT/'constraints.json').read_text(encoding='utf-8')),global_constraints=True)
    selected_self=raw.copy();graph=fixed.copy();self_j=np.full(len(d),-1,int);graph_deployed=np.zeros(len(d),bool)
    selections=[];self_selections=[];candidates=[];models=[];covered=np.zeros(len(d),int)
    for month,boundary,train_mask,val_context,test_context in month_windows(times,h):
        train_mask=train_mask&valid;val=val_context&valid;test=test_context&valid
        if not test.any():continue
        covered[test]+=1
        j=PENALTIES.index(10.);losses=[]
        if val.sum()>=50:
            wval=weights(d.loc[val]);losses=[float(wval@((self_preds[val,k]-y[val])**2)) for k in range(len(PENALTIES))]
            j=max(k for k,loss in enumerate(losses) if loss<=min(losses)+1e-12)
        selected_self[test_context]=self_preds[test_context,j];self_j[test_context]=j
        self_selections.append(dict(month=month,horizon=h,penalty=PENALTIES[j],validation_rows=int(val.sum()),validation_mse_grid=losses))
        info=dict(month=month,horizon=h,train_rows=int(train_mask.sum()),validation_rows=int(val.sum()),test_rows=int(test.sum()),
            max_train_label=int((times[train_mask]+60*h).max()) if train_mask.any() else None,
            max_validation_label=int((times[val]+60*h).max()) if val.any() else None,
            month_start=int(boundary),validation_start=int(boundary-30*86400),method=FULL)
        assert not train_mask.any() or info['max_train_label']<info['validation_start']
        assert not val.any() or info['max_validation_label']<boundary
        if train_mask.sum()<50 or val.sum()<50:
            selections.append(dict(**info,alpha=0.,penalty=None,status='insufficient_history_fixed_projection'))
            continue
        wtrain,wval=weights(d.loc[train_mask]),weights(d.loc[val])
        ref_mse=float(wval@((fixed[val]-y[val])**2));ref_mae=float(wval@abs(fixed[val]-y[val]))
        local=[];fitted={}
        for penalty in PENALTIES:
            model=fit_ridge(x[train_mask],(y-raw)[train_mask],wtrain,penalty);fitted[penalty]=model
            delta=np.zeros(len(d));delta[val_context]=predict_ridge(model,x[val_context])
            for alpha in ALPHAS:
                if alpha==0:mse,mae=ref_mse,ref_mae
                else:
                    pred=projector.apply(np.clip(raw+alpha*delta,0,1),val);err=pred[val]-y[val]
                    mse,mae=float(wval@(err**2)),float(wval@abs(err))
                local.append(dict(**info,penalty=penalty,alpha=alpha,mse=mse,mae=mae,reference_mse=ref_mse,reference_mae=ref_mae,
                    guard_mse=bool(mse<=ref_mse*1.01),guard_mae=bool(mae<=ref_mae*1.01)))
        candidates.extend(local);chosen=select_candidate(local);assert chosen is not None
        penalty,alpha=chosen['penalty'],chosen['alpha'];delta=np.zeros(len(d))
        delta[test_context]=predict_ridge(fitted[penalty],x[test_context])
        pred=projector.apply(np.clip(raw+alpha*delta,0,1),test)
        graph[test_context]=pred[test_context];graph_deployed[test_context]=alpha>0
        selections.append(dict(chosen,status='validation_selected'))
        mean,sd,center,beta=fitted[penalty]
        models.append(dict(**info,penalty=penalty,alpha=alpha,model=dict(mean=mean.tolist(),sd=sd.tolist(),center=center,beta=beta.tolist())))
        print('Graph fit',h,month,'train/val/test',int(train_mask.sum()),int(val.sum()),int(test.sum()),'alpha',alpha,'penalty',penalty,flush=True)
    assert (covered[valid]==1).all() and (self_j[valid]>=0).all()
    result=d[['vid','component','event_component','event_id','market_id','origin','raw','event_group','topic','source','bootstrap_cluster']].copy()
    result['target']=y;result['scored']=valid;result['self_train_n']=counts;result['self_penalty_index']=self_j;result['graph_nonzero_alpha']=graph_deployed
    pred={'Market Persistence':raw,'Self-Ridge':selected_self,'Fixed Projection':fixed,FULL:graph}
    for name,values in pred.items():
        assert np.isfinite(values).all() and ((values>=0)&(values<=1)).all();result[name]=values
    for name in ['Fixed Projection',FULL]:assert projector.residual(pred[name],valid)['max_residual']<=1e-8
    result.to_parquet(OUT/f'predictions_{h}.parquet',index=False)
    pd.DataFrame(selections).to_csv(OUT/f'selections_{h}.csv',index=False)
    pd.DataFrame(candidates).to_csv(OUT/f'candidates_{h}.csv',index=False)
    dump(OUT/f'self_selections_{h}.json',self_selections);dump(OUT/f'models_{h}.json',models)
    dump(OUT/f'TRAINING_{h}.json',dict(horizon=h,scored_rows=int(valid.sum()),scored_events=int(d.loc[valid,'event_id'].nunique()),
        exact_shared_mask=True,monthly_full_v2_graph_fit=True,monthly_full_v2_self_penalty_selection=True,causal_time_boundaries_passed=True,
        self_fallback_scored_rows=int((valid&(counts<50)).sum()),self_fitted_scored_rows=int((valid&(counts>=50)).sum()),
        graph_nonzero_alpha_scored_rows=int((valid&graph_deployed).sum()),graph_changed_from_fixed_scored_rows=int((valid&(abs(graph-fixed)>1e-10)).sum()),
        graph_projection_audit=projector.residual(graph,valid),predictions_sha256=sha(OUT/f'predictions_{h}.parquet'),seconds=time.perf_counter()-started))
    print('HORIZON COMPLETE',h,round(time.perf_counter()-started,1),'seconds',flush=True)

def summarize():
    rows=[];intervals=[];bygroup=[]
    original=pd.read_csv(C/'metrics.csv').set_index('horizon')
    for h in H:
        d=pd.read_parquet(OUT/f'predictions_{h}.parquet');source=pd.read_parquet(C/f'predictions_{h}.parquet')
        for col in ['vid','component','event_id','market_id','origin','raw','target','scored','Market Persistence','Fixed Projection']:
            pd.testing.assert_series_equal(d[col],source[col],check_dtype=False)
        z=d[d.scored].copy();y=z.target.to_numpy();values={m:score(z,y,z[m].to_numpy()) for m in METHODS}
        assert abs(values['Fixed Projection']['mse']-original.loc[h,'mse'])<1e-12
        assert abs(values['Market Persistence']['mse']-original.loc[h,'baseline_mse'])<1e-12
        for method in METHODS:
            r=values[method];rows.append(dict(horizon=h,method=method,**r,
                mse_gain_vs_persistence_pct=100*(1-r['mse']/values['Market Persistence']['mse']),
                mse_gain_vs_self_ridge_pct=100*(1-r['mse']/values['Self-Ridge']['mse'])))
        for a,b in [(FULL,'Market Persistence'),(FULL,'Fixed Projection'),(FULL,'Self-Ridge')]:
            intervals.append(dict(horizon=h,method=a,baseline=b,**paired(z,y,z[a].to_numpy(),z[b].to_numpy())))
        w=weights(z)
        for method in ['Fixed Projection',FULL]:
            g=z[['event_group']].copy();g['w']=w;g['a']=w*(z[method].to_numpy()-y)**2;g['b']=w*(z['Market Persistence'].to_numpy()-y)**2
            g=g.groupby('event_group').sum();delta=(g.a-g.b)/g.w
            g['status']=np.select([delta<-1e-12,delta>1e-12],['positive','negative'],default='flat')
            g['mse_gain_pct']=100*(1-g.a/g.b);g['horizon']=h;g['method']=method;bygroup.append(g.reset_index())
    table=pd.DataFrame(rows);table.to_csv(OUT/'metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False,encoding='utf-8-sig');pd.concat(bygroup).to_csv(OUT/'group_metrics.csv',index=False,encoding='utf-8-sig')
    saved=json.loads((OUT/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'))
    assert all(sha(Path(p))==digest for p,digest in saved['source_sha256'].items())
    audit=dict(status='complete',source_inputs_unchanged=True,counts=[374,550,5609,47],same_rows_targets_masks_weights=True,
        model_selection_before_scoring=True,independent_test=False,model_columns_complete=True,
        horizons={str(h):json.loads((OUT/f'TRAINING_{h}.json').read_text(encoding='utf-8')) for h in H})
    dump(OUT/'COMPARISON_AUDIT.json',audit)
    print(table[['horizon','method','mse','mae','mse_gain_vs_persistence_pct','mse_gain_vs_self_ridge_pct']].to_string(index=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','prepare','train','summary'],default='all');parser.add_argument('--horizons',type=int,nargs='*',default=H);args=parser.parse_args()
    with threadpool_limits(limits=1):
        lock()
        if args.stage in ['all','prepare']:prepare()
        if args.stage in ['all','train']:
            for h in args.horizons:train(h)
        if args.stage in ['all','summary']:summarize()

if __name__=='__main__':main()

"""Eight-horizon full-v2 comparison, without changing the frozen data or 5/15 results."""
from pathlib import Path
import sys,json,shutil,tempfile,time
sys.path.insert(0,str(Path.home()/'.cache/polymarket-experiment-deps'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import train_candidate374_models as previous
import expanded_guarded_fit as fitter
from expanded_guarded_core import sha
from evaluate_final288_multihorizon import dump

ROOT=Path(__file__).resolve().parents[1]
C=previous.C;F=C/'four_method_comparison';OLD=previous.OLD
OUT=C/'multihorizon_report_20261005'
H=[1,5,15,30,60,180,360,1440]
NEW=[1,30,60,180,360,1440]

def prepare():
    OUT.mkdir(exist_ok=True)
    sources=[F/'observations.parquet',F/'own_features.npy',F/'geometry_features.npy',F/'graph_features.npy',F/'constraints.json',F/'fixed_projection.npy',
        F/'predictions_5.parquet',F/'predictions_15.parquet',C/'events.csv',C/'event_edges.csv',C/'contract_catalog.csv',C/'metrics.csv',
        *[C/f'predictions_{h}.parquet' for h in H],*[OLD/f'self_{h}.npz' for h in H],OLD/'self_fit_audit_1_5_15_30_60_180_360_1440.parquet']
    previous.H=H
    protocol=previous.protocol();protocol['fit_scope']='Eight requested horizons; frozen 5/15 artifacts reused exactly, six other horizons freshly fit using the same protocol'
    protocol['source_anomalies']='125 raw source read failures and 10 inherited timing anomalies disclosed; source history and prior models remain unchanged; possible training-context effects not tested'
    protocol['secondary_analyses']='Self-Ridge+Projection; shrinkage .1/.25/.5; vid-mean smoothing .1/.25/.5; own-five-minute-trend common support; terminal metrics; conflict bands; group distribution; common-eight-horizon and trained-only support'
    signature=dict(protocol=protocol,source_sha256={str(p):sha(p) for p in sources},entrypoint_sha256=sha(Path(__file__)))
    lock=OUT/'PROTOCOL_LOCK.json'
    if lock.exists():
        saved=json.loads(lock.read_text());assert saved['source_sha256']==signature['source_sha256'] and saved['protocol']==protocol
    else:dump(lock,dict(**signature,locked_at_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    if (OUT/'PREPARATION.json').exists():return
    # Existing 7+ GiB free is far above this job's bounded <0.5 GiB output. The
    # original fitter's 10 GiB safety reserve is respected by fitting in C: temp.
    assert shutil.disk_usage(OUT).free>2*1024**3
    d=pd.read_parquet(F/'observations.parquet')
    for h in H:
        p=pd.read_parquet(C/f'predictions_{h}.parquet')
        pd.testing.assert_frame_equal(p[['vid','component','market_id','origin']],d[['vid','component','market_id','origin']])
        d[f'target_{h}']=p.target;d[f'complete_{h}']=p.scored
    d.to_parquet(OUT/'observations.parquet',index=False)
    for name in ['own_features.npy','geometry_features.npy','graph_features.npy','fixed_projection.npy','constraints.json']:
        shutil.copyfile(F/name,OUT/name)
    for h in [5,15]:
        for name in [f'predictions_{h}.parquet',f'models_{h}.json',f'self_{h}.npz',f'selections_{h}.csv',f'candidates_{h}.csv',f'self_selections_{h}.json',f'TRAINING_{h}.json']:
            shutil.copyfile(F/name,OUT/name)
    extra=pd.read_parquet(C/'added_observations.parquet');hist,meta,history_audit=previous.new_histories(extra)
    scratch=Path(tempfile.gettempdir())/'polymarket_v2_multihorizon_20261005';folder=scratch/'added_contracts';folder.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(folder).free>10*1024**3
    extra.to_parquet(folder/'observations.parquet',index=False)
    shutil.copyfile(F/'new_contract_self/own_features.npy',folder/'own_features.npy')
    fitter.histories=lambda folder=None:hist;fitter.metadata=lambda folder=None:({},meta,{})
    fitter.baseline(scratch,'added_contracts',NEW)
    core=pd.read_parquet(OLD/'self_fit_audit_1_5_15_30_60_180_360_1440.parquet');core['fit_source']='reused_contract_local_grid'
    prior=pd.read_parquet(F/'self_fit_audit.parquet');prior=prior[prior.fit_source=='new_contract_local_grid']
    added=pd.read_parquet(folder/'self_fit_audit_1_30_60_180_360_1440.parquet');added['fit_source']='new_contract_local_grid'
    audit=pd.concat([core,prior,added],ignore_index=True)
    assert not audit.duplicated(['market_id','horizon','day']).any()
    assert (audit.max_label_query<audit.day).all()
    audit.to_parquet(OUT/'self_fit_audit.parquet',index=False)
    for h in NEW:
        a=np.load(OLD/f'self_{h}.npz');b=np.load(folder/f'self_{h}.npz')
        pred=np.vstack([a['forecasts'],b['forecasts']]);count=np.r_[a['counts'],b['counts']]
        assert pred.shape==(len(d),5)
        np.testing.assert_allclose(pred[count<50],np.repeat(d.raw.to_numpy()[count<50,None],5,axis=1),atol=0,rtol=0)
        np.savez_compressed(OUT/f'self_{h}.npz',forecasts=pred,counts=count)
    dump(OUT/'PREPARATION.json',dict(rows=len(d),horizons=H,source_rows_unchanged=True,features_unchanged=True,
        existing_5_15_predictions_unchanged=True,daily_contract_fit_scratch=str(folder),history_audit=history_audit))

def main():
    with threadpool_limits(limits=1):
        prepare();previous.OUT=OUT;previous.H=H
        for h in NEW:previous.train(h)
        previous.summarize()
        for h in [5,15]:assert sha(OUT/f'predictions_{h}.parquet')==sha(F/f'predictions_{h}.parquet')

if __name__=='__main__':main()

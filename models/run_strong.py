"""Stage original AutoARIMA / LightGBM / Chronos-2 code in a new run folder."""
from pathlib import Path
import argparse,shutil,subprocess,sys
from rcf.verify import sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['AutoARIMA','LightGBM','Chronos-2'],required=True)
    ap.add_argument('--data',default='data');ap.add_argument('--output',required=True)
    ap.add_argument('--chronos-weights');ap.add_argument('--recover-arima',action='store_true')
    a=ap.parse_args();root=Path(__file__).resolve().parent;out=Path(a.output).resolve();d=Path(a.data).resolve()
    out.mkdir(parents=True,exist_ok=True);(out/'inputs').mkdir(exist_ok=True)
    sources={'answers.parquet':d/'core/answers.parquet','history.npz':d/'core/history.npz',
             'training_observations.parquet':d/'training/observations.parquet','training_history.npz':d/'training/history.npz'}
    for n,p in sources.items():
        dest=out/'inputs'/n
        if dest.exists():assert sha(dest)==sha(p)
        else:shutil.copy2(p,dest)
    for n in ['worker_pmd.py','numerics.py','repair_arima.py','repair_arima_v2.py','PROTOCOL_LOCK.json']:
        dest=out/('worker.py' if n=='worker_pmd.py' else n);src=root/'strong'/n
        if dest.exists():assert sha(dest)==sha(src)
        else:shutil.copy2(src,dest)
    if a.model=='Chronos-2':
        if not a.chronos_weights:raise ValueError('Provide pinned Chronos-2 snapshot with --chronos-weights (see MODEL_RUNS.md)')
        dst=out/'models/chronos-2';dst.parent.mkdir(exist_ok=True)
        if not dst.exists():shutil.copytree(Path(a.chronos_weights),dst)
    subprocess.run([sys.executable,str(out/('repair_arima_v2.py' if a.recover_arima else 'worker.py'))]+([] if a.recover_arima else [a.model]),check=True,cwd=out)
if __name__=='__main__':main()

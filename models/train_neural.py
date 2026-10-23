"""Explicitly start a fresh parent-pool neural training job. No auto-launch."""
from pathlib import Path
import argparse,os,json,shutil,subprocess

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['LSTM','DLinear','PatchTST','iTransformer'],required=True)
    ap.add_argument('--data',default='data');ap.add_argument('--output',required=True);ap.add_argument('--device',default='cpu')
    ap.add_argument('--months',nargs='+');ap.add_argument('--fetch-author-code',action='store_true')
    ap.add_argument('--prepare-only',action='store_true')
    a=ap.parse_args();root=Path(__file__).resolve().parents[1];out=Path(a.output).resolve();data=Path(a.data).resolve()
    os.environ['RCF_DATA']=str(data);os.environ['RCF_RUN']=str(out)
    from plan1_cross_model.protocol import load_config,write_json,digest,object_hash
    out.mkdir(parents=True,exist_ok=True);(out/'inputs').mkdir(exist_ok=True)
    target=out/'inputs/history.npz';source=data/'training/history.npz'
    if target.exists():assert digest(target)==digest(source)
    else:shutil.copy2(source,target)
    write_json(out/'inputs/INPUT_LOCK.json',{'history_sha256':digest(target)})
    pins=json.loads((root/'configs/author_sources.json').read_text());write_json(out/'author_sources.json',pins)
    if a.fetch_author_code and a.model!='LSTM':
        pin=pins[a.model];dest=root/'third_party'/a.model;dest.parent.mkdir(exist_ok=True)
        if not dest.exists():subprocess.run(['git','clone',pin['origin'],str(dest)],check=True)
        subprocess.run(['git','-C',str(dest),'checkout','--detach',pin['commit']],check=True)
    identity={'configuration':load_config(),'inputs':{str(p):digest(p) for p in [source,data/'training/observations.parquet']},'scope':'fresh execution on original 198449-row training pool; score retained303 by original row_id'}
    identity['protocol_hash']=object_hash(identity);lock=out/'PROTOCOL_LOCK.json'
    if lock.exists():assert json.loads(lock.read_text())==identity,'Different existing run; choose new --output'
    else:write_json(lock,identity)
    if a.prepare_only:
        print('Prepared historical training inputs and protocol:',out);return
    from plan1_cross_model.training import train
    train(a.model,months=a.months,device=a.device)
if __name__=='__main__':main()

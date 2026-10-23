"""Prepare and explicitly run one local NF4 model on the archived S3 cohort (Linux)."""
from pathlib import Path
import argparse,shutil,subprocess,sys,json,gzip,hashlib
from rcf.verify import sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['qwen1_8','qwen7','llama7','mistral7'],required=True)
    ap.add_argument('--data',default='data');ap.add_argument('--output',required=True)
    ap.add_argument('--weights',required=True,help='Local pinned snapshot, never a model-name alias')
    ap.add_argument('--batch-size',type=int,default=1);ap.add_argument('--memory-gib',type=float,default=40)
    ap.add_argument('--prepare-only',action='store_true')
    a=ap.parse_args();root=Path(__file__).resolve().parent;out=Path(a.output).resolve();d=Path(a.data).resolve()/'llm'
    if out.exists():raise FileExistsError('Use a new directory; invoke staged worker directly to resume')
    shutil.copytree(root/'llm_runtime',out);(out/'inputs').mkdir();(out/'results').mkdir();(out/'models').mkdir()
    for n in ['CONTEXT_LOCK.json','answers.parquet','systems.json','neighbor_messages.npz','history.npz']:
        shutil.copy2(d/n,out/'inputs'/n)
    with gzip.open(d/'requests.jsonl.gz','rb') as src,(out/'inputs/requests.jsonl').open('wb') as dst:shutil.copyfileobj(src,dst)
    for n in ['STRICT_FORMAT_AMENDMENT.json','QUANTIZATION_PROTOCOL.json']:shutil.copy2(d/n,out/n)
    (out/'REUSED_RESPONSES.json').write_text(json.dumps({'files':[],'note':'fresh strict rerun; archived heterogeneous Qwen1.8 responses are not overwritten'}))
    pins=json.loads((root.parent/'configs/llm_models.json').read_text());pin=pins[a.model]
    weights=Path(a.weights).resolve()
    # Match revision in Hugging Face cache, or an explicit snapshot receipt.
    if weights.name!=pin['revision']:
        receipt=json.loads((weights/'REVISION.json').read_text())
        assert receipt['revision']==pin['revision'] and receipt['repo_id']==pin['repo_id']
    receipt={'repo_id':pin['repo_id'],'revision':pin['revision'],'files':{p.relative_to(weights).as_posix():sha(p) for p in sorted(weights.rglob('*')) if p.is_file()}}
    (out/'models'/f'{a.model}.VERIFIED.json').write_text(json.dumps(receipt))
    import os
    os.environ['POLYMARKET_MODELS_DIR']=str(out/'models')
    (out/'models'/a.model).symlink_to(weights,target_is_directory=True)
    cap=int(a.memory_gib*2**30);alloc=json.dumps({'batch_size':a.batch_size,'cpu_threads':8,'memory_cap_bytes':cap,'max_memory':{'0':cap}})
    (out/'RUN_ARGUMENTS.json').write_text(json.dumps({'model':a.model,'allocation':json.loads(alloc)},indent=2))
    print('Prepared',out,flush=True)
    if not a.prepare_only:
        subprocess.run([sys.executable,str(out/'orchestration/infer_strict20.py'),'--model',a.model,'--allocation',alloc],check=True,cwd=out)
if __name__=='__main__':main()

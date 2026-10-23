"""Portable paths for the unchanged historical neural training loop."""
from pathlib import Path
import os,json,hashlib,shutil,importlib.util
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def object_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def load_config():
    c=json.loads((ROOT/'configs/neural.json').read_text())
    c['historical_report']=str(Path(os.environ.get('RCF_DATA',ROOT/'data')).resolve()/'training')
    c['output']=str(Path(os.environ.get('RCF_RUN',ROOT/'runs/neural')).resolve())
    c['author_sources']=str(Path(os.environ.get('RCF_AUTHOR_SOURCES',ROOT/'third_party')).resolve())
    return c
def output_dir(config=None):return Path((config or load_config())['output'])
def disk_guard(path,config=None):
    p=Path(path)
    while not p.exists():p=p.parent
    if shutil.disk_usage(p).free<(config or load_config())['disk_guard_gib']*2**30:raise RuntimeError('Less than 10 GiB available')
def read_lock():
    p=output_dir()/'PROTOCOL_LOCK.json';x=json.loads(p.read_text())
    assert x['configuration']==load_config()
    for name,h in x['inputs'].items():assert digest(name)==h,name
    return x
def code_receipt():return {p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
def numerics():
    from rcf import numerics
    return numerics

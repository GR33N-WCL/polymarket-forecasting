import sys,math
from pathlib import Path
from functools import lru_cache
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from common import *
from relaxed_parser import extract
@lru_cache(maxsize=1)
def context():return read(ROOT/'inputs/CONTEXT_LOCK.json')
@lru_cache(maxsize=1)
def reused():return read(ROOT/'REUSED_RESPONSES.json')['files']
def eligible_requests():return requests()
def check_record(x,r,key):
    legacy=key=='qwen1_8' and r['id']+'.json' in reused()
    expected=digest(legacy_messages(r) if legacy else messages(r))
    assert x['request_id']==r['id'] and x['model']==key
    assert x['input_hash']==expected and x['row_ids']==r['row_ids']
    assert x['revision']==context()['model_revisions'][key]
    assert x['precision']=='NF4' and x['compute_dtype']=='float16' and x['double_quant']
    p=x['predictions'][0];assert len(p)==8
    assert all(v is None or type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in p)
    n=sum(v is not None for v in p)
    assert x['status']==('ok' if n==8 else 'partial' if n else 'parse_failed')
    parsed,_=extract(x['attempts'][-1]['body']);assert parsed==p
    return n
def recover_model(key):return {}

from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def verify(data):
    data=Path(data);manifest=json.loads((data/'MANIFEST.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        assert sha(data/name)==expected,name
    f=pd.read_parquet(data/'core/answers.parquet');assert len(f)==183077 and f.row_id.is_unique
    assert f.event_id.nunique()==303 and f.market_id.nunique()==4617 and f.event_group.nunique()==43
    ids=f.row_id.to_numpy()
    for name in ['history','neighbor_messages']:
        z=np.load(data/'core'/f'{name}.npz',allow_pickle=False);assert np.array_equal(z['row_id'],ids)
    graph=json.loads((data/'core/event_graph.json').read_text());assert len(graph['nodes'])==303 and len(graph['edges'])==345
    print(json.dumps(dict(status='passed',files=len(manifest['files']),rows=len(f),events=303,contracts=4617,groups=43)))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data');a=ap.parse_args();verify(a.data)

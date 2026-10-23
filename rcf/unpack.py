from pathlib import Path,PurePosixPath
import argparse,tarfile,json
from .verify import sha
def main():
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--output',default='.')
    a=p.parse_args();archive=Path(a.archive);root=Path(a.output).resolve()
    manifest=json.loads((Path(__file__).parents[1]/'configs/release.json').read_text())
    assert sha(archive)==manifest['sha256'],'Archive checksum mismatch'
    with tarfile.open(archive,'r:gz') as t:
        for m in t.getmembers():
            rel=PurePosixPath(m.name)
            if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0]!='data' or not m.isfile():
                raise ValueError('Unexpected archive member: '+m.name)
            dst=(root/Path(*rel.parts)).resolve()
            if not dst.is_relative_to(root) or dst.exists():raise FileExistsError(str(dst))
        for m in t.getmembers():
            dst=root/Path(*PurePosixPath(m.name).parts);dst.parent.mkdir(parents=True,exist_ok=True)
            with t.extractfile(m) as src,dst.open('xb') as out:
                import shutil
                shutil.copyfileobj(src,out)
    print('Extracted verified archive to',root)
if __name__=='__main__':main()

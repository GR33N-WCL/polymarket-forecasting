"""Extract identifiable forecasts; formatting mistakes are not prediction failures."""
import json,math,re
H=[1,5,15,30,60,180,360,1440]
def value(x):
    if type(x) in (int,float):return float(x) if math.isfinite(x) and 0<=x<=1 else None
    if isinstance(x,str):
        s=x.strip()
        try:
            z=float(s[:-1])/100 if s.endswith('%') else float(s)
            return z if math.isfinite(z) and 0<=z<=1 else None
        except ValueError:return None
    return None
def horizon(k):
    s=str(k).strip().lower()
    m=re.fullmatch(r'(?:horizon[_ ]*)?(\d+)\s*(?:m|min|mins|minute|minutes|分钟)?',s)
    return int(m[1]) if m and int(m[1]) in H else None
def is_horizon_list(a):
    try:return [float(t) for t in a]==[float(h) for h in H]
    except (ValueError,TypeError):return False

def extract(body):
    candidates=[]
    def add(a,method):
        if len(a)==8 and any(x is not None for x in a):candidates.append((a,method))
    def visit(x,method):
        if isinstance(x,list):
            if len(x)==8 and all(not isinstance(t,(list,dict)) for t in x):
                if not is_horizon_list(x):add([value(t) for t in x],method)
            elif len(x)==1:visit(x[0],method)
        if isinstance(x,dict):
            a=[None]*8
            for k,v in x.items():
                h=horizon(k)
                if h is not None:a[H.index(h)]=value(v)
            add(a,method+'_keyed')
            for k in ['predictions','prices','forecast','forecasts','prediction']:
                if k in x:visit(x[k],method+'_named')
    text=body.strip();decoder=json.JSONDecoder()
    for i,c in enumerate(text):
        if c not in '[{':continue
        try:x,_=decoder.raw_decode(text[i:]);visit(x,'json_extracted')
        except (ValueError,TypeError):pass
    a=[None]*8;conflicts=set()
    pattern=r'(?<![\d.])(1440|360|180|60|30|15|5|1)\s*(?:minutes?|mins?|min|m|分钟)\s*[:=：|\-]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?)'
    for match in re.finditer(pattern,text,re.I):
        j=H.index(int(match[1]));v=value(match[2])
        if a[j] is not None and v is not None and a[j]!=v:conflicts.add(j)
        else:a[j]=v
    for j in conflicts:a[j]=None
    add(a,'explicit_horizon_text')
    # A stand-alone comma/semicolon separated line can identify eight ordered prices.
    for line in text.splitlines():
        if re.fullmatch(r'\s*[\d.eE+%\-]+\s*(?:[,;，]\s*[\d.eE+%\-]+\s*){7}',line):
            parts=re.split(r'[,;，]',line)
            if not is_horizon_list(parts):add([value(t) for t in parts],'standalone_numeric_line')
    if not candidates:return [None]*8,'unextractable'
    # Prefer a unique complete output; never choose between contradictory complete answers.
    n=max(sum(v is not None for v in x) for x,_ in candidates)
    best={tuple(x):method for x,method in candidates if sum(v is not None for v in x)==n}
    if len(best)!=1:return [None]*8,'ambiguous_outputs'
    x,method=next(iter(best.items()));return list(x),method

def tests():
    a=[.1,.2,.3,.4,.5,.6,.7,.8]
    assert extract('Here is my prediction:\n```json\n'+json.dumps(a)+'\n```')[0]==a
    assert extract('0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8')[0]==a
    assert extract('{"5min":"20%","15min":0.3}')[0]==[None,.2,.3,None,None,None,None,None]
    assert extract('[0.1,2,0.3,null,0.5,0.6,0.7,0.8]')[0]==[.1,None,.3,None,.5,.6,.7,.8]
    assert extract('[0.1,0.2,0.3]')[0]==[None]*8
    assert extract(str(H))[0]==[None]*8
    assert extract(','.join(map(str,H)))[0]==[None]*8
    assert extract(json.dumps(list(map(str,H))))[0]==[None]*8
    assert extract('[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8] [0,0,0,0,0,0,0,0]')[0]==[None]*8
    assert extract('5 minutes: 0.2\n30 minutes: 0.4')[0]==[None,.2,None,.4,None,None,None,None]
    assert extract('5 minutes: 0.2\n5 minutes: 0.9')[0]==[None]*8
if __name__=='__main__':tests();print('Parser tests passed')

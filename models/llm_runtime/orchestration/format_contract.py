"""Strict prediction shape, unambiguous format recovery and numeric grammar."""
import json,math,re
from functools import lru_cache

NUMBER=r'(0(\.[0-9]{1,12})?|1(\.0{1,12})?)'
ROW=r'\['+','.join([NUMBER]*8)+r'\]'

def grammar(n,mode):
    if mode=='individual':
        assert n==1
        return ' ?'+ROW
    assert n>=1
    return r' ?\['+','.join([ROW]*n)+r'\]'

def validate(value,n,mode):
    matrix=[value] if mode=='individual' else value
    if not isinstance(matrix,list) or len(matrix)!=n:raise ValueError('wrong contract count')
    for row in matrix:
        if not isinstance(row,list) or len(row)!=8:raise ValueError('wrong horizon count')
        if any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in row):raise ValueError('invalid price')
    return matrix

def recover(body,n,mode):
    text=body.strip();fence=re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```',text,re.I)
    if fence:text=fence.group(1).strip()
    try:return validate(json.loads(text),n,mode),'strict_json'
    except (ValueError,TypeError):pass
    decoder=json.JSONDecoder();candidates=[]
    for pos,char in enumerate(text):
        if char!='[':continue
        try:
            value,_=decoder.raw_decode(text[pos:]);candidates.append(validate(value,n,mode))
        except (ValueError,TypeError):continue
    if len(candidates)!=1:raise ValueError('no unique complete schema-valid price array')
    return candidates[0],'unique_array_with_surrounding_text'

def tokenizer_data(tokenizer,key):
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    if key.startswith('qwen'):
        # An explicit existing ChatML EOS is used only by the grammar builder.
        # No tokenizer file is saved or modified.
        tokenizer.eos_token_id=tokenizer.im_end_id
    return build_token_enforcer_tokenizer_data(tokenizer)

@lru_cache(maxsize=128)
def regex_parser(n,mode):
    from lmformatenforcer import RegexParser
    return RegexParser(grammar(n,mode))

def prefix_function(data,n,mode):
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    return build_transformers_prefix_allowed_tokens_fn(data,regex_parser(n,mode))

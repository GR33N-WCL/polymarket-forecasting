import os,sys,time,json,argparse,fcntl,traceback
from collections import deque
from itertools import islice
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from common import *
from acceptance_store import check_record
from relaxed_parser import extract
from format_contract import tokenizer_data,prefix_function
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--allocation',required=True);args=ap.parse_args();key=args.model
    lock=open(ROOT/'results'/(key+'.worker.lock'),'a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
    ctx=read(ROOT/'inputs/CONTEXT_LOCK.json');rr=requests();rr.sort(key=lambda r:(ctx['token_counts'][key][r['id']],r['id']))
    done=full=partial=failed=valid_values=format_failures=0;pending=[];batch_size=json.loads(args.allocation)['batch_size'];start=time.time();timed_requests=0;timed_seconds=0.
    for r in rr:
        p=ROOT/'results'/key/'responses'/(r['id']+'.json')
        if p.exists():
            n=check_record(read(p),r,key);assert n==8;done+=1;full+=int(n==8);partial+=int(0<n<8);failed+=int(n==0);valid_values+=n
        else:pending.append(r)
    def status(stage='running',current=None):
        save(ROOT/'results'/(key+'.status.json'),dict(stage=stage,pid=os.getpid(),request_total=len(rr),attempted_requests=done,successful_requests=full,partial_requests=partial,parse_failed_requests=failed,pending_requests=len(rr)-done,valid_horizon_values=valid_values,format_failed_attempts=format_failures,completion_requires_full_eight=True,fallback_requests=0,batch_size=batch_size,current=current,time=time.time(),elapsed_seconds=time.time()-start,timed_requests=timed_requests,timed_generation_seconds=timed_seconds,successful_requests_per_second=timed_requests/timed_seconds if timed_seconds else None,reused_requests=len(read(ROOT/'REUSED_RESPONSES.json')['files']) if key=='qwen1_8' else 0))
    if not pending:status('complete');return
    import torch
    from transformers import AutoModelForCausalLM,GenerationConfig,BitsAndBytesConfig
    alloc=json.loads(args.allocation);torch.set_num_threads(alloc.get('cpu_threads',8));torch.manual_seed(13)
    torch.cuda.set_per_process_memory_fraction(alloc['memory_cap_bytes']/torch.cuda.get_device_properties(0).total_memory,0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert (ROOT/'models'/(key+'.VERIFIED.json')).exists()
    tok=load_tok(key);memory={int(k) if k!='cpu' else k:int(v) for k,v in alloc['max_memory'].items()}
    kw=dict(torch_dtype=torch.float16,device_map={'':0},low_cpu_mem_usage=True,local_files_only=True,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
    if key.startswith('qwen'):kw.update(trust_remote_code=True,fp16=True,bf16=False,fp32=False,use_flash_attn=False)
    else:kw.update(attn_implementation='sdpa')
    status('loading')
    model=AutoModelForCausalLM.from_pretrained(Path(os.environ.get('POLYMARKET_MODELS_DIR',str(ROOT/'models')))/key,**kw).eval()
    if key=='qwen1_8':
        from qwen_mask_compat import install
        assert read(ROOT/'orchestration/QWEN18_PADDED_CHECK.json')['status']=='passed';install(model)
        # NF4 Accelerate binds a per-instance _old_forward before the class fix.
        import types
        for block in model.transformer.h:
            if hasattr(block.attn,'_old_forward'):block.attn._old_forward=types.MethodType(type(block.attn).forward,block.attn)
    import bitsandbytes as bnb
    quant_layers=[name for name,m in model.named_modules() if isinstance(m,bnb.nn.Linear4bit)]
    assert quant_layers and getattr(model,'is_loaded_in_4bit',False),'Actual NF4 layers required'
    assert not any(p.device.type in ['cpu','meta'] for p in model.parameters()),'CPU/disk offload prohibited'
    for name,m in model.named_modules():
        if isinstance(m,bnb.nn.Linear4bit):
            assert m.compute_dtype==torch.float16 and m.weight.quant_state.quant_type=='nf4'
    # In generation, only the final position's vocabulary logits are consumed.
    # Verify the cropped LM-head against full logits before using it; no token,
    # attention, position encoding, weight or decoding criterion is changed.
    first=pending[0];probe_ids=torch.tensor([encode(tok,key,messages(first))],device='cuda')
    with torch.inference_mode():before=model(input_ids=probe_ids,attention_mask=torch.ones_like(probe_ids),use_cache=False).logits[:,-1,:].float()
    hook=model.lm_head.register_forward_pre_hook(lambda module,args:(args[0][:,-1:,:],)+args[1:])
    with torch.inference_mode():after=model(input_ids=probe_ids,attention_mask=torch.ones_like(probe_ids),use_cache=False).logits[:,-1,:].float()
    rel=float((after-before).square().mean().sqrt()/before.square().mean().sqrt().clamp_min(1e-8))
    head_audit=dict(relative_rms=rel,max_abs=float((after-before).abs().max()),top1_equal=bool((after.argmax(-1)==before.argmax(-1)).all()))
    assert rel<.003 and head_audit['top1_equal'],'Last-position LM-head equivalence failed'
    del before,after,probe_ids;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    save(ROOT/'results'/(key+'.LOAD.json'),dict(device_map={k:str(v) for k,v in getattr(model,'hf_device_map',{'':'cuda:0'}).items()},precision='NF4',compute_dtype='float16',double_quant=True,quantized_linear_layers=len(quant_layers),lm_head_audit=head_audit,pid=os.getpid(),allocation=alloc,versions=dict(torch=torch.__version__,transformers=__import__('transformers').__version__,bitsandbytes=bnb.__version__),time=time.time()))
    data=tokenizer_data(tok,key)
    pad=tok.eod_id if key.startswith('qwen') else tok.eos_token_id
    eos=[tok.im_end_id,tok.eod_id] if key.startswith('qwen') else tok.eos_token_id
    gen=GenerationConfig(do_sample=False,num_beams=1,use_cache=True,pad_token_id=pad,eos_token_id=eos,bos_token_id=tok.bos_token_id,repetition_penalty=1.0)
    if key.startswith('qwen'):gen.chat_format='chatml';gen.max_window_size=4096
# Appended by build_runtime.py below the existing unchanged loading implementation.
    source_hash=digest({str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'code/common.py',ROOT/'orchestration/relaxed_parser.py',ROOT/'orchestration/format_contract.py']})
    fixed=dict(quantization_protocol_sha256=sha(ROOT/'QUANTIZATION_PROTOCOL.json'),weights_receipt_sha256=sha(ROOT/'models'/(key+'.VERIFIED.json')),protocol_hash=sha(ROOT/'inputs/CONTEXT_LOCK.json'),acceptance_protocol_hash=sha(ROOT/'STRICT_FORMAT_AMENDMENT.json'))
    pending=deque(pending)
    while pending:
        if (ROOT/'results'/(key+'.YIELD')).exists():status('yielded');return
        batch=list(islice(pending,0,batch_size));ms=[messages(r) for r in batch]
        seq=[encode(tok,key,m) for m in ms];width=max(map(len,seq));reserve=256
        assert width+reserve<=4096,'Input truncation prohibited'
        ids=torch.full((len(seq),width),pad,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
        for j,s in enumerate(seq):ids[j,-len(s):]=torch.tensor(s,device='cuda');mask[j,-len(s):]=1
        fn=prefix_function(data,1,'individual')
        status(current=[r['id'] for r in batch]);beg=time.time()
        try:
            with torch.inference_mode():answers=model.generate(input_ids=ids,attention_mask=mask,generation_config=gen,max_new_tokens=reserve,prefix_allowed_tokens_fn=fn)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size>1:batch_size=max(1,batch_size//2);status('running');continue
            raise
        elapsed=time.time()-beg;retry=[]
        for r,answer,m,s in zip(batch,answers,ms,seq):
            body=tok.decode(answer[width:],skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()
            pred,method=extract(body);n=sum(x is not None for x in pred)
            state='ok' if n==8 else 'partial' if n else 'parse_failed'
            x=dict(**fixed,precision='NF4',compute_dtype='float16',double_quant=True,request_id=r['id'],model=key,mode='individual',row_ids=r['row_ids'],bundle_id=r['bundle_id'],revision=ctx['model_revisions'][key],input_hash=digest(m),inference_source_hash=source_hash,input_tokens=len(s),reserved_output=reserve,saved_at=time.time(),attempts=[dict(body=body,kind='guided_numeric_generation',started_at=beg,finished_at=time.time())],status=state,predictions=[pred],acceptance_method=method,valid_horizons=[v is not None for v in pred],input_representation='lossless_rle_or_original_when_shorter',error=None if n==8 else 'Only identifiable valid horizons retained')
            if n!=8:
                attempt_dir=ROOT/'results'/key/'guided_attempts'/r['id'];attempt_dir.mkdir(parents=True,exist_ok=True)
                save(attempt_dir/(str(time.time_ns())+'.json'),x)
                format_failures+=1;retry.append(r);continue
            check_record(x,r,key)
            dest=ROOT/'results'/key/'responses'/(r['id']+'.json');assert not dest.exists();save(dest,x)
            if n:
                folder='accepted_responses' if n==8 else 'partial_responses';target=ROOT/'results'/key/folder/(r['id']+'.json');target.parent.mkdir(parents=True,exist_ok=True);os.link(dest,target)
            done+=1;full+=int(n==8);partial+=int(0<n<8);failed+=int(n==0);valid_values+=n
        for _ in batch:pending.popleft()
        pending.extend(retry)
        timed_requests+=len(batch)-len(retry);timed_seconds+=elapsed;status()
        if format_failures>=10:
            status('pending_retry');raise RuntimeError('Ten strict format failures; no failed response counted complete')
    assert done==len(rr);status('complete')

if __name__=='__main__':
    try:main()
    except Exception:traceback.print_exc();raise

"""Minimal runtime compatibility fix for the pinned 2023 Qwen cached attention.

No weight, tokenizer, configuration, prompt, or generation parameter is changed.
The number of query positions is read from query instead of a possibly absent
triangular mask. The padding mask is retained in both prefill and decode.
"""
import inspect
import textwrap


def install(model):
    cls = type(model.transformer.h[0].attn)
    original = cls.forward
    src = textwrap.dedent(inspect.getsource(original))
    old = '-1, -1, causal_mask.size(2), -1'
    assert src.count(old) == 1, 'Pinned Qwen implementation changed'
    src = src.replace(old, '-1, -1, query.size(-2), -1')
    scope = {}
    exec(compile(src, '<audited-qwen-mask-compat>', 'exec'), original.__globals__, scope)
    cls.forward = scope['forward']
    return original

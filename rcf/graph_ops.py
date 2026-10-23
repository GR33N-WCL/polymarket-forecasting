from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve




def normalized_rows(a, b):
    scale = np.maximum(np.abs(a).sum(axis=1), 1e-15)
    return a / scale[:, None], b / scale


def signature(a, b):
    a, b = normalized_rows(a, b)
    return tuple(sorted(set(tuple(np.round(np.r_[row, rhs], 12)) for row, rhs in zip(a, b))))


def canonical_relations(a, b):
    """Group the two opposite inequality rows of an equality as ONE relation."""
    an, bn = normalized_rows(a, b)
    used, relations = set(), []
    for i in range(len(b)):
        if i in used:
            continue
        rows = [i]
        used.add(i)
        for j in range(i + 1, len(b)):
            if j not in used and np.allclose(an[j], -an[i], atol=1e-12, rtol=0) and abs(bn[j] + bn[i]) <= 1e-12:
                rows.append(j)
                used.add(j)
                break
        relations.append(tuple(rows))
    return relations


def relation_residuals(values, a, b):
    an, bn = normalized_rows(a, b)
    residual = np.maximum(np.asarray(values) @ an.T - bn, 0)
    relations = canonical_relations(a, b)
    return np.stack([residual[..., list(rows)].max(axis=-1) for rows in relations], axis=-1)


def permuted_columns(a, b, seed):
    permutation = np.random.default_rng(seed).permutation(a.shape[1])
    changed = signature(a[:, permutation], b) != signature(a, b)
    return a[:, permutation], changed, permutation


def mask_relations(a, b, fraction, seed, targeted=False):
    relations = canonical_relations(a, b)
    count = len(relations)
    if count < 2:
        return a, b, np.zeros(count, bool), False
    rng = np.random.default_rng(seed)
    n = min(count - 1, max(1, int(np.ceil(fraction * count))))
    omitted = np.zeros(count, bool)
    keep = None
    supported = not targeted
    if targeted:
        supports = [set(np.flatnonzero(a[rows[0]])) for rows in relations]
        pairs = [(i, j) for i in range(count) for j in range(i + 1, count) if supports[i] & supports[j]]
        if not pairs:
            return a, b, omitted, False
        pair = pairs[rng.integers(len(pairs))]
        remove, keep = pair if rng.integers(2) else pair[::-1]
        omitted[remove] = True
        supported = True
    candidates = [i for i in range(count) if i != keep and not omitted[i]]
    rng.shuffle(candidates)
    omitted[candidates[:n - int(omitted.sum())]] = True
    kept_rows = [row for i, rows in enumerate(relations) if not omitted[i] for row in rows]
    return a[kept_rows], b[kept_rows], omitted, supported


def smooth_batch(z, a, b, strength):
    if not strength or not len(b):
        return np.array(z, copy=True)
    n = a.shape[1]
    laplacian = np.zeros((n, n))
    for rows in canonical_relations(a, b):
        members = np.flatnonzero(a[rows[0]])
        k = len(members)
        if k < 2:
            continue
        weight = 2 / (k * (k - 1))
        for ix, i in enumerate(members):
            for j in members[ix + 1:]:
                laplacian[i, i] += weight
                laplacian[j, j] += weight
                laplacian[i, j] -= weight
                laplacian[j, i] -= weight
    # Inverse of I+2 lambda L is nonnegative and row-stochastic: [0,1] is preserved.
    answer = cho_solve(cho_factor(np.eye(n) + 2 * strength * laplacian), np.asarray(z).T).T
    return np.clip(answer, 0, 1)



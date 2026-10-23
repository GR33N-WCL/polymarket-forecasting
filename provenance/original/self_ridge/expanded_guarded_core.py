"""Numerical primitives for the rolling, post-projection guarded experiment."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'papers/data_audit_tools'))
sys.path.insert(0, str(Path.home() / '.cache/polymarket-experiment-deps'))
import numpy as np
import pandas as pd
from audit_six_relations_frozen326_v1 import matrix, project, violation

HORIZONS = [5, 15, 30, 60, 180, 360]
PENALTIES = [.01, .1, 1., 10., 100.]
ALPHAS = [0., .01, .025, .05, .1, .2, .35, .5, .75, 1.]


def guard_disk(path):
    free = shutil.disk_usage(Path(path).anchor).free
    if free < 10 * 1024**3:
        raise OSError(f'Free space below 10 GiB: {free}; no source files deleted')


def dump(path, obj):
    path = Path(path)
    guard_disk(path)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_npz(path, **arrays):
    path = Path(path)
    guard_disk(path)
    temporary = path.with_name(path.stem+'.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def asof(history, query, max_age=60):
    """Freshness is measured from the real trade, availability from bucket end."""
    t, last, price = history
    query = np.asarray(query)
    ix = np.searchsorted(t, query, side='right') - 1
    safe = np.maximum(ix, 0)
    age = query - last[safe]
    valid = (ix >= 0) & (age >= 0) & (age <= max_age)
    return np.where(valid, price[safe], np.nan), np.where(valid, age, np.nan)


def own_features(history, query):
    value, age = asof(history, query, 300)
    columns = [np.nan_to_num(value), np.nan_to_num(age, nan=300) / 300]
    for minutes in (1, 5, 15, 60):
        old, _ = asof(history, np.asarray(query)-minutes*60, 300)
        columns.extend([np.nan_to_num(value-old), np.isfinite(old).astype(float)])
    t = history[0]
    count = np.searchsorted(t, query, side='right') - np.searchsorted(t, np.asarray(query)-3600, side='right')
    return np.column_stack(columns+[np.log1p(count)])


def weights(frame):
    """Original component identities are fixed, regardless of added graph edges."""
    sizes = frame.groupby('vid').market_id.transform('size').to_numpy()
    counts = frame[['component', 'vid']].drop_duplicates().groupby('component').size()
    w = 1. / sizes / frame.component.map(counts).to_numpy()
    return w / w.sum()


def score(frame, y, pred):
    if not np.isfinite(y).all() or not np.isfinite(pred).all() or len(frame) == 0:
        raise ValueError('Scores require nonempty, finite targets and predictions')
    w = weights(frame)
    e = np.asarray(pred)-np.asarray(y)
    return dict(mse=float(w@(e*e)), mae=float(w@abs(e)), micro_mse=float(np.mean(e*e)),
                micro_mae=float(np.mean(abs(e))), observations=len(e), events=int(frame.event_id.nunique()),
                contracts=int(frame.market_id.nunique()), vectors=int(frame.vid.nunique()))


def fit_ridge(x, y, w, penalty):
    # Weighted sum objective with mean weight one, so the penalty is scale-stable.
    w = np.asarray(w) / np.mean(w)
    mean = np.average(x, axis=0, weights=w)
    sd = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    sd[sd < 1e-8] = 1.
    z = np.clip((x-mean)/sd, -10, 10)
    center = float(np.average(y, weights=w))
    beta = np.linalg.solve(z.T@(w[:, None]*z)+penalty*np.eye(z.shape[1]), z.T@(w*(y-center)))
    return mean, sd, center, beta


def predict_ridge(model, x):
    mean, sd, center, beta = model
    return np.clip(center+np.clip((x-mean)/sd, -10, 10)@beta, -.1, .1)


def month_windows(times, horizon):
    dates = pd.to_datetime(times, unit='s', utc=True)
    months = sorted(set(dates.strftime('%Y-%m')))
    for month in months:
        start = pd.Timestamp(month+'-01', tz='UTC')
        end = start + pd.offsets.MonthBegin(1)
        t, stop = int(start.timestamp()), int(end.timestamp())
        train = (times >= t-90*86400) & (times+horizon*60 < t-30*86400)
        val = (times >= t-30*86400) & (times+horizon*60 < t)
        test = (times >= t) & (times < stop)
        yield month, t, train, val, test


def project_batch(values, a, b, tol=1e-10, iterations=3000):
    """Euclidean projection by Dykstra; explicit error on infeasibility/nonconvergence."""
    original = np.asarray(values, dtype=float)
    if not np.isfinite(original).all() or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Projection input contains nonfinite values')
    x = np.clip(original, 0, 1).copy()
    if not len(a):
        return x
    bad = np.max(x@a.T-b, axis=1) > tol
    if not bad.any():
        return x
    z = original[bad].copy()
    corrections = np.zeros((len(a), len(z)))
    box = np.zeros_like(z)
    norm = np.sum(a*a, axis=1)
    if np.any(norm == 0):
        if np.any(b[norm == 0] < -tol):
            raise ValueError('Infeasible zero constraint')
        return project_batch(original, a[norm > 0], b[norm > 0], tol, iterations)
    converged = False
    for cycle in range(iterations):
        before = z.copy()
        for j, row in enumerate(a):
            y = z + corrections[j, :, None]*row
            step = np.maximum((y@row-b[j])/norm[j], 0)
            z = y - step[:, None]*row
            corrections[j] = step
        y = z+box
        z = np.clip(y, 0, 1)
        box = y-z
        if np.max(abs(z-before)) <= tol*.1 and np.max(z@a.T-b) <= tol:
            converged = True
            break
    if not converged:
        # Small pathological batches get an independent constrained solver, never a silent pass.
        for k in range(len(z)):
            z[k], ok = project(original[bad][k], a, b)
            if not ok or violation(z[k], a, b) > 1e-8:
                raise RuntimeError('Projection failed feasibility/convergence check')
    x[bad] = z
    if np.max(x@a.T-b) > 1e-8:
        raise RuntimeError('Post-projection residual exceeds 1e-8')
    return x


class Projector:
    """Batch identical constraint systems without changing vector/sample identities."""
    def __init__(self, data, rules, global_constraints=False, participation_mask=None):
        bymarket = defaultdict(set)
        for i, rule in enumerate(rules):
            for mid in rule['members']:
                bymarket[str(mid)].add(i)
        grouped = {}
        self.rule_witness = defaultdict(int)
        self.event_pairs = defaultdict(int)
        mids = data.market_id.to_numpy()
        times = data.origin.to_numpy()
        eids = data.event_id.to_numpy()
        self.global_constraints = global_constraints
        if global_constraints:
            parent = {m: m for m in set(mids)}
            def find(m):
                while parent[m] != m:
                    parent[m] = parent[parent[m]]
                    m = parent[m]
                return m
            for rule in rules:
                present = [m for m in rule['members'] if m in parent]
                for m in present[1:]:
                    parent[find(m)] = find(present[0])
            grouping = pd.DataFrame(dict(origin=times, root=[find(m) if bymarket[m] else None for m in mids])).groupby(['root', 'origin'], sort=False).indices
        else:
            grouping = data.groupby('vid', sort=False).indices
        for _, ids in grouping.items():
            if len(ids) < 2:
                continue
            local = set().union(*(bymarket[m] for m in mids[ids]))
            a, b, names = matrix([rules[i] for i in sorted(local)], mids[ids], int(times[ids[0]]))
            if not len(a):
                continue
            a, b = np.asarray(a, float), np.asarray(b, float)
            key = (len(ids), a.shape, a.tobytes(), b.tobytes())
            if key not in grouped:
                grouped[key] = [a, b, []]
            grouped[key][2].append(ids)
            for row, name in zip(a if participation_mask is None or participation_mask[ids].any() else [], names):
                self.rule_witness[name] += 1
                events = sorted(set(eids[ids][np.abs(row) > 0]))
                for j, event in enumerate(events):
                    for other in events[j+1:]:
                        self.event_pairs[(event, other)] += 1
        self.systems = [(a, b, np.asarray(ids)) for a, b, ids in grouped.values()]
        self.n = len(data)

    def apply(self, predictions, mask=None):
        if not np.isfinite(predictions).all():
            raise ValueError('Nonfinite forecast cannot be projected')
        output = np.clip(np.asarray(predictions, dtype=float), 0, 1).copy()
        for a, b, ids in self.systems:
            if mask is not None:
                present = mask[ids]
                if not self.global_constraints and np.any(present.any(axis=1) != present.all(axis=1)):
                    raise ValueError('Projection/scoring mask splits a joint vector')
                ids = ids[present.any(axis=1) if self.global_constraints else present.all(axis=1)]
            if len(ids):
                output[ids] = project_batch(output[ids], a, b)
        return output

    def residual(self, values, mask=None):
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite forecasts cannot count as consistent')
        maximum = 0.
        count = bad = 0
        for a, b, ids in self.systems:
            if mask is not None:
                ids = ids[mask[ids].any(axis=1) if self.global_constraints else mask[ids].all(axis=1)]
            if not len(ids):
                continue
            v = values[ids]
            r = np.maximum.reduce([np.max(v@a.T-b, axis=1), np.max(-v, axis=1), np.max(v-1, axis=1)])
            maximum = max(maximum, float(r.max()))
            count += len(ids)
            bad += int((r > 1e-8).sum())
        return dict(max_residual=maximum, constrained_vectors=count, violating_vectors=bad,
                    consistency_pct=100*(1-bad/count) if count else None)

    def context_mask(self, score_mask):
        result = np.asarray(score_mask).copy()
        for _, _, ids in self.systems:
            selected = ids[score_mask[ids].any(axis=1)]
            result[selected.ravel()] = True
        return result


def select_candidate(records):
    eligible = [r for r in records if r['guard_mse'] and r['guard_mae']]
    if not eligible:
        return None
    minimum = min(r['mse'] for r in eligible)
    return min((r for r in eligible if r['mse'] <= minimum+1e-12),
               key=lambda r: (r['alpha'], -r['penalty']))

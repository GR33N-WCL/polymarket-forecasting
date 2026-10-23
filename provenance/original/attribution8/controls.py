from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

METHODS = ('Shrink', 'Calibrate', 'Isotonic', 'Histogram', 'CurrentCombination')
REGS = (0., 1e-6, 1e-4, .01, 1., 100.)
SHRINK = (0., .1, .25, .5, .75, 1.)
BINS = (5, 10, 20)
PSEUDO_COUNTS = (0., 20., 100.)
MIN_ROWS = 20


def candidates(method):
    # Identity first, and strict improvement required: deterministic tie handling.
    result = [{'kind': 'identity'}]
    if method == 'Shrink':
        result += [{'kind': method, 'beta': b} for b in SHRINK if b]
    elif method in ('Calibrate', 'CurrentCombination'):
        result += [{'kind': method, 'regularization': r} for r in REGS]
    elif method == 'Isotonic':
        result += [{'kind': method}]
    elif method == 'Histogram':
        result += [{'kind': method, 'bins': n, 'pseudo_count': p}
                   for n in BINS for p in PSEUDO_COUNTS]
    else:
        raise ValueError(method)
    return result


def fit(spec, z, current, y, w):
    z, current, y, w = map(np.asarray, (z, current, y, w))
    w = w / w.sum()
    model = dict(spec)
    kind = spec['kind']
    if kind in ('identity', 'Shrink'):
        return model
    if kind in ('Calibrate', 'CurrentCombination'):
        design = np.column_stack((z, np.ones(len(z)))) if kind == 'Calibrate' else np.column_stack((z, current, np.ones(len(z))))
        center = np.zeros(design.shape[1]); center[0] = 1.
        r = spec['regularization']
        coef = np.linalg.lstsq(design.T @ (w[:, None] * design) + r*np.eye(len(center)),
                               design.T @ (w*y) + r*center, rcond=None)[0]
        model['coefficients'] = coef.tolist()
    elif kind == 'Isotonic':
        iso = IsotonicRegression(y_min=0., y_max=1., out_of_bounds='clip')
        iso.fit(z, y, sample_weight=w)
        model.update(x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist())
    elif kind == 'Histogram':
        edges = np.unique(np.quantile(z, np.linspace(0, 1, spec['bins']+1)[1:-1]))
        bins = np.searchsorted(edges, z, side='right')
        count = np.bincount(bins, minlength=len(edges)+1)
        mass = np.bincount(bins, weights=w*len(w), minlength=len(edges)+1)
        weighted = np.bincount(bins, weights=w*len(w)*y, minlength=len(edges)+1)
        means = np.divide(weighted, mass, out=np.zeros_like(mass), where=mass>0)
        model.update(edges=edges.tolist(), count=count.tolist(), mass=mass.tolist(), means=means.tolist())
    return model


def predict(model, z, current):
    z, current = np.asarray(z), np.asarray(current)
    kind = model['kind']
    if kind == 'identity':
        q = z.copy()
    elif kind == 'Shrink':
        q = (1-model['beta'])*z + model['beta']*current
    elif kind in ('Calibrate', 'CurrentCombination'):
        co = model['coefficients']
        q = co[0]*z + co[-1]
        if kind == 'CurrentCombination':
            q = q + co[1]*current
    elif kind == 'Isotonic':
        q = np.interp(z, model['x'], model['y'])
    elif kind == 'Histogram':
        ids = np.searchsorted(model['edges'], z, side='right')
        mass, means, count = [np.asarray(model[x])[ids] for x in ('mass', 'means', 'count')]
        pseudo = model['pseudo_count']
        q = np.divide(mass*means+pseudo*z, mass+pseudo, out=z.copy(), where=mass+pseudo>0)
        q[count < MIN_ROWS] = z[count < MIN_ROWS]
    else:
        raise ValueError(kind)
    return np.clip(q, 0., 1.)


def choose(method, fit_data, select_data):
    z, p, y, w = select_data
    records, winner = [], {'kind': 'identity'}
    best = float(w @ ((z-y)**2))
    for spec in candidates(method):
        fitted = fit(spec, *fit_data)
        score = float(w @ ((predict(fitted, z, p)-y)**2))
        records.append({'spec': spec, 'selection_mse': score})
        if score < best - 1e-15:
            winner, best = spec, score
    return winner, best, records

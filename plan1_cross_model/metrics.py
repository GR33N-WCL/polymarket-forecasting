from __future__ import annotations

import numpy as np
import pandas as pd

from .protocol import load_config


def weights(frame):
    if frame.empty:
        return np.empty(0)
    if frame.groupby("vid").component.nunique().max() != 1:
        raise ValueError("A vid crosses original scoring components")
    contracts = frame.groupby("vid").vid.transform("size").to_numpy()
    vids = frame[["vid", "component"]].drop_duplicates().groupby("component").size()
    w = 1 / contracts / frame.component.map(vids).to_numpy()
    return w / w.sum()


def directions(change):
    return np.where(np.abs(change) <= load_config()["direction_tolerance"], 0, np.sign(change))


def metric_vectors(raw, final, y, current, task="price"):
    result = {"MSE": ((raw-y)**2, (final-y)**2), "MAE": (np.abs(raw-y), np.abs(final-y))}
    if task == "terminal":
        return {"Brier": result["MSE"]}
    truth = directions(y-current)
    result["DG"] = ((directions(raw-current) == truth).astype(float),
                    (directions(final-current) == truth).astype(float))
    return result


def paired_scores(frame, raw, final, y, mask, task="price", bootstrap=True):
    chosen = np.asarray(mask) & np.isfinite(y)
    if not chosen.any():
        return []
    f = frame.loc[chosen]
    w = weights(f)
    a, b, target, current = [np.asarray(v)[chosen] for v in (raw, final, y, frame.raw)]
    groups, group_idx = np.unique(f.event_group, return_inverse=True)
    c = load_config()
    draws = None
    if bootstrap:
        draws = np.random.default_rng(c["bootstrap_seed"]).multinomial(len(groups), np.full(len(groups), 1/len(groups)), c["bootstrap_replicates"])
    result = []
    vectors = metric_vectors(a, b, target, current, task)
    if task == "price":
        moving = directions(target-current) != 0
        vectors["DG_moving"] = vectors["DG"]
    for metric, (vraw, vfinal) in vectors.items():
        ww = w.copy()
        if metric == "DG_moving":
            ww *= moving
            if ww.sum() == 0:
                continue
            ww /= ww.sum()
        mraw, mfinal = float(ww @ vraw), float(ww @ vfinal)
        direction = 1 if metric.startswith("DG") else -1
        gain = direction * (vfinal-vraw)
        lo = hi = None
        if draws is not None:
            mass = np.bincount(group_idx, weights=ww, minlength=len(groups))
            delta = np.bincount(group_idx, weights=ww*gain, minlength=len(groups))
            denom = draws @ mass
            samples = (draws[denom > 0] @ delta) / denom[denom > 0]
            lo, hi = map(float, np.quantile(samples, [.025, .975]))
        result.append({"metric": metric, "raw": mraw, "final": mfinal,
                       "gain": float(ww @ gain), "gain_ci_low": lo, "gain_ci_high": hi,
                       "relative_error_reduction_percent": (100*(mraw-mfinal)/mraw if mraw and not metric.startswith("DG") else None),
                       "direction_gain_percentage_points": 100*(mfinal-mraw) if metric.startswith("DG") else None,
                       "rows": len(f), "groups": len(groups), "components": f.component.nunique(),
                       "moving_row_fraction": float(moving.mean()) if task == "price" else None,
                       "moving_weight_fraction": float(w @ moving) if task == "price" else None})
    return result


def conflict_thresholds(system_records, month_start, selected_bundles=None):
    c = load_config()
    if system_records.empty:
        return tuple(c["conflict_fallback_thresholds"]), "insufficient_validation"
    chosen = system_records.origin.between(month_start - 30*86400, month_start, inclusive="left")
    if selected_bundles is not None:
        chosen &= system_records.bundle_id.isin(selected_bundles)
    positive = system_records.loc[chosen & (system_records.max_violation > c["solver_tolerance"]), "max_violation"]
    if len(positive) < c["minimum_conflict_validation_systems"]:
        return tuple(c["conflict_fallback_thresholds"]), "insufficient_validation"
    return tuple(np.quantile(positive, [1/3, 2/3])), "validation_tertiles"


def conflict_bands(values, thresholds):
    lo, hi = thresholds
    return np.where(values <= load_config()["solver_tolerance"], "none",
                    np.where(values <= lo, "low", np.where(values <= hi, "medium", "high")))

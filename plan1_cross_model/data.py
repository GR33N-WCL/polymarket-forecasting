from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd

from .protocol import ROOT, load_config, output_dir, object_hash, write_json, disk_guard, digest


def observations():
    c = load_config()
    frame = pd.read_parquet(ROOT / c["historical_report"] / "observations.parquet")
    frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    frame["month"] = pd.to_datetime(frame.origin, unit="s", utc=True).dt.strftime("%Y-%m")
    frame["bundle_id"] = frame.event_group.astype(str) + "@" + frame.origin.astype(str)
    return frame


def month_start(month):
    return int(pd.Timestamp(str(month) + "-01", tz="UTC").timestamp())


def time_masks(frame, month, horizons, task="price", labels=None):
    t = month_start(month)
    end = int((pd.Timestamp(str(month) + "-01", tz="UTC") + pd.offsets.MonthBegin()).timestamp())
    train_end = t - 30 * 86400
    bounds = [(t - 90 * 86400, train_end), (train_end, t), (t, end)]
    result = {}
    for name, (lo, hi) in zip(("train", "validation", "evaluation"), bounds):
        rows = frame.origin.between(lo, hi, inclusive="left").to_numpy()
        if task == "price":
            # Minute-history prices become observable at bucket end; +60 is conservative
            # when exact target availability was not exported with frozen observations.
            mask = np.column_stack([frame[f"complete_{h}"].to_numpy() &
                                    np.isfinite(frame[f"target_{h}"]) for h in horizons])
            if name != "evaluation":
                mask &= frame.origin.to_numpy()[:, None] + np.array(horizons)[None, :] * 60 + 60 < hi
        else:
            if labels is None:
                raise ValueError("Terminal labels must come from the evidence ledger")
            mask = labels.verified.to_numpy()[:, None] & np.isfinite(labels.label.to_numpy()[:, None])
            if name != "evaluation":
                mask &= (labels.resolution_available_at.to_numpy()[:, None] < hi)
                mask &= labels.time_verified.to_numpy()[:, None]
        result[name] = (rows, mask & rows[:, None])
    return result


def fair_allocate(capacities, budget):
    """Equal integer allocation; redistribute shortages, ordered ID tie breaking."""
    keys = sorted(capacities)
    allocated = dict.fromkeys(keys, 0)
    remaining = min(int(budget), sum(capacities.values()))
    while remaining:
        for key in keys:
            if allocated[key] < capacities[key]:
                allocated[key] += 1
                remaining -= 1
                if not remaining:
                    break
    return allocated


def sample_bundles(frame, cap, seed=13):
    bundles = frame[["bundle_id", "event_group", "month", "origin"]].drop_duplicates("bundle_id")
    group_capacity = bundles.groupby("event_group").size().to_dict()
    allocation = fair_allocate(group_capacity, cap)
    chosen = []
    for group, count in allocation.items():
        part = bundles[bundles.event_group == group]
        months = fair_allocate(part.groupby("month").size().to_dict(), count)
        for month, n in months.items():
            ids = part.loc[part.month == month, "bundle_id"].tolist()
            ids.sort(key=lambda s: (hashlib.sha256(f"{seed}:{s}".encode()).hexdigest(), s))
            chosen.extend(ids[:n])
    return sorted(chosen)


def freeze_panels(frame):
    c = load_config()
    out = output_dir() / "panels"
    out.mkdir(parents=True, exist_ok=True)
    evaluation = sample_bundles(frame, c["evaluation_bundle_cap"], c["sample_seed"])
    validation = {}
    for month in sorted(frame.month.unique()):
        t = month_start(month)
        past = frame[frame.origin.between(t - 30 * 86400, t, inclusive="left")]
        validation[month] = sample_bundles(past, c["validation_bundle_cap"], c["sample_seed"])
    record = {"evaluation_bundles": evaluation, "validation_bundles_by_month": validation,
              "selection": "graph group, then month, equal quotas; SHA256(seed:bundle_id)"}
    record["panel_hash"] = object_hash(record)
    lock = out / "PANEL_LOCK.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != record:
        raise RuntimeError("Frozen panel changed")
    if not lock.exists():
        write_json(lock, record)
    selected = frame[frame.bundle_id.isin(evaluation)]
    selected[["row_id", "bundle_id", "market_id", "origin", "vid", "component", "event_group", "month"]].to_parquet(out / "evaluation_rows.parquet", index=False)
    write_json(out / "coverage.json", {
        "bundles": len(evaluation), "rows": len(selected), "events": selected.event_id.nunique(),
        "contracts": selected.market_id.nunique(), "groups": selected.event_group.nunique(),
        "months": sorted(selected.month.unique()),
        "scored_rows": {str(h): int(selected[f"complete_{h}"].sum()) for h in c["horizons"]}})
    return record


def load_panels():
    return json.loads((output_dir() / "panels/PANEL_LOCK.json").read_text(encoding="utf-8"))


def terminal_labels(frame):
    """No inferred publication dates. Empty ledger is an explicit blocked task."""
    path = output_dir() / "terminal_label_evidence.csv"
    columns = ["market_id", "label", "verified", "evidence_source", "resolution_available_at",
               "time_verified", "time_evidence_source", "state"]
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    ledger = pd.read_csv(path, dtype={"market_id": str})
    if ledger.market_id.duplicated().any():
        raise ValueError("Duplicate terminal evidence")
    result = frame[["market_id"]].merge(ledger, how="left", on="market_id", validate="many_to_one")
    for flag in ("verified", "time_verified"):
        result[flag] = result[flag].astype(str).str.lower().eq("true")
    result["label"] = pd.to_numeric(result.label, errors="coerce")
    result["resolution_available_at"] = pd.to_numeric(result.resolution_available_at, errors="coerce")
    result.verified &= result.label.isin([0, 1]) & result.evidence_source.fillna("").ne("") & result.state.eq("resolved")
    result.time_verified &= result.time_evidence_source.fillna("").ne("") & np.isfinite(result.resolution_available_at)
    return result


def build_history(frame):
    """Input-only as-of lookup; anomalous timestamps are flagged and excluded causally."""
    c = load_config()
    out = output_dir() / "inputs"
    out.mkdir(parents=True, exist_ok=True)
    disk_guard(out)
    x = np.full((len(frame), c["history_steps"]), .5, np.float32)
    observed = np.zeros_like(x, bool)
    stale = np.ones_like(x, bool)
    age = np.full_like(x, np.inf)
    anomalies = 0
    offsets = np.arange(c["history_steps"] - 1, -1, -1) * c["history_step_seconds"]
    for event, group in frame.groupby("event_id", sort=True):
        p = ROOT / c["dataset"] / "trades/minute_history" / f"{event}.parquet"
        history = pd.read_parquet(p)
        history.market_id = history.market_id.astype(str)
        for market, rows in group.groupby("market_id", sort=True):
            hist = history[history.market_id == str(market)].sort_values("available_at")
            valid = (hist.last_trade_timestamp <= hist.available_at) & np.isfinite(hist.price)
            anomalies += int((~valid).sum())
            hist = hist[valid]
            if hist.empty:
                continue
            available = hist.available_at.to_numpy()
            trade = hist.last_trade_timestamp.to_numpy()
            prices = hist.price.to_numpy()
            ix = rows.row_id.to_numpy()
            queries = rows.origin.to_numpy()[:, None] - offsets
            pos = np.searchsorted(available, queries, side="right") - 1
            found = pos >= 0
            safe = np.maximum(pos, 0)
            point_age = queries - trade[safe]
            found &= point_age >= 0
            x[ix] = np.where(found, prices[safe], .5)
            age[ix] = np.where(found, point_age, np.inf)
            stale[ix] = ~found | (point_age > c["history_stale_seconds"])
            observed[ix] = found
    # Frozen current values have already passed the package's strict 60-second audit.
    x[:, -1] = frame.raw.to_numpy()
    observed[:, -1] = True
    stale[:, -1] = False
    age[:, -1] = 0  # explicitly aligned current observation; not a recovered trade timestamp
    eligible = (observed & ~stale).sum(axis=1) >= c["minimum_history_points"]
    np.savez_compressed(out / "history.npz", prices=x, observed=observed, stale=stale, age=age, eligible=eligible)
    write_json(out / "history_audit.json", {"rows": len(frame), "eligible_rows": int(eligible.sum()),
               "excluded_anomalous_history_rows": anomalies, "source_modified": False,
               "valid_history_point": "as-of value with age <=300s; stale values retained as inputs",
               "last_point": "aligned to frozen current price; age stored as alignment sentinel 0"})
    return x, observed, eligible


def load_history():
    path = output_dir() / "inputs/history.npz"
    lock = json.loads((output_dir() / "inputs/INPUT_LOCK.json").read_text(encoding="utf-8"))
    if digest(path) != lock["history_sha256"]:
        raise RuntimeError("Historical input cache changed after sealing")
    return np.load(path)

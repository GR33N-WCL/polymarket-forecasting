from __future__ import annotations

import numpy as np
import pandas as pd
from .protocol import output_dir, read_lock, write_json, digest, disk_guard, code_receipt


def cache_path(model, seed, task):
    return output_dir() / "predictions" / model.replace("/", "__") / str(seed) / task


def save_cache(frame, model, seed, task, values, status, checkpoint_hash, provenance):
    path = cache_path(model, seed, task)
    disk_guard(path)
    path.mkdir(parents=True, exist_ok=True)
    horizons = read_lock()["configuration"]["horizons"] if task == "price" else ["terminal"]
    if values.shape != (len(frame), len(horizons)) or status.shape != values.shape:
        raise ValueError("Cache shape does not match the frozen row list")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Record invalid numeric outputs as explicit fallbacks before caching")
    lock_hash = read_lock()["protocol_hash"]
    hashes = {}
    for j, horizon in enumerate(horizons):
        f = frame[["row_id", "market_id", "origin", "vid", "component", "event_group", "bundle_id"]].copy()
        f["task"], f["model"], f["seed"], f["horizon"] = task, model, seed, str(horizon)
        f["raw_prediction"] = values[:, j]
        f["final_prediction"] = values[:, j]
        f["status"] = status[:, j]
        f["checkpoint_hash"] = checkpoint_hash
        f["graph_variant"] = "Raw"
        f["protocol_hash"] = lock_hash
        target = path / f"{horizon}.parquet"
        if target.exists():
            previous = pd.read_parquet(target)
            if not previous.equals(f):
                raise RuntimeError(f"Refusing to overwrite a different frozen cache: {target}")
        else:
            f.to_parquet(target, index=False)
        hashes[str(horizon)] = digest(target)
    write_json(path / "MANIFEST.json", {"protocol_hash": lock_hash, "files": hashes,
               "provenance": provenance, "code_sha256": code_receipt(),
               "status_counts": dict(zip(*[a.tolist() for a in np.unique(status, return_counts=True)]))})
    return path


def load_cache(frame, model, seed, task, horizon):
    import json
    path = cache_path(model, seed, task)
    manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
    p = path / f"{horizon}.parquet"
    if digest(p) != manifest["files"][str(horizon)]:
        raise RuntimeError("Raw prediction cache changed after freezing")
    f = pd.read_parquet(p)
    if f.protocol_hash.nunique() != 1 or f.protocol_hash.iloc[0] != read_lock()["protocol_hash"]:
        raise RuntimeError("Mismatched cache protocol")
    identity = ["row_id", "market_id", "origin", "vid", "component", "event_group", "bundle_id"]
    expected = frame.set_index("row_id").loc[f.row_id].reset_index()[identity]
    if not expected.equals(f[identity]):
        raise RuntimeError("Cache changed frozen scoring identities or row ordering")
    return f

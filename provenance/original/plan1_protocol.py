from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/plan1_cross_model_20261010.json"
_VERIFIED_INPUTS = False


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def load_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def output_dir(config=None):
    return ROOT / (config or load_config())["output"]


def disk_guard(path, config=None):
    path = Path(path)
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < (config or load_config())["disk_guard_gib"] * 2**30:
        raise RuntimeError("Less than 10 GiB free; preserve frozen artifacts and use another output volume")


def numerics():
    path = ROOT / load_config()["dataset"] / "code/numerics.py"
    spec = importlib.util.spec_from_file_location("plan1_frozen_numerics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze(config=None):
    """Freeze before outcome computation. Existing locks cannot silently be replaced."""
    config = config or load_config()
    out = output_dir(config)
    disk_guard(out, config)
    source = ROOT / config["historical_report"]
    dataset = ROOT / config["dataset"]
    paths = [source / "observations.parquet", source / "constraints.json",
             dataset / "graph/projection_systems.npz", dataset / "DATASET_MANIFEST.json",
             dataset / "metadata/market_metadata.json", dataset / "code/numerics.py"]
    paths += [source / f"predictions_{h}.parquet" for h in config["horizons"]]
    files = {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in paths}
    payload = {"configuration": config, "input_sha256": files,
               "dataset_status": "development; v1 unchanged",
               "scoring": "contract -> original vid -> original component, equal at each level",
               "residual_normalization": "max(Aq-b,0)/L1(row); equality opposite rows counted once",
               "missing_rounding": "ceil(fraction * relations), capped at relations-1; <2 relations unsupported"}
    payload["protocol_hash"] = object_hash(payload)
    path = out / "PROTOCOL_LOCK.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("Protocol/source changed after freeze; create a separately named version")
    if not path.exists():
        write_json(path, payload)
    return payload


def read_lock():
    global _VERIFIED_INPUTS
    path = output_dir() / "PROTOCOL_LOCK.json"
    if not path.exists():
        raise RuntimeError("Run freeze first")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["configuration"] != load_config():
        raise RuntimeError("Configuration differs from frozen protocol")
    if not _VERIFIED_INPUTS:
        for filename, expected in lock["input_sha256"].items():
            if digest(ROOT / filename) != expected:
                raise RuntimeError(f"Frozen input changed: {filename}")
        _VERIFIED_INPUTS = True
    return lock


def code_receipt():
    return {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p)
            for p in sorted((ROOT / "plan1_cross_model").glob("*.py"))}

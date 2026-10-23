"""Thin adapters: use a fresh process per author repository (generic layers imports)."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from types import SimpleNamespace
import torch
from torch import nn

from .protocol import ROOT, load_config, output_dir, digest, write_json, object_hash

ENTRIES = {"DLinear": "models/DLinear.py", "PatchTST": "PatchTST_supervised/models/PatchTST.py",
           "iTransformer": "model/iTransformer.py"}
_LOADED = None


def source_receipt(name):
    repo = ROOT / load_config()["author_sources"] / name
    # Read only this explicitly identified migrated checkout; no global Git setting.
    git = ["git", "-c", "safe.directory="+repo.as_posix(), "-C", str(repo)]
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    origin = subprocess.check_output(git + ["remote", "get-url", "origin"], text=True).strip()
    hashes = {p.relative_to(repo).as_posix(): digest(p) for p in sorted(repo.rglob("*.py")) if ".git" not in p.parts}
    return {"model": name, "commit": commit, "origin": origin, "python_source_sha256": hashes}


def freeze_sources():
    import json
    path = output_dir() / "author_sources.json"
    receipts = {name: source_receipt(name) for name in ENTRIES}
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != receipts:
        raise RuntimeError("Frozen author source changed")
    write_json(path, receipts)
    return receipts


class LSTM(nn.Module):
    def __init__(self, width, outputs):
        super().__init__()
        self.encoder = nn.LSTM(1, width, num_layers=2, dropout=.1, batch_first=True)
        self.head = nn.Linear(width, outputs)

    def forward(self, x):
        hidden, _ = self.encoder(x)
        return self.head(hidden[:, -1]).unsqueeze(-1)


class Forecaster(nn.Module):
    def __init__(self, name, width=64, task="price"):
        super().__init__()
        global _LOADED
        self.name, self.task = name, task
        outputs = 8 if task == "price" else 1
        if name == "LSTM":
            self.core = LSTM(width, outputs)
            self.receipt = {"model": name, "layers": 2, "width": width}
        else:
            import json
            repo = ROOT / load_config()["author_sources"] / name
            pinned = json.loads((output_dir() / "author_sources.json").read_text(encoding="utf-8"))[name]
            if source_receipt(name) != pinned:
                raise RuntimeError("Author code hash mismatch")
            if _LOADED not in (None, name):
                raise RuntimeError("Launch each author model in its own Python worker")
            _LOADED = name
            package = repo / "PatchTST_supervised" if name == "PatchTST" else repo
            for generic in ("layers", "models", "utils"):
                mod = sys.modules.get(generic)
                if mod and getattr(mod, "__file__", None) and not __import__("pathlib").Path(mod.__file__).resolve().is_relative_to(package.resolve()):
                    raise RuntimeError("Foreign generic author package already imported")
            if str(package) not in sys.path:
                sys.path.insert(0, str(package))
            spec = importlib.util.spec_from_file_location("plan1_author_entry", repo / ENTRIES[name])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            config = SimpleNamespace(seq_len=72, pred_len=outputs, individual=False, enc_in=1,
                c_out=1, d_model=width, d_ff=4*width, n_heads=4, e_layers=2, dropout=.1,
                fc_dropout=.1, head_dropout=0., patch_len=12, stride=6, padding_patch="end",
                revin=1, affine=0, subtract_last=0, decomposition=0, kernel_size=25,
                output_attention=False, use_norm=True, embed="timeF", freq="t", factor=1,
                activation="gelu", class_strategy="projection")
            self.core = module.Model(config)
            self.receipt = {"model": name, "source_hash": object_hash(pinned), "config": vars(config),
                            "cross_contract_inputs": name == "iTransformer"}
        self.receipt.update({"task": task, "outputs": outputs,
                             "horizon_mapping": load_config()["horizons"] if task == "price" else ["terminal"]})

    def forward(self, x):
        y = self.core(x, None, None, None) if self.name == "iTransformer" else self.core(x)
        if isinstance(y, tuple):
            y = y[0]
        if y.shape != (x.shape[0], self.receipt["outputs"], x.shape[2]):
            raise ValueError(f"Unexpected author output {y.shape}")
        return torch.sigmoid(y) if self.task == "terminal" else y

"""Execution helpers; scientific sampling and configuration stay frozen."""
from __future__ import annotations

import os
from pathlib import Path
import torch

from .protocol import disk_guard


def atomic_torch_save(value, path):
    path = Path(path)
    disk_guard(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def capture_rng():
    import random
    import numpy as np
    state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": {"name": state[0], "keys": state[1].tolist(),
                      "position": state[2], "has_gauss": state[3], "cached_gaussian": state[4]},
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(record):
    import random
    import numpy as np
    random.setstate(record["python"])
    state = record["numpy"]
    np.random.set_state((state["name"], np.asarray(state["keys"], dtype=np.uint32),
                         state["position"], state["has_gauss"], state["cached_gaussian"]))
    torch.set_rng_state(record["torch"].cpu())
    if record["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(record["cuda"])

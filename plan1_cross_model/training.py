from __future__ import annotations

import itertools
import hashlib
import json
import time
from pathlib import Path
import random
import numpy as np
import torch

from .protocol import load_config, output_dir, write_json, digest, disk_guard, object_hash, read_lock
from .data import observations, time_masks, terminal_labels, load_history
from .metrics import weights
from .models import Forecaster
from .cache import save_cache
from .execution import atomic_torch_save, capture_rng, restore_rng


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configurations(name):
    c = load_config()
    widths = [64] if name == "DLinear" else c["widths"]
    decays = c["dlinear_weight_decay"] if name == "DLinear" else c["weight_decay"]
    return [dict(lr=lr, width=width, decay=decay) for lr, width, decay in itertools.product(c["learning_rates"], widths, decays)]


def batches(frame, row_mask, multivariate, batch_size, seed=None):
    if multivariate:
        # Group membership is input information, never inferred from target availability.
        pool = frame.loc[row_mask].sort_values("market_id").groupby("bundle_id", sort=True).row_id.apply(list)
        buckets = {}
        for ids in pool:
            buckets.setdefault(len(ids), []).append(ids)
    else:
        buckets = {1: [[int(i)] for i in frame.row_id[row_mask]]}
    chunks = []
    rng = np.random.default_rng(seed)
    for width in sorted(buckets):
        rows = buckets[width]
        if seed is not None:
            rng.shuffle(rows)
        chunks.extend(np.asarray(rows[i:i+batch_size], dtype=int) for i in range(0, len(rows), batch_size))
    if seed is not None:
        rng.shuffle(chunks)
    return chunks


def target_weights(frame, mask):
    result = np.zeros(mask.shape, dtype=float)
    for j in range(mask.shape[1]):
        if mask[:, j].any():
            result[mask[:, j], j] = weights(frame.loc[mask[:, j]])
    return result


def run_epoch(model, x, y, mask, ww, chunks, device, optimizer=None):
    model.train(optimizer is not None)
    sum_error, mass = np.zeros(y.shape[1]), np.zeros(y.shape[1])
    horizon_count = max(1, int((ww.sum(axis=0) > 0).sum()))
    for ids in chunks:
        history = torch.as_tensor(x[ids].transpose(0, 2, 1), device=device)
        target = torch.as_tensor(np.nan_to_num(y[ids]).transpose(0, 2, 1), dtype=torch.float32, device=device)
        weight = torch.as_tensor(ww[ids].transpose(0, 2, 1), dtype=torch.float32, device=device)
        if not (weight > 0).any():
            continue
        with torch.set_grad_enabled(optimizer is not None):
            pred = model(history)
            errors = (pred-target).square() * weight
            if not torch.isfinite(errors).all():
                raise FloatingPointError("Nonfinite training/validation loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                # Same frozen training weights, equal horizon mass over a full epoch.
                loss = errors.sum() * len(chunks) / horizon_count
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            else:
                errors = (pred.clamp(0, 1)-target).square() * weight
        sum_error += errors.detach().sum(dim=(0, 2)).cpu().numpy()
        mass += weight.sum(dim=(0, 2)).cpu().numpy()
    valid = mass > 0
    return float(np.mean(sum_error[valid]/mass[valid])) if valid.any() else float("inf")


def fit_one(name, task, config, seed, frame, x, y, masks, eligible, device, destination, epochs=None):
    destination = Path(destination)
    legal = (masks["train"][1] | masks["validation"][1]) & eligible[:, None]
    identity = {"protocol_hash": read_lock()["protocol_hash"], "model": name, "task": task,
                "config": config, "seed": seed, "epoch_limit": epochs or load_config()["epochs"],
                "training_code_sha256": digest(Path(__file__)),
                "model_adapter_sha256": digest(Path(__file__).with_name("models.py")),
                "history_sha256": json.loads((output_dir() / "inputs/INPUT_LOCK.json").read_text(encoding="utf-8"))["history_sha256"],
                "target_sha256": hashlib.sha256(np.ascontiguousarray(np.where(legal, y, 0.)).tobytes()).hexdigest(),
                "mask_sha256": hashlib.sha256(masks["train"][1].tobytes() + masks["validation"][1].tobytes()).hexdigest(),
                "execution": {"device": str(device), "torch_version": str(torch.__version__),
                              "torch_threads": torch.get_num_threads()}}
    receipt_path = destination.with_suffix(".json")
    resume_path = destination.with_suffix(".resume.pt")
    if receipt_path.exists():
        if not destination.exists():
            raise RuntimeError("Incomplete checkpoint write; preserve it and use a fresh checkpoint directory")
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if previous.get("identity") != identity or previous.get("checkpoint_sha256") != digest(destination):
            raise RuntimeError("Checkpoint identity changed; use a new version/directory instead of overwriting")
        return previous["best_validation"]
    if destination.exists() and not resume_path.exists():
        raise RuntimeError("Unidentified incomplete checkpoint; preserve it and use a fresh directory")
    set_seed(seed)
    model = Forecaster(name, config["width"], task).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["decay"])
    train_rows, train_mask = masks["train"]
    val_rows, val_mask = masks["validation"]
    train_mask = train_mask & eligible[:, None]
    val_mask = val_mask & eligible[:, None]
    train_w, val_w = target_weights(frame, train_mask), target_weights(frame, val_mask)
    c = load_config()
    best, stale, best_epoch = float("inf"), 0, -1
    trace, first_epoch, best_state = [], 0, None
    if resume_path.exists():
        resumed = torch.load(resume_path, map_location=device, weights_only=True)
        if resumed["identity"] != identity:
            raise RuntimeError("Interrupted-fit identity changed; preserve it and use a new checkpoint directory")
        model.load_state_dict(resumed["state_dict"])
        optimizer.load_state_dict(resumed["optimizer"])
        best, stale, best_epoch = resumed["best"], resumed["stale"], resumed["best_epoch"]
        trace, first_epoch, best_state = resumed["trace"], resumed["next_epoch"], resumed["best_state_dict"]
        restore_rng(resumed["rng"])
        print(f"RESUME {name} {task} {destination.parent.name} {destination.stem} epoch={first_epoch}", flush=True)
    val_chunks = batches(frame, val_rows, name == "iTransformer", c["batch_size"])
    started = time.monotonic()
    progress_path = output_dir() / "training" / name / task / "progress.json"
    for epoch in range(first_epoch, epochs or c["epochs"]):
        if stale >= c["patience"]:
            break
        before = time.monotonic()
        train_chunks = batches(frame, train_rows, name == "iTransformer", c["batch_size"], seed+epoch)
        train_loss = run_epoch(model, x, y, train_mask, train_w, train_chunks, device, optimizer)
        val_loss = run_epoch(model, x, y, val_mask, val_w, val_chunks, device)
        trace.append({"epoch": epoch, "train_mse": train_loss, "validation_mse": val_loss,
                      "seconds": round(time.monotonic()-before, 4)})
        if val_loss < best - 1e-12:
            best, stale, best_epoch = val_loss, 0, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        # One atomic state file is authoritative even if a later progress write is interrupted.
        atomic_torch_save({"identity": identity, "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(), "best_state_dict": best_state,
            "best": best, "stale": stale, "best_epoch": best_epoch,
            "trace": trace, "next_epoch": epoch+1, "rng": capture_rng()}, resume_path)
        write_json(progress_path, {"status": "training", "model": name, "task": task,
            "month": destination.parent.name, "checkpoint": str(destination), "seed": seed,
            "config": config, "completed_epochs": epoch+1, "best_validation": best,
            "stale_epochs": stale, "latest_epoch_seconds": trace[-1]["seconds"],
            "elapsed_this_fit_seconds": round(time.monotonic()-started, 2)})
        if epoch == first_epoch or (epoch+1) % 5 == 0 or stale >= c["patience"]:
            print(f"EPOCH {name} {task} {destination.parent.name} {destination.stem} "
                  f"{epoch+1}/{epochs or c['epochs']} val={val_loss:.8g} "
                  f"best={best:.8g} stale={stale} seconds={trace[-1]['seconds']}", flush=True)
    if best_state is None:
        raise RuntimeError("No valid training epoch; do not report a trained checkpoint")
    atomic_torch_save({"state_dict": best_state, "config": config, "receipt": model.receipt,
                      "seed": seed, "task": task}, destination)
    write_json(destination.with_suffix(".json"), {"identity": identity, "checkpoint_sha256": digest(destination), "trace": trace, "best_epoch": best_epoch,
        "best_validation": best, "training_rows": int(train_mask.any(axis=1).sum()),
        "validation_rows": int(val_mask.any(axis=1).sum()),
        "train_labels_per_output": train_mask.sum(axis=0).tolist(),
        "validation_labels_per_output": val_mask.sum(axis=0).tolist(),
        "training_ids_hash": object_hash(frame.row_id[train_mask.any(axis=1)].tolist()),
        "validation_ids_hash": object_hash(frame.row_id[val_mask.any(axis=1)].tolist())})
    return best


def train(name, task="price", months=None, device="cpu", checkpoint_root=None):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; use --device cpu for the full protocol")
    c = load_config()
    f = observations()
    inputs = load_history()
    x, eligible = inputs["prices"], inputs["eligible"]
    labels = terminal_labels(f) if task == "terminal" else None
    if labels is not None and not (labels.verified & labels.time_verified).any():
        raise RuntimeError("Terminal training blocked: no verified label-publication timestamps")
    y = f[[f"target_{h}" for h in c["horizons"]]].to_numpy() if task == "price" else labels.label.to_numpy()[:, None]
    months = months or sorted(f.month.unique())
    root = Path(checkpoint_root) if checkpoint_root else output_dir() / "checkpoints"
    predictions = {s: np.repeat(f.raw.to_numpy()[:, None], y.shape[1], axis=1) for s in c["seeds"]}
    states = {s: np.full(y.shape, "month_not_requested", dtype=object) for s in c["seeds"]}
    hashes = {s: np.full(len(f), "no_checkpoint", dtype=object) for s in c["seeds"]}
    audit = []
    print(f"START {name} {task} device={device} threads={torch.get_num_threads()} "
          f"months={len(months)} configurations={len(configurations(name))} seeds={c['seeds']}", flush=True)
    for month_index, month in enumerate(months, 1):
        masks = time_masks(f, month, c["horizons"], task, labels)
        evaluation = masks["evaluation"][0]
        ntrain = int((masks["train"][1] & eligible[:, None]).any(axis=1).sum())
        nval = int((masks["validation"][1] & eligible[:, None]).any(axis=1).sum())
        print(f"MONTH {month_index}/{len(months)} {month} train={ntrain} validation={nval} "
              f"evaluation={int(evaluation.sum())}", flush=True)
        if ntrain < c["minimum_train_rows"] or nval < c["minimum_validation_rows"]:
            for seed in c["seeds"]:
                states[seed][evaluation] = "insufficient_past_data_fallback"
            audit.append({"month": month, "train_rows": ntrain, "validation_rows": nval, "status": "fallback"})
            write_json(output_dir() / "training" / name / task / "selection.json", audit)
            continue
        directory = root / name / task / month
        trials = []
        for i, config in enumerate(configurations(name)):
            print(f"TRIAL {month} {i+1}/{len(configurations(name))} {config}", flush=True)
            dest = directory / f"trial_{i}_seed13.pt"
            score = fit_one(name, task, config, 13, f, x, y, masks, eligible, device, dest)
            trials.append((score, i, config, dest))
        score, index, config, best_path = min(trials, key=lambda item: (item[0], item[1]))
        audit.append({"month": month, "selected": config, "trial": index, "validation": score,
                      "train_rows": ntrain, "validation_rows": nval,
                      "trials": [{"validation": item[0], "index": item[1], "configuration": item[2]}
                                 for item in trials]})
        for seed in c["seeds"]:
            print(f"SELECTED {month} seed={seed} configuration={config}", flush=True)
            dest = best_path if seed == 13 else directory / f"selected_seed{seed}.pt"
            if seed != 13:
                fit_one(name, task, config, seed, f, x, y, masks, eligible, device, dest)
            checkpoint = torch.load(dest, map_location=device, weights_only=True)
            model = Forecaster(name, config["width"], task).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            with torch.no_grad():
                for ids in batches(f, evaluation, name == "iTransformer", c["batch_size"]):
                    prediction = model(torch.as_tensor(x[ids].transpose(0, 2, 1), device=device)).cpu().numpy().transpose(0, 2, 1)
                    good = eligible[ids] & np.isfinite(prediction).all(axis=2)
                    predictions[seed][ids[good]] = np.clip(prediction[good], 0, 1)
                    states[seed][ids[good]] = "model_generated"
                    states[seed][ids[~good]] = "insufficient_history_or_nonfinite_fallback"
            hashes[seed][evaluation] = digest(dest)
            del model
        write_json(output_dir() / "training" / name / task / "selection.json", audit)
    # Incomplete month jobs remain visibly incomplete and cannot be scored as full runs.
    for seed in c["seeds"]:
        save_cache(f, name, seed, task, predictions[seed], states[seed], hashes[seed],
                   {"months": months, "monthly_training": audit, "complete_month_coverage": set(months) == set(f.month.unique())})
    write_json(output_dir() / "training" / name / task / "progress.json",
               {"status": "raw_caches_complete", "model": name, "task": task, "months": months,
                "complete_month_coverage": set(months) == set(f.month.unique()),
                "checkpoint_root": str(root), "device": str(device), "torch_threads": torch.get_num_threads()})
    print(f"COMPLETE raw predictions {name} {task}", flush=True)


def smoke_model(name):
    """Actual forward/backward through author code, not a surrogate predictor."""
    set_seed(13)
    torch.set_num_threads(2)
    checks = []
    for task in ("price", "terminal"):
        model = Forecaster(name, 64, task)
        channels = [3, 5] if name == "iTransformer" else [1]
        for n in channels:
            x = torch.rand(4, 72, n)
            y = model(x)
            (y-.4).square().mean().backward()
            assert torch.isfinite(y).all()
            assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            checks.append({"task": task, "channels": n, "shape": list(y.shape), "backward": "passed"})
    write_json(output_dir() / "checks" / f"smoke_{name}.json", {"checks": checks, "scope": "CPU adapter smoke; not trained research results"})
    return checks

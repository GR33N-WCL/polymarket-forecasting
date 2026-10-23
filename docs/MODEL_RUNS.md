# Model execution

Run commands from the repository root after extracting the public release data. Choose a new output directory for each configuration. Model execution is explicit; downloading or opening this repository starts no training.

See [the non-LLM model guide](NON_LLM_MODELS.md) for architecture settings and the distinction between training seeds and cache identifiers.

## Verified cache replay

```bash
python -m rcf.replay --data data --output runs/attribution8
```

Models: Self-Ridge, AutoARIMA, LightGBM, Chronos-2, LSTM, DLinear, PatchTST and iTransformer. Neural models have five seeds (13,37,73,101,137); the remaining models have one cache each. See `configs/benchmark.json`.

## Fresh neural training

```bash
python -m pip install -r requirements-neural.txt
python -m models.train_neural --model LSTM --data data --output runs/lstm --device cuda:0
python -m models.train_neural --model DLinear --data data --output runs/dlinear --device cuda:0 --fetch-author-code
```

PatchTST and iTransformer use the same entry point. The option `--fetch-author-code` fetches the repository and exact commit recorded in `configs/author_sources.json`; source hashes are checked by the original adapter. Keep each author model in a separate process because its upstream package names overlap.

Training uses the original 198449-row parent pool, causal monthly train/validation windows, original configurations, early stopping and five seeds. Output caches preserve explicit generation/fallback status. For evaluation on the test dataset, join the new cache to `core/answers.parquet` using original `row_id` and retain only `model_generated` statuses; recompute complete-vector and common-success masks for the new runs. Do not impose the archived benchmark mask on new failures.

`--months YYYY-MM ...` limits execution to specified months and marks incomplete month coverage. `--prepare-only` stages inputs and seals a run identity without training. The bundled reference scores are from the original completed runs, not from this release's staging check.

## Supplementary models

```bash
python -m pip install -r requirements-strong.txt
python -m models.run_strong --model AutoARIMA --data data --output runs/arima
python -m models.run_strong --model AutoARIMA --data data --output runs/arima --recover-arima
python -m models.run_strong --model LightGBM --data data --output runs/lightgbm
python -m models.run_strong --model Chronos-2 --data data --output runs/chronos --chronos-weights /path/to/pinned/snapshot
```

AutoARIMA uses the archived `pmdarima` worker and its failed-window recovery, preserving valid primary outputs. `worker_standard.py` is a historical StatsForecast alternative and is not the final execution entry point. Chronos-2 uses repository `amazon/chronos-2`, revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, zero-shot univariate median predictions with cross-learning disabled. Supply weights locally. LightGBM uses the original past-window training pool and validation selection.

Self-Ridge's frozen outputs are verified through the cache replay. Its original daily-fit reconstruction and selection programs are retained under `provenance/original/self_ridge/`; they still depend on historical project assets and are **not** advertised as a portable from-scratch command in v0.1.0.

## Local large language models

The strict worker targets Linux/CUDA, uses NF4 double quantization with FP16 computation and forbids CPU offloading. Install a CUDA-matched PyTorch build and `requirements-llm.txt`. Weight/model revisions are in `configs/llm_models.json`; acquire snapshots separately under the providers' terms.

```bash
CUDA_VISIBLE_DEVICES=0 python -m models.run_llm \
  --model mistral7 --data data --output runs/mistral_strict \
  --weights /path/to/huggingface/snapshots/EXACT_REVISION \
  --batch-size 1 --memory-gib 40
```

Available keys are `qwen1_8`, `qwen7`, `llama7`, `mistral7`. Qwen14 is not included in this paper release. There is no DeepSeek/API execution entry point.

If the snapshot folder is renamed, provide `REVISION.json` containing its `repo_id` and pinned `revision`. The launcher hashes weight files into a local run receipt. It preserves the original prompt/encoding and strict response checks: exactly eight finite prices in [0,1], raw response retained, request and input identities matched. No truncation or current-price fallback is used. Ten format failures halt a worker for diagnosis; failed attempts remain pending.

The launcher starts a fresh strict rerun. It does not reuse heterogeneous archived Qwen1.8 outputs. `--prepare-only` prepares the run without inference. To resume a staged run, read `RUN_ARGUMENTS.json` and call the staged `orchestration/infer_strict20.py` with the same model and allocation; set `POLYMARKET_MODELS_DIR` to that run's models directory.

## Environment boundaries

Core replay needs NumPy, pandas, SciPy and PyArrow. Fresh inference additionally requires the original model libraries and external weights. Dependency files specify the usable package families, while archived protocol/source receipts identify the scientific settings. The release audit does not assert bitwise reproduction across CUDA, PyTorch or GPU versions.

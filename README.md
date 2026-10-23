# Polymarket Forecasting

Dataset and reproducibility code accompanying **Beyond Isolated Markets: Relational Contract Price Forecasting in Prediction Markets**.

**Paper status: submitted; under review.** This repository provides the dataset and supporting model execution and evaluation code for the submitted paper.

**论文配套数据集（已投稿）**：本仓库提供上述论文使用的数据集，以及配套的模型运行与评价代码。

This repository separates the current 303-event evaluation cohort from its historical training and context pool. It contains the numerical correction code, runnable model entry points, archived source provenance, and a versioned data asset. It does not start remote jobs or call a language-model API.

## Quick start

Python 3.10 or later is required. The numerical replay was checked locally with Python 3.12.

```bash
git clone https://github.com/GR33N-WCL/polymarket-forecasting.git
cd polymarket-forecasting
python -m venv .venv
# Activate .venv using your shell's usual command.
python -m pip install -r requirements.txt
gh release download v0.1.0 --repo GR33N-WCL/polymarket-forecasting --pattern 'dataset-v0.1.0.tar.gz'
python -m rcf.unpack dataset-v0.1.0.tar.gz --output .
python -m rcf.verify --data data
python -m rcf.construct --data data --output runs/data_replay.json
python -m rcf.replay --data data --output runs/attribution8
python -m unittest discover -s tests -v
```

The dataset is available from the public [v0.1.0 release](https://github.com/GR33N-WCL/polymarket-forecasting/releases/tag/v0.1.0). `rcf.unpack` checks the archive checksum against the committed release manifest before extraction and refuses to overwrite existing files.

`rcf.replay` reproduces **24 model/seed units × 8 horizons × 6 variants = 1,152 metric rows** on the eight non-LLM models' common support. It verifies the MSE, MAE and improvement/harm counts against the archived reference results. It computes seed means of losses, not ensembles of predictions. The main paper's S1, S2 and S3 tables use different paired supports; this replay must not be relabelled as their complete reconstruction.

To check one model first:

```bash
python -m rcf.replay --model Self-Ridge --data data --output runs/ridge_check
```

## Dataset

| Scope | Content |
| --- | --- |
| Current evaluation cohort | 303 events, 345 event edges, 43 event-graph groups |
| Contract catalogue | 5,033 contracts |
| Forecast-origin table | 183,077 rows, 4,617 distinct contracts |
| Any valid target before model-success filtering | 4,507 contracts |
| Historical training pool | 198,449 rows; preserved separately |
| Archived S3 input cohort | 36,616 requests per model; 303 events, 3,790 contracts, 43 groups |
| Forecast horizons | 1, 5, 15, 30, 60, 180, 360 and 1,440 minutes |

The source archive period reported in the manuscript (2022-11-21 to 2026-05-04) and the actual forecast-origin range (2024-01-06 to 2026-09-25 UTC) describe different scopes. The latter includes the historical expanded cohort. See [the construction account](docs/DATASET_CONSTRUCTION_zh.md) before describing a single data period.

## Contents

- `rcf/`: portable replay, price lookup, input verification and joint projection.
- `models/`: neural training, three supplementary predictors and strict local LLM inference entry points.
- `plan1_cross_model/`: original neural training/model adapters with a portable path/configuration layer.
- `configs/`: model/source revisions and experiment settings.
- `docs/`: construction process, data dictionary and execution instructions.
- `provenance/`: original source snapshots and selection/quality records. Files here are historical records, not portable entry points.
- `results/reference/`: saved group-distribution statistics and verification receipts.
- `data/`: downloaded release asset, excluded from Git.

## Model execution and replication scope

See [MODEL_RUNS.md](docs/MODEL_RUNS.md). Fresh training/inference writes to new run directories. No pretrained model weights are distributed here. Access to gated model weights remains governed by their original providers.

The current release has **numerically verified cache replay**. Training entry points, data staging and source compilation were checked; the release process did not retrain all models or run new GPU inference. Self-Ridge's daily-fit reconstruction remains in the original source archive; its frozen predictions are fully included and replayable. This distinction is recorded in [VALIDATION.md](docs/VALIDATION.md).

## Version and interpretation

The 303-event roster is derived from a retrospectively examined development cohort. Removed groups and inherited source anomalies are documented. Frozen targets and scoring identities are retained. Failed forecasts are excluded using explicit success masks and complete logical-vector rules.

The S3 asset is the locally verified archived sample. Its manifest preserves pre-existing successes, whole-bundle closure and randomized completion. A later author-reported pure-random rerun on another machine has not been linked to an input manifest in this repository. These cohorts must not be silently substituted for one another.

## Access and rights

This is the public dataset and reproducibility repository for the submitted paper. No new redistribution licence is assigned to third-party market data or model code by this release. Author implementations are fetched from pinned upstream repositories; retain their original licences and model conditions. The original research repository and source data are unchanged.

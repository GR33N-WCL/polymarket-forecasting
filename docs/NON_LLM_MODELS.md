# Non-LLM models: architectures, seeds and execution

The released 303-event dataset is used as the **test dataset**. Model fitting and validation use earlier observations in `data/training/`; these historical inputs are separate from the test evaluation role. The test targets are not used to choose monthly hyperparameters. All models predict prices at 1, 5, 15, 30, 60, 180, 360 and 1,440 minutes.

## Implementations and seeds

| Model | Architecture / settings in this repository | Seed / execution |
| --- | --- | --- |
| Self-Ridge | Contract-local ridge regression; original implementation under `provenance/original/self_ridge/` | One frozen cache, ID `0`; verified replay available. Portable fresh-fit entry point is not yet complete. |
| AutoARIMA | Nonseasonal ARIMA; stepwise AICc selection, p/q up to 3, d up to 2, max order 5; 72-point window | One fit per unique history; cache ID `0` is not a random seed. |
| LightGBM | Gradient-boosted trees; 7/15/31 leaves × 20/80 minimum child samples; learning rate 0.03, at most 1,000 trees, early stopping 50 | Actual `random_state=13`; cache ID is `0`. |
| LSTM | Two LSTM layers, hidden width 64/128, dropout 0.1, linear output head | Final training seeds: 13, 37, 73, 101, 137. |
| DLinear | Pinned author decomposition-linear implementation; input length 72, output length 8, shared channel parameters | Same five training seeds. |
| PatchTST | Pinned author implementation; patch length 12, stride 6; two encoder layers, four attention heads, width 64/128, FFN width 4× model width | Same five training seeds. |
| iTransformer | Pinned author implementation with variate tokens; two encoder layers, four attention heads, width 64/128, FFN width 4× model width | Same five training seeds. |
| Chronos-2 | Pretrained `amazon/chronos-2`; zero-shot univariate median forecast, context 72, forecast length 288 five-minute steps; no cross-learning or fine-tuning | One inference cache, ID `0`; no task-training seed. |

Actual adapters are in `plan1_cross_model/models.py`; neural settings are in `configs/neural.json`. Author source URLs, commits and file hashes are pinned in `configs/author_sources.json`. Supplementary predictor settings are in `models/strong/PROTOCOL_LOCK.json` and their implementation in `models/strong/worker_pmd.py`. Chronos-2's pinned revision is `29ec3766d36d6f73f0696f85560a422f50e8498c`.

## Neural training settings

Inputs contain 72 five-minute history points. For a test month starting at T, fitting uses [T−90 days, T−30 days), and validation uses [T−30 days, T). Labels must be available before the corresponding window ends. Training uses batch size 32, at most 100 epochs and patience 10. Learning rates are 0.0001, 0.0003 and 0.001. LSTM/PatchTST/iTransformer search widths 64/128 and weight decay 0/0.0001. DLinear searches decay 0/0.000001/0.0001/0.01. Configuration selection uses seed 13; final training runs all five seeds above. The launcher reads these settings rather than requiring a separate `--seed` argument.

Bootstrap seed 13 and the ten relation-control randomization seeds are evaluation settings, not additional model-training seeds. Cache replay contains 24 model/seed units: four neural models × five seeds, plus four single-cache predictors.

## Run the supplied test predictions

After downloading and extracting the dataset as described in the root README:

```bash
python -m pip install -r requirements.txt
python -m rcf.verify --data data
python -m rcf.replay --data data --output runs/test_replay
python -m rcf.replay --model Self-Ridge --data data --output runs/self_ridge_test
```

Replay evaluates the supplied base predictions and correction variants. It is distinct from fitting a new model.

## Train the four neural models

Run each command in a separate process and use a new output directory:

```bash
python -m pip install -r requirements-neural.txt
python -m models.train_neural --model LSTM --data data --output runs/lstm --device cuda:0
python -m models.train_neural --model DLinear --data data --output runs/dlinear --device cuda:0 --fetch-author-code
python -m models.train_neural --model PatchTST --data data --output runs/patchtst --device cuda:0 --fetch-author-code
python -m models.train_neural --model iTransformer --data data --output runs/itransformer --device cuda:0 --fetch-author-code
```

`--months YYYY-MM ...` restricts execution to selected months. `--prepare-only` prepares a run without training. Join newly generated predictions to test records by original `row_id`, keep `model_generated` outputs, and recompute complete-vector/common-success masks as described in `MODEL_RUNS.md`. The saved benchmark mask must not conceal failures in a new run.

## Run the other predictors

```bash
python -m pip install -r requirements-strong.txt
python -m models.run_strong --model AutoARIMA --data data --output runs/arima
python -m models.run_strong --model AutoARIMA --data data --output runs/arima --recover-arima
python -m models.run_strong --model LightGBM --data data --output runs/lightgbm
python -m models.run_strong --model Chronos-2 --data data --output runs/chronos --chronos-weights /path/to/pinned/snapshot
```

Chronos-2 weights must be obtained separately. Self-Ridge can be evaluated immediately with the replay command above; its archived fresh-fit implementation still needs historical project assets. Model code and launchers are included, but this release's validation did not retrain all models. See `VALIDATION.md` for the completed checks.

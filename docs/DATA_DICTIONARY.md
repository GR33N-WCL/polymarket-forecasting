# Data dictionary

All timestamps are UTC Unix seconds. Arrays use the frozen row order unless an explicit `row_id` is supplied. Join by `row_id`, not by positional indices from another version.

| Path | Fields and meaning |
| --- | --- |
| `core/answers.parquet` | `row_id`, event/market IDs, `origin`, `raw` current price, `target_h`, `complete_h`, actual `last_trade_h`, original `vid` and `component`, graph `event_group`, `bundle_id` |
| `core/history.npz` | `row_id`; `prices`, `observed`, `stale`, `age` of shape [183077,72]; per-row `eligible` |
| `core/neighbor_messages.npz` | `row_id`, `message`, `support`, each horizon in [1,5,15,30,60,180,360,1440] order |
| `core/semantic_edge_arrays.npz` | Accepted directed historical slots, target positional index, source identity, valid-horizon mask, source signal including age decay, reliability, signed amplitude and aggregate denominators |
| `core/systems.json` | `system_id`, matrix `A`, vector `b`, complete `row_ids` vectors; project each complete vector under `Aq <= b` plus price bounds |
| `core/event_graph.json` | Retained 303 event nodes and 345 undirected edges |
| `core/retained_contracts.csv` | 5033-entry catalogue, distinct from the 4617 contracts represented at forecast origins |
| `core/market_metadata.json` | Retrospective market descriptions/outcome definitions; historical text-version availability is not established |
| `benchmark/raw/` | Eight non-LLM models, 24 model/seed arrays; `row_id`, `Raw`, horizon-specific `success` |
| `benchmark/COMMON_SUPPORT.npz` | Joint-vector-closed success, scoring mask and calibration legality; the scoring mask varies by horizon |
| `benchmark/expected/` | Saved per-seed MSE/MAE and exact changed/improved/harmed counts used in replay verification |
| `training/` | Original parent observations and 72-point histories for fresh neural/LightGBM runs |
| `history/` | 374-event parent minute histories for historical context; does not enlarge the current 303-event scored roster |
| `parent_graph/` | Historical graph, edge roles, catalogue and metadata; retained for provenance/context only |
| `llm/` | Archived S3 cohort and original request snapshots, quantization settings, acceptance changes and sample manifest |

`raw` in the observations means the observed current price; `Raw` in a model cache means that model's uncorrected forecast. The current-price baseline is not inserted into the paper's main model table by this repository.

`complete_h` can be false even when a numerical `target_h` is present. Always apply the stored target-validity and model-success rules. Failure/missing states are not replaced by the current price during scoring.

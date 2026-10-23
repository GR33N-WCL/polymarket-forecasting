# Evidence and terminology

| Statement | Evidence | Status |
| --- | --- | --- |
| 303 events / 183077 observations | `data/core/answers.parquet`, retained roster and scope audit | Recounted |
| 345 edges / 43 groups | Filtered parent graph and retained `event_group` identities | Recounted |
| 5033 catalogue contracts versus 4617 represented contracts | Retained catalogue versus observations | Recounted |
| 553620 price values replay exactly | `results/reference/DATA_REPLAY_AUDIT.json` | Recomputed |
| Eight-model attribution replay | `results/reference/CACHE_REPLAY_AUDIT.json` | Recomputed |
| Pure-random new S3 rerun | Later author statement; no matching local manifest located | Separate version awaiting linkage |

| Canonical term | Meaning |
| --- | --- |
| Event / 事件 | Event-level question and metadata |
| Contract / 合约 | A specified market outcome with a price series |
| Contract–origin record / 合约—起点记录 | One contract observed at one prediction origin |
| Event-graph group / 事件图组 | Graph connected group; used for clustered diagnostics |
| Scoring component / 评分分量 | Original aggregation identity; distinct from event-graph group |
| Raw forecast / 基础预测 | Model output before either correction |
| Current price / 当前价格 | Observed input, `raw` in the observation table |
| Semantic correction / 语义修正 | Historical-neighbor information updates |
| Joint logical correction / 联合逻辑修正 | Shared-variable projection under audited constraints |

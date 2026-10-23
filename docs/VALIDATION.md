# Validation of v0.1.0

Verified on 23 October 2026:

| Check | Result |
| --- | --- |
| Current 303-event identities | 183077 rows, unique original row IDs, 4617 represented contracts, 43 graph groups |
| Event graph | 303 nodes, 345 edges |
| Current-price and valid-target replay | 553620 values; maximum absolute difference 0 |
| Semantic-message aggregation | 71508 accepted slots; maximum difference 4.440892098500626e-16 |
| Eight-model cache replay | 24 model/seed units, 1152 metric rows; all MSE/MAE comparisons and exact sign/change counts pass |
| Largest metric discrepancy | 1.8908485888147197e-16 |
| Unit checks | Minute availability, overlapping constraints and original hierarchical weights pass |
| Neural entry point | Training module imports; parent-pool input staging and protocol preparation pass |
| Supplementary and LLM entry points | Argument parsers and Python source syntax checked |
| Selected source/text credential scan | No credential-pattern matches |

Receipts are in `results/reference/`. The data archive has a SHA-256 digest in `configs/release.json` and a file-by-file manifest inside `data/MANIFEST.json`.

The source code for optional GPU training/inference is included, but no new training or inference was performed to publish this repository. The portable Self-Ridge daily-fit reconstruction is not complete in this first release; use its verified frozen cache or the archived original implementation with historical assets. Historical web-assisted event discovery is represented by its final roster/graph and available source records, not an invented complete prompt trace.

Before a public release, reconcile the author's later pure-random S3 rerun with its actual input manifest and update the data-period wording to distinguish the original archive from expanded observations. These points do not alter the preserved numerical assets.

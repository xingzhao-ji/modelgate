# Architecture and contracts

The implementation is intentionally small enough to audit. There is one process for training and one for serving; no scheduler, queue, hosted registry, or hidden model API is required.

| Module | Ownership |
|---|---|
| `download.py` | Exact public snapshot, hashes, ZIP member selection, provenance |
| `data.py` | Fixed feature contract, no silent coercion, duration/label exclusion, row partitions |
| `training.py` | Train-only sklearn pipelines, candidate selection, disjoint calibration and policy |
| `metrics.py` | AP/AUC/Brier, fixed-bin calibration, confusion matrix, explicit policy costs |
| `evaluation.py` | Reserved holdout, block-bootstrap intervals, slices, offline gate |
| `drift.py` | Train-fixed distributions and stateless PSI comparisons |
| `artifacts.py` | Atomic bundle publication, content hashes, schema/version/type checks |
| `api.py` | HTTP contracts, body bounds, health, request IDs, counters |
| `experiment.py` | Real subprocess HTTP measurements and negative-path experiments |
| `cli.py` | Explicit inputs/outputs and nonzero exit status on invalid work |

## Statistical isolation

The five row windows have different responsibilities. Training and category/median/scaler fitting use only the first window; selection sees only the second; calibration sees only the third; policy sees only the fourth. Training slices away the final window before feature/target parsing. Evaluating the holdout does not call `fit`, change thresholds, or modify an artifact.

Candidate selection uses AP instead of accuracy because positive labels are rare in the development windows. AP is sensitive to prevalence, so the report shows each window's prevalence and constant-score reference. The reported value is sklearn `average_precision_score`, not trapezoidal integration of interpolated precision-recall points.

Sigmoid calibration fits a second binary model on clipped raw log odds. It uses its own labels and reports its later-window behavior. This makes the distinction between discrimination and calibration observable. The policy minimizes an explicit cost on a separate window using sorted cumulative counts, with ties handled as groups; it does not test every threshold against holdout outcomes.

## Artifact boundary

A bundle contains exactly the files named in the manifest: `model.skops`, `metadata.json`, and `reference.json`; the manifest itself has a fixed format version. The model ID is the first 16 hexadecimal characters of the SHA-256 of the canonical file-hash map. Metadata contains feature order, threshold, calibration coefficients, source data provenance, training decisions, and exact numpy/sklearn/skops/pandas versions. The reference profile stores aggregate bins and frequencies, not raw training records.

Writers stage all files under the destination parent, validate the exact staged bundle, and publish it with an atomic directory rename. A preexisting artifact is rejected. This protects readers from partial publication, but is not a transactional multi-writer registry or a crash-durable remote object store. No fsync/power-loss guarantee is claimed. The service loads one immutable bundle at startup and requires a restart to change it.

The loader reads each bounded file once, checks hashes before parsing or deserializing, rejects unexpected manifest names and symlinks, verifies dependency and schema compatibility, validates reference distributions, and reviews unknown skops types against a fixed code-defined allowlist. It never loads arbitrary pickle/joblib objects or automatically trusts a type list from the artifact. The allowlist includes numpy dtype and specific sklearn histogram-tree/loss internals needed by the supported candidates. See [skops persistence guidance](https://skops.readthedocs.io/en/stable/persistence.html).

Checksums can be recomputed by someone who controls the whole directory; they provide integrity checking, not authenticity. This remains a trusted-local-artifact system. There is no artifact-upload endpoint. Malicious compressed artifacts and hostile local file writers are outside the local demonstration boundary.

## API and resource behavior

The body limiter counts chunks before JSON parsing and refuses more than 2 MiB. Scoring accepts at most 512 records, drift at most 5,000, and the CLI server limits concurrent connections to 32. No caller can select a model file or execute training through HTTP. Values are checked at every scoring boundary; a missing key in one row cannot hide behind another row's dataframe columns. Explicit null values invoke documented imputations. Numeric range checks are broad sanity limits, not domain completeness.

Model computation runs in FastAPI's synchronous worker threads with one numeric-library thread per call; the recorded benchmark uses one server process. Counters are guarded by a lock and are process-local; they reset on restart and are not a durable monitoring store. The service exposes request IDs and elapsed time and avoids request-value logging. There is no authentication or Internet-facing deployment. Readiness returns 503 when loading fails; valid requests continue after malformed requests return 422.

Drift uses training quantile cuts plus edge and missing bins for numerical fields, and exact training category vocabulary plus unseen bucket for categorical fields. Identical distributions give PSI zero. A fixed epsilon 0.0001 makes zero-frequency comparisons finite. PSI > 0.25 is a heuristic alert, not a p-value; correlated columns can all fire together. Comparisons require at least 100 records. No production stream is retained; callers provide a window to compare.

## Reproducibility limits

The fixed dataset, seed, splits, dependency lock, and hyperparameters make the experiment reproducible. The source-content inventory in the evidence folder identifies the reviewed code. OS numeric libraries and package wheels can cause small cross-platform numerical differences. skops ZIP metadata can alter a bundle hash across serializations. Latency depends on host load and HTTP client overhead. CI runs the complete behavior on a fresh Ubuntu environment when published as a standalone repository, but local results are the only executed evidence supplied here.

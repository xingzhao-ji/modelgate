# Verification notes

The suite tests data leakage and service behavior using actual sklearn estimators, serialized artifacts, filesystem writes, and HTTP. Tiny generated datasets are deterministic fixtures for fast tests; all portfolio results come from the complete public UCI snapshot.

The implementation began with behavior tests and explicit unimplemented API boundaries. The first run produced 19 expected failures for split isolation, schema validation, train-only preprocessing, probability metrics, threshold choice, and training. These passed after implementation. Artifact/drift and HTTP features followed the same red/green sequence. The actual-HTTP test first failed because no module CLI existed; after implementation it starts an independent uvicorn process and compares HTTP probabilities against batch CLI output. A reference-distribution corruption test was observed failing before load-time profile validation was added.

The complete suite now runs with **all warnings treated as errors**. A system issue was reduced to `numpy.ones((600, 27)) @ numpy.zeros(27)` falsely emitting floating-point warnings under numpy 2.2.6 / Apple Accelerate. Pinning numpy 2.3.4 corrected that reproduction; warnings are not suppressed. See the upstream [NumPy issue](https://github.com/numpy/numpy/issues/29820) for the Accelerate behavior.

## Behavior covered

- Ordered disjoint exhaustive partitions and minimum dataset size.
- Exclusion of `duration` and `y`; strict target values.
- Missing/extraneous keys, numeric strings, booleans, infinity, ranges, category types/lengths.
- Explicit null support, unseen categories, training-only preprocessing statistics/vocabulary.
- Deterministic validation winner and complete model/report invariance after changing holdout values.
- Stable-feature ablation invariance when economic/calendar inputs are changed.
- Independently hand-computed Brier, confusion counts, threshold costs, and one-class metric behavior.
- Known block-bootstrap outcomes, plus public-download hash rejection.
- Exact probability roundtrip through skops; immutable artifact destinations.
- A nonlinear fixture where gradient boosting wins and survives the same artifact roundtrip.
- Corrupted model/metadata/reference bytes, malicious manifest paths, invalid threshold and reference profiles with recomputed hashes.
- Missing provenance, inconsistent/fractional partition boundaries, and enormous numeric artifact metadata, even when hashes are recomputed.
- Same-distribution PSI=0, injected distribution changes, numeric missingness, unknown categories, and undersized drift windows.
- Healthy/unready startup, real prediction/model/drift endpoints, malformed requests followed by successful requests.
- Raw JSON with a 401-digit integer or nonstandard NaN is rejected before pandas can coerce or overflow.
- A real TCP server in a subprocess, plus batch scoring in a separate interpreter.
- Full HTTP latency/failure drill against a fitted model, including corrupt-artifact service readiness.

Run the normal suite with `.venv/bin/pytest -q -W error`. The `integration` marker selects the two real-process end-to-end tests; they are included by default. `make test` also checks lint and formatting. The public-data training, final evaluation, and larger HTTP benchmark are separate reproducible commands, exercised again by the CI template.

Tests intentionally avoid asserting exact latency or a fixed holdout score: those would be brittle hardware or platform checks. Accuracy is an empirical result documented in the frozen report. The suite does not claim exhaustive fuzzing, adversarial artifact safety, authentication, multi-process observability, group-disjoint client identification, or cloud performance.

On macOS, Python ignores `.pth` files carrying the filesystem `UF_HIDDEN` flag. Do not recursively set that flag on virtualenv contents. A dot-prefixed `.venv` directory is already hidden from normal listings; it does not need recursive Finder flags. If another tool sets flags and an editable install disappears from `sys.path`, inspect that flag before reinstalling packages.

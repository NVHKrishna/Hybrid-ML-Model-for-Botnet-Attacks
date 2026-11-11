# NetSentinel Complete Testing and Validation Report

**Audit date:** 2026-11-03 (Asia/Kolkata)  
**Project:** NetSentinel – UNSW-NB15 botnet/attack detection  
**Final run inspected:** [`runs/NetSentinel_20261103_113830/`](../runs/NetSentinel_20261103_113830/)  
**Scope:** Read-only inspection and inference/data validation. No model training or source changes were performed. New evidence is isolated under [`reports/testing_audits/NetSentinel_Test_20261103_190808/`](./testing_audits/NetSentinel_Test_20261103_190808/).

## 1. Executive summary

**Verdict: READY FOR A CONTROLLED FINAL DEMONSTRATION; SUBMISSION READINESS IS CONDITIONAL.** The saved run loads, all four base checkpoints and ACLR produce finite probabilities, end-to-end base-to-ACLR inference reproduces the saved predictions exactly, and recomputed held-out metrics agree with the run metrics after rounding. Dataset size/schema, target distribution, partition sizes, saved labels and train-only scaler fit were independently checked.

There are **3 FAIL, 38 PASS and 1 NOT TESTED** checks in the executed audit harness. The three failures are the same input-contract weakness observed for the CNN, LSTM and RNN: direct model calls accepted 41 features. The normal preprocessing path enforces fitted feature columns and thus provides a guard, but model entry points themselves do not reliably enforce the stated 42-feature contract. The existing [`tests/test_pipeline.py`](../tests/test_pipeline.py) was **not run** because it trains models, creates checkpoints and writes/deletes files; this would conflict with the instruction not to retrain or modify project artifacts. The previous 34-check verification log is historical evidence, not a substitute for tests in this audit.

Full-data external K-fold evaluation remains **NOT TESTED**. The five-fold OOF generation recorded in the completed training run is ACLR meta-training input construction; it is not external K-fold model evaluation.

## 2. Environment and project inventory

| Item | Status | Evidence / result |
|---|---|---|
| Python and core ML stack | PASS | Python 3.13.7; TensorFlow 2.21.0; NumPy 2.5.1; pandas 3.0.3; scikit-learn 1.9.1. See [test_results.json](./testing_audits/NetSentinel_Test_20261103_190808/test_results.json). |
| Requirements compatibility | PASS | Installed TensorFlow, scikit-learn, NumPy, pandas, matplotlib, seaborn, joblib, SciPy and PyYAML meet the minimums in [`requirements.txt`](../requirements.txt). |
| Dataset path | PASS | Config selects `dataset/UNSW_NB15_training-set.csv`; file exists. Config: [`config.yaml`](../configs/config.yaml). |
| Run checkpoints | PASS | Run-local `ann_best.keras`, `cnn_best.keras`, `lstm_best.keras`, `rnn_best.keras`, `aclr_meta.keras` all exist and load. Run-local preprocessor exists and loads. |
| Older root model artifacts | PASS with risk | Legacy `*_final.keras` files remain under [`models/`](../models/). The final run uses checkpoints in its own run directory; current training code creates a run-specific output directory by default. Keep using run-local artifact paths to avoid selecting legacy models accidentally. |
| Full run completion | PASS | `run_metadata.json` says `completed`; all configured epochs are recorded complete (ANN 25, CNN 30, LSTM 30, RNN 15, ACLR 25). Training log ends with `Execution complete.` |
| Hardware | Informational | CPU-only; TensorFlow logged that native Windows GPU support is unavailable. This affects runtime, not the observed inference results. |

The source uses project-root-relative resolution in [`train.py`](../src/train.py) and config-relative dataset paths; the inspected source/config did not show a hard-coded absolute project path. The saved run metadata intentionally contains absolute paths for provenance. Old root `models/` and `outputs/` artifacts and the October smoke report are stale relative to the final run, so they should not be presented as final-run results.

## 3. Methodology and safeguards

- Inspected the project tree, source, config, requirements, README and reports; enumerated datasets, checkpoints, run outputs and logs.
- Loaded the configured dataset and reconstructed the configured deterministic 70/30 outer split plus 15% validation carve-out from outer training. Compared the resulting held-out labels with the saved test labels.
- Loaded the saved run preprocessor and all five saved Keras models. Ran inference on validation and held-out test partitions; also tested single/multiple rows and repeated calls. No `.fit`, `fit_oof`, or training command was invoked by this audit.
- Recomputed accuracy, precision, recall, F1, ROC-AUC, average precision (PR-AUC), and confusion matrices from saved `y_test.npy` and each saved test probability array. Binary metrics used threshold 0.5; ROC/PR metrics used continuous probabilities.
- Probed unseen categories, column ordering, missing columns, NaNs, unexpected numeric types, malformed dimensions and nonexistent checkpoint loading. Kept logs and JSON evidence in the timestamped audit folder.
- Did not execute the project test script because it fits four base models and an ACLR model and writes/deletes checkpoints. This is **NOT TESTED**, not a pass.

## 4. Dataset and preprocessing

| Test | Status | Result |
|---|---|---|
| Dataset row/column/feature counts | PASS | 82,332 rows; 45 CSV columns; 42 model features after dropping `id`, `attack_cat`, `label`. |
| Target distribution | PASS | Label 0: 37,000; label 1: 45,332. |
| Missing/duplicate/infinite values | PASS | 0 missing cells; 0 duplicate rows; 0 infinite numeric feature values in the raw configured dataset. |
| Feature leakage columns | PASS | `label`, `attack_cat`, and `id` are excluded from the model feature list. Target is separately returned as `y`. |
| Partition dimensions | PASS | Train 48,987; validation 8,645; test 24,700. The indices are pairwise disjoint. |
| Preprocessor fit isolation | PASS | Reconstructed scaler saw 48,987 samples, equal to the train partition only; saved scaler also records 48,987. Encoders and scaler are fitted in `build_paper_splits()` on `X_train_raw` ([`data_preprocessing.py`](../src/data_preprocessing.py)). |
| Saved split labels | PASS | Saved `y_test.npy` equals the deterministic reconstructed held-out labels element-for-element. |
| Feature ordering | PASS | Reordering input DataFrame columns still yields the fitted model feature order and identical transformed values. |
| Unseen categorical value | PASS | An unseen `service` category maps to a finite reserved code. |
| Invalid raw input handling | PASS | Missing feature column, missing numeric value and unexpected string in a numeric feature were rejected by the saved preprocessor. |

No target or `attack_cat` leakage was found in the model feature list or prediction input flow. Static code inspection shows test predictions are generated after fitting/evaluation on validation in [`train.py`](../src/train.py); test labels are used to calculate evaluation metrics and are not passed into model inference.

## 5. Individual model tests

All saved best checkpoints loaded. Each produced `(24,700,)` test probabilities and `(8,645,)` validation probabilities. Outputs were finite and in `[0,1]`. Each model produced `(1,)` and `(8,)` predictions for single and batch inputs, and repeated inference on the same five rows had zero maximum absolute difference. Current recomputed predictions were elementwise identical to the saved run arrays (`max abs diff = 0.0`).

| Model | Status | Input → output | Test accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ANN | PASS | `(N,42) → (N,1)` | 0.957611 | 0.958507 | 0.964779 | 0.961633 | 0.992915 | 0.994635 |
| CNN | FAIL (input validation) | `(N,42,1) → (N,1)` | 0.956680 | 0.950655 | 0.971765 | 0.961094 | 0.993307 | 0.994971 |
| LSTM | FAIL (input validation) | `(N,42,1) → (N,1)` | 0.958745 | 0.962571 | 0.962500 | 0.962535 | 0.993126 | 0.994832 |
| RNN | FAIL (input validation) | `(N,42,1) → (N,1)` | 0.952429 | 0.958115 | 0.955368 | 0.956739 | 0.991169 | 0.993231 |

The FAIL status above concerns the malformed-dimension probe, not checkpoint loading, valid-shape inference, or metrics. An ANN rejected a 41-feature input; CNN, LSTM and RNN accepted a 41-step sequence when called directly. The preprocessor rejects inputs missing a fitted feature and outputs the correct 42 columns, so this was not observed in the tested end-to-end path. Still, callers bypassing preprocessing could silently submit the wrong feature count. Enforce shape/schema at the inference boundary before treating the project as robust for arbitrary external inputs.

Confusion matrices use `[[TN, FP], [FN, TP]]` at threshold 0.5:

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| ANN | 10,532 | 568 | 479 | 13,121 |
| CNN | 10,414 | 686 | 384 | 13,216 |
| LSTM | 10,591 | 509 | 510 | 13,090 |
| RNN | 10,532 | 568 | 607 | 12,993 |
| ACLR | 10,484 | 616 | 402 | 13,198 |

## 6. ACLR stacking tests

| Test | Status | Result |
|---|---|---|
| Meta-model architecture | PASS | Saved ACLR loads with input `(None,4)` and output `(None,1)`. |
| Base features | PASS | Test base probability matrix has shape `(24,700,4)` and uses ANN/CNN/LSTM/RNN outputs in that order. |
| Validation/test inference | PASS | Full base-to-ACLR run gives validation `(8,645,)`, test `(24,700,)`; test output is finite and bounded in `[0,1]`. |
| Saved ACLR output comparison | PASS | Recomputed stack probabilities equal `prob_aclr_test.npy` exactly (`max abs diff = 0.0`). |
| Single/batch stack call | PASS | Meta model accepts four-feature rows and returns `(1,1)` / `(8,1)`. |
| OOF fold evidence | PASS (training process evidence) | Training log records folds 1–5 and OOF matrix `(48,987,4)`; run metadata reports five folds. |
| OOF isolation review | PASS (source review) | `fit_oof` trains each fold model on `tr_idx` and predicts only its `val_idx`; meta fitting uses the resulting OOF matrix and `y_train`. Validation features are generated separately from `X_val`. See [`stacking.py`](../src/stacking.py). |
| External full-data K-fold evaluation | NOT TESTED | No full-data external K-fold evaluation was run. OOF generation for meta-training is not equivalent to external K-fold evaluation. |

The current run’s training log records the OOF fold stages but does not persist the full OOF matrix as a standalone artifact. Therefore fold-level *data lineage* is supported by source plus execution logs; this audit did not reproduce OOF training because that would retrain models.

## 7. End-to-end behavior

| Scenario | Status | Evidence |
|---|---|---|
| Dataset → saved preprocessor → each base model → ACLR | PASS | Recomputed base probabilities and stack output matched all saved probability arrays exactly. |
| Normal example | PASS | First saved held-out normal example (`y=0`, index 0): predictions were class 0 for ANN/CNN/LSTM/RNN/ACLR; actual probabilities are recorded in [test_results.json](./testing_audits/NetSentinel_Test_20261103_190808/test_results.json) for inference shapes and saved probabilities are available in the run outputs. |
| Attack example | PASS | First saved held-out attack example (`y=1`, index 1): all five models predicted class 1 at threshold 0.5. |
| Single row / batch | PASS | All models accepted single and 8-row batches at the valid 42-feature shape. |
| Repeated predictions | PASS | Identical across repeated inference calls (max difference 0). |
| Invalid feature count | FAIL | ANN rejects 41 features; CNN/LSTM/RNN accept a 41-step input when called directly. |
| Missing feature / NaN / invalid numeric type | PASS | Saved preprocessor raises on missing fitted columns, non-finite transformed output and non-numeric conversion. |
| Missing checkpoint | PASS | Loading a deliberately nonexistent checkpoint raised an error; no file was created. |
| Existing test suite | NOT TESTED | Not run because it retrains and writes files (see above). |

Observed held-out probabilities (normal index 0 / attack index 1; class threshold 0.5): ANN 0.170473 / 1.000000; CNN 0.118898 / 0.999994; LSTM 0.411998 / 1.000000; RNN 0.163432 / 0.999832; ACLR 0.186392 / 0.998839. The labels and probabilities come from saved held-out arrays, not fabricated expectations.

## 8. Security, reliability and source findings

| Category | Status | Finding |
|---|---|---|
| Test-label/model-input leakage | PASS | Inference functions receive features only; test labels are used for metrics. Training flow uses validation for checkpointing and final test evaluation occurs after training. |
| Train/validation/test separation | PASS | Stratified outer split, inner validation carve-out, and pairwise index checks; scaler fit limited to training rows. |
| Stale checkpoint reuse in final run | PASS with operational caution | Final run checkpoints are isolated under its run directory. Old root checkpoints remain and could confuse manual selection; always select the run path explicitly. |
| Unsafe path handling | PASS with caution | Run directory refuses non-empty paths and paths resolve under project root by default. Config or CLI paths are user-controlled local paths; do not point them at valuable existing directories because utilities may create subdirectories. |
| Direct model input validation | FAIL | CNN/LSTM/RNN accept malformed 41-feature sequences; the preprocessor enforces 42 feature columns but model APIs lack consistent guards. |
| Error handling | PASS for tested cases | Missing/malformed data and a nonexistent checkpoint fail with errors rather than silently yielding predictions. A full fault-injection campaign was not run. |
| Dependency compatibility | PASS in current environment | Required modules import and saved Keras artifacts load under installed versions. This does not prove compatibility with every version allowed by broad minimum constraints. |
| Preprocessing/model consistency | PASS | Run-local serialized preprocessor feature order is consistent with 42-feature saved checkpoints and reproduced saved predictions. |
| Deterministic inference | PASS | Repeated calls gave identical outputs in this process/environment. Cross-platform bitwise reproducibility was not tested. |
| External K-fold metrics | NOT TESTED | Not part of the completed run or this audit. |

## 9. Metric verification

Metrics were independently recomputed from the saved ground truth and probability arrays using threshold 0.5 for accuracy/precision/recall/F1 and continuous probabilities for ROC-AUC/average precision. All five models matched the run’s `metrics_test.json` to within `5e-7` absolute difference (saved values are rounded to 6 decimals). The same independent results are stored in [independent_metrics.json](./testing_audits/NetSentinel_Test_20261103_190808/independent_metrics.json).

The older root [`reports/metrics_test.json`](./metrics_test.json) differs slightly from the final run’s metrics because it predates the run-local evidence and has minor metric deltas. For example root ACLR ROC-AUC is 0.993823 versus final run 0.993818; final-run probability arrays reproduce the final run report. Use [`runs/NetSentinel_20261103_113830/reports/metrics_test.json`](../runs/NetSentinel_20261103_113830/reports/metrics_test.json) as the authoritative run-specific metric file.

## 10. Failures and blocked/not-tested items

- **FAIL (3 checks):** CNN, LSTM and RNN accepted a 41-feature input through direct model invocation. This is a contract validation gap. The preprocessor path did reject missing schema and produced 42 features in normal end-to-end testing.
- **NOT TESTED:** Existing `tests/test_pipeline.py`; running it would train models, create/delete checkpoints and mutate project paths, contrary to the explicit no-retraining/no-modification constraint.
- **NOT TESTED:** Full-data external K-fold evaluation; no claim is made that OOF meta-training substitutes for it.
- **NOT TESTED:** Exhaustive malformed-type, resource-exhaustion, concurrency, deployment/API and cross-platform tests. One unexpected numeric type, missing column, NaN, malformed feature count and missing checkpoint were exercised.
- **BLOCKED:** None. No test depended on unavailable external service or missing artifact.

A first invocation of the audit-only harness itself had a metric-call bug (continuous probabilities passed to a binary classification metric); that attempt stopped before completing the checks. The harness was corrected to threshold binary metrics and rerun. It did not affect source or project artifacts; final results in the timestamped evidence are from the corrected run.

## 11. Exact commands executed

Project inventory and dependency inspection:

```powershell
Get-ChildItem -Force
rg --files -g '!runs/**' -g '!dataset/**' -g '!models/**' -g '!outputs/**' -g '!reports/backups/**'
Get-ChildItem -Recurse -File runs\NetSentinel_20261103_113830, models, outputs, dataset, configs, reports
py --version
py -m pip list --format=freeze
```

Read-only inference/data audit (the command created a timestamped folder and executed only the audit harness contained in that folder; the final corrected run is recorded in its `console.log`):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TF_CPP_MIN_LOG_LEVEL='2'
py reports\testing_audits\NetSentinel_Test_20261103_190808\run_audit.py
```

Additional exact commands executed: `py --version`; `py -m pip list --format=freeze`; `py reports\\testing_audits\\NetSentinel_Test_20261103_190808\\run_audit.py`. The saved normal/attack examples were read by a `py -c` NumPy inspection. No pytest, training, smoke-training, source edit, model save, or model deletion command was run.

## 12. Files created and evidence

- This report: [`NetSentinel_Complete_Testing_Report.md`](./NetSentinel_Complete_Testing_Report.md).
- Audit-only script: [`run_audit.py`](./testing_audits/NetSentinel_Test_20261103_190808/run_audit.py).
- Corrected run console: [`console.log`](./testing_audits/NetSentinel_Test_20261103_190808/console.log).
- Test outcomes: [`test_results.json`](./testing_audits/NetSentinel_Test_20261103_190808/test_results.json).
- Independent metrics/confusion matrices: [`independent_metrics.json`](./testing_audits/NetSentinel_Test_20261103_190808/independent_metrics.json).

No source, configuration, dataset, checkpoint, original report, or saved prediction was changed. Temporary inference artifacts were not written into the model/run directories.

## 13. Recommendations and readiness

1. Before accepting arbitrary production inputs, enforce exactly 42 features for each base model and the expected four probability features for ACLR. Re-run the malformed-dimension probes after a source change.
2. Keep run-specific checkpoints, preprocessor, metrics and predictions paired; avoid selecting similarly named legacy root `models/*_final.keras` files.
3. If the submission requires the paper’s external K-fold table, run and document a distinct full-data external K-fold evaluation as a separate experiment. Do not label the existing OOF process as external CV.
4. Run the project’s training-oriented automated test suite only in a disposable copy or after making its output paths fully isolated; it was not run here.
5. Include the final run’s own reports and artifacts, not the October smoke output or stale root metrics, in the submission bundle.

**Readiness verdict:** The saved run is **ready for a controlled demonstration** through its tested preprocessing and run-local inference path. It is **conditionally ready for final submission**: the main numerical results are reproducible from saved artifacts and no test-label input leakage was observed, but the direct model input dimension guard should be fixed and retested, and any required external K-fold result must be clearly marked not performed or completed as a separate study. This audit did not modify source and therefore does not claim the input-validation issue is fixed.



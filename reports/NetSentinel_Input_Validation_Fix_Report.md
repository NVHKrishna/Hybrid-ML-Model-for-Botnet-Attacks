# NetSentinel Input Validation Fix Report

**Date:** 2026-11-03  
**Project:** `D:\Study\Projects\NetSentinel`  
**Scope:** Fix the three input-dimension failures identified in the complete testing audit. No model training or checkpoint writes were performed.

## Summary

The CNN, LSTM, and RNN now reject an invalid feature dimension at their inference boundary with an informative `ValueError`. A regression audit executed **67 checks: 67 PASS, 0 FAIL**. This includes the prior 41-feature failures against newly built models and saved checkpoints, extra features, incorrect tensor ranks/channels, valid inference, output ranges, checkpoint integrity, and end-to-end preprocessing-to-ACLR prediction comparisons.

The existing [`tests/test_pipeline.py`](../tests/test_pipeline.py) was **not run** because it fits ANN/CNN/LSTM/RNN and ACLR models and writes/deletes checkpoint artifacts. Running it would violate the explicit “Do not run model training” and preserve-artifacts instructions. A separate read-only regression harness was executed instead; see evidence below. Accordingly, the project’s training-oriented full test suite remains **NOT RUN**.

## Root cause

The architectures declare input dimensions in their Keras `Input` layers, but those dimensions were not consistently enforced at the model prediction boundary. Keras could accept an alternate sequence length for CNN/LSTM/RNN, while ANN happened to fail later inside a Dense operation. That behavior was inconsistent and could produce a backend error rather than a clear validation error. Loaded checkpoints also do not persist the runtime guard added here.

ACLR has a different interface by design: it consumes **four base-model probabilities**, not 42 raw dataset features. The test therefore validates four-feature ACLR input and rejection of incompatible widths.

## Files changed and backups

| File | Change | Backup |
|---|---|---|
| [`src/models.py`](../src/models.py) | Added a model input dimension guard and applied it to builders and the shared probability prediction helper. | [`models.py`](./backups/input_validation_20261103_194029/src/models.py) |
| [`NetSentinel_Input_Validation_Fix_Report.md`](./NetSentinel_Input_Validation_Fix_Report.md) | This report. | Not applicable. |
| Regression evidence files | New, isolated audit script and results only; no project model/data outputs. | [`NetSentinel_Fix_20261103_194424/`](./input_validation_tests/NetSentinel_Fix_20261103_194424/) |

No dataset, historical results, preprocessor, saved checkpoint, or training report was modified. Only `src/models.py` was changed among the application source files.

## Exact code change

In [`_enforce_input_dimensions`](../src/models.py:44), the guard reads the model’s declared non-batch input shape, checks rank and each known dimension, and raises `ValueError` that reports expected and received shapes. It wraps both the model call and `predict` methods. The shared [`predict_proba`](../src/models.py:219) also validates the input before inference and installs the runtime guard on a loaded checkpoint, so subsequent direct Keras `predict` calls use the same validation.

The ANN/CNN/LSTM/RNN layers, weights, optimizer configuration and preprocessing behavior were not changed. Guard installation is runtime-only and does not save or rewrite checkpoints.

## Regression results

Executed [`regression_audit.py`](./input_validation_tests/NetSentinel_Fix_20261103_194424/regression_audit.py) against both newly built models and the saved run checkpoints. Full output is in [`console.log`](./input_validation_tests/NetSentinel_Fix_20261103_194424/console.log); machine-readable evidence is in [`results.json`](./input_validation_tests/NetSentinel_Fix_20261103_194424/results.json).

| Test category | Status | Result |
|---|---|---|
| ANN/CNN/LSTM/RNN valid 42-feature inference | PASS | Correct output shape; values finite and in `[0,1]`. Sequence models received `(N,42,1)` after reshaping. |
| Prior failure: 41 features | PASS | All four base models reject malformed input with `ValueError`; prior CNN/LSTM/RNN failures resolved. |
| Extra feature (43) | PASS | All four base models reject the input with `ValueError`. |
| Incorrect ranks/channels | PASS | Rank-1 and malformed rank/channel inputs are rejected. |
| Saved checkpoint loads and inference | PASS | All five run-local checkpoints loaded successfully; base outputs and ACLR output have the expected shapes. |
| ACLR input width | PASS | Four base-probability columns accepted; 41/5 columns and malformed ranks rejected. ACLR is not a 42-raw-feature model. |
| Prediction probabilities | PASS | Outputs were finite and remained within `[0,1]`. |
| Full held-out inference outputs vs saved arrays | PASS | ANN/CNN/LSTM maximum absolute difference `0.0`; RNN difference `2.2054e-6`; ACLR difference `3.5763e-7`. These small differences are within the declared `1e-5` tolerance and reflect inference floating-point execution. |
| Checkpoint integrity | PASS | SHA-256 hashes for all five checkpoint files were unchanged before and after the audit. |
| Python syntax compilation | PASS | `py -m py_compile src\models.py` completed successfully. |
| Existing training-oriented automated suite | NOT RUN | It performs model fitting and mutates temporary checkpoint paths; not run to honor the no-training/no-overwrite constraints. |

**Before/after for the three failures:** before, CNN/LSTM/RNN accepted a 41-step input in the audit probe; after, all three raise `ValueError` reporting expected `(batch, 42, 1)` and the received dimensions. All 67 assertions in the completed regression audit passed. Two earlier exploratory harness runs (folders `NetSentinel_Fix_20261103_194111` and `NetSentinel_Fix_20261103_194256`) exposed that an initial Keras `InputSpec`-based attempt did not guard `predict()` consistently and that a tolerance was too strict. That approach was replaced with the explicit call/predict guard; the final completed run is the authoritative result.

## Exact commands executed

The regression harness was created in a new timestamped test-output folder and run with:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:TF_CPP_MIN_LOG_LEVEL='3'
py reports\input_validation_tests\NetSentinel_Fix_20261103_194424\regression_audit.py
py -m py_compile src\models.py
```

The regression harness only loads checkpoints, performs inference, reconstructs/transforms data in memory, checks shapes and hashes, and writes its own files under the timestamped audit folder. It does not call model `fit`, OOF training, or `save`.

## Remaining issues

- The existing automated suite is still not run in this audit because it trains models. Its previously recorded 34 passing checks remain historical evidence only.
- Full-data external K-fold evaluation remains unperformed; it is distinct from ACLR’s OOF meta-training.
- Inference may differ from persisted arrays by a few millionths across executions because of floating-point kernels; all measured differences were below `1e-5`.
- ACLR expects four probability inputs, not raw 42-feature rows; callers should use the full `ACLRStacker` path for raw model features.

## Conclusion

**All three reported input-dimension failures are resolved and regression-tested.** Valid input behavior remained consistent with the saved run’s predictions within the stated numerical tolerance, and all saved checkpoint hashes remained unchanged. No models were retrained. The full existing automated suite remains **NOT RUN** to comply with the no-training instruction.


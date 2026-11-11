# NetSentinel Progress Recovery Report

**Inspection date:** 2026-11-03  
**Scope:** Read-only inspection of source, reports, logs, saved models, outputs, and tests. No training or test command was run, and no existing project artifact was modified.

## Executive findings

| Question | Finding |
|---|---|
| Was test-set leakage fixed? | **Yes in the current Python pipeline.** Base checkpoints monitor the dedicated validation partition; final test inference occurs after training. The archived October 29 preliminary smoke run and older audit report describe the pre-fix leak and must be treated as historical. |
| Are train, validation, and test properly separated? | **Yes in current code.** It constructs a stratified outer 70/30 split and splits only the outer training portion again for validation, checks disjoint indices, and fits preprocessing on train only. |
| Does ACLR use OOF predictions correctly? | **Yes for meta-training.** Stratified K-fold OOF predictions are generated only over X_train, with each fold's model trained on fold-train and predicting its held-out fold. The meta-model is trained on these OOF features and may monitor validation predictions. |
| Are ANN/CNN/LSTM/RNN complete? | **Yes, implemented and represented by saved checkpoints and training-history plots.** The architecture builders and training/inference functions are present. |
| Did smoke tests pass? | **A smoke execution is evidenced by archived 1,000-sample metrics and the older audit's explicit “SMOKE TEST COMPLETED” status.** These artifacts are from the pre-fix leakage run. The current test script contains 2-epoch checks, but no raw test-run log or captured exit status is present to independently confirm a post-fix smoke run. |
| Was full training started/completed? | **Full corrected pipeline appears completed through held-out test evaluation.** Final probability/label arrays contain 24,700 samples each; full test metrics and test plots were saved on Oct 31, 2026. Training-history plots and current model checkpoints also exist. No raw execution log or history objects were found, so exact epoch counts and process exit status cannot be proven from the available files. |
| Last completed task | **The strongest evidence points to final full-size test evaluation and report/output writing, completed Oct 31, 2026 at 16:58:47.** This is the timestamp of the test metrics, prediction arrays, and final test plots. There is no later execution log to identify a subsequent task or prove the run's final console exit. |

## 1. Leakage correction and split integrity

- [src/train.py](../src/train.py#L151) labels base-model training as validation on X_val. The call passes X_val/y_val to train_model at lines 164–175; validation predictions and metrics are computed on X_val at lines 178–185.
- [src/train.py](../src/train.py#L198) gives ACLR only X_train/y_train for OOF generation. Final base models receive X_val/y_val as validation inputs (lines 208–217), and ACLR meta-training receives X_val/y_val (lines 219–228).
- The held-out test set is used only in the final inference/evaluation section of [src/train.py](../src/train.py#L244): base-model probabilities are generated on X_test at lines 252–263 and ACLR test probabilities at lines 265–272. Reports and prediction arrays are written afterward at lines 274–291.
- [src/data_preprocessing.py](../src/data_preprocessing.py#L180) creates a stratified outer train/test split. Lines 197–212 split outer training into train and validation. Lines 214–217 assert all three index intersections are empty. Lines 224–229 fit the preprocessor on X_train_raw only and transform validation/test with that fitted object.
- [src/models.py](../src/models.py#L132) documents the train_model contract: held-out test data must not be supplied; the checkpoint monitors validation loss when validation data is present. Its implementation builds validation_data from X_val/y_val (lines 158–171).
- [tests/test_pipeline.py](../tests/test_pipeline.py#L61) contains explicit partition-size, disjointness, and train-only scaler-fit checks (lines 77–94); it also trains with X_val_t, not X_test_t (lines 129–164).
- Historical evidence differs: [reports/preliminary_smoke_results/README.md](./preliminary_smoke_results/README.md#L1) says those October 29 smoke files predate the leakage correction and records the original X_val=X_test behavior. [reports/netsentinel_audit_report.md](./netsentinel_audit_report.md#L224) also asserts the old leak at train.py lines 159–160. Its line references and conclusions are stale relative to current source; its smoke-specific findings remain relevant only to those archived artifacts.

**Conclusion:** The current source fixes the checkpoint-selection leakage and maintains distinct train/validation/test partitions. The archived initial smoke metrics are not a clean evaluation under that protocol.

## 2. ACLR out-of-fold stacking

- [src/stacking.py](../src/stacking.py#L93) builds a StratifiedKFold over X_train/y_train (lines 105–109), allocates one prediction slot per training row and base model (lines 110–112), and for each fold trains on tr_idx then predicts val_idx (lines 116–138). Every training row receives a prediction from a model that did not train on that row.
- [src/stacking.py](../src/stacking.py#L194) requires generated OOF features before fit_meta and fits the meta-model on self.oof_preds_ and y_train (lines 211–239). Optional validation features are generated separately using base_predict(X_val), not used as meta-training rows.
- [src/train.py](../src/train.py#L188) invokes OOF generation with X_train/y_train, then fits the meta-model with X_val/y_val (lines 198–228). Test predictions are only requested in final evaluation (lines 265–268).

**Conclusion:** OOF use is structurally appropriate for leakage-free meta-training. One caveat: when final base checkpoints already exist, load_or_fit_final_base loads them (stacking.py lines 160–165); that is fine for producing validation/test predictions, while meta-training still uses the separately generated OOF matrix.

## 3. Model implementation status

[src/models.py](../src/models.py#L44) defines all four base models and a shared compile routine:

- **ANN:** Dense 64 → Dense 32 → sigmoid output, lines 46–60.
- **CNN:** two Conv1D/MaxPooling blocks, Flatten, sigmoid output, lines 65–84.
- **LSTM:** LSTM(64), Dense(32), sigmoid output, lines 89–102.
- **RNN:** SimpleRNN(64), sigmoid output, lines 107–118.
- Shared train_model and predict_proba utilities are implemented at lines 132–190.

Saved artifacts include [ann_best.keras](../models/ann_best.keras), [cnn_best.keras](../models/cnn_best.keras), [lstm_best.keras](../models/lstm_best.keras), and [rnn_best.keras](../models/rnn_best.keras), with history plots under [outputs/plots/](../outputs/plots). Older *_final.keras files have Oct 29 timestamps and are legacy artifacts; do not assume they correspond to the corrected run. The *_best.keras files are dated Oct 31.

**Conclusion:** The four implementations are complete in source and have saved trained checkpoints. The artifact inventory alone cannot prove their exact epoch counts.

## 4. Smoke-test evidence

- [src/train.py](../src/train.py#L45) defines smoke sizes (3,000 train, 600 validation, 1,000 test), 2 epochs, and 2 OOF folds; smoke mode applies these settings at lines 110–117 and 129–142.
- [tests/test_pipeline.py](../tests/test_pipeline.py#L129) includes a two-epoch validation-training check for each base model, and lines 180–200 check ACLR shape/OOF generation.
- Archived [preliminary smoke metrics](./preliminary_smoke_results/metrics_test.json) and [CSV](./preliminary_smoke_results/test_results.csv) contain results for five models; the audit report [records model smoke completion](./netsentinel_audit_report.md#L180). The archive README explicitly identifies the data as pre-correction and warns against treating it as final.
- No standalone console log or test runner output with a timestamp/exit code was found. Therefore a **pre-correction smoke run is evidenced as completed**, while a **post-correction smoke pass is not independently established** from saved evidence.

## 5. Full-run status and most recent work

Evidence for a completed full-size corrected evaluation:

- [reports/metrics_test.json](./metrics_test.json) and [reports/test_results.csv](./test_results.csv) were last written **2026-10-31 16:58:47**.
- The six arrays [outputs/y_test.npy](../outputs/y_test.npy) and [outputs/prob_*_test.npy](../outputs) each have shape **(24,700,)** and dtype float32, matching the configured held-out test partition size (config.yaml test_split 0.30 and test assertions at test_pipeline.py lines 77–80).
- Current test metrics report ACLR accuracy 0.958785, ROC-AUC 0.993823, and PR-AUC 0.995367 in [reports/metrics_test.json](./metrics_test.json). Test reports and arrays are written by the current pipeline at train.py lines 279–291.
- Current model checkpoints were last modified Oct 31 between 15:58 and 16:09; the current preprocessor was saved at 15:57. Training-history plots are dated 15:58–16:09; final test confusion matrices, ROC/PR plots, metrics, and arrays are dated 16:58.
- [reports/metrics_val.json](./metrics_val.json) and [reports/val_results.csv](./val_results.csv) are dated 16:58:47 and 16:58:27, respectively; they hold validation metrics for all five models.

The source's paper mode uses configured ANN/CNN/LSTM/RNN epochs of 25/30/30/15 and ACLR 25, with 5 OOF folds ([configs/config.yaml](../configs/config.yaml#L35), [src/train.py](../src/train.py#L125)). Artifacts support that the corrected full pipeline reached final test reporting, but absent run logs mean the exact invocation, observed epoch counts, and clean exit status cannot be authenticated.

## 6. Recovery point

The last completed task supported by timestamps and artifact contents is **full-size held-out evaluation plus serialization of final metrics, labels/probabilities, and plots on Oct 31, 2026 at 16:58:47**. The repository contains no execution logs to show a later task or an interrupted command. The older audit report's claim that full training had not run was written Oct 29, before the Oct 31 artifacts and is superseded by them.

**Inspection boundary:** No source, saved model, dataset, log, test file, or existing report was edited during this inspection. Only this requested report is being added.


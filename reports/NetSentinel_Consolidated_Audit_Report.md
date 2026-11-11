# NetSentinel Consolidated Audit Report

**Date:** 2026-11-03  
**Audit scope:** Source modules, config, tests, both notebooks, datasets, saved checkpoints, reports, predictions, plots, environment and disk. Corrections were backed up before editing. Existing root-level checkpoints/results were preserved.

## Overall status

The current source separates fit, validation and held-out test data; preprocessing fits on training rows only; base model checkpoint selection uses validation data; and ACLR meta-training receives out-of-fold predictions from training data. Dataset-loader and fold-isolation defects were corrected, and run outputs now go to a fresh timestamped directory.

The full paper-mode run completed successfully on CPU (exit code 0). All four base models completed their configured epochs, five OOF folds generated a (48,987, 4) feature matrix, ACLR completed 25 epochs, and held-out evaluation completed on 24,700 samples. The full results and metric reconciliation are in the [final training report](./NetSentinel_Final_Training_Report.md).

## Dataset and preprocessing

The configured paper dataset, [UNSW_NB15_training-set.csv](../dataset/UNSW_NB15_training-set.csv), has 82,332 rows, binary target label counts of 37,000 normal and 45,332 attack, 45 raw columns, and 42 input features after dropping id, attack_cat and label. There are no raw missing values. The split builder creates a stratified 70/30 outer split (57,632/24,700), then carves 15% of the outer training portion (8,645 rows) for validation, leaving 48,987 model-fit rows. This internal validation split is an extra methodological step beyond the paper’s reported 70/30 train/test division; it supports checkpoint selection without using test labels.

Current implementation evidence:

- [load_dataset](../src/data_preprocessing.py#L40) validates candidate row counts and schema, and fails closed if neither expected CSV has 82,332 rows.
- [PaperPreprocessor](../src/data_preprocessing.py#L86) preserves training feature order, fits LabelEncoders and StandardScaler using only the training partition, and maps unseen categories to a reserved code instead of collapsing them into the first known category.
- [build_paper_splits](../src/data_preprocessing.py#L175) creates stratified disjoint partitions, asserts index non-overlap, fits preprocessing on X_train only, and transforms validation and test with that fitted object.
- [tests/test_pipeline.py](../tests/test_pipeline.py#L61) checks dataset sizes, class counts, disjointness, train-only scaler fitting, finite outputs, unknown-category handling and feature ordering.

## Model and training audit

All four architectures are implemented in [models.py](../src/models.py): ANN (64→32→1), CNN (Conv1D 64 → pool → Conv1D 32 → pool → flatten → 1), LSTM(64)→Dense(32)→1, and SimpleRNN(64)→1. The models use Adam at 0.001, binary crossentropy and batch size 32. Configured epochs are ANN 25, CNN 30, LSTM 30, RNN 15, and ACLR meta 25 ([config.yaml](../configs/config.yaml#L35)).

The current pipeline sends the internal X_val/y_val to base checkpoint selection and ACLR validation; X_test is used in the final evaluation path only ([train.py](../src/train.py#L164), [train.py](../src/train.py#L244)). Best checkpoints are reloaded for inference ([models.py](../src/models.py#L123)). The run-directory guard refuses a non-empty output directory, preventing silent reuse or overwrite of old project checkpoints ([train.py](../src/train.py#L73)).

## ACLR stacking audit

[fit_oof](../src/stacking.py#L93) uses StratifiedKFold over the model-fit training partition. For each fold and each base model, it trains on the fold’s in-sample indices and writes predictions only for that fold’s excluded indices. ACLR meta-training then uses the OOF matrix and corresponding training labels; optional validation features are generated separately from X_val ([fit_meta](../src/stacking.py#L207)). The held-out test set is not used in OOF generation or meta fitting.

The optional external CV helper now reserves an inner validation subset inside each outer training fold, isolates each fold’s checkpoints in its own temporary directory, restores saved best models, and evaluates only on that fold’s held-out rows ([run_kfold_cv](../src/stacking.py#L289)). Full-run OOF stacking uses the configured five folds; external 5-fold evaluation is separately optional via --kfold.

## Issues found and disposition

| Severity | Finding | Disposition |
|---|---|---|
| Critical | Loader previously returned an unintended row count when both candidate files missed the expected size. | Fixed: strict size/schema checks and safe failure. |
| High | External CV previously could use its held-out fold as validation data, contaminating fold evaluation. | Fixed: inner validation split is drawn only from the outer fold’s training data. |
| High | Shared/default checkpoint folders risked stale model reuse across runs/folds. | Fixed: run-specific outputs, per-fold temporary checkpoint directories, checkpoint shape checks. |
| Medium | ACLR final-base training retained last-epoch weights in memory after ModelCheckpoint selected a best epoch. | Fixed: reload best checkpoint after fitting. |
| Medium | Unseen categorical values silently mapped to the first known category. | Fixed: reserve a documented unknown integer code. |
| Low | Existing notebooks contain stale error outputs and Kaggle-specific paths. | Left as historical artifacts; the Python pipeline is the maintained entry point. Main notebook and backup are byte-identical, and their saved outputs include three errors (missing id, ndarray.to_numpy, uncompiled model). |
| Low | TensorFlow has no GPU on this native Windows installation. | Full run uses CPU; no architecture or hyperparameter changes were made. |

## Changed files and reasons

Pre-change copies of these files are preserved in [reports/backups/20261103_113411/](./backups/20261103_113411/).

| File | Change |
|---|---|
| [src/data_preprocessing.py](../src/data_preprocessing.py) | Fail closed on wrong dataset sizes/schema; reject non-binary labels and row-count changes; use an explicit reserved code for unseen categories and retain feature-order checks. |
| [src/stacking.py](../src/stacking.py) | Reload best trained checkpoints, reject incompatible shapes and stale ACLR meta checkpoints, and isolate optional external-CV folds with a separate validation subset. |
| [src/train.py](../src/train.py) | Write each run to a fresh timestamped directory; snapshot config; save logs, histories, validation metrics and comparison plots; record run metadata and optionally execute external CV. |
| [src/utils.py](../src/utils.py) | Add optional UTF-8 file logging for experiment logs. |
| [src/evaluate.py](../src/evaluate.py) | Add held-out test accuracy and precision/recall/F1 comparison charts. |
| [tests/test_pipeline.py](../tests/test_pipeline.py) | Keep test-generated artifacts in a temporary directory; cover dataset size rejection, unknown categories, feature order and best-checkpoint reload. |
| [README.md](../README.md) | Document corrected execution commands, timestamped outputs, current verified run, and the added validation split. |

## Verification and preserved artifacts

- Syntax compilation passed for all source modules and the test suite after corrections.
- Automated checks: **34 passed, 0 failed**. Tests wrote checkpoints only under a system temporary directory and removed them at exit.
- End-to-end smoke mode: **exit code 0**, all four bases plus ACLR trained/evaluated on the configured smoke subsets. Run directory and console output are in the OS temporary folder.
- A separate synthetic two-fold CV smoke check passed and exercised isolated fold directories and inner validation.
- The full training completed under [runs/NetSentinel_20261103_113830/](../runs/NetSentinel_20261103_113830/). Test predictions all had 24,700 finite entries; metrics independently recomputed from predictions matched the saved JSON within rounding tolerance; all five checkpoints and the preprocessor loaded successfully.
- Prior root-level models, reports and predictions were not overwritten.
- Pre-change copies of modified files are in [reports/backups/20261103_113411/](./backups/20261103_113411/).

## Base paper reference

Ali et al., “Hybrid Machine Learning Model for Efficient Botnet Attack Detection in IoT Environment,” IEEE Access, vol. 12, pp. 40682–40699, 2024, DOI: [10.1109/ACCESS.2024.3376400](https://doi.org/10.1109/ACCESS.2024.3376400). The article describes use of the 82,332-record UNSW-NB15 training CSV and a 70/30 split; it reports accuracies ANN 0.7568, CNN 0.9440, LSTM 0.9651, RNN 0.9522, ACLR 0.9698, and five-fold ACLR accuracy 0.9749. See the separate [base paper comparison](./NetSentinel_Base_Paper_Comparison.md) for per-model metrics and methodological differences.


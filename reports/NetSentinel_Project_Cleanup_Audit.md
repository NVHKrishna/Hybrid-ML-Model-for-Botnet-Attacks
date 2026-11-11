# NetSentinel Project Cleanup and Folder Organization Audit

**Project:** `D:\Study\Projects\NetSentinel`  
**Audit mode:** Read-only. No existing files were moved, renamed, deleted or overwritten. This requested report is the only file created for this audit. No source code was changed and no models were retrained.

## Executive summary

The existing layout is serviceable: source, configs, datasets, tests, reports, and a self-contained timestamped final run. Do not perform a wholesale restructure. Preserve the final run bundle and archive historical files only after explicit approval and reference updates.

| Measure | Result |
|---|---:|
| Files inventoried recursively (report excluded from its own snapshot) | 141 |
| Total size at inventory time | 655,123,724 bytes (~624.56 MiB) |
| Directories | 32 |
| Empty directories | 0 |
| Exact duplicate SHA-256 groups | 14 groups; 7,305,577 redundant bytes if one copy per group is retained |
| Python bytecode caches | 8 files; 92,573 bytes |
| Potential savings | Up to 7,398,150 bytes (~7.06 MiB) from verified exact duplicates and regenerable caches, only if approved and removed. No archive size is counted as reclaimed space. |

A duplicate is not by itself a deletion instruction. Exact config and preprocessor copies preserve backup/run provenance. File references below are lexical filename/path matches in source and documentation, not proof of active runtime use.

## Complete inventory summary by type

| Type | Files | Bytes |
|---|---:|---:|
| `.csv` | 13 | 634,061,527 |
| `.ipynb` | 2 | 327,132 |
| `.joblib` | 2 | 8,002 |
| `.json` | 16 | 157,140 |
| `.keras` | 14 | 3,285,147 |
| `.log` | 9 | 13,494,626 |
| `.md` | 10 | 96,984 |
| `.npy` | 12 | 1,187,136 |
| `.pdf` | 1 | 290,858 |
| `.png` | 30 | 1,428,895 |
| `.py` | 19 | 157,034 |
| `.pyc` | 8 | 92,573 |
| `.txt` | 1 | 201 |
| `.yaml` | 3 | 11,490 |
| `.zip` | 1 | 524,979 |
| **Total** | **141** | **655,123,724** |

The detailed 141-file manifest follows. Paths are absolute; times are local (Asia/Calcutta); sizes are bytes. “Inference required” is for the current run-local inference path. “Reproducibility” indicates value for reconstructing or verifying results.

| Full path | Type | Bytes | Last modified | Source/docs reference | Inference required | Reproducibility | Classification / recommended action |
|---|---|---:|---|---|---|---|---|
| `D:\Study\Projects\NetSentinel\netsentinel.ipynb` | `.ipynb` | 163566 | 2026-10-29 10:28:54 | Yes (text match) | No | Yes | **ARCHIVE** — Retain historical original in notebooks/archive if approved. |
| `D:\Study\Projects\NetSentinel\netsentinel_BACKUP.ipynb` | `.ipynb` | 163566 | 2026-10-29 10:28:54 | Yes (text match) | No | Yes | **ARCHIVE** — Byte-identical backup; archive, do not delete without approval. |
| `D:\Study\Projects\NetSentinel\README.md` | `.md` | 7821 | 2026-11-03 13:09:39 | Yes (text match) | No | Yes | **KEEP** — Preserve project entrypoint/dependencies. |
| `D:\Study\Projects\NetSentinel\requirements.txt` | `.txt` | 201 | 2026-10-29 10:54:03 | Yes (text match) | No | Yes | **KEEP** — Preserve project entrypoint/dependencies. |
| `D:\Study\Projects\NetSentinel\configs\config.yaml` | `.yaml` | 3830 | 2026-10-29 12:02:03 | Yes (text match) | No | Yes | **KEEP** — Preserve config and run snapshots. |
| `D:\Study\Projects\NetSentinel\dataset\NUSW-NB15_features.csv` | `.csv` | 4044 | 2026-10-29 10:30:47 | Yes (text match) | No | Yes | **KEEP** — Preserve datasets and schema/provenance files. |
| `D:\Study\Projects\NetSentinel\dataset\UNSW-NB15_1.csv` | `.csv` | 168979718 | 2026-10-29 10:30:50 | Yes (text match) | No | Yes | **KEEP** — Preserve datasets and schema/provenance files. |
| `D:\Study\Projects\NetSentinel\dataset\UNSW-NB15_2.csv` | `.csv` | 165221021 | 2026-10-29 10:30:56 | Yes (text match) | No | Yes | **KEEP** — Preserve datasets and schema/provenance files. |
| `D:\Study\Projects\NetSentinel\dataset\UNSW-NB15_3.csv` | `.csv` | 154588103 | 2026-10-29 10:31:04 | Yes (text match) | No | Yes | **KEEP** — Preserve datasets and schema/provenance files. |
| `D:\Study\Projects\NetSentinel\dataset\UNSW-NB15_4.csv` | `.csv` | 97588754 | 2026-10-29 10:31:07 | Yes (text match) | No | Yes | **KEEP** — Preserve datasets and schema/provenance files. |
| `D:\Study\Projects\NetSentinel\dataset\UNSW-NB15_LIST_EVENTS.csv` | `.csv` | 4639 | 2026-10-29 10:31:08 | Yes (text match) | No | Yes | **KEEP** — Preserve datasets and schema/provenance files. |
| `D:\Study\Projects\NetSentinel\dataset\UNSW_NB15_testing-set.csv` | `.csv` | 32293018 | 2026-10-29 10:31:09 | Yes (text match) | No | Yes | **KEEP** — Preserve datasets and schema/provenance files. |
| `D:\Study\Projects\NetSentinel\dataset\UNSW_NB15_training-set.csv` | `.csv` | 15380800 | 2026-10-29 10:31:09 | Yes (text match) | No | Yes | **KEEP** — Preserve datasets and schema/provenance files. |
| `D:\Study\Projects\NetSentinel\models\aclr_meta.keras` | `.keras` | 55508 | 2026-10-31 16:58:22 | Yes (text match) | Legacy only | Yes | **ARCHIVE** — Archive only after confirming legacy references; keep final run models in place. |
| `D:\Study\Projects\NetSentinel\models\ann_best.keras` | `.keras` | 84686 | 2026-10-31 15:58:18 | Yes (text match) | Legacy only | Yes | **ARCHIVE** — Archive only after confirming legacy references; keep final run models in place. |
| `D:\Study\Projects\NetSentinel\models\ann_final.keras` | `.keras` | 1155172 | 2026-10-29 11:01:27 | Yes (text match) | Legacy only | Yes | **ARCHIVE** — Archive only after confirming legacy references; keep final run models in place. |
| `D:\Study\Projects\NetSentinel\models\cnn_best.keras` | `.keras` | 114306 | 2026-10-31 16:00:15 | Yes (text match) | Legacy only | Yes | **ARCHIVE** — Archive only after confirming legacy references; keep final run models in place. |
| `D:\Study\Projects\NetSentinel\models\cnn_final.keras` | `.keras` | 440703 | 2026-10-29 11:01:32 | Yes (text match) | Legacy only | Yes | **ARCHIVE** — Archive only after confirming legacy references; keep final run models in place. |
| `D:\Study\Projects\NetSentinel\models\lstm_best.keras` | `.keras` | 258016 | 2026-10-31 16:07:52 | Yes (text match) | Legacy only | Yes | **ARCHIVE** — Archive only after confirming legacy references; keep final run models in place. |
| `D:\Study\Projects\NetSentinel\models\lstm_final.keras` | `.keras` | 387727 | 2026-10-29 11:01:57 | Yes (text match) | Legacy only | Yes | **ARCHIVE** — Archive only after confirming legacy references; keep final run models in place. |
| `D:\Study\Projects\NetSentinel\models\preprocessor.joblib` | `.joblib` | 4001 | 2026-10-31 15:57:29 | Yes (text match) | Legacy only | Yes | **ARCHIVE** — Archive only after confirming legacy references; keep final run models in place. |
| `D:\Study\Projects\NetSentinel\models\rnn_best.keras` | `.keras` | 76374 | 2026-10-31 16:09:13 | Yes (text match) | Legacy only | Yes | **ARCHIVE** — Archive only after confirming legacy references; keep final run models in place. |
| `D:\Study\Projects\NetSentinel\models\rnn_final.keras` | `.keras` | 123765 | 2026-10-29 11:02:09 | Yes (text match) | Legacy only | Yes | **ARCHIVE** — Archive only after confirming legacy references; keep final run models in place. |
| `D:\Study\Projects\NetSentinel\outputs\prob_aclr_test.npy` | `.npy` | 98928 | 2026-10-31 16:58:47 | Yes (text match) | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\prob_ann_test.npy` | `.npy` | 98928 | 2026-10-31 16:58:47 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\prob_cnn_test.npy` | `.npy` | 98928 | 2026-10-31 16:58:47 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\prob_lstm_test.npy` | `.npy` | 98928 | 2026-10-31 16:58:47 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\prob_rnn_test.npy` | `.npy` | 98928 | 2026-10-31 16:58:47 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\y_test.npy` | `.npy` | 98928 | 2026-10-31 16:58:47 | Yes (text match) | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\cm_aclr.png` | `.png` | 27953 | 2026-10-29 11:02:19 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\cm_aclr_test.png` | `.png` | 30539 | 2026-10-31 16:58:47 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\cm_ann.png` | `.png` | 27314 | 2026-10-29 11:10:42 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\cm_ann_test.png` | `.png` | 29591 | 2026-10-31 16:58:29 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\cm_cnn.png` | `.png` | 27334 | 2026-10-29 11:10:51 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\cm_cnn_test.png` | `.png` | 29300 | 2026-10-31 16:58:31 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\cm_lstm.png` | `.png` | 30443 | 2026-10-29 10:59:44 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\cm_lstm_test.png` | `.png` | 30038 | 2026-10-31 16:58:35 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\cm_rnn.png` | `.png` | 29784 | 2026-10-29 10:59:57 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\cm_rnn_test.png` | `.png` | 30624 | 2026-10-31 16:58:37 | No match | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\history_ann.png` | `.png` | 75463 | 2026-10-31 15:58:23 | Yes (text match) | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\history_cnn.png` | `.png` | 73218 | 2026-10-31 16:00:32 | Yes (text match) | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\history_lstm.png` | `.png` | 81245 | 2026-10-31 16:07:54 | Yes (text match) | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\history_rnn.png` | `.png` | 85353 | 2026-10-31 16:09:25 | Yes (text match) | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\pr_curves.png` | `.png` | 46571 | 2026-10-31 16:58:47 | Yes (text match) | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\outputs\plots\roc_curves.png` | `.png` | 63142 | 2026-10-31 16:58:47 | Yes (text match) | No | Yes | **ARCHIVE** — Archive as prior root-level outputs; some are duplicates, others historical. |
| `D:\Study\Projects\NetSentinel\reports\metrics_test.json` | `.json` | 847 | 2026-10-31 16:58:47 | Yes (text match) | No | Yes | **ARCHIVE** — Preserve older root-level result; current metrics are run-local. |
| `D:\Study\Projects\NetSentinel\reports\metrics_val.json` | `.json` | 826 | 2026-10-29 11:02:33 | Yes (text match) | No | Yes | **ARCHIVE** — Preserve older root-level result; current metrics are run-local. |
| `D:\Study\Projects\NetSentinel\reports\NetSentinel_Audit_Evidence.zip` | `.zip` | 524979 | 2026-10-29 11:40:48 | No match | No | Yes | **ARCHIVE** — Keep as dated historical evidence; mark as superseded. |
| `D:\Study\Projects\NetSentinel\reports\netsentinel_audit_report.md` | `.md` | 23962 | 2026-10-29 11:38:16 | Yes (text match) | No | Yes | **ARCHIVE** — Keep as dated historical evidence; mark as superseded. |
| `D:\Study\Projects\NetSentinel\reports\NetSentinel_Base_Paper_Comparison.md` | `.md` | 4489 | 2026-11-03 13:08:59 | Yes (text match) | No | Yes | **KEEP** — Preserve current final report. |
| `D:\Study\Projects\NetSentinel\reports\NetSentinel_Complete_Testing_Report.md` | `.md` | 20011 | 2026-11-03 19:12:35 | Yes (text match) | No | Yes | **KEEP** — Preserve current final report. |
| `D:\Study\Projects\NetSentinel\reports\NetSentinel_Consolidated_Audit_Report.md` | `.md` | 8907 | 2026-11-03 13:10:13 | Yes (text match) | No | Yes | **KEEP** — Preserve current final report. |
| `D:\Study\Projects\NetSentinel\reports\NetSentinel_Final_Training_Report.md` | `.md` | 7584 | 2026-11-03 13:10:30 | Yes (text match) | No | Yes | **KEEP** — Preserve current final report. |
| `D:\Study\Projects\NetSentinel\reports\NetSentinel_Implementation_Audit.pdf` | `.pdf` | 290858 | 2026-10-29 11:40:48 | No match | No | Yes | **ARCHIVE** — Keep as dated historical evidence; mark as superseded. |
| `D:\Study\Projects\NetSentinel\reports\NetSentinel_Input_Validation_Fix_Report.md` | `.md` | 7571 | 2026-11-03 19:45:53 | Yes (text match) | No | Yes | **KEEP** — Preserve current final report. |
| `D:\Study\Projects\NetSentinel\reports\NetSentinel_PostTraining_Verification.log` | `.log` | 24608 | 2026-11-03 13:06:26 | No match | No | Yes | **ARCHIVE** — Keep as dated historical evidence; mark as superseded. |
| `D:\Study\Projects\NetSentinel\reports\NetSentinel_PreTraining_Verification.log` | `.log` | 24628 | 2026-11-03 11:36:33 | No match | No | Yes | **ARCHIVE** — Keep as dated historical evidence; mark as superseded. |
| `D:\Study\Projects\NetSentinel\reports\NetSentinel_Progress_Recovery_Report.md` | `.md` | 10499 | 2026-11-03 11:15:35 | No match | No | Yes | **ARCHIVE** — Keep as dated historical evidence; mark as superseded. |
| `D:\Study\Projects\NetSentinel\reports\test_results.csv` | `.csv` | 288 | 2026-10-31 16:58:47 | Yes (text match) | No | Yes | **ARCHIVE** — Preserve older root-level result; current metrics are run-local. |
| `D:\Study\Projects\NetSentinel\reports\val_results.csv` | `.csv` | 286 | 2026-10-31 16:58:27 | Yes (text match) | No | Yes | **ARCHIVE** — Preserve older root-level result; current metrics are run-local. |
| `D:\Study\Projects\NetSentinel\reports\backups\20261103_113411\README.md` | `.md` | 5701 | 2026-10-29 11:19:39 | Yes (text match) | No | Yes | **KEEP** — Preserve rollback provenance copies. |
| `D:\Study\Projects\NetSentinel\reports\backups\20261103_113411\configs\config.yaml` | `.yaml` | 3830 | 2026-10-29 12:02:03 | Yes (text match) | No | Yes | **KEEP** — Preserve rollback provenance copies. |
| `D:\Study\Projects\NetSentinel\reports\backups\20261103_113411\src\data_preprocessing.py` | `.py` | 8810 | 2026-10-29 12:04:42 | Yes (text match) | No | Yes | **KEEP** — Preserve rollback provenance copies. |
| `D:\Study\Projects\NetSentinel\reports\backups\20261103_113411\src\evaluate.py` | `.py` | 6984 | 2026-10-29 11:08:12 | Yes (text match) | No | Yes | **KEEP** — Preserve rollback provenance copies. |
| `D:\Study\Projects\NetSentinel\reports\backups\20261103_113411\src\stacking.py` | `.py` | 11346 | 2026-10-29 12:10:16 | Yes (text match) | No | Yes | **KEEP** — Preserve rollback provenance copies. |
| `D:\Study\Projects\NetSentinel\reports\backups\20261103_113411\src\train.py` | `.py` | 11733 | 2026-10-29 12:12:30 | Yes (text match) | No | Yes | **KEEP** — Preserve rollback provenance copies. |
| `D:\Study\Projects\NetSentinel\reports\backups\20261103_113411\src\utils.py` | `.py` | 1631 | 2026-10-29 10:53:13 | Yes (text match) | No | Yes | **KEEP** — Preserve rollback provenance copies. |
| `D:\Study\Projects\NetSentinel\reports\backups\20261103_113411\tests\test_pipeline.py` | `.py` | 9539 | 2026-10-29 12:16:49 | Yes (text match) | No | Yes | **KEEP** — Preserve rollback provenance copies. |
| `D:\Study\Projects\NetSentinel\reports\backups\input_validation_20261103_194029\src\models.py` | `.py` | 6622 | 2026-10-29 12:07:08 | Yes (text match) | No | Yes | **KEEP** — Preserve rollback provenance copies. |
| `D:\Study\Projects\NetSentinel\reports\input_validation_tests\NetSentinel_Fix_20261103_194111\console.log` | `.log` | 89480 | 2026-11-03 19:41:53 | Yes (text match) | No | Yes | **ARCHIVE** — Archive superseded exploratory attempts; retain final passing audit. |
| `D:\Study\Projects\NetSentinel\reports\input_validation_tests\NetSentinel_Fix_20261103_194111\regression_audit.py` | `.py` | 5860 | 2026-11-03 19:41:11 | Yes (text match) | No | Yes | **ARCHIVE** — Archive superseded exploratory attempts; retain final passing audit. |
| `D:\Study\Projects\NetSentinel\reports\input_validation_tests\NetSentinel_Fix_20261103_194111\results.json` | `.json` | 42385 | 2026-11-03 19:41:53 | Yes (text match) | No | Yes | **ARCHIVE** — Archive superseded exploratory attempts; retain final passing audit. |
| `D:\Study\Projects\NetSentinel\reports\input_validation_tests\NetSentinel_Fix_20261103_194256\console.log` | `.log` | 95552 | 2026-11-03 19:43:31 | Yes (text match) | No | Yes | **ARCHIVE** — Archive superseded exploratory attempts; retain final passing audit. |
| `D:\Study\Projects\NetSentinel\reports\input_validation_tests\NetSentinel_Fix_20261103_194256\regression_audit.py` | `.py` | 5862 | 2026-11-03 19:42:57 | Yes (text match) | No | Yes | **ARCHIVE** — Archive superseded exploratory attempts; retain final passing audit. |
| `D:\Study\Projects\NetSentinel\reports\input_validation_tests\NetSentinel_Fix_20261103_194256\results.json` | `.json` | 45429 | 2026-11-03 19:43:31 | Yes (text match) | No | Yes | **ARCHIVE** — Archive superseded exploratory attempts; retain final passing audit. |
| `D:\Study\Projects\NetSentinel\reports\input_validation_tests\NetSentinel_Fix_20261103_194424\console.log` | `.log` | 31072 | 2026-11-03 19:44:54 | Yes (text match) | No | Yes | **KEEP** — Preserve final audit evidence. |
| `D:\Study\Projects\NetSentinel\reports\input_validation_tests\NetSentinel_Fix_20261103_194424\regression_audit.py` | `.py` | 5979 | 2026-11-03 19:44:24 | Yes (text match) | No | Yes | **KEEP** — Preserve final audit evidence. |
| `D:\Study\Projects\NetSentinel\reports\input_validation_tests\NetSentinel_Fix_20261103_194424\results.json` | `.json` | 13181 | 2026-11-03 19:44:54 | Yes (text match) | No | Yes | **KEEP** — Preserve final audit evidence. |
| `D:\Study\Projects\NetSentinel\reports\preliminary_smoke_results\metrics_test.json` | `.json` | 831 | 2026-10-29 11:19:15 | Yes (text match) | No | Yes | **ARCHIVE** — Retain as pre-correction historical smoke evidence. |
| `D:\Study\Projects\NetSentinel\reports\preliminary_smoke_results\README.md` | `.md` | 439 | 2026-10-29 12:00:18 | Yes (text match) | No | Yes | **ARCHIVE** — Retain as pre-correction historical smoke evidence. |
| `D:\Study\Projects\NetSentinel\reports\preliminary_smoke_results\test_results.csv` | `.csv` | 283 | 2026-10-29 11:19:15 | Yes (text match) | No | Yes | **ARCHIVE** — Retain as pre-correction historical smoke evidence. |
| `D:\Study\Projects\NetSentinel\reports\testing_audits\NetSentinel_Test_20261103_190808\console.log` | `.log` | 32330 | 2026-11-03 19:10:25 | Yes (text match) | No | Yes | **KEEP** — Preserve final audit evidence. |
| `D:\Study\Projects\NetSentinel\reports\testing_audits\NetSentinel_Test_20261103_190808\independent_metrics.json` | `.json` | 4439 | 2026-11-03 19:10:25 | Yes (text match) | No | Yes | **KEEP** — Preserve final audit evidence. |
| `D:\Study\Projects\NetSentinel\reports\testing_audits\NetSentinel_Test_20261103_190808\run_audit.py` | `.py` | 12643 | 2026-11-03 19:09:12 | Yes (text match) | No | Yes | **KEEP** — Preserve final audit evidence. |
| `D:\Study\Projects\NetSentinel\reports\testing_audits\NetSentinel_Test_20261103_190808\test_results.json` | `.json` | 14344 | 2026-11-03 19:10:25 | Yes (text match) | No | Yes | **KEEP** — Preserve final audit evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830.console.log` | `.log` | 6595356 | 2026-11-03 13:05:11 | No match | No | Yes | **ARCHIVE** — Exact duplicate of nested console log; retain canonical run copy. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\config.yaml` | `.yaml` | 3830 | 2026-10-29 12:02:03 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\histories\aclr.json` | `.json` | 6498 | 2026-11-03 13:04:44 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\histories\ann.json` | `.json` | 6502 | 2026-11-03 11:40:01 | No match | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\histories\cnn.json` | `.json` | 7755 | 2026-11-03 11:43:19 | No match | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\histories\lstm.json` | `.json` | 7753 | 2026-11-03 11:56:35 | No match | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\histories\rnn.json` | `.json` | 4002 | 2026-11-03 12:04:37 | No match | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\logs\console.log` | `.log` | 6595356 | 2026-11-03 13:05:11 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\logs\train.log` | `.log` | 6244 | 2026-11-03 13:05:11 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\models\aclr_meta.keras` | `.keras` | 55508 | 2026-11-03 13:04:42 | Yes (text match) | Yes | Yes | **KEEP** — Preserve final run checkpoints and preprocessor. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\models\ann_best.keras` | `.keras` | 84686 | 2026-11-03 11:39:54 | Yes (text match) | Yes | Yes | **KEEP** — Preserve final run checkpoints and preprocessor. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\models\cnn_best.keras` | `.keras` | 114306 | 2026-11-03 11:43:00 | Yes (text match) | Yes | Yes | **KEEP** — Preserve final run checkpoints and preprocessor. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\models\lstm_best.keras` | `.keras` | 258016 | 2026-11-03 11:56:33 | Yes (text match) | Yes | Yes | **KEEP** — Preserve final run checkpoints and preprocessor. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\models\preprocessor.joblib` | `.joblib` | 4001 | 2026-11-03 11:38:41 | Yes (text match) | Yes | Yes | **KEEP** — Preserve final run checkpoints and preprocessor. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\models\rnn_best.keras` | `.keras` | 76374 | 2026-11-03 12:04:22 | Yes (text match) | Yes | Yes | **KEEP** — Preserve final run checkpoints and preprocessor. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\prob_aclr_test.npy` | `.npy` | 98928 | 2026-11-03 13:05:11 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\prob_ann_test.npy` | `.npy` | 98928 | 2026-11-03 13:05:11 | No match | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\prob_cnn_test.npy` | `.npy` | 98928 | 2026-11-03 13:05:11 | No match | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\prob_lstm_test.npy` | `.npy` | 98928 | 2026-11-03 13:05:11 | No match | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\prob_rnn_test.npy` | `.npy` | 98928 | 2026-11-03 13:05:11 | No match | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\y_test.npy` | `.npy` | 98928 | 2026-11-03 13:05:11 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\cm_aclr_test.png` | `.png` | 30539 | 2026-11-03 13:05:10 | No match | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\cm_ann_test.png` | `.png` | 30110 | 2026-11-03 13:04:50 | No match | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\cm_cnn_test.png` | `.png` | 29300 | 2026-11-03 13:04:52 | No match | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\cm_lstm_test.png` | `.png` | 30038 | 2026-11-03 13:04:57 | No match | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\cm_rnn_test.png` | `.png` | 30624 | 2026-11-03 13:05:00 | No match | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\comparison_accuracy.png` | `.png` | 25163 | 2026-11-03 13:05:11 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\comparison_precision_recall_f1.png` | `.png` | 27135 | 2026-11-03 13:05:11 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\history_aclr.png` | `.png` | 82615 | 2026-11-03 13:04:45 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\history_ann.png` | `.png` | 75463 | 2026-11-03 11:40:02 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\history_cnn.png` | `.png` | 73218 | 2026-11-03 11:43:20 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\history_lstm.png` | `.png` | 81245 | 2026-11-03 11:57:31 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\history_rnn.png` | `.png` | 85353 | 2026-11-03 12:04:40 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\pr_curves.png` | `.png` | 47123 | 2026-11-03 13:05:11 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\outputs\plots\roc_curves.png` | `.png` | 63057 | 2026-11-03 13:05:11 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\reports\metrics_test.json` | `.json` | 847 | 2026-11-03 13:05:11 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\reports\metrics_val.json` | `.json` | 845 | 2026-11-03 13:04:48 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\reports\run_metadata.json` | `.json` | 656 | 2026-11-03 13:05:11 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\reports\test_results.csv` | `.csv` | 287 | 2026-11-03 13:05:11 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\runs\NetSentinel_20261103_113830\reports\val_results.csv` | `.csv` | 286 | 2026-11-03 13:04:48 | Yes (text match) | No | Yes | **KEEP** — Preserve final run artifacts and evidence. |
| `D:\Study\Projects\NetSentinel\src\data_preprocessing.py` | `.py` | 10241 | 2026-11-03 11:34:24 | Yes (text match) | Yes | Yes | **KEEP** — Preserve source module. |
| `D:\Study\Projects\NetSentinel\src\evaluate.py` | `.py` | 8100 | 2026-11-03 11:35:06 | Yes (text match) | No | Yes | **KEEP** — Preserve source module. |
| `D:\Study\Projects\NetSentinel\src\models.py` | `.py` | 8904 | 2026-11-03 19:44:05 | Yes (text match) | Yes | Yes | **KEEP** — Preserve source module. |
| `D:\Study\Projects\NetSentinel\src\stacking.py` | `.py` | 13718 | 2026-11-03 11:40:54 | Yes (text match) | Yes | Yes | **KEEP** — Preserve source module. |
| `D:\Study\Projects\NetSentinel\src\train.py` | `.py` | 15076 | 2026-11-03 11:41:07 | Yes (text match) | No | Yes | **KEEP** — Preserve source module. |
| `D:\Study\Projects\NetSentinel\src\utils.py` | `.py` | 1956 | 2026-11-03 11:35:45 | Yes (text match) | No | Yes | **KEEP** — Preserve source module. |
| `D:\Study\Projects\NetSentinel\src\__init__.py` | `.py` | 26 | 2026-10-29 10:54:48 | Yes (text match) | No | Yes | **KEEP** — Preserve source module. |
| `D:\Study\Projects\NetSentinel\src\__pycache__\data_preprocessing.cpython-313.pyc` | `.pyc` | 13546 | 2026-11-03 13:05:37 | No match | No | No | **DELETE CANDIDATE** — Regenerable Python bytecode cache; deletion requires approval. |
| `D:\Study\Projects\NetSentinel\src\__pycache__\evaluate.cpython-313.pyc` | `.pyc` | 12503 | 2026-11-03 13:05:37 | No match | No | No | **DELETE CANDIDATE** — Regenerable Python bytecode cache; deletion requires approval. |
| `D:\Study\Projects\NetSentinel\src\__pycache__\models.cpython-313.pyc` | `.pyc` | 11082 | 2026-11-03 19:45:01 | No match | No | No | **DELETE CANDIDATE** — Regenerable Python bytecode cache; deletion requires approval. |
| `D:\Study\Projects\NetSentinel\src\__pycache__\stacking.cpython-313.pyc` | `.pyc` | 16824 | 2026-11-03 13:05:37 | No match | No | No | **DELETE CANDIDATE** — Regenerable Python bytecode cache; deletion requires approval. |
| `D:\Study\Projects\NetSentinel\src\__pycache__\train.cpython-313.pyc` | `.pyc` | 19465 | 2026-11-03 13:05:37 | No match | No | No | **DELETE CANDIDATE** — Regenerable Python bytecode cache; deletion requires approval. |
| `D:\Study\Projects\NetSentinel\src\__pycache__\utils.cpython-313.pyc` | `.pyc` | 3684 | 2026-11-03 13:05:37 | No match | No | No | **DELETE CANDIDATE** — Regenerable Python bytecode cache; deletion requires approval. |
| `D:\Study\Projects\NetSentinel\src\__pycache__\__init__.cpython-313.pyc` | `.pyc` | 142 | 2026-10-29 10:56:45 | No match | No | No | **DELETE CANDIDATE** — Regenerable Python bytecode cache; deletion requires approval. |
| `D:\Study\Projects\NetSentinel\tests\test_pipeline.py` | `.py` | 12004 | 2026-11-03 11:41:55 | Yes (text match) | No | Yes | **KEEP** — Preserve tests; suite includes training behavior. |
| `D:\Study\Projects\NetSentinel\tests\__pycache__\test_pipeline.cpython-313.pyc` | `.pyc` | 15327 | 2026-11-03 13:05:37 | No match | No | No | **DELETE CANDIDATE** — Regenerable bytecode cache; deletion requires approval. |

## Duplicate and historical artifact findings

### Confirmed byte-identical groups

| Group | Exact paths | Redundant bytes if one copy is retained |
|---|---|---:|
| Notebook copies | `netsentinel.ipynb`; `netsentinel_BACKUP.ipynb` | 163,566 |
| Config snapshots | `configs/config.yaml`; `reports/backups/20261103_113411/configs/config.yaml`; `runs/NetSentinel_20261103_113830/config.yaml` | 7,660 (but retain snapshots for provenance) |
| Preprocessor copies | `models/preprocessor.joblib`; `runs/NetSentinel_20261103_113830/models/preprocessor.joblib` | 4,001 |
| Test labels | `outputs/y_test.npy`; `runs/NetSentinel_20261103_113830/outputs/y_test.npy` | 98,928 |
| Test confusion plots | Root `outputs/plots/cm_{aclr,cnn,lstm,rnn}_test.png` and same-named final run plots | 120,501 |
| History plots | Root `outputs/plots/history_{ann,cnn,lstm,rnn}.png` and same-named final run plots | 315,279 |
| Validation CSV | `reports/val_results.csv`; `runs/NetSentinel_20261103_113830/reports/val_results.csv` | 286 |
| Console log | `runs/NetSentinel_20261103_113830.console.log`; `runs/NetSentinel_20261103_113830/logs/console.log` | 6,595,356 |
| **Total** | **14 SHA-256 duplicate groups** | **7,305,577** |

The root `outputs/` folder is older than the final run. It contains some byte-identical files above, plus unique older plots, probability arrays, and reports. Do not remove it as a single unit based on age. Root `models/` includes older `*_final.keras` files and Oct 31 best checkpoints; the authoritative Nov 3 models are in the final run bundle. The two notebooks and two console logs were confirmed byte-identical by SHA-256. Pixel comparison found no additional image duplicates beyond the same byte-identical files.

### Caches, notebooks and unused files

- Eight `.pyc` files under `src/__pycache__/` and `tests/__pycache__/` are regenerable cache files; they are DELETE CANDIDATE only after approval.
- No empty directories were found.
- Neither notebook is imported by the current Python package or test runner. README and historical audit documentation mention them. Keep as archive research material, not deletion candidates.
- The raw UNSW component CSVs, event list, and misspelled `NUSW-NB15_features.csv` are not consumed by the current run config. They may support provenance or extended experiments: KEEP/REVIEW.
- No other standalone unused project scripts were identified beyond active `src/`, `tests/`, and audit scripts stored with their evidence.
- Two early input-validation regression folders are superseded exploratory results; their existence is explained in the final input-validation report. Archive them only after preserving the final passing evidence folder.

## Files that must be preserved

Treat these as KEEP; do not delete them as part of any cleanup:

- **Datasets:** `dataset/UNSW_NB15_training-set.csv`, `dataset/UNSW_NB15_testing-set.csv`, `dataset/UNSW-NB15_1.csv` through `_4.csv`, `dataset/UNSW-NB15_LIST_EVENTS.csv`, and `dataset/NUSW-NB15_features.csv`.
- **Source/config/tests:** all of `src/`, `configs/config.yaml`, `tests/test_pipeline.py`, `README.md`, and `requirements.txt`.
- **Final run:** all of `runs/NetSentinel_20261103_113830/`; especially its ANN/CNN/LSTM/RNN/ACLR checkpoints, `preprocessor.joblib`, test predictions and labels, metrics, metadata, histories, plots, and logs.
- **Current reports:** `reports/NetSentinel_Final_Training_Report.md`, `reports/NetSentinel_Consolidated_Audit_Report.md`, `reports/NetSentinel_Base_Paper_Comparison.md`, `reports/NetSentinel_Input_Validation_Fix_Report.md`, and `reports/NetSentinel_Complete_Testing_Report.md`.
- **Evidence and rollback:** `reports/input_validation_tests/NetSentinel_Fix_20261103_194424/`, `reports/testing_audits/NetSentinel_Test_20261103_190808/`, and all of `reports/backups/`.
- **Historical evidence:** retain the preliminary smoke report, old audit report/PDF/ZIP, and root-level Oct 31 checkpoints/results in an archive if they are moved out of the main view.

The minimal final inference payload is the matching run-local preprocessor, base and meta checkpoints, and source prediction path. Metrics, ground-truth/prediction arrays, confusion matrices, ROC/PR plots and reports are result-verification evidence and should remain paired with the run.

## Proposed folder structure

Preserve the project’s existing layout; keep the timestamped run self-contained instead of scattering its artifacts into a second `results/` hierarchy:

```text
NetSentinel/
├── README.md
├── requirements.txt
├── configs/config.yaml
├── dataset/                         # original dataset and schema/provenance
├── src/                              # current Python package
├── tests/                            # test suite (includes training tests)
├── runs/
│   └── NetSentinel_20261103_113830/  # canonical final experiment; keep intact
│       ├── config.yaml
│       ├── models/                   # five checkpoints + preprocessor
│       ├── histories/
│       ├── logs/
│       ├── outputs/plots/
│       ├── outputs/                  # labels and probabilities
│       └── reports/                  # metrics and metadata
├── reports/
│   ├── current final reports
│   ├── backups/                      # rollback copies
│   ├── archive/                      # proposed dated historical reports/results
│   └── audit evidence folders
├── notebooks/archive/                # proposed home for research notebooks
├── models/ and outputs/              # retain until legacy references are updated
```

This is an organizational target, not an instruction to move files. The current training script uses per-run output paths, while README, config, tests and historical notebooks/reports still refer to root `models/` or `outputs/`. Avoid restructuring until those references are updated and verified.

## Cleanup proposal (not executed)

### KEEP

Keep all files named above; all source, config and tests; all datasets; the entire final run; current reports; rollback backups; and final audit/regression evidence. Keep duplicate config and preprocessor snapshots when they document distinct backup/run provenance.

### ARCHIVE

After approval, move (do not delete) these to explicit archive locations while preserving the final run bundle:

- Root `models/` contents, including Oct 29 `*_final.keras` and Oct 31 best checkpoints/preprocessor.
- Entire root `outputs/` tree as older results and plots.
- `netsentinel_BACKUP.ipynb`; optionally keep `netsentinel.ipynb` beside it in `notebooks/archive/`.
- `runs/NetSentinel_20261103_113830.console.log`, an exact copy of the canonical nested `logs/console.log`.
- `reports/preliminary_smoke_results/`, `reports/netsentinel_audit_report.md`, `reports/NetSentinel_Implementation_Audit.pdf`, `reports/NetSentinel_Audit_Evidence.zip`, `reports/NetSentinel_Progress_Recovery_Report.md`, and `reports/NetSentinel_PreTraining_Verification.log`.
- `reports/NetSentinel_PostTraining_Verification.log`, root `reports/metrics_*.json` and root `reports/*_results.csv` as prior-run outputs.
- Exploratory audit folders `reports/input_validation_tests/NetSentinel_Fix_20261103_194111/` and `...194256/`; retain final passing folder `...194424/`.

### DELETE CANDIDATE (approval required)

Only the following regenerable bytecode cache files are proposed for deletion:

- `src/__pycache__/data_preprocessing.cpython-313.pyc`
- `src/__pycache__/evaluate.cpython-313.pyc`
- `src/__pycache__/models.cpython-313.pyc`
- `src/__pycache__/stacking.cpython-313.pyc`
- `src/__pycache__/train.cpython-313.pyc`
- `src/__pycache__/utils.cpython-313.pyc`
- `src/__pycache__/__init__.cpython-313.pyc`
- `tests/__pycache__/test_pipeline.cpython-313.pyc`

No deletion is authorized by this audit.

### REVIEW

- Raw dataset components, event-list and schema CSVs: unused by current run, but potentially necessary for provenance/extended work.
- Root `models/` and `outputs/`: archive only after confirming their historical value and updating references.
- The old PDF and evidence ZIP: verify submission requirements and archive contents before moving.
- Both notebooks: preserve at least the original even though the backup is byte-identical.

## Required reference updates before approved moves

1. Update the directory tree in README, which documents root outputs/models and both notebooks.
2. Search references in source, tests, configs, notebooks and reports before moving anything. Keep historical reports as dated snapshots; use a new archive index/redirect rather than rewriting old conclusions.
3. Review `configs/config.yaml` root `paths` values. Current `src/train.py` creates a run directory and supplies it explicitly, but helper defaults and old notebook flows may still assume root folders.
4. `tests/test_pipeline.py` writes a preprocessor artifact under root `models/` and performs training. Do not move that location until the test behavior is separately updated and validated.
5. Preserve all links from the final training report and paper comparison into `runs/NetSentinel_20261103_113830/`.
6. If deduplicating, keep one canonical copy and record hashes/locations first; do not alter datasets, checkpoints or run outputs.

## Safe execution plan and risks

No execution is authorized. If approved later:

1. Make an external/versioned backup and record hashes before moving or deleting anything.
2. Leave the final run, datasets, source, tests, current reports, and rollback backups untouched.
3. Archive root legacy models/outputs and dated material to non-overwriting archive paths.
4. Update only current README/index links; retain historical reports intact and label them historical.
5. Remove only the eight listed bytecode caches after verifying resolved targets remain inside the project.
6. Re-run reference checks and compare final checkpoint/prediction hashes with the pre-cleanup manifest.

Risks include default/helper/test paths depending on root `models/`, broken README/report links, loss of notebook/research provenance, and formal submission requirements for the old PDF/ZIP. No candidate is an unconditional deletion.

## Final disposition

This report is a proposal only. No existing file was moved, renamed, deleted or overwritten; no source code or model was modified. Await explicit approval before cleanup operations.

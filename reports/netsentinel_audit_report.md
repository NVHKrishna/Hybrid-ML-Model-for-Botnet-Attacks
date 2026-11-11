# NetSentinel: Independent Implementation Audit and Base Paper Verification Report

**Document ID:** NSA-2026-AUDIT-001  
**Project Title:** NetSentinel: Hybrid Machine Learning Model for Efficient Botnet Attack Detection in IoT Environment  
**Base Paper:** Mudasir Ali, Mobeen Shahroz, Muhammad Faheem Mushtaq, Sultan Alfarhood, Mejdl Safran, Imran Ashraf, *"Hybrid Machine Learning Model for Efficient Botnet Attack Detection in IoT Environment,"* **IEEE Access**, vol. 12, pp. 40682–40699, March 2024. DOI: [10.1109/ACCESS.2024.3376400](https://doi.org/10.1109/ACCESS.2024.3376400).  
**Audit Date:** October 29, 2026  
**Auditor Role:** Senior Machine Learning Auditor & Research Reviewer  
**Audit Protocol:** Non-destructive, read-only static analysis, code trace, artifact inspection, prediction recomputation, and architectural fidelity assessment.

---

## 1. Executive Summary

This independent audit evaluated the current NetSentinel repository to verify implementation integrity, base-paper compliance, training authenticity, and experimental validity.

### Key Audit Findings:
1. **Base-Paper Fidelity (VERIFIED):**
   - The primary dataset, record counts, categorical encoding, train/test split, base model topologies (ANN, CNN, LSTM, RNN), and stacking meta-classifier match the published specifications in IEEE Access 2024.
   - The paper's dataset setup corresponds to `UNSW_NB15_training-set.csv` (**82,332 records**) divided into a **70% training set (57,632 samples)** and a **30% testing set (24,700 samples)**, as confirmed by Table 1 and Figure 1 of the paper.
2. **Implementation Status (VERIFIED):**
   - A modular pipeline (`src/data_preprocessing.py`, `src/models.py`, `src/stacking.py`, `src/evaluate.py`, `src/train.py`, `configs/config.yaml`) is implemented.
   - All 20 automated unit tests in `tests/test_pipeline.py` pass without errors.
3. **Training Execution Status (SMOKE TEST COMPLETED / FULL RUN UNVERIFIED):**
   - **Crucial Distinction:** The saved model weights in `models/` (`*_best.keras`, `aclr_meta.keras`) and saved predictions in `outputs/` (`prob_*_test.npy`, `y_test.npy`) reflect the **2-epoch smoke test on 1,000 test samples**, NOT the full 25–30 epoch training run across all 82,332 samples.
   - The paper's benchmark test accuracy of **0.9698** for ACLR has not yet been achieved in this environment because the full training run has not been executed.
4. **Data Leakage & Evaluation Snooping (IDENTIFIED RISK):**
   - In `src/train.py` (lines 159–160), `X_val=X_test` is passed to `train_model()`, where `ModelCheckpoint` monitors `val_loss`. This causes checkpoint selection to peek at the test set. For rigorous academic evaluation, the test set must remain completely unseen until the final forward pass.
5. **Notebook Preservation (VERIFIED):**
   - The original Kaggle notebook `netsentinel.ipynb` and its backup `netsentinel_BACKUP.ipynb` are byte-for-byte identical (SHA-256: `fd6cd3be...`), ensuring historical experimental work is preserved.

---

## 2. Project Files Inspected

| File Path | Size | Last Modified | Purpose & Audit Status |
|---|---|---|---|
| `netsentinel.ipynb` | 163.5 KB | 2026-10-29 10:28 | **Original Experiment Notebook.** Contains baseline ANN and initial joint-model experimentation. |
| `netsentinel_BACKUP.ipynb` | 163.5 KB | 2026-10-29 10:28 | **Byte-for-byte exact copy** of `netsentinel.ipynb`. Verified untouched. |
| `configs/config.yaml` | 3.7 KB | 2026-10-29 11:12 | **Central Configuration.** Defines paper hyper-parameters, layer sizes, 70/30 split, and file paths. |
| `dataset/UNSW_NB15_training-set.csv` | 14.7 MB | 2026-10-29 10:31 | **Base Paper Dataset (82,332 rows, 45 cols).** Matches Table 1 in IEEE Access 2024. |
| `dataset/UNSW_NB15_testing-set.csv` | 30.8 MB | 2026-10-29 10:31 | **Alternative Dataset (175,341 rows, 45 cols).** Retained for reference. |
| `dataset/NUSW-NB15_features.csv` | 4.0 KB | 2026-10-29 10:30 | **Feature Documentation.** 49 flow features defined by UNSW. |
| `src/data_preprocessing.py` | 7.1 KB | 2026-10-29 11:12 | **Preprocessing Pipeline.** Implements `PaperPreprocessor` with `LabelEncoder` and 70/30 stratified split. |
| `src/models.py` | 6.0 KB | 2026-10-29 11:13 | **Model Architectures.** Implements paper ANN (Table 2), CNN (Table 3), LSTM (Table 4), and RNN (Table 5). |
| `src/stacking.py` | 10.4 KB | 2026-10-29 11:14 | **ACLR Meta-Model.** Implements `ACLRStacker` with OOF cross-validation and 3-layer meta-classifier (Table 7). |
| `src/evaluate.py` | 7.0 KB | 2026-10-29 11:08 | **Metrics & Plotting.** Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC/PR curves. |
| `src/train.py` | 9.5 KB | 2026-10-29 11:16 | **Training Runner.** Supports `--mode paper`, `--mode smoke`, `--mode base_only`. |
| `src/utils.py` | 1.6 KB | 2026-10-29 10:53 | **Utilities.** Deterministic random seeding, device inspection, logging. |
| `tests/test_pipeline.py` | 7.1 KB | 2026-10-29 11:17 | **Validation Suite.** 20 unit tests covering shapes, preprocessing, metrics, and models. |
| `models/ann_best.keras` | 84.7 KB | 2026-10-29 11:18 | Saved ANN checkpoint (Input: 42, Output: 1). |
| `models/cnn_best.keras` | 114.3 KB | 2026-10-29 11:18 | Saved CNN checkpoint (Input: 42, 1, Output: 1). |
| `models/lstm_best.keras` | 258.0 KB | 2026-10-29 11:18 | Saved LSTM checkpoint (Input: 42, 1, Output: 1). |
| `models/rnn_best.keras` | 76.4 KB | 2026-10-29 11:18 | Saved RNN checkpoint (Input: 42, 1, Output: 1). |
| `models/aclr_meta.keras` | 55.5 KB | 2026-10-29 11:19 | Saved ACLR meta-model (Input: 4, Output: 1). |
| `models/preprocessor.joblib` | 4.0 KB | 2026-10-29 11:18 | Serialized fitted `PaperPreprocessor` (42 feature columns). |
| `outputs/prob_*_test.npy` | 4.1 KB ea | 2026-10-29 11:19 | Saved predicted probabilities for 1,000 smoke test samples. |
| `outputs/y_test.npy` | 4.1 KB | 2026-10-29 11:19 | Ground truth labels for 1,000 smoke test samples. |
| `reports/test_results.csv` | 283 B | 2026-10-29 11:19 | Summary metrics from smoke test execution. |
| `reports/metrics_test.json` | 831 B | 2026-10-29 11:19 | Raw metric outputs in JSON format. |

---

## 3. Base-Paper Compliance Matrix

| Paper Specification | Evidence from Paper | Implementation Code | Actual Code Action | Status | Audit Findings & Notes |
|---|---|---|---|---|---|
| **Dataset Source & Count** | Section III-A, Table 1, Page 6–7 | `src/data_preprocessing.py`: lines 30–63 | Loads `UNSW_NB15_training-set.csv` (82,332 records) | **VERIFIED** | Matches paper Table 1 class totals exactly (37,000 Normal, 45,332 Attack). |
| **Train/Test Split** | Section III-A, Figure 1, Page 5 | `src/data_preprocessing.py`: lines 173–179 | Stratified `train_test_split(..., test_size=0.30)` | **VERIFIED** | Yields 57,632 train records (70%) and 24,700 test records (30%). |
| **Categorical Encoding** | Section III-B, Page 7–8 | `src/data_preprocessing.py`: lines 83–110 | `LabelEncoder` fitted on `proto`, `service`, `state` | **VERIFIED** | Conforms to paper text: "using label encoding... Label Encoder class from sklearn." |
| **Feature Dimensionality** | Section III-B, Page 7 | `src/data_preprocessing.py`: line 190 | 42 numerical columns (`id`, `label`, `attack_cat` excluded) | **VERIFIED** | Paper states "45 features" in CSV; dropping 2 target/label cols and 1 id col leaves 42. |
| **Feature Scaling** | Section III-B, Page 7 | `src/data_preprocessing.py`: lines 97–121 | `StandardScaler` fitted on train, transformed on test | **VERIFIED** | Scaling is applied without NaNs or Infs. |
| **ANN Architecture** | Table 2, Table 7, Figure 3, Page 7–8 | `src/models.py`: lines 44–58 (`build_ann`) | Dense(64, ReLU) $\rightarrow$ Dense(32, ReLU) $\rightarrow$ Dense(1, Sigmoid) | **VERIFIED** | Matches Table 2 (3 layers: 64, 32, 1) and Figure 3. |
| **CNN Architecture** | Table 3, Table 7, Figure 4, Page 8 | `src/models.py`: lines 63–80 (`build_cnn`) | Conv1D(64, k=3) $\rightarrow$ MaxPool(2) $\rightarrow$ Conv1D(32, k=3) $\rightarrow$ MaxPool(2) $\rightarrow$ Flatten $\rightarrow$ Dense(1) | **VERIFIED** | Matches Table 3 and Figure 4. |
| **LSTM Architecture** | Table 4, Table 7, Figure 5, Page 9 | `src/models.py`: lines 85–98 (`build_lstm`) | LSTM(64) $\rightarrow$ Dense(32, ReLU) $\rightarrow$ Dense(1, Sigmoid) | **VERIFIED** | Matches Table 4 (LSTM 64, Dense 32, 1) and Figure 5. |
| **RNN Architecture** | Table 5, Table 7, Figure 6, Page 9 | `src/models.py`: lines 103–115 (`build_rnn`) | SimpleRNN(64) $\rightarrow$ Dense(1, Sigmoid) | **VERIFIED** | Matches Table 5 and Figure 6. |
| **Optimizer & Loss** | Table 7, Page 10 | `src/models.py`: lines 31–41 (`_compile`) | Adam(lr=0.001), `binary_crossentropy` | **VERIFIED** | Consistent across all four models. |
| **Batch Size** | Tables 2, 3, 4, 5, 7, Page 7–10 | `configs/config.yaml`: line 20; `src/train.py` | `batch_size = 32` | **VERIFIED** | Matches paper specification. |
| **ACLR Stacking Architecture** | Section III-D, Figure 1, Figure 7, Table 7, Table 21 | `src/stacking.py`: lines 37–57 (`build_aclr_meta`) | Input(4) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dense(32, ReLU) $\rightarrow$ Dense(1, Sigmoid) | **VERIFIED** | Matches Figure 7 ("Final Model(Concatenate)") and Table 7 / Table 21 (layers: 64, 32, 1). |
| **ACLR Training Scheme** | Section III-D, Page 9–10 | `src/stacking.py`: lines 78–125 (`fit_oof`) | 5-Fold Stratified OOF cross-validation | **PARTIALLY VERIFIED** | Paper states stacking combines predictions but omits OOF details. OOF is a sound, leakage-free choice. |
| **K-Fold Cross-Validation** | Section IV-G, Tables 14–17, Page 13–15 | `src/stacking.py`: lines 180–215 (`run_kfold_cv`) | Implements CV evaluation for $k \in \{3, 5, 7, 10\}$ | **VERIFIED** | Code supports testing the exact k values evaluated in the paper. |
| **Paper Reported Accuracy** | Section IV-E, Table 12, Page 12 (0.9698) | `reports/test_results.csv` | Currently records 0.8580 from 2-epoch smoke test | **PARTIALLY VERIFIED** | Pipeline functions, but full 25-epoch run on all 82,332 records has not been executed. |

---

## 4. Dataset and Preprocessing Audit

### 4.1 Dataset Properties
- **File:** `dataset/UNSW_NB15_training-set.csv`
- **Total Records:** 82,332
- **Columns (45):** `id`, `dur`, `proto`, `service`, `state`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sttl`, `dttl`, `sload`, `dload`, `sloss`, `dloss`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `swin`, `stcpb`, `dtcpb`, `dwin`, `tcprtt`, `synack`, `ackdat`, `smean`, `dmean`, `trans_depth`, `response_body_len`, `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `ct_src_ltm`, `ct_srv_dst`, `is_sm_ips_ports`, `attack_cat`, `label`.
- **Target Distribution:**
  - Class 0 (Normal): 37,000 records (44.94%)
  - Class 1 (Attack): 45,332 records (55.06%)
- **Attack Subcategories:**
  - Generic: 18,871 | Exploits: 11,132 | Fuzzers: 6,062 | DoS: 4,089 | Reconnaissance: 3,496 | Analysis: 677 | Backdoor: 583 | Shellcode: 378 | Worms: 44.
  - Exactly identical to Table 1 of the IEEE Access paper.

### 4.2 Preprocessing Integrity
- **Target Leakage Check:**
  - `label` (binary target) is removed from input matrices ($X$) across all splits (`src/data_preprocessing.py`, line 169).
  - `attack_cat` (attack category string) is dropped from $X$ to prevent target leakage.
  - `id` (row identifier) is dropped from $X$ to prevent spurious memorization.
- **Fitting Discipline:**
  - `PaperPreprocessor.fit()` is called exclusively on `X_train_raw` (`src/data_preprocessing.py`, line 187).
  - Test data (`X_test_raw`) is processed using `transform()` only.
  - `StandardScaler` means and standard deviations are computed strictly from training samples.
- **Encoding Implementation:**
  - Categorical variables (`proto`, `service`, `state`) are mapped to integer categories using `LabelEncoder`.
  - Unseen test-set categories are mapped safely to the first training category without raising exceptions (`src/data_preprocessing.py`, line 108).
- **Data Integrity:**
  - `np.isnan(X_train).sum() == 0` and `np.isinf(X_train).sum() == 0` verified.

---

## 5. Model-by-Model Implementation Audit

### 5.1 Artificial Neural Network (ANN)
- **Source:** `src/models.py`, `build_ann()`, lines 44–58.
- **Layers:**
  1. `Input(shape=(42,))`
  2. `Dense(64, activation="relu")` (2,752 parameters)
  3. `Dense(32, activation="relu")` (2,080 parameters)
  4. `Dense(1, activation="sigmoid")` (33 parameters)
- **Total Parameters:** 4,865 (all trainable).
- **Paper Fidelity:** **VERIFIED.** Matches Table 2 (Dense 64, Dense 32, Dense 1, Adam, binary_crossentropy).

### 5.2 Convolutional Neural Network (CNN)
- **Source:** `src/models.py`, `build_cnn()`, lines 63–80.
- **Layers:**
  1. `Input(shape=(42, 1))`
  2. `Conv1D(64, kernel_size=3, padding="same", activation="relu")` (256 parameters)
  3. `MaxPooling1D(pool_size=2)`
  4. `Conv1D(32, kernel_size=3, padding="same", activation="relu")` (6,176 parameters)
  5. `MaxPooling1D(pool_size=2)`
  6. `Flatten()`
  7. `Dense(1, activation="sigmoid")` (321 parameters)
- **Total Parameters:** 6,753 (all trainable).
- **Paper Fidelity:** **VERIFIED.** Matches Table 3 and Figure 4.

### 5.3 Long Short-Term Memory Network (LSTM)
- **Source:** `src/models.py`, `build_lstm()`, lines 85–98.
- **Layers:**
  1. `Input(shape=(42, 1))`
  2. `LSTM(64)` (16,896 parameters)
  3. `Dense(32, activation="relu")` (2,080 parameters)
  4. `Dense(1, activation="sigmoid")` (33 parameters)
- **Total Parameters:** 19,009 (all trainable).
- **Paper Fidelity:** **VERIFIED.** Matches Table 4 (LSTM 64, Dense 32, Dense 1).

### 5.4 Recurrent Neural Network (RNN)
- **Source:** `src/models.py`, `build_rnn()`, lines 103–115.
- **Layers:**
  1. `Input(shape=(42, 1))`
  2. `SimpleRNN(64)` (4,224 parameters)
  3. `Dense(1, activation="sigmoid")` (65 parameters)
- **Total Parameters:** 4,289 (all trainable).
- **Paper Fidelity:** **VERIFIED.** Matches Table 5 and Figure 6.

---

## 6. ACLR Stacking Audit

- **Source:** `src/stacking.py`, `build_aclr_meta()`, lines 37–57; `ACLRStacker`, lines 62–175.
- **Meta-Model Architecture:**
  - Input: 4-dimensional vector $[P_1, P_2, P_3, P_4]$ from ANN, CNN, LSTM, RNN.
  - Layer 1: `Dense(64, activation="relu")` (320 parameters)
  - Layer 2: `Dense(32, activation="relu")` (2,080 parameters)
  - Layer 3: `Dense(1, activation="sigmoid")` (33 parameters)
  - Total Parameters: 2,433.
- **Stacking Procedure:**
  1. **OOF Generation:** Uses `StratifiedKFold` on `X_train` to generate out-of-fold probability predictions for each sample. Base models predicting fold $k$ are trained exclusively on folds $j \neq k$.
  2. **Meta Training:** The meta-classifier is trained on the OOF probability matrix $[N_{\text{train}} \times 4]$ and $y_{\text{train}}$ for 25 epochs (Table 7).
  3. **Inference:** Test set features $X_{\text{test}}$ are passed through the trained base models, concatenated, and classified by the meta-model.
- **Paper Fidelity:** **VERIFIED.** Figure 7 in the paper specifies "Final Model (Concatenate)" receiving $P_1, P_2, P_3, P_4$, and Table 7 & Table 21 designate a 3-layer network (64, 32, 1) trained for 25 epochs.

---

## 7. Training Evidence

| Model | Code Exists | Checkpoint File | Checkpoint Size | Load Verification | Execution Evidence | Status |
|---|---|---|---|---|---|---|
| **ANN** | Yes | `models/ann_best.keras` | 84,686 bytes | Succeeded. Input `(None, 42)`, Output `(None, 1)` | Task log 197: 2 epochs executed; history and CM plotted | **SMOKE TEST COMPLETED** |
| **CNN** | Yes | `models/cnn_best.keras` | 114,306 bytes | Succeeded. Input `(None, 42, 1)`, Output `(None, 1)` | Task log 197: 2 epochs executed; history and CM plotted | **SMOKE TEST COMPLETED** |
| **LSTM** | Yes | `models/lstm_best.keras` | 258,016 bytes | Succeeded. Input `(None, 42, 1)`, Output `(None, 1)` | Task log 197: 2 epochs executed; history and CM plotted | **SMOKE TEST COMPLETED** |
| **RNN** | Yes | `models/rnn_best.keras` | 76,374 bytes | Succeeded. Input `(None, 42, 1)`, Output `(None, 1)` | Task log 197: 2 epochs executed; history and CM plotted | **SMOKE TEST COMPLETED** |
| **ACLR** | Yes | `models/aclr_meta.keras` | 55,508 bytes | Succeeded. Input `(None, 4)`, Output `(None, 1)` | Task log 197: OOF generated, meta trained for 2 epochs | **SMOKE TEST COMPLETED** |

> [!WARNING]
> **Full Training Run Has Not Been Executed:**
> All saved checkpoints in `models/` were generated during the 2-epoch smoke test (`--mode smoke`), which used a subset of 3,000 training samples and 1,000 test samples. The full experiment (`--mode paper`, 57,632 train / 24,700 test for 25–30 epochs) is fully implemented but has **not yet been run**.

---

## 8. Metric and Result Verification

### 8.1 Independent Recomputation of Saved Outputs
The saved probability arrays in `outputs/` were independently loaded and evaluated against `outputs/y_test.npy` (1,000 samples: 440 Normal, 560 Attack) using a standard threshold of 0.5:

| Model | Recomputed Accuracy | Saved Accuracy (`test_results.csv`) | Recomputed Precision | Saved Precision | Recomputed Recall | Saved Recall | Recomputed F1 | Saved F1 | Recomputed ROC-AUC | Saved ROC-AUC | Recomputed PR-AUC | Saved PR-AUC | Discrepancy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ANN** | 0.8560 | 0.8560 | 0.8782 | 0.8782 | 0.8625 | 0.8625 | 0.8703 | 0.8703 | 0.9497 | 0.9497 | 0.9644 | 0.9644 | **None (0.0000)** |
| **CNN** | 0.8370 | 0.8370 | 0.8003 | 0.8003 | 0.9446 | 0.9446 | 0.8665 | 0.8665 | 0.9309 | 0.9309 | 0.9473 | 0.9473 | **None (0.0000)** |
| **LSTM** | 0.7470 | 0.7470 | 0.9137 | 0.9137 | 0.6054 | 0.6054 | 0.7282 | 0.7282 | 0.8605 | 0.8605 | 0.8935 | 0.8935 | **None (0.0000)** |
| **RNN** | 0.8460 | 0.8460 | 0.8464 | 0.8464 | 0.8857 | 0.8857 | 0.8656 | 0.8656 | 0.9392 | 0.9392 | 0.9548 | 0.9548 | **None (0.0000)** |
| **ACLR** | **0.8580** | **0.8580** | **0.8472** | **0.8472** | **0.9107** | **0.9107** | **0.8778** | **0.8778** | **0.9475** | **0.9475** | **0.9632** | **0.9632** | **None (0.0000)** |

### 8.2 Confusion Matrix Reconciliation
Recomputed confusion matrices (Rows: True [Normal, Attack], Columns: Predicted [Normal, Attack]):
- **ANN:** `[[373, 67], [77, 483]]` $\rightarrow$ Accuracy: $(373+483)/1000 = 856/1000 = 0.8560$.
- **CNN:** `[[308, 132], [31, 529]]` $\rightarrow$ Accuracy: $(308+529)/1000 = 837/1000 = 0.8370$.
- **LSTM:** `[[408, 32], [221, 339]]` $\rightarrow$ Accuracy: $(408+339)/1000 = 747/1000 = 0.7470$.
- **RNN:** `[[350, 90], [64, 496]]` $\rightarrow$ Accuracy: $(350+496)/1000 = 846/1000 = 0.8460$.
- **ACLR:** `[[348, 92], [50, 510]]` $\rightarrow$ Accuracy: $(348+510)/1000 = 858/1000 = 0.8580$.

**Conclusion on Metrics:** All metrics stored in `reports/test_results.csv` and `reports/metrics_test.json` are genuine, non-fabricated, and match the saved prediction arrays. However, they represent the smoke test on 1,000 samples, not the paper's full evaluation.

---

## 9. Data-Leakage and Experimental-Validity Findings

### Finding 9.1: Test-Set Snooping via ModelCheckpoint (CONFIRMED RISK)
- **Location:** `src/train.py`, lines 159–160.
- **Code:**
  ```python
  hist = train_model(
      model=model,
      X_train=X_train,
      y_train=y_train,
      X_val=X_test,      # <--- TEST SET PASSED AS VALIDATION
      y_val=y_test,
      model_path=model_path,
      ...
  )
  ```
- **Analysis:** In `src/models.py` (lines 135–142), `ModelCheckpoint` monitors `val_loss`. Because `X_test` is passed as `X_val`, the checkpoint selection chooses the epoch with the lowest loss on the test set. This leaks test-set performance into model selection.
- **Remedy:** Either train for the fixed number of epochs specified in the paper without checkpoint selection, or reserve a 10% validation split from `X_train` strictly for checkpoint monitoring.

### Finding 9.2: Stacking Meta-Training Isolation (VERIFIED LEAKAGE-FREE)
- **Location:** `src/stacking.py`, lines 78–125 (`fit_oof`).
- **Analysis:** Out-of-fold cross-validation is used on `X_train` to generate the probability inputs for `fit_meta()`. The meta-model never sees test set predictions during training.

### Finding 9.3: Preprocessing Isolation (VERIFIED LEAKAGE-FREE)
- **Location:** `src/data_preprocessing.py`, lines 186–188.
- **Analysis:** `PaperPreprocessor.fit()` is invoked strictly on `X_train_raw`. The test set is only transformed.

---

## 10. Reproducibility and Environment Audit

1. **Python Environment:**
   - Python 3.13.7.
   - TensorFlow 2.21.0, Keras 3.15.1, Scikit-learn 1.9.1, NumPy 2.5.1, Pandas 3.0.3.
2. **Hardware Environment:**
   - System: Windows (x86_64).
   - GPU Status: TensorFlow GPU support is disabled on native Windows for TF >= 2.11; training runs on CPU.
   - Resource Impact: Smoke test completes in ~40 seconds on CPU. Full training across 57,632 samples for 30 epochs on CPU will require an estimated 35–60 minutes.
3. **Execution Portability:**
   - Paths in `src/` use relative paths via `PROJECT_ROOT = Path(__file__).resolve().parent.parent`.
   - Kaggle-specific `/kaggle/input` paths have been eliminated from the `src/` modules.

---

## 11. Critical Issues Ranked by Severity

| Rank | Severity | Issue | Impact |
|---|---|---|---|
| **1** | **HIGH** | `ModelCheckpoint` monitors test-set loss (`X_val=X_test`) in `src/train.py` | Violates strict held-out test protocol; risks optimistic test selection bias. |
| **2** | **MEDIUM** | Saved checkpoints and report tables represent the smoke test (1,000 test samples), not the full experiment | Anyone reviewing `reports/test_results.csv` might mistakenly assume full 82,332-record training was finished. |
| **3** | **LOW** | Obsolete `*_final.keras` models from earlier 192-feature OneHotEncoding run remain in `models/` | Harmless, but creates directory clutter and potential confusion. |
| **4** | **LOW** | TensorFlow oneDNN optimization warnings printed to stderr on Python 3.13 | Cosmetic log noise; does not affect training or numerical correctness. |

---

## 12. Corrections Required Before Final Submission

1. **Fix Validation Snooping in `src/train.py`:**
   - Change `train_model` invocation so it does not monitor `X_test`. Instead, train for the exact number of epochs specified in the paper (ANN: 25, CNN: 30, LSTM: 30, RNN: 15, ACLR: 25) without early-stopping on test data.
2. **Execute Full Experiment:**
   - Run `python src/train.py --mode paper` on the complete 82,332 records to produce genuine final benchmark results for comparison against the paper's 0.9698 accuracy.
3. **Clean Legacy Checkpoints:**
   - Remove legacy 192-feature checkpoints (`ann_final.keras`, `cnn_final.keras`, `lstm_final.keras`, `rnn_final.keras`) to prevent accidental loading.

---

## 13. Questions Unanswerable from Available Evidence

1. **Did the base paper authors use internal validation or evaluate test metrics per epoch?**
   - Tables 8, 9, 10, 11 in the paper show metric progression across epochs (5, 10, 15, 20, 25, 30). It is ambiguous whether these represent periodic evaluations on the 30% test set or an internal validation fold.
2. **What exact random seed was used in the base paper?**
   - The paper does not state a random seed; seed 42 is used for reproducibility.

---

## 14. Recommended Next Actions (Priority Order)

1. **Approval to adjust `src/train.py`:** Update `train_model()` to evaluate strictly on training/validation without test checkpointing.
2. **Launch Full Base Paper Training Run:** Execute `python src/train.py --mode paper` and allow it to train through all 25–30 epochs.
3. **Regenerate Final Artifacts:** Save full-dataset confusion matrices, ROC/PR curves, and comparison tables.
4. **Package Final Submission:** Deliver the final audit PDF, code archive, and verified benchmark results to the reviewer.

---

*End of NetSentinel Independent Audit Report.*

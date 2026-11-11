# NetSentinel: Hybrid Machine Learning Model for Efficient Botnet Attack Detection in IoT Environment

Implementation and reproduction of the base research paper:
> **IEEE Access (Volume 12, 2024)**  
> *"Hybrid Machine Learning Model for Efficient Botnet Attack Detection in IoT Environment"*  
> **Authors:** Mudasir Ali, Mobeen Shahroz, Muhammad Faheem Mushtaq, Sultan Alfarhood, Mejdl Safran, Imran Ashraf  
> **DOI:** `10.1109/ACCESS.2024.3376400`

---

## 1. Base Paper Methodology & Parameter Alignment

Every design decision, hyperparameter, and architectural specification in this project is directly aligned with the published IEEE Access 2024 paper:

| Aspect | Base Paper Specification | NetSentinel Implementation | Reference in Paper |
|---|---|---|---|
| **Dataset** | UNSW-NB15 | `dataset/UNSW_NB15_training-set.csv` (82,332 records) | Section III-A, Table 1 |
| **Data Split** | 70% Train (57,632), 30% Test (24,700) | Stratified 70/30 train_test_split | Section III-A, Figure 1 |
| **Categorical Encoding** | `LabelEncoder` (sklearn) on categorical columns | `PaperPreprocessor` with `LabelEncoder` for `proto`, `service`, `state` | Section III-B |
| **Feature Scaling** | StandardScaler / normalization across all 42 input features | Standard scaling on 42 features | Section III-B |
| **Input Shape** | 42 features (ANN), (42, 1) for sequence/conv models | Exact 42 features | Table 7, Figures 3-6 |
| **ANN Architecture** | 3 Dense layers: 64 $\rightarrow$ 32 $\rightarrow$ 1 (ReLU, Sigmoid) | `build_ann(n_features=42)` | Table 2, Table 7, Figure 3 |
| **CNN Architecture** | Conv1D(64, k=3) $\rightarrow$ MaxPool(2) $\rightarrow$ Conv1D(32, k=3) $\rightarrow$ MaxPool(2) $\rightarrow$ Flatten $\rightarrow$ Dense(1, Sigmoid) | `build_cnn(n_features=42)` | Table 3, Table 7, Figure 4 |
| **LSTM Architecture** | LSTM(64) $\rightarrow$ Dense(32, ReLU) $\rightarrow$ Dense(1, Sigmoid) | `build_lstm(n_features=42)` | Table 4, Table 7, Figure 5 |
| **RNN Architecture** | SimpleRNN(64) $\rightarrow$ Dense(1, Sigmoid) | `build_rnn(n_features=42)` | Table 5, Table 7, Figure 6 |
| **ACLR Stacking** | Concatenate P1, P2, P3, P4 $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dense(32, ReLU) $\rightarrow$ Dense(1, Sigmoid) | `ACLRStacker` with `build_aclr_meta(4)` | Section III-D, Figure 1, Figure 7, Table 7 |
| **Optimizer & Loss** | Adam (lr=0.001), Binary Cross-Entropy | Adam, `binary_crossentropy` across all models | Table 7 |
| **Batch Size** | 32 | `batch_size = 32` | Tables 2, 3, 4, 5, 7 |
| **Epochs** | ANN: 25, CNN: 30, LSTM: 30, RNN: 15, ACLR: 25 | Configurable matching Table 7 & Table 21 | Table 7, 21 |
| **K-Fold Validation** | $k \in \{3, 5, 7, 10\}$ (best $k=5$ achieves 0.9749 accuracy) | Supported in `src/stacking.py` via `run_kfold_cv` | Section IV-G, Tables 14-17 |

---

## 2. Base Paper Reported Benchmark Results (Table 13, Table 18)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **ANN** | 0.7568 | 0.7817 | 0.7568 | 0.7559 | — | — |
| **CNN** | 0.9440 | 0.9444 | 0.9440 | 0.9439 | — | — |
| **LSTM** | 0.9651 | 0.9651 | 0.9651 | 0.9651 | — | — |
| **RNN** | 0.9522 | 0.9523 | 0.9522 | 0.9521 | — | — |
| **Proposed ACLR** | **0.9698** | **0.9691** | **0.9693** | **0.9692** | **0.9934** | **0.9950** |
| **ACLR (5-Fold CV)** | **0.9749** | **0.9770** | **0.9717** | **0.9723** | — | — |

---

## 3. Project Organization

```
NetSentinel/
├── configs/
│   └── config.yaml             # All paper hyperparameters and paths
├── dataset/
│   ├── UNSW_NB15_training-set.csv  # 82,332 records (Base paper main dataset)
│   ├── UNSW_NB15_testing-set.csv   # 175,341 records
│   └── NUSW-NB15_features.csv      # Schema & documentation
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Paper dataset loader, LabelEncoder & 70/30 split
│   ├── models.py               # Paper-faithful ANN, CNN, LSTM, RNN builders
│   ├── stacking.py             # Paper ACLR stacking meta-model & K-fold CV
│   ├── evaluate.py             # Accuracy, Prec, Rec, F1, ROC-AUC, PR-AUC, plots
│   ├── utils.py                # Logging, seeds, device info
│   └── train.py                # Complete runnable pipeline
├── tests/
│   └── test_pipeline.py        # 20 automated unit sanity checks
├── models/                     # Saved .keras models and preprocessor.joblib
├── outputs/
│   └── plots/                  # Generated ROC curves, PR curves, Confusion matrices
├── reports/
│   ├── test_results.csv        # Metrics comparison table
│   └── metrics_test.json       # Exact scores in JSON
├── netsentinel.ipynb           # Original Kaggle notebook
├── netsentinel_BACKUP.ipynb    # Preserved original backup
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation
```

---

## 4. Setup and Execution

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Automated Unit Tests (All 20 tests pass)
```bash
python tests/test_pipeline.py
```

### Step 3: Fast Smoke Test (~45 seconds)
Validates the entire end-to-end pipeline with subsampled data:
```bash
python src/train.py --mode smoke
```

### Step 4: Run the Complete Base Paper Reproduction
Runs the full training of ANN, CNN, LSTM, RNN and the ACLR stacking model on the complete 82,332 records with 70/30 split:
```bash
python src/train.py --mode paper
```

### Step 5: Train Base Models Only (Optional)
```bash
python src/train.py --mode base_only
```

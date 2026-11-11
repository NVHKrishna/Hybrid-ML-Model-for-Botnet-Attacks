"""
src/data_preprocessing.py
NetSentinel — Data loading and preprocessing pipeline aligned with Base Paper.

Base Paper Reference:
  IEEE Access 2024: "Hybrid Machine Learning Model for Efficient Botnet Attack
  Detection in IoT Environment" (Mudasir Ali et al.)
  Section III-A & III-B:
    - Dataset: UNSW-NB15 dataset file with 82,332 records (UNSW_NB15_training-set.csv)
    - Train/Test Split: 70% training (57,632 samples) / 30% testing (24,700 samples)
    - Preprocessing: Removal of null values, Label Encoding for categorical columns (proto, service, state),
      Standard scaling across the 42 resulting numeric features.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

log = logging.getLogger(__name__)

# Base paper dataset record count: 82,332 records
PAPER_TOTAL_RECORDS = 82_332

# Columns to drop before building input feature matrix
DROP_COLS = ["label", "attack_cat", "id"]

# Categorical features in UNSW-NB15
CATEGORICAL_COLS = ["proto", "service", "state"]


def load_dataset(
    dataset_path: str = "dataset/UNSW_NB15_training-set.csv",
) -> pd.DataFrame:
    """
    Load the dataset specified in the paper.
    If the file path has 82,332 records, it is directly loaded.
    If paths are swapped, auto-detects the 82,332-record file.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    if len(df) != PAPER_TOTAL_RECORDS:
        # Check if the other CSV in dataset/ has 82,332 rows
        alt_path = path.parent / ("UNSW_NB15_testing-set.csv" if "training" in path.name else "UNSW_NB15_training-set.csv")
        if alt_path.exists():
            alt_df = pd.read_csv(alt_path)
            if len(alt_df) == PAPER_TOTAL_RECORDS:
                log.info("Loaded base paper dataset (%d records) from %s", len(alt_df), alt_path)
                return alt_df

    log.info("Loaded dataset with %d records from %s", len(df), path)
    return df


class PaperPreprocessor:
    """
    Paper-faithful preprocessor using Label Encoding and Standard Scaling.
    
    Steps:
      1. Drop target columns ('label', 'attack_cat') and identifier ('id').
      2. LabelEncode categorical columns ('proto', 'service', 'state').
      3. StandardScale all features so neural networks train stably.
    """

    def __init__(self, categorical_cols: List[str] = None):
        self.categorical_cols = categorical_cols or CATEGORICAL_COLS
        self.label_encoders_: Dict[str, LabelEncoder] = {}
        self.scaler_ = StandardScaler()
        self.feature_columns_: List[str] = []
        self._fitted = False

    def fit(self, X_df: pd.DataFrame) -> "PaperPreprocessor":
        X = X_df.copy()
        self.feature_columns_ = [c for c in X.columns if c not in DROP_COLS]

        # Fit LabelEncoder on each categorical column
        for col in self.categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                # Ensure string type
                vals = X[col].astype(str)
                le.fit(vals)
                self.label_encoders_[col] = le

        # Transform to fit scaler
        X_num = self._encode_categoricals(X)
        self.scaler_.fit(X_num)
        self._fitted = True
        return self

    def _encode_categoricals(self, X_df: pd.DataFrame) -> np.ndarray:
        X = X_df[self.feature_columns_].copy()
        for col, le in self.label_encoders_.items():
            if col in X.columns:
                # Handle potential unseen labels gracefully
                classes = set(le.classes_)
                vals = X[col].astype(str).map(lambda s: s if s in classes else le.classes_[0])
                X[col] = le.transform(vals)
        return X.to_numpy(dtype=np.float32)

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        X_num = self._encode_categoricals(X_df)
        X_scaled = self.scaler_.transform(X_num).astype(np.float32)
        
        # Verify no NaN or Inf
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            raise ValueError("Preprocessed data contains NaN or Inf values.")
        return X_scaled

    def fit_transform(self, X_df: pd.DataFrame) -> np.ndarray:
        return self.fit(X_df).transform(X_df)

    @property
    def n_features(self) -> int:
        return len(self.feature_columns_)

    def save(self, path: str) -> None:
        joblib.dump({
            "label_encoders": self.label_encoders_,
            "scaler": self.scaler_,
            "feature_columns": self.feature_columns_,
            "categorical_cols": self.categorical_cols,
        }, path)
        log.info("Saved preprocessor to %s", path)

    @classmethod
    def load(cls, path: str) -> "PaperPreprocessor":
        d = joblib.load(path)
        obj = cls(categorical_cols=d["categorical_cols"])
        obj.label_encoders_ = d["label_encoders"]
        obj.scaler_ = d["scaler"]
        obj.feature_columns_ = d["feature_columns"]
        obj._fitted = True
        return obj


def build_paper_splits(
    dataset_path: str = "dataset/UNSW_NB15_training-set.csv",
    test_split: float = 0.30,
    val_split: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, object]:
    """
    Prepares three distinct, strictly disjoint data partitions:
      1. Outer 70/30 split (paper Figure 1 & Section III-A):
         - Outer Train (70%): 57,632 records
         - Held-out Test (30%): 24,700 records (held untouched until final evaluation)
      2. Inner train/validation split of the 70% outer train:
         - Training partition (e.g. 85% of outer train = 48,987 records): used to fit model parameters
         - Validation partition (e.g. 15% of outer train = 8,645 records): used for checkpoint selection / early stopping
      3. Preprocessor fitting:
         - PaperPreprocessor is fitted ONLY on the training partition (X_train_raw).
         - Validation and Test partitions are transformed using the fitted preprocessor.
    """
    df = load_dataset(dataset_path)

    # Check for null values as mentioned in the paper
    null_count = int(df.isnull().sum().sum())
    if null_count > 0:
        df = df.dropna()
        log.info("Removed %d rows with null values", null_count)

    X_raw = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y_raw = df["label"].to_numpy(dtype=np.float32)
    indices = np.arange(len(df))

    # 1. Stratified 70/30 outer split (paper Figure 1 & Section III-A)
    (
        X_outer_train_raw,
        X_test_raw,
        y_outer_train,
        y_test,
        outer_train_idx,
        test_idx,
    ) = train_test_split(
        X_raw,
        y_raw,
        indices,
        test_size=test_split,
        random_state=random_seed,
        stratify=y_raw,
    )

    # 2. Stratified inner split of the 70% training portion into train and validation
    (
        X_train_raw,
        X_val_raw,
        y_train,
        y_val,
        train_idx,
        val_idx,
    ) = train_test_split(
        X_outer_train_raw,
        y_outer_train,
        outer_train_idx,
        test_size=val_split,
        random_state=random_seed,
        stratify=y_outer_train,
    )

    # Verify mutual exclusion of index sets
    assert len(set(train_idx) & set(val_idx)) == 0, "Train and validation indices overlap!"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train and test indices overlap!"
    assert len(set(val_idx) & set(test_idx)) == 0, "Validation and test indices overlap!"

    log.info(
        "Three-way Partition: Train=%d, Val=%d, Held-out Test=%d (Total=%d)",
        len(y_train), len(y_val), len(y_test), len(df)
    )

    # 3. Fit preprocessor STRICTLY on the training partition
    preprocessor = PaperPreprocessor()
    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_val   = preprocessor.transform(X_val_raw)
    X_test  = preprocessor.transform(X_test_raw)

    log.info("Feature count: %d features (42 numeric after LabelEncoder)", preprocessor.n_features)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val":   X_val,
        "y_val":   y_val,
        "X_test":  X_test,
        "y_test":  y_test,
        "train_indices": train_idx,
        "val_indices":   val_idx,
        "test_indices":  test_idx,
        "preprocessor": preprocessor,
        "n_features": preprocessor.n_features,
        "feature_names": preprocessor.feature_columns_,
    }

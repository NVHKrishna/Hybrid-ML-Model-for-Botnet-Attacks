"""
src/stacking.py
NetSentinel — ACLR Stacking Model aligned with Base Paper.

Base Paper Reference:
  IEEE Access 2024: "Hybrid Machine Learning Model for Efficient Botnet Attack
  Detection in IoT Environment" (Mudasir Ali et al.)
  Section III-D, Figure 1, Figure 7, Table 7, Table 12, Table 21:
    - Base models: ANN, CNN, LSTM, RNN output probabilities P1, P2, P3, P4
    - Stacking meta-classifier ("Final Model (Concatenate)" in Figure 7):
        Input(4) -> Dense(64, relu) -> Dense(32, relu) -> Dense(1, sigmoid)
    - Optimizer: Adam (lr=0.001)
    - Loss: binary_crossentropy
    - Batch size: 32
    - Epochs: 25 -> achieves 0.9698 test accuracy
    - K-fold Cross-Validation evaluated for k in [3, 5, 7, 10] (Table 14-17)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Sequential, layers as L
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

log = logging.getLogger(__name__)


# ── ACLR Meta-Learner Architecture (Figure 7, Table 7, Table 21) ─────────────

def build_aclr_meta(n_base: int = 4, lr: float = 1e-3) -> Model:
    """
    ACLR Meta-classifier:
      Input shape: (4,) for predictions [P1, P2, P3, P4]
      Dense(64, activation='relu')
      Dense(32, activation='relu')
      Dense(1, activation='sigmoid')
    """
    model = Sequential([
        L.Input(shape=(n_base,), name="aclr_input"),
        L.Dense(64, activation="relu", name="aclr_dense1"),
        L.Dense(32, activation="relu", name="aclr_dense2"),
        L.Dense(1, activation="sigmoid", name="aclr_output"),
    ], name="ACLR_Meta")

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )
    return model


# ── ACLR Stacking Pipeline ───────────────────────────────────────────────────

class ACLRStacker:
    """
    Stacking architecture combining ANN, CNN, LSTM, and RNN predictions
    as described in Section III-D and Figure 7 of the base paper.
    """

    def __init__(
        self,
        model_builders: Dict[str, callable],
        n_folds: int = 5,
        random_seed: int = 42,
        models_dir: str = "models",
    ):
        self.model_builders = model_builders
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.base_names: List[str] = list(model_builders.keys())
        self.seq_models = {"CNN", "LSTM", "RNN"}

        self.oof_preds_: Optional[np.ndarray] = None
        self.final_base_models_: Dict[str, Model] = {}
        self.meta_model_: Optional[Model] = None

    def fit_oof(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_features: int,
        batch_size: int = 32,
        epochs: int = 30,
        model_epochs: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        Generate out-of-fold predictions to prevent data leakage during meta-training.
        """
        kf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_seed,
        )
        N = len(y_train)
        n_base = len(self.base_names)
        oof = np.zeros((N, n_base), dtype=np.float32)

        epochs_map = model_epochs or {"ANN": 25, "CNN": 30, "LSTM": 30, "RNN": 15}

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            log.info("Generating OOF predictions - Fold %d/%d", fold + 1, self.n_folds)
            Xtr, Xvl = X_train[tr_idx], X_train[val_idx]
            ytr, yvl = y_train[tr_idx], y_train[val_idx]

            for b_idx, name in enumerate(self.base_names):
                tf.keras.utils.set_random_seed(self.random_seed + fold)
                model = self.model_builders[name](n_features)
                is_seq = name in self.seq_models

                Xtr_in = np.expand_dims(Xtr, -1) if is_seq else Xtr
                Xvl_in = np.expand_dims(Xvl, -1) if is_seq else Xvl
                ep = epochs_map.get(name, epochs)

                model.fit(
                    Xtr_in, ytr,
                    validation_data=(Xvl_in, yvl),
                    epochs=ep,
                    batch_size=batch_size,
                    verbose=0,
                )
                oof[val_idx, b_idx] = model.predict(Xvl_in, verbose=0).ravel()
                tf.keras.backend.clear_session()

        self.oof_preds_ = oof
        log.info("OOF predictions generated. Shape: %s", oof.shape)
        return oof

    def load_or_fit_final_base(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        n_features: int = 42,
        batch_size: int = 32,
        epochs: int = 30,
        model_epochs: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Load trained base models from models_dir if available, or train them.
        """
        epochs_map = model_epochs or {"ANN": 25, "CNN": 30, "LSTM": 30, "RNN": 15}

        for name in self.base_names:
            best_path = self.models_dir / f"{name.lower()}_best.keras"
            if best_path.exists() and best_path.stat().st_size == 0:
                raise ValueError(f"Refusing to load empty checkpoint: {best_path}")
            if best_path.exists():
                log.info("Loading pre-trained base model for %s from %s", name, best_path)
                loaded = tf.keras.models.load_model(str(best_path))
                expected_input = (None, n_features, 1) if name in self.seq_models else (None, n_features)
                if tuple(loaded.input_shape) != expected_input or tuple(loaded.output_shape) != (None, 1):
                    raise ValueError(
                        f"Incompatible checkpoint for {name} at {best_path}: "
                        f"input={loaded.input_shape}, output={loaded.output_shape}; "
                        f"expected input={expected_input}, output={(None, 1)}."
                    )
                self.final_base_models_[name] = loaded
                continue

            log.info("Training final base model: %s", name)
            tf.keras.utils.set_random_seed(self.random_seed)
            model = self.model_builders[name](n_features)
            is_seq = name in self.seq_models
            Xtr = np.expand_dims(X_train, -1) if is_seq else X_train
            val_data = (np.expand_dims(X_val, -1) if is_seq else X_val, y_val) if X_val is not None else None
            path = str(best_path)

            ep = epochs_map.get(name, epochs)
            model.fit(
                Xtr, y_train,
                validation_data=val_data,
                epochs=ep,
                batch_size=batch_size,
                callbacks=[
                    ModelCheckpoint(
                        filepath=path,
                        monitor="val_loss" if val_data else "loss",
                        save_best_only=True,
                        verbose=0,
                    )
                ],
                verbose=1,
            )
            # ModelCheckpoint saves the best validation epoch; restore it before inference.
            if not best_path.exists():
                raise RuntimeError(f"Training did not create the expected checkpoint: {best_path}")
            self.final_base_models_[name] = tf.keras.models.load_model(str(best_path))
            log.info("Saved %s to %s", name, path)

    def fit_meta(
        self,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        lr: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 25,
    ) -> Dict[str, List[float]]:
        """
        Train ACLR meta-model on OOF prediction features (Table 7 & Table 12: 25 epochs).

        LEAKAGE-FREE ENFORCEMENT:
          - The meta-model is trained strictly on out-of-fold predictions of the training partition.
          - Validation partition (X_val, y_val) may optionally be used for validation loss monitoring.
          - The held-out test set MUST NEVER be passed during meta-model fitting.
        """
        if self.oof_preds_ is None:
            raise RuntimeError("Run fit_oof() before fit_meta().")

        tf.keras.utils.set_random_seed(self.random_seed)
        meta = build_aclr_meta(n_base=len(self.base_names), lr=lr)

        val_data = None
        cb_list = []
        path = str(self.models_dir / "aclr_meta.keras")
        if Path(path).exists():
            raise FileExistsError(
                f"Refusing to reuse an existing ACLR meta checkpoint: {path}. "
                "Choose a fresh run directory to prevent stale checkpoint reuse."
            )

        if X_val is not None and y_val is not None:
            val_stack_feats = self.base_predict(X_val)
            val_data = (val_stack_feats, y_val)
            cb_list.append(ModelCheckpoint(
                filepath=path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ))

        history = meta.fit(
            self.oof_preds_,
            y_train,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb_list,
            verbose=1,
        )
        if not cb_list:
            meta.save(path)
        else:
            meta = tf.keras.models.load_model(path)

        self.meta_model_ = meta
        log.info("ACLR meta-model saved to %s", path)
        return {key: [float(value) for value in values] for key, values in history.history.items()}

    def base_predict(self, X: np.ndarray) -> np.ndarray:
        """Collect base model predictions [P1, P2, P3, P4]."""
        preds = []
        for name in self.base_names:
            model = self.final_base_models_[name]
            is_seq = name in self.seq_models
            Xin = np.expand_dims(X, -1) if is_seq else X
            preds.append(model.predict(Xin, verbose=0).ravel())
        return np.column_stack(preds).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Full ACLR inference: Base Models -> Concatenate [P1..P4] -> ACLR Meta -> Final Probability."""
        if self.meta_model_ is None:
            raise RuntimeError("Run fit_meta() before predict_proba().")
        stack_feats = self.base_predict(X)
        return self.meta_model_.predict(stack_feats, verbose=0).ravel()


# ── Base Paper K-Fold Cross-Validation Evaluation (Section IV-G) ─────────────

def run_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_builders: Dict[str, callable],
    k: int = 5,
    batch_size: int = 32,
    epochs: int = 25,
    random_seed: int = 42,
    model_epochs: Optional[Dict[str, int]] = None,
    meta_epochs: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluates ACLR under K-fold Cross Validation matching Section IV-G (Tables 14-17).
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_outer_tr, X_te = X[train_idx], X[test_idx]
        y_outer_tr, y_te = y[train_idx], y[test_idx]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_outer_tr,
            y_outer_tr,
            test_size=0.15,
            random_state=random_seed + fold,
            stratify=y_outer_tr,
        )

        # Isolate every fold's saved base/meta checkpoints. A fold must never
        # load another fold's model or a pre-existing project checkpoint.
        with tempfile.TemporaryDirectory(prefix=f"netsentinel_cv_fold_{fold}_") as fold_dir:
            stacker = ACLRStacker(
                model_builders,
                n_folds=3,
                random_seed=random_seed + fold,
                models_dir=fold_dir,
            )
            stacker.fit_oof(
                X_tr, y_tr, n_features=X.shape[1], batch_size=batch_size,
                epochs=epochs, model_epochs=model_epochs,
            )
            stacker.load_or_fit_final_base(
                X_tr, y_tr, X_val=X_val, y_val=y_val,
                n_features=X.shape[1], batch_size=batch_size, epochs=epochs,
                model_epochs=model_epochs,
            )
            stacker.fit_meta(
                y_tr, X_val=X_val, y_val=y_val,
                batch_size=batch_size, epochs=meta_epochs or epochs,
            )
            prob = stacker.predict_proba(X_te)
        pred = (prob >= 0.5).astype(int)

        scores["accuracy"].append(accuracy_score(y_te, pred))
        scores["precision"].append(precision_score(y_te, pred, zero_division=0))
        scores["recall"].append(recall_score(y_te, pred, zero_division=0))
        scores["f1"].append(f1_score(y_te, pred, zero_division=0))

    return {metric: float(np.mean(vals)) for metric, vals in scores.items()}

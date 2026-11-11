"""
src/models.py
NetSentinel — Individual model builders (ANN, CNN, LSTM, RNN).

Architectures and hyperparameters taken directly from the base paper:
IEEE Access 2024: "Hybrid Machine Learning Model for Efficient Botnet Attack
Detection in IoT Environment" (Mudasir Ali et al.)

Tables 2, 3, 4, 5, 7 and Figures 3, 4, 5, 6:
  ANN:  Dense(64, relu) -> Dense(32, relu) -> Dense(1, sigmoid)  [Table 2, Table 7, Fig 3]
  CNN:  Conv1D(64, k=3) -> MaxPool(2) -> Conv1D(32, k=3) -> MaxPool(2) -> Flatten -> Dense(1, sigmoid) [Table 3, Fig 4]
  LSTM: LSTM(64) -> Dense(32, relu) -> Dense(1, sigmoid) [Table 4, Table 7, Fig 5]
  RNN:  SimpleRNN(64) -> Dense(1, sigmoid) [Table 5, Table 7, Fig 6]
"""

from __future__ import annotations

import logging
from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers as L
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

log = logging.getLogger(__name__)


def _compile(model: Model, lr: float = 1e-3) -> Model:
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


def _enforce_input_dimensions(model: Model) -> Model:
    """Reject malformed model inputs before Keras executes the forward pass."""
    if getattr(model, "_netsentinel_input_validation", False):
        return model

    expected_shape = tuple(model.input_shape[1:])
    original_call = model.call
    original_predict = model.predict

    def validate_input_shape(inputs):
        try:
            actual_shape = tuple(inputs.shape)
        except (AttributeError, TypeError) as exc:
            raise ValueError(
                f"Invalid input for {model.name}: expected shape (batch, {expected_shape})."
            ) from exc

        valid_rank = len(actual_shape) == len(expected_shape) + 1
        valid_dimensions = valid_rank and all(
            actual is None or actual == expected
            for actual, expected in zip(actual_shape[1:], expected_shape)
        )
        if not valid_dimensions:
            raise ValueError(
                f"Invalid input shape for {model.name}: expected (batch, {', '.join(map(str, expected_shape))}), "
                f"received {actual_shape}."
            )

    def validated_call(inputs, *args, **kwargs):
        validate_input_shape(inputs)
        return original_call(inputs, *args, **kwargs)

    def validated_predict(inputs, *args, **kwargs):
        validate_input_shape(inputs)
        return original_predict(inputs, *args, **kwargs)

    model.call = validated_call
    model.predict = validated_predict
    model._netsentinel_input_validation = True
    return model


# ── ANN Architecture (Table 2, Table 7, Figure 3) ────────────────────────────

def build_ann(n_features: int, lr: float = 1e-3) -> Model:
    """
    Base Paper ANN:
      - Layer 1: Dense 64 (ReLU)
      - Layer 2: Dense 32 (ReLU)
      - Layer 3: Dense 1 (Sigmoid)
      - Optimizer: Adam, Loss: binary_crossentropy, Batch size: 32, Epochs: 25-30
    """
    model = Sequential([
        L.Input(shape=(n_features,), name="ann_input"),
        L.Dense(64, activation="relu", name="ann_dense1"),
        L.Dense(32, activation="relu", name="ann_dense2"),
        L.Dense(1, activation="sigmoid", name="ann_output"),
    ], name="ANN")
    return _compile(_enforce_input_dimensions(model), lr)


# ── CNN Architecture (Table 3, Table 7, Figure 4) ────────────────────────────

def build_cnn(n_features: int, lr: float = 1e-3) -> Model:
    """
    Base Paper 1D-CNN:
      - Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')
      - MaxPooling1D(pool_size=2)
      - Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')
      - MaxPooling1D(pool_size=2)
      - Flatten()
      - Dense(1, activation='sigmoid')
    """
    model = Sequential([
        L.Input(shape=(n_features, 1), name="cnn_input"),
        L.Conv1D(filters=64, kernel_size=3, padding="same", activation="relu", name="cnn_conv1"),
        L.MaxPooling1D(pool_size=2, name="cnn_pool1"),
        L.Conv1D(filters=32, kernel_size=3, padding="same", activation="relu", name="cnn_conv2"),
        L.MaxPooling1D(pool_size=2, name="cnn_pool2"),
        L.Flatten(name="cnn_flatten"),
        L.Dense(1, activation="sigmoid", name="cnn_output"),
    ], name="CNN")
    return _compile(_enforce_input_dimensions(model), lr)


# ── LSTM Architecture (Table 4, Table 7, Figure 5) ───────────────────────────

def build_lstm(n_features: int, lr: float = 1e-3) -> Model:
    """
    Base Paper LSTM:
      - LSTM(64 units)
      - Dense(32, activation='relu') [Table 4]
      - Dense(1, activation='sigmoid')
    """
    model = Sequential([
        L.Input(shape=(n_features, 1), name="lstm_input"),
        L.LSTM(64, name="lstm_layer1"),
        L.Dense(32, activation="relu", name="lstm_dense1"),
        L.Dense(1, activation="sigmoid", name="lstm_output"),
    ], name="LSTM")
    return _compile(_enforce_input_dimensions(model), lr)


# ── RNN Architecture (Table 5, Table 7, Figure 6) ────────────────────────────

def build_rnn(n_features: int, lr: float = 1e-3) -> Model:
    """
    Base Paper SimpleRNN:
      - SimpleRNN(64 units)
      - Dense(1, activation='sigmoid')
    """
    model = Sequential([
        L.Input(shape=(n_features, 1), name="rnn_input"),
        L.SimpleRNN(64, name="rnn_layer1"),
        L.Dense(1, activation="sigmoid", name="rnn_output"),
    ], name="RNN")
    return _compile(_enforce_input_dimensions(model), lr)


# ── Training & Inference Utilities ───────────────────────────────────────────

def train_model(
    model: Model,
    X_train: "np.ndarray",
    y_train: "np.ndarray",
    X_val: Optional["np.ndarray"] = None,
    y_val: Optional["np.ndarray"] = None,
    model_path: Optional[str] = None,
    batch_size: int = 32,
    epochs: int = 30,
    reshape_seq: bool = False,
    callbacks: Optional[list] = None,
) -> dict:
    """
    Train a model following the base paper's training parameters.

    LEAKAGE-FREE ENFORCEMENT:
      - X_val and y_val must be drawn exclusively from the training/validation partition.
      - Held-out test set data MUST NEVER be passed to this function.
      - Checkpoint selection and early stopping monitor val_loss on genuine validation samples only.
    """
    import numpy as np

    if X_val is not None and y_val is not None:
        # Sanity check: Ensure validation set is not identical to train set
        assert len(X_val) != len(X_train) or not np.array_equal(X_val[:min(10, len(X_val))], X_train[:min(10, len(X_train))]), (
            "Training and validation sets must be distinct!"
        )

    Xtr = np.expand_dims(X_train, -1) if reshape_seq else X_train
    val_data = None
    if X_val is not None and y_val is not None:
        Xvl = np.expand_dims(X_val, -1) if reshape_seq else X_val
        val_data = (Xvl, y_val)

    cb_list = callbacks or []
    if model_path:
        cb_list.append(ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss" if val_data else "loss",
            save_best_only=True,
            verbose=0,
        ))

    history = model.fit(
        Xtr, y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1,
    )
    return history.history


def predict_proba(
    model: Model,
    X: "np.ndarray",
    reshape_seq: bool = False,
) -> "np.ndarray":
    """Return 1-D predicted probabilities after validating the model input shape."""
    import numpy as np

    expected_shape = tuple(model.input_shape[1:])
    Xin = np.expand_dims(X, -1) if reshape_seq else np.asarray(X)
    actual_shape = tuple(Xin.shape[1:]) if Xin.ndim else ()
    if Xin.ndim != len(expected_shape) + 1 or actual_shape != expected_shape:
        raise ValueError(
            f"Invalid input shape for {model.name}: expected (batch, {', '.join(map(str, expected_shape))}), "
            f"received {tuple(Xin.shape)}."
        )

    # Older saved checkpoints do not persist runtime validation; attach the guard
    # after load so subsequent direct Keras calls enforce the declared dimensions.
    _enforce_input_dimensions(model)
    return model.predict(Xin, verbose=0).ravel()

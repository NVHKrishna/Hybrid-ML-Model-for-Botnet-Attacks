"""
src/train.py
NetSentinel — Main training and evaluation pipeline aligned with Base Paper.

Base Paper:
  IEEE Access 2024: "Hybrid Machine Learning Model for Efficient Botnet Attack
  Detection in IoT Environment" (Mudasir Ali et al.)

Strict Leakage-Free Architecture:
  1. Outer 70/30 split creates held-out Test Set (30%, 24,700 samples).
  2. Training portion (70%, 57,632 samples) is partitioned into:
     - Training set (85% of outer train = 48,987 samples): parameter fitting & OOF generation.
     - Validation set (15% of outer train = 8,645 samples): model checkpointing & validation loss monitoring.
  3. Preprocessor is fitted STRICTLY on the training set.
  4. Test set is accessed ONLY for final post-training evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import build_paper_splits
from src.evaluate import (
    comparison_table, compute_metrics, plot_confusion_matrix,
    plot_pr_curves, plot_roc_curves, plot_training_history, print_metrics,
    print_confusion,
)
from src.models import build_ann, build_cnn, build_lstm, build_rnn, train_model, predict_proba
from src.stacking import ACLRStacker
from src.utils import ensure_dirs, gpu_info, load_yaml, set_seeds, setup_logging

log = logging.getLogger(__name__)

# Smoke test parameters
SMOKE_N_TRAIN = 3_000
SMOKE_N_VAL   = 600
SMOKE_N_TEST  = 1_000
SMOKE_EPOCHS  = 2
SMOKE_FOLDS   = 2


def parse_args():
    p = argparse.ArgumentParser(description="NetSentinel Training Pipeline (Base Paper Fidelity)")
    p.add_argument(
        "--mode",
        choices=["paper", "smoke", "base_only"],
        default="paper",
        help="Run mode: 'paper' (full base paper reproduction), 'smoke' (quick test), 'base_only' (skip stacking)",
    )
    p.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    p.add_argument("--kfold", action="store_true", help="Run K-fold cross-validation (k=5) as in Table 15")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    cfg = load_yaml(args.config)

    log.info("=" * 70)
    log.info("NetSentinel: Hybrid Machine Learning Model for Botnet Attack Detection")
    log.info("Base Paper Fidelity Implementation (IEEE Access 2024)")
    log.info("Mode: %s | Hardware: %s", args.mode, gpu_info())
    log.info("=" * 70)

    seed = cfg["training"]["random_seed"]
    set_seeds(seed)

    # Setup directories
    models_dir = cfg["paths"]["models_dir"]
    outputs_dir = cfg["paths"]["outputs_dir"]
    plots_dir = cfg["paths"]["plots_dir"]
    reports_dir = cfg["paths"]["reports_dir"]
    ensure_dirs(models_dir, outputs_dir, plots_dir, reports_dir)

    # ── 1. Data Loading & Preprocessing (Three Strictly Disjoint Partitions) ──
    dataset_path = cfg["data"]["dataset_file"]
    test_split = cfg["data"]["test_split"]
    val_split = cfg["data"].get("val_split", 0.15)

    log.info("Loading UNSW-NB15 dataset and creating 3 disjoint partitions (Train / Val / Held-out Test)...")
    splits = build_paper_splits(
        dataset_path=dataset_path,
        test_split=test_split,
        val_split=val_split,
        random_seed=seed,
    )

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]
    n_features = splits["n_features"]

    # Verify complete partition disjointness
    assert len(set(splits["train_indices"]) & set(splits["val_indices"])) == 0, "Train & Val indices overlap!"
    assert len(set(splits["train_indices"]) & set(splits["test_indices"])) == 0, "Train & Test indices overlap!"
    assert len(set(splits["val_indices"]) & set(splits["test_indices"])) == 0, "Val & Test indices overlap!"

    if args.mode == "smoke":
        log.info("Running in SMOKE mode with subsampled data...")
        idx_tr = np.random.choice(len(y_train), min(SMOKE_N_TRAIN, len(y_train)), replace=False)
        idx_vl = np.random.choice(len(y_val),   min(SMOKE_N_VAL,   len(y_val)),   replace=False)
        idx_te = np.random.choice(len(y_test),  min(SMOKE_N_TEST,  len(y_test)),  replace=False)
        X_train, y_train = X_train[idx_tr], y_train[idx_tr]
        X_val,   y_val   = X_val[idx_vl],   y_val[idx_vl]
        X_test,  y_test  = X_test[idx_te],  y_test[idx_te]

    log.info("Partitions: Train=%s, Val=%s, Held-out Test=%s, Features=%d",
             X_train.shape, X_val.shape, X_test.shape, n_features)

    # Save fitted preprocessor
    splits["preprocessor"].save(os.path.join(models_dir, "preprocessor.joblib"))

    # ── 2. Model Hyperparameters from Base Paper ──────────────────────────────
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]

    if args.mode == "smoke":
        model_epochs = {"ANN": 2, "CNN": 2, "LSTM": 2, "RNN": 2}
        meta_epochs = 2
        oof_folds = SMOKE_FOLDS
    else:
        model_epochs = {
            "ANN":  cfg["models"]["ann"]["epochs"],
            "CNN":  cfg["models"]["cnn"]["epochs"],
            "LSTM": cfg["models"]["lstm"]["epochs"],
            "RNN":  cfg["models"]["rnn"]["epochs"],
        }
        meta_epochs = cfg["stacking"]["epochs"]
        oof_folds = cfg["stacking"]["default_k"]

    model_builders = {
        "ANN":  lambda nf: build_ann(nf, lr=lr),
        "CNN":  lambda nf: build_cnn(nf, lr=lr),
        "LSTM": lambda nf: build_lstm(nf, lr=lr),
        "RNN":  lambda nf: build_rnn(nf, lr=lr),
    }
    seq_models = {"CNN", "LSTM", "RNN"}

    # ── 3. Train Individual Base Models (Validated on X_val, NEVER X_test) ────
    import tensorflow as tf
    histories = {}
    val_results = {}
    val_preds = {}

    for name, builder in model_builders.items():
        log.info("--- Training %s (%d epochs, batch_size=%d) ---", name, model_epochs[name], batch_size)
        tf.keras.utils.set_random_seed(seed)
        model = builder(n_features)
        is_seq = name in seq_models
        model_path = os.path.join(models_dir, f"{name.lower()}_best.keras")

        # Train model with ModelCheckpoint monitoring validation loss on X_val
        hist = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,          # Genuine validation set
            y_val=y_val,
            model_path=model_path,
            batch_size=batch_size,
            epochs=model_epochs[name],
            reshape_seq=is_seq,
        )
        histories[name] = hist

        # Predict on Validation Set for model selection / validation logging
        model = tf.keras.models.load_model(model_path)
        val_prob = predict_proba(model, X_val, reshape_seq=is_seq)
        val_preds[name] = val_prob
        val_results[name] = compute_metrics(y_val, val_prob)

        print_metrics(f"{name} — Validation Set", val_results[name])
        plot_training_history(hist, name, plots_dir)
        tf.keras.backend.clear_session()

    # ── 4. ACLR Stacking Model (Leakage-Free OOF on X_train) ───────────────────
    if args.mode != "base_only":
        log.info("--- Training Proposed ACLR Hybrid Stacking Model ---")
        stacker = ACLRStacker(
            model_builders=model_builders,
            n_folds=oof_folds,
            random_seed=seed,
            models_dir=models_dir,
        )

        # Step A: OOF predictions from base models on X_train ONLY
        log.info("Generating Out-of-Fold features from training set only...")
        stacker.fit_oof(
            X_train=X_train,
            y_train=y_train,
            n_features=n_features,
            batch_size=batch_size,
            model_epochs=model_epochs,
        )

        # Step B: Load existing trained base models (or fit with X_val)
        stacker.load_or_fit_final_base(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_features=n_features,
            batch_size=batch_size,
            model_epochs=model_epochs,
        )

        # Step C: Train ACLR meta-model (validates on X_val, NEVER X_test)
        log.info("Training ACLR Meta-Classifier (%d epochs)...", meta_epochs)
        stacker.fit_meta(
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            lr=lr,
            batch_size=batch_size,
            epochs=meta_epochs,
        )

        # Validation evaluation for ACLR
        aclr_val_prob = stacker.predict_proba(X_val)
        val_preds["ACLR"] = aclr_val_prob
        val_results["ACLR"] = compute_metrics(y_val, aclr_val_prob)
        print_metrics("Proposed ACLR Model — Validation Set", val_results["ACLR"])

    # Save validation comparison table
    val_table = comparison_table(val_results)
    val_table.to_csv(os.path.join(reports_dir, "val_results.csv"))
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS (Model Selection)")
    print("=" * 60)
    print(val_table.to_string())

    # ── 5. Final Evaluation on Untouched Held-Out Test Set ────────────────────
    log.info("\n" + "=" * 70)
    log.info("FINAL EVALUATION ON HELD-OUT TEST SET (ACCESSED ONLY ONCE)")
    log.info("=" * 70)

    test_results = {}
    test_preds = {}

    for name in model_builders:
        is_seq = name in seq_models
        model_path = os.path.join(models_dir, f"{name.lower()}_best.keras")
        model = tf.keras.models.load_model(model_path)
        prob = predict_proba(model, X_test, reshape_seq=is_seq)
        test_preds[name] = prob
        test_results[name] = compute_metrics(y_test, prob)

        print_metrics(f"{name} — Held-Out Test Set", test_results[name])
        print_confusion(y_test, prob)
        plot_confusion_matrix(y_test, prob, f"{name}_test", plots_dir)
        tf.keras.backend.clear_session()

    if args.mode != "base_only":
        aclr_test_prob = stacker.predict_proba(X_test)
        test_preds["ACLR"] = aclr_test_prob
        test_results["ACLR"] = compute_metrics(y_test, aclr_test_prob)

        print_metrics("Proposed ACLR Model — Held-Out Test Set", test_results["ACLR"])
        print_confusion(y_test, aclr_test_prob)
        plot_confusion_matrix(y_test, aclr_test_prob, "ACLR_test", plots_dir)

    # ── 6. Save Plots & Final Test Reports ─────────────────────────────────────
    roc_data = {name: {"y_true": y_test, "y_prob": prob} for name, prob in test_preds.items()}
    plot_roc_curves(roc_data, plots_dir)
    plot_pr_curves(roc_data, plots_dir)

    test_table = comparison_table(test_results)
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS COMPARISON (Table 13 Benchmark)")
    print("=" * 60)
    print(test_table.to_string())

    test_table.to_csv(os.path.join(reports_dir, "test_results.csv"))
    with open(os.path.join(reports_dir, "metrics_test.json"), "w") as f:
        json.dump({k: {m: round(v, 6) for m, v in mv.items()} for k, mv in test_results.items()}, f, indent=2)

    np.save(os.path.join(outputs_dir, "y_test.npy"), y_test)
    for name, prob in test_preds.items():
        np.save(os.path.join(outputs_dir, f"prob_{name.lower()}_test.npy"), prob)

    log.info("=" * 70)
    log.info("Execution complete. Final results saved in %s and %s", reports_dir, plots_dir)
    log.info("=" * 70)


if __name__ == "__main__":
    main()

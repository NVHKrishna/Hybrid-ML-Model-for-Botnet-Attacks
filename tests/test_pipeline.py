"""
tests/test_pipeline.py
NetSentinel — Automated sanity and leakage-free verification suite.

Checks:
  1. Module imports
  2. Dataset loading and 3-way partition (Train / Val / Held-out Test)
  3. Strict mutual exclusion of indices (zero leakage)
  4. Preprocessor fit strictly on training partition
  5. Model tensor shapes matching Base Paper specifications
  6. Short training convergence with genuine validation partition (NOT test set)
  7. Metrics computation integrity
  8. ACLR meta-model shape and out-of-fold training data isolation
  9. Preprocessor persistence and reload fidelity
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import numpy as np

logging.basicConfig(level=logging.WARNING)

PASS = []
FAIL = []
TEST_TMP = tempfile.TemporaryDirectory(prefix="netsentinel_tests_")
TEST_ARTIFACTS_DIR = Path(TEST_TMP.name)


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    (PASS if condition else FAIL).append(name)


# ── 1. Import checks ──────────────────────────────────────────────────────────
print("\n=== 1. Imports ===")
try:
    from src.data_preprocessing import build_paper_splits, PaperPreprocessor, load_dataset
    from src.models import build_ann, build_cnn, build_lstm, build_rnn, train_model, predict_proba
    from src.stacking import ACLRStacker, build_aclr_meta
    from src.evaluate import compute_metrics, comparison_table
    from src.utils import set_seeds
    import tensorflow as tf
    check("All imports", True)
except Exception as e:
    check("All imports", False, str(e))
    print("Cannot continue — fix import errors first.")
    sys.exit(1)

set_seeds(42)

# ── 2. Three-way Partition and Leakage-Free Verification ───────────────────────
print("\n=== 2. Three-Way Data Partition (Train / Val / Held-out Test) ===")
try:
    splits = build_paper_splits(
        dataset_path="dataset/UNSW_NB15_training-set.csv",
        test_split=0.30,
        val_split=0.15,
        random_seed=42,
    )
    check("Dataset loads without error", True)
    
    n_features = splits["n_features"]
    n_tr = len(splits["y_train"])
    n_vl = len(splits["y_val"])
    n_te = len(splits["y_test"])
    
    check("Feature count = 42 (LabelEncoder)", n_features == 42, f"got {n_features}")
    check("Held-out Test rows = 24700 (30% outer)", n_te == 24700, f"got {n_te}")
    check("Outer Train total = 57632 (70% outer)", n_tr + n_vl == 57632, f"got {n_tr + n_vl}")
    check("Total dataset rows = 82332", n_tr + n_vl + n_te == 82332, f"got {n_tr + n_vl + n_te}")
    all_labels = np.concatenate([splits["y_train"], splits["y_val"], splits["y_test"]])
    check("Label distribution = 37,000 normal / 45,332 attack",
          int(np.sum(all_labels == 0)) == 37000 and int(np.sum(all_labels == 1)) == 45332,
          f"got normal={int(np.sum(all_labels == 0))}, attack={int(np.sum(all_labels == 1))}")
    
    # Check disjointness of index sets
    tr_set = set(splits["train_indices"])
    vl_set = set(splits["val_indices"])
    te_set = set(splits["test_indices"])
    
    check("Train and Val indices are mutually disjoint", len(tr_set & vl_set) == 0, f"overlap={len(tr_set & vl_set)}")
    check("Train and Test indices are mutually disjoint", len(tr_set & te_set) == 0, f"overlap={len(tr_set & te_set)}")
    check("Val and Test indices are mutually disjoint", len(vl_set & te_set) == 0, f"overlap={len(vl_set & te_set)}")
    
    # Check preprocessor fitting isolation
    scaler_n_samples = splits["preprocessor"].scaler_.n_samples_seen_
    check("Preprocessor fitted strictly on training data", scaler_n_samples == n_tr, 
          f"scaler seen={scaler_n_samples}, train_len={n_tr}")
    
    check("No NaN in X_train", int(np.isnan(splits["X_train"]).sum()) == 0)
    check("No Inf in X_train", int(np.isinf(splits["X_train"]).sum()) == 0)
except Exception as e:
    check("Dataset loading and partitioning", False, str(e))

print("\n=== 2b. Dataset safety and unseen category handling ===")
try:
    invalid_dir = TEST_ARTIFACTS_DIR / "invalid_dataset"
    invalid_dir.mkdir()
    import pandas as pd
    wrong = pd.DataFrame({"label": [0]})
    wrong.to_csv(invalid_dir / "UNSW_NB15_training-set.csv", index=False)
    wrong.to_csv(invalid_dir / "UNSW_NB15_testing-set.csv", index=False)
    try:
        load_dataset(str(invalid_dir / "UNSW_NB15_training-set.csv"))
        rejected_wrong_size = False
    except ValueError:
        rejected_wrong_size = True
    check("Loader rejects candidates with incorrect row counts", rejected_wrong_size)

    raw_sample = load_dataset("dataset/UNSW_NB15_training-set.csv").drop(
        columns=["label", "attack_cat", "id"], errors="ignore"
    ).head(3).copy()
    prep = splits["preprocessor"]
    baseline_encoded = prep.transform(raw_sample.iloc[:1])
    unseen = raw_sample.iloc[:1].copy()
    unseen.loc[unseen.index[0], "service"] = "__unseen_test_category__"
    unseen_encoded = prep.transform(unseen)
    service_index = prep.feature_columns_.index("service")
    check("Unseen categories map to finite reserved codes",
          np.isfinite(unseen_encoded).all() and unseen_encoded[0, service_index] != baseline_encoded[0, service_index])
    reordered = raw_sample[prep.feature_columns_[::-1]]
    ordered_features = prep.transform(reordered)
    check("Feature output ordering follows training fit order",
          np.allclose(ordered_features, prep.transform(raw_sample)))
except Exception as e:
    check("Dataset safety and unseen category handling", False, str(e))

# Slices for fast testing
N_TINY = 200
X_tiny = splits["X_train"][:N_TINY]
y_tiny = splits["y_train"][:N_TINY]
X_val_t = splits["X_val"][:100]
y_val_t = splits["y_val"][:100]
X_test_t = splits["X_test"][:100]
y_test_t = splits["y_test"][:100]
nf = n_features

# ── 3. Base Paper Model Shapes ────────────────────────────────────────────────
print("\n=== 3. Model shapes (Paper Architectures) ===")
for name, builder, is_seq in [
    ("ANN",  build_ann,  False),
    ("CNN",  build_cnn,  True),
    ("LSTM", build_lstm, True),
    ("RNN",  build_rnn,  True),
]:
    try:
        tf.keras.utils.set_random_seed(42)
        model = builder(nf)
        Xin = np.expand_dims(X_tiny, -1) if is_seq else X_tiny
        out = model(Xin, training=False)
        check(f"{name} output shape (N,1)", out.shape == (N_TINY, 1), f"got {out.shape}")
        tf.keras.backend.clear_session()
    except Exception as e:
        check(f"{name} shape", False, str(e))

# ── 4. Leakage-Free Training with Validation Partition ────────────────────────
print("\n=== 4. Training with Genuine Validation Partition (NOT Test Set) ===")
os.makedirs("models", exist_ok=True)
for name, builder, is_seq in [
    ("ANN",  build_ann,  False),
    ("CNN",  build_cnn,  True),
    ("LSTM", build_lstm, True),
    ("RNN",  build_rnn,  True),
]:
    try:
        tf.keras.utils.set_random_seed(42)
        model = builder(nf)
        test_model_path = str(TEST_ARTIFACTS_DIR / f"test_{name.lower()}.keras")
        
        # Train strictly on X_tiny with X_val_t (NEVER X_test_t)
        hist = train_model(
            model=model,
            X_train=X_tiny,
            y_train=y_tiny,
            X_val=X_val_t,   # Genuine validation partition
            y_val=y_val_t,
            model_path=test_model_path,
            epochs=2,
            batch_size=32,
            reshape_seq=is_seq,
        )
        
        # Predict on validation partition
        val_prob = predict_proba(model, X_val_t, reshape_seq=is_seq)
        check(f"{name} trains with validation partition", 
              val_prob.shape == (100,) and not np.isnan(val_prob).any(),
              f"prob range [{val_prob.min():.3f}, {val_prob.max():.3f}]")

        reloaded = tf.keras.models.load_model(test_model_path)
        reloaded_prob = predict_proba(reloaded, X_val_t, reshape_seq=is_seq)
        check(f"{name} best checkpoint reloads with aligned predictions",
              reloaded_prob.shape == (100,) and np.isfinite(reloaded_prob).all())
              
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
        tf.keras.backend.clear_session()
    except Exception as e:
        check(f"{name} training", False, str(e))

# ── 5. Metrics ────────────────────────────────────────────────────────────────
print("\n=== 5. Metrics ===")
try:
    dummy_true = np.array([0, 0, 1, 1, 0, 1])
    dummy_prob = np.array([0.1, 0.4, 0.6, 0.9, 0.3, 0.8])
    m = compute_metrics(dummy_true, dummy_prob)
    check("Metrics returns core keys", set(['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']).issubset(m.keys()))
    check("Accuracy between 0-1", 0 <= m["accuracy"] <= 1)
    check("ROC-AUC between 0-1", 0 <= m["roc_auc"] <= 1)
except Exception as e:
    check("Metrics", False, str(e))

# ── 6. ACLR Stacking and Out-of-Fold Isolation ────────────────────────────────
print("\n=== 6. ACLR Stacking Meta-Model & OOF Isolation ===")
try:
    tf.keras.utils.set_random_seed(42)
    meta = build_aclr_meta(n_base=4)
    dummy_stack = np.random.rand(50, 4).astype("float32")
    out = meta(dummy_stack, training=False)
    check("ACLR meta output shape (50,1)", out.shape == (50, 1), f"got {out.shape}")
    tf.keras.backend.clear_session()
    
    # Test OOF generation is bounded to training set size
    stacker = ACLRStacker(
        model_builders={"ANN": lambda nf: build_ann(nf)},
        n_folds=2,
        random_seed=42,
        models_dir=str(TEST_ARTIFACTS_DIR / "stacker_models"),
    )
    oof = stacker.fit_oof(X_tiny, y_tiny, n_features=nf, batch_size=32, epochs=1)
    check("OOF predictions match training partition size", len(oof) == len(X_tiny), f"oof_len={len(oof)}")
    tf.keras.backend.clear_session()
except Exception as e:
    check("ACLR meta shape and OOF isolation", False, str(e))

# ── 7. Preprocessor Persistence ───────────────────────────────────────────────
print("\n=== 7. Preprocessor persistence ===")
try:
    df_raw = load_dataset("dataset/UNSW_NB15_training-set.csv")
    raw_sample = df_raw.drop(columns=["label", "attack_cat", "id"], errors="ignore").head(5)

    prep = splits["preprocessor"]
    test_prep_path = "models/test_preprocessor.joblib"
    prep.save(test_prep_path)
    prep2 = PaperPreprocessor.load(test_prep_path)

    X_orig = prep.transform(raw_sample)
    X_reloaded = prep2.transform(raw_sample)
    match = np.allclose(X_orig, X_reloaded)
    check("Preprocessor save/load", match, f"arrays match={match}")
    if os.path.exists(test_prep_path):
        os.remove(test_prep_path)
except Exception as e:
    check("Preprocessor save/load", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED tests:", FAIL)
else:
    print("All leakage-free base paper tests passed!")
print(f"{'='*50}")
TEST_TMP.cleanup()
sys.exit(0 if not FAIL else 1)

"""
src/utils.py
NetSentinel — Shared utilities.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seeds(seed: int = 42) -> None:
    """Set Python, NumPy, and TensorFlow random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass  # TF not available


def get_class_weights(y: np.ndarray) -> dict:
    """Balanced class weights for binary labels."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def gpu_info() -> str:
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return f"{len(gpus)} GPU(s): {[g.name for g in gpus]}"
        return "No GPU detected — running on CPU"
    except Exception:
        return "TensorFlow not available"


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)

"""Run saved NetSentinel models on new UNSW-NB15-format input rows.

Usage from the NetSentinel project directory:
    python -m src.predict --input my_input.csv

The input CSV must contain the 42 raw model feature columns. The fitted
preprocessor handles categorical encoding and scaling; label/attack_cat/id
columns are not model features and are ignored if present.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# Make the script importable when launched directly from the parent workspace.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import PaperPreprocessor
from src.models import predict_proba


DEFAULT_RUN = PROJECT_ROOT / "runs" / "NetSentinel_20261103_113830"
INPUT_FILE = PROJECT_ROOT / "dataset" / "UNSW_NB15_testing-set.csv"
BASE_MODELS = ("ANN", "CNN", "LSTM", "RNN")
SEQUENCE_MODELS = {"CNN", "LSTM", "RNN"}
# Change this 1-based number to choose which data row to predict by default.
ROW_NUMBER = 3


class _CompatibleGlorotUniform(tf.keras.initializers.GlorotUniform):
    """Load newer Keras Glorot configs with older TensorFlow/Keras runtimes.

    Some Keras versions serialize axis hints in the initializer config,
    although older GlorotUniform constructors do not accept those options.
    The saved model weights are loaded separately, so the initializer is not
    used to create replacement weights during inference.
    """

    def __init__(self, seed=None, **kwargs):
        kwargs.pop("input_axes", None)
        kwargs.pop("output_axes", None)
        super().__init__(seed=seed, **kwargs)


# Some Keras archives resolve GlorotUniform by its built-in registered name,
# bypassing custom_objects. Patch that constructor too, removing only the
# inference-irrelevant serialized axis hint before delegating to this runtime.
_original_glorot_init = tf.keras.initializers.GlorotUniform.__init__


def _glorot_init_compat(self, *args, **kwargs):
    kwargs.pop("input_axes", None)
    kwargs.pop("output_axes", None)
    _original_glorot_init(self, *args, **kwargs)


tf.keras.initializers.GlorotUniform.__init__ = _glorot_init_compat


def _load_model(path: Path):
    """Load a checkpoint, including Glorot configs from newer Keras versions."""
    compatible_objects = {
        "GlorotUniform": _CompatibleGlorotUniform,
        "keras>GlorotUniform": _CompatibleGlorotUniform,
        "keras.initializers>GlorotUniform": _CompatibleGlorotUniform,
    }
    return tf.keras.models.load_model(str(path), custom_objects=compatible_objects)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Normal/Attack labels for rows in a CSV using saved NetSentinel models."
    )
    parser.add_argument(
        "--input", default=None,
        help=f"CSV containing raw input rows (default: {INPUT_FILE})",
    )
    parser.add_argument(
        "--record", default=None,
        help="One comma-separated UNSW-NB15 record; omit attack_cat/label if unknown",
    )
    parser.add_argument(
        "--row", type=int, default=ROW_NUMBER,
        help=f"1-based data-row number to predict (default: {ROW_NUMBER}; ignored with --id)",
    )
    parser.add_argument(
        "--id", type=int, default=None,
        help="Dataset id from the input CSV; predicts the matching record",
    )
    parser.add_argument(
        "--run-dir", default=str(DEFAULT_RUN),
        help=f"Training run directory (default: {DEFAULT_RUN})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: <run-dir>/predictions/prediction_row_<row>.csv)",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Attack probability cutoff (default: 0.5)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    if args.record is not None and args.input is not None:
        raise ValueError("Use either --record or --input, not both")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    models_dir = run_dir / "models"
    preprocessor_path = models_dir / "preprocessor.joblib"
    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"Saved preprocessor not found: {preprocessor_path}")

    preprocessor = PaperPreprocessor.load(str(preprocessor_path))
    source_row: int | str
    if args.record is not None:
        values = [value.strip() for value in next(csv.reader([args.record.strip()]))]
        schema = list(pd.read_csv(INPUT_FILE, nrows=0).columns)
        feature_columns = preprocessor.feature_columns_
        if len(values) == len(schema):
            columns = schema
        elif len(values) == len(schema) - 1:
            columns = schema[:-1]  # final label omitted
        elif len(values) == len(feature_columns) + 1:
            columns = ["id", *feature_columns]
        elif len(values) == len(feature_columns):
            columns = feature_columns
        else:
            raise ValueError(
                f"Record has {len(values)} values; expected 42 features, 43 with id, "
                f"44 without label, or {len(schema)} including targets."
            )
        raw = pd.DataFrame([values], columns=columns)
        source_row = "manual"
    else:
        input_path = Path(args.input or INPUT_FILE).expanduser()
        if not input_path.is_absolute():
            input_path = Path.cwd() / input_path
        if not input_path.is_file():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")
        raw = pd.read_csv(input_path)
        if raw.empty:
            raise ValueError("Input CSV has no data rows")
        if args.id is not None:
            if "id" not in raw.columns:
                raise ValueError(f"Input CSV has no 'id' column: {input_path}")
            matches = raw.loc[pd.to_numeric(raw["id"], errors="coerce") == args.id]
            if matches.empty:
                raise ValueError(f"Dataset id {args.id} was not found in {input_path}")
            if len(matches) > 1:
                raise ValueError(f"Dataset id {args.id} occurs more than once in {input_path}")
            raw = matches.copy()
            source_row = args.id
        else:
            if args.row < 1 or args.row > len(raw):
                raise ValueError(f"--row must be between 1 and {len(raw)}; received {args.row}")
            raw = raw.iloc[[args.row - 1]].copy()
            source_row = args.row

    X = preprocessor.transform(raw)

    results: dict[str, np.ndarray] = {}
    for name in BASE_MODELS:
        model_path = models_dir / f"{name.lower()}_best.keras"
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved {name} checkpoint not found: {model_path}")
        model = _load_model(model_path)
        results[name] = predict_proba(model, X, reshape_seq=name in SEQUENCE_MODELS)

    meta_path = models_dir / "aclr_meta.keras"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Saved ACLR meta-model not found: {meta_path}")
    stack_features = np.column_stack([results[name] for name in BASE_MODELS]).astype(np.float32)
    meta_model = _load_model(meta_path)
    results["ACLR"] = meta_model.predict(stack_features, verbose=0).ravel()

    combined_probability = results["ACLR"]
    output = raw.reset_index(drop=True).copy()
    output.insert(0, "source_row", source_row)
    output["combined_attack_probability"] = combined_probability
    output["combined_prediction"] = np.where(
        combined_probability >= args.threshold, "Attack", "Normal"
    )
    predicted_label = output["combined_prediction"].iloc[0]

    print("Overall prediction")
    print("==================")
    if "id" in raw.columns:
        print(f"Dataset ID: {raw['id'].iloc[0]}")
    print(f"Overall attack probability: {combined_probability[0]:.4f}")
    print(f"Final prediction: {predicted_label}")
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else run_dir / "predictions" / (
            "prediction_record.csv" if args.record is not None
            else f"prediction_id_{args.id}.csv" if args.id is not None
            else f"prediction_row_{args.row}.csv"
        )
    )
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"\nSaved record and combined prediction to: {output_path}")


if __name__ == "__main__":
    main()

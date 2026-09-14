"""Camera-ready prediction-head implementation for CIKM 2026."""

from .features import (
    PROBING_FEATURES,
    SUPERVISED_FEATURES,
    UNSUPERVISED_FEATURES,
    get_prediction_head_input_columns,
)
from .mlp import build_prediction_head
from .sequential import build_sequential_dataframe, split_train_test
from .train import fit_prediction_head, prepare_prediction_head_data
from .evaluate import evaluate_predictions

__all__ = [
    "UNSUPERVISED_FEATURES",
    "SUPERVISED_FEATURES",
    "PROBING_FEATURES",
    "get_prediction_head_input_columns",
    "build_prediction_head",
    "build_sequential_dataframe",
    "split_train_test",
    "prepare_prediction_head_data",
    "fit_prediction_head",
    "evaluate_predictions",
]

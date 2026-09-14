import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_predictions(y_true, y_pred):
    """Evaluate a partial-quality or partial-utility prediction head."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "pearson": pearsonr(y_true, y_pred)[0],
        "spearman": spearmanr(y_true, y_pred)[0],
        "kendall": kendalltau(y_true, y_pred)[0],
    }

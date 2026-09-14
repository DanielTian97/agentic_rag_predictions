from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import get_prediction_head_input_columns
from .mlp import build_prediction_head
from .sequential import build_sequential_dataframe, split_train_test


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "prediction_results"

FEATURE_GROUP_ALIASES = {
    ("unsupervised", "supervised"): "unsup_sup",
    ("unsupervised", "supervised", "probing"): "unsup_sup_prob",
}


def prediction_output_dir(
    signal_groups,
    output_root=DEFAULT_OUTPUT_ROOT,
    pipeline=None,
    dataset=None,
    window_size=None,
):
    """Return the directory used to persist one prediction-head configuration."""
    groups = tuple(signal_groups)
    group_name = FEATURE_GROUP_ALIASES.get(groups, "_".join(groups))

    path = Path(output_root) / group_name
    if pipeline is not None:
        path /= str(pipeline)
    if dataset is not None:
        path /= str(dataset)
    if window_size is not None:
        path /= f"w{window_size}"
    return path


def save_prediction_output(
    predictions,
    target,
    signal_groups,
    output_root=DEFAULT_OUTPUT_ROOT,
    pipeline=None,
    dataset=None,
    window_size=None,
):
    """Persist continuous prediction-head outputs for later controller use."""
    if target not in {"performance", "utility"}:
        raise ValueError("target must be 'performance' or 'utility'")

    output_dir = prediction_output_dir(
        signal_groups=signal_groups,
        output_root=output_root,
        pipeline=pipeline,
        dataset=dataset,
        window_size=window_size,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{target}.csv"
    predictions.to_csv(output_path, index=False)
    return output_path


def prepare_prediction_head_data(
    dataframe,
    target,
    window_size=1,
    signal_groups=("unsupervised", "supervised", "probing"),
    include_type=True,
    include_iteration=False,
):
    """Prepare the sequential feature matrix consumed by the prediction head."""
    if target not in {"performance", "utility"}:
        raise ValueError("target must be 'performance' or 'utility'")

    input_columns = get_prediction_head_input_columns(
        signal_groups=signal_groups,
        include_iteration=include_iteration,
    )

    missing = [c for c in input_columns if c not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing prediction-head features: {missing}")

    seq_df, sequential_columns = build_sequential_dataframe(
        dataframe=dataframe,
        input_columns=input_columns,
        window_size=window_size,
        target=target,
        include_type=include_type,
    )
    return seq_df, sequential_columns


def fit_prediction_head(
    dataframe,
    target,
    window_size=1,
    signal_groups=("unsupervised", "supervised", "probing"),
    include_type=True,
    include_iteration=False,
    random_state=42,
    save_output=False,
    output_root=DEFAULT_OUTPUT_ROOT,
    pipeline=None,
    dataset=None,
):
    """Fit the camera-ready MLP prediction head and return test predictions.

    Set ``save_output=True`` to persist the continuous predictions as CSV so
    that controller experiments can load them independently later.
    """
    seq_df, input_columns = prepare_prediction_head_data(
        dataframe=dataframe,
        target=target,
        window_size=window_size,
        signal_groups=signal_groups,
        include_type=include_type,
        include_iteration=include_iteration,
    )

    train_df, test_df, X_train, y_train, X_test, y_test = split_train_test(
        seq_df=seq_df,
        input_columns=input_columns,
        target=target,
    )

    if not pd.api.types.is_numeric_dtype(y_train):
        raise TypeError(f"{target} must be numeric")

    model = build_prediction_head(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Keep the explicit iteration id in the prediction output so downstream
    # controller code never has to infer trajectory order from qid strings.
    metadata = dataframe[["qid", "original_qid", "sub_qid"]].drop_duplicates()
    predictions = test_df[["qid", "original_qid", target]].merge(
        metadata,
        on=["qid", "original_qid"],
        how="left",
        validate="one_to_one",
    )
    if predictions["sub_qid"].isna().any():
        raise ValueError("prediction output is missing sub_qid metadata")

    predictions[f"predicted_{target}"] = y_pred
    predictions["sub_qid"] = predictions["sub_qid"].astype(int)

    output_path = None
    if save_output:
        output_path = save_prediction_output(
            predictions=predictions,
            target=target,
            signal_groups=signal_groups,
            output_root=output_root,
            pipeline=pipeline,
            dataset=dataset,
            window_size=window_size,
        )

    return model, predictions, (y_test.to_numpy(), y_pred), output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the CIKM 2026 MLP prediction head and emit held-out predictions."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Merged feature CSV. It must include a split column with train/test rows.",
    )
    parser.add_argument(
        "--target",
        choices=["performance", "utility"],
        required=True,
        help="Prediction target: partial answer quality (performance) or partial utility.",
    )
    parser.add_argument("--window-size", type=int, choices=[1, 3, 5], default=1)
    parser.add_argument(
        "--signal-groups",
        nargs="+",
        choices=["unsupervised", "supervised", "probing"],
        default=["unsupervised", "supervised", "probing"],
    )
    parser.add_argument(
        "--include-iteration",
        action="store_true",
        help="Include the optional iteration-index feature (disabled in the main Table 3 signal groups).",
    )
    parser.add_argument(
        "--no-type",
        action="store_true",
        help="Do not include the dataset question-type indicator when present.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pipeline", default=None)
    parser.add_argument("--dataset", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    dataframe = pd.read_csv(args.input_csv)

    _, predictions, _, output_path = fit_prediction_head(
        dataframe=dataframe,
        target=args.target,
        window_size=args.window_size,
        signal_groups=tuple(args.signal_groups),
        include_type=not args.no_type,
        include_iteration=args.include_iteration,
        random_state=args.random_state,
        save_output=args.save_output,
        output_root=args.output_root,
        pipeline=args.pipeline,
        dataset=args.dataset,
    )

    print(f"Generated {len(predictions)} held-out {args.target} predictions.")
    if output_path is not None:
        print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()

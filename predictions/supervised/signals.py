"""Merge qid-aligned supervised regression outputs into the six paper features."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RELATION_SHORT = {
    "adjacent_think": "at",
    "long_distance": "ld",
    "intra_iteration": "ii",
}

SUPERVISED_SIGNAL_COLUMNS = [
    "u_at_pred",
    "u_ld_pred",
    "u_ii_pred",
    "p_at_pred",
    "p_ld_pred",
    "p_ii_pred",
]


def _load_signal(path: Path, feature_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"qid", "pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    out = df[["qid", "pred"]].rename(columns={"pred": feature_name})
    if out["qid"].duplicated().any():
        raise ValueError(f"duplicate qid in {path}")
    return out


def merge_supervised_signals(
    *,
    utility_adjacent: Path,
    utility_long_distance: Path,
    utility_intra_iteration: Path,
    performance_adjacent: Path,
    performance_long_distance: Path,
    performance_intra_iteration: Path,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    sources = {
        "u_at_pred": utility_adjacent,
        "u_ld_pred": utility_long_distance,
        "u_ii_pred": utility_intra_iteration,
        "p_at_pred": performance_adjacent,
        "p_ld_pred": performance_long_distance,
        "p_ii_pred": performance_intra_iteration,
    }

    merged = None
    for feature_name, path in sources.items():
        frame = _load_signal(Path(path), feature_name)
        merged = frame if merged is None else merged.merge(frame, on="qid", how="inner")

    if merged is None or merged.empty:
        raise ValueError("no aligned supervised rows were produced")

    merged = merged[["qid", *SUPERVISED_SIGNAL_COLUMNS]]
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_csv, index=False)
    return merged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utility-adjacent", type=Path, required=True)
    parser.add_argument("--utility-long-distance", type=Path, required=True)
    parser.add_argument("--utility-intra-iteration", type=Path, required=True)
    parser.add_argument("--performance-adjacent", type=Path, required=True)
    parser.add_argument("--performance-long-distance", type=Path, required=True)
    parser.add_argument("--performance-intra-iteration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    merge_supervised_signals(
        utility_adjacent=args.utility_adjacent,
        utility_long_distance=args.utility_long_distance,
        utility_intra_iteration=args.utility_intra_iteration,
        performance_adjacent=args.performance_adjacent,
        performance_long_distance=args.performance_long_distance,
        performance_intra_iteration=args.performance_intra_iteration,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()

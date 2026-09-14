"""Prepare continuous P_i/U_i regression data for the three trajectory relations."""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


RELATION_COLUMNS = {
    "adjacent_think": [
        "orig_qid", "sub_qid", "prev_r", "prev_q", "curr_r", "curr_q", "query",
    ],
    "intra_iteration": [
        "orig_qid", "sub_qid", "query", "curr_r", "curr_q", "curr_title", "curr_t",
    ],
    "long_distance": [
        "orig_qid", "sub_qid", "orig_r", "curr_r", "query", "curr_q",
    ],
}


def _truncate(series: pd.Series, length: int) -> pd.Series:
    return series.fillna("").astype(str).str.slice(0, length)


def _parse_numeric_sequence(value, *, take_first: bool = False) -> list[float]:
    if isinstance(value, (list, tuple)):
        parsed = value
    else:
        try:
            parsed = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            parsed = None

    if take_first and isinstance(parsed, (list, tuple)) and parsed:
        if isinstance(parsed[0], (list, tuple)):
            parsed = parsed[0]

    if isinstance(parsed, (list, tuple)):
        try:
            return [float(x) for x in parsed]
        except (TypeError, ValueError):
            pass

    numbers = re.findall(
        r"tensor\(\s*([-+]?(?:\d*\.\d+|\d+\.?)(?:e[-+]?\d+)?)",
        str(value),
        flags=re.IGNORECASE,
    )
    if not numbers:
        raise ValueError(f"cannot parse numeric sequence: {value!r}")
    return [float(x) for x in numbers]


def build_r1s_targets(generation_csv: Path, evaluation_csv: Path, dataset: str) -> pd.DataFrame:
    """Recover continuous P_i and U_i targets from R1-Searcher probing traces."""
    generation = pd.read_csv(generation_csv)
    evaluation = pd.read_csv(evaluation_csv)
    merged = generation.merge(evaluation, on="qid")

    iteration_col = "search_iterations" if "search_iterations" in merged.columns else "iteration"
    required = {"qid", iteration_col, "probed_f1_after_think"}
    missing = required.difference(merged.columns)
    if missing:
        raise ValueError(f"R1-Searcher probing files are missing columns: {sorted(missing)}")

    rows = []
    skipped = 0
    for row in merged.itertuples(index=False):
        try:
            f1 = _parse_numeric_sequence(getattr(row, "probed_f1_after_think"))
            n = int(getattr(row, iteration_col))
            if len(f1) != n + 1:
                raise ValueError("probe count does not match iteration count")

            for i in range(n):
                rows.append(
                    {
                        "qid": f"{dataset}-{row.qid}-{i}",
                        "utility": f1[i + 1] - f1[i],
                        "performance": f1[i + 1],
                    }
                )
        except (ValueError, TypeError, AttributeError):
            skipped += 1

    if not rows:
        raise ValueError("no valid R1-Searcher target rows were produced")
    result = pd.DataFrame(rows)
    print(f"[prepare] built {len(result)} target rows; skipped {skipped} malformed traces")
    return result


def build_r1_targets(dataframe_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(dataframe_csv)
    required = {"qid", "utility", "performance"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Search-R1 target dataframe is missing columns: {sorted(missing)}")
    return df[["qid", "utility", "performance"]].copy()


def build_relation_frame(think_csv: Path, relation: str, dataset: str) -> pd.DataFrame:
    """Build one relation-specific input dataframe from parsed trajectory states."""
    if relation not in RELATION_COLUMNS:
        raise ValueError(f"unknown relation: {relation}")

    df = pd.read_csv(think_csv).sort_values(["orig_qid", "sub_qid"]).copy()
    required = {"orig_qid", "sub_qid", "query", "r", "q"}
    if relation == "intra_iteration":
        required.update({"title", "text"})
    if relation == "long_distance":
        required.add("orig_r")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"think dataframe is missing columns: {sorted(missing)}")

    df["curr_r"] = _truncate(df["r"], 256)
    df["curr_q"] = _truncate(df["q"], 256)

    if relation == "adjacent_think":
        df["prev_r"] = _truncate(df.groupby("orig_qid")["r"].shift(1), 256)
        df["prev_q"] = _truncate(df.groupby("orig_qid")["q"].shift(1), 256)
    elif relation == "intra_iteration":
        df["curr_title"] = _truncate(df["title"], 128)
        df["curr_t"] = _truncate(df["text"], 256)
    else:
        df["orig_r"] = _truncate(df["orig_r"], 256)

    # sub_qid=0 is the reasoning state before any retrieval. Supervised rows
    # correspond to retrieval-reasoning iterations, so start from sub_qid=1.
    result = df.loc[df["sub_qid"] >= 1, RELATION_COLUMNS[relation]].copy()
    result["sub_qid"] = result["sub_qid"].astype(int) - 1
    result["qid"] = result.apply(
        lambda row: f"{dataset}-{row.orig_qid}-{row.sub_qid}", axis=1
    )
    return result


def prepare_dataset(
    *,
    think_csv: Path,
    relation: str,
    dataset: str,
    rag_model: str,
    output_csv: Path,
    target_dataframe: Path | None = None,
    generation_csv: Path | None = None,
    evaluation_csv: Path | None = None,
) -> pd.DataFrame:
    inputs = build_relation_frame(think_csv, relation, dataset)

    if rag_model == "r1":
        if target_dataframe is None:
            raise ValueError("target_dataframe is required for rag_model='r1'")
        targets = build_r1_targets(target_dataframe)
    elif rag_model == "r1s":
        if generation_csv is None or evaluation_csv is None:
            raise ValueError("generation_csv and evaluation_csv are required for rag_model='r1s'")
        targets = build_r1s_targets(generation_csv, evaluation_csv, dataset)
    else:
        raise ValueError(f"unknown rag model: {rag_model}")

    result = inputs.merge(targets, on="qid", how="inner")
    if result.empty:
        raise ValueError("input/target merge produced zero rows")

    result["utility"] = result["utility"].astype(float)
    result["performance"] = result["performance"].astype(float)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def merge_datasets(inputs: Iterable[Path], output_csv: Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(path).drop(columns=["orig_qid", "sub_qid"], errors="ignore")
        for path in inputs
    ]
    if not frames:
        raise ValueError("at least one input is required")
    merged = pd.concat(frames, ignore_index=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--relation", choices=sorted(RELATION_COLUMNS), required=True)
    prep.add_argument("--dataset", required=True)
    prep.add_argument("--rag-model", choices=["r1", "r1s"], required=True)
    prep.add_argument("--think-csv", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--target-dataframe", type=Path)
    prep.add_argument("--generation-csv", type=Path)
    prep.add_argument("--evaluation-csv", type=Path)

    merge = sub.add_parser("merge")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        prepare_dataset(
            think_csv=args.think_csv,
            relation=args.relation,
            dataset=args.dataset,
            rag_model=args.rag_model,
            output_csv=args.output,
            target_dataframe=args.target_dataframe,
            generation_csv=args.generation_csv,
            evaluation_csv=args.evaluation_csv,
        )
    else:
        merge_datasets(args.input, args.output)


if __name__ == "__main__":
    main()

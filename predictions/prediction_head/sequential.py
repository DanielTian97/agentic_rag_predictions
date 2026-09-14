from functools import reduce

import numpy as np
import pandas as pd


def build_sequential_dataframe(
    dataframe,
    input_columns,
    window_size,
    target,
    extra_columns=None,
    include_type=False,
):
    """Construct the fixed-length sequential feature window used by the paper."""
    if extra_columns is None:
        extra_columns = []

    df_list = []
    sequential_input_columns = []

    for i in range(window_size):
        temp_df = dataframe[input_columns + ["qid", "sub_qid"]].copy()

        temp_df["sub_qid"] = temp_df["sub_qid"].apply(
            lambda x: x + (i + 1 - window_size)
        )

        temp_df.loc[temp_df["sub_qid"] < 0, input_columns] = np.nan

        renamed_columns = {c: f"{c}_{i}" for c in input_columns}
        temp_df = temp_df.rename(columns=renamed_columns)
        temp_df = temp_df.drop(columns=["sub_qid"])

        df_list.append(temp_df)
        sequential_input_columns += list(renamed_columns.values())

    seq_df = reduce(
        lambda left, right: pd.merge(left, right, on=["qid"]),
        df_list,
    )

    merge_columns = ["qid", "original_qid", target, "split"] + extra_columns
    if include_type and "type" not in merge_columns:
        merge_columns.append("type")

    seq_df = pd.merge(seq_df, dataframe[merge_columns], on=["qid"])

    if include_type and "type" in seq_df.columns:
        seq_df = pd.get_dummies(seq_df, columns=["type"], prefix="type")
        bool_cols = [c for c in seq_df.columns if seq_df[c].dtype == "bool"]
        seq_df[bool_cols] = seq_df[bool_cols].astype(int)
        sequential_input_columns += bool_cols

    return seq_df, sequential_input_columns


def split_train_test(seq_df, input_columns, target):
    train_df = seq_df.query("split == 'train'")
    test_df = seq_df.query("split == 'test'")

    X_train = train_df[input_columns]
    y_train = train_df[target]
    X_test = test_df[input_columns]
    y_test = test_df[target]

    return train_df, test_df, X_train, y_train, X_test, y_test

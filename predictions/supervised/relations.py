"""Relation-specific text construction for the supervised regressors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class RelationSpec:
    special_tokens: tuple[str, ...]
    formatter: Callable[[pd.Series], str]


def _adjacent(row: pd.Series) -> str:
    return (
        f"{row['query']} [SEP] [PREV] {row['prev_r']} [PREV_SUBQ] {row['prev_q']} "
        f"[SEP] [CURR] {row['curr_r']} [CURR_SUBQ] {row['curr_q']}"
    )


def _long_distance(row: pd.Series) -> str:
    return (
        f"{row['query']} [SEP] [ORIG] {row['orig_r']} [SEP] "
        f"[CURR_SUBQ] {row['curr_q']} [CURR] {row['curr_r']}"
    )


def _intra_iteration(row: pd.Series) -> str:
    return (
        f"{row['query']} [SEP] {row['curr_r']} [CURR_SUBQ] {row['curr_q']} "
        f"[CURR_TITLE] {row['curr_title']} [CURR_INFO] {row['curr_t']}"
    )


RELATIONS = {
    "adjacent_think": RelationSpec(
        special_tokens=("[PREV]", "[CURR]", "[PREV_SUBQ]", "[CURR_SUBQ]"),
        formatter=_adjacent,
    ),
    "long_distance": RelationSpec(
        special_tokens=("[ORIG]", "[CURR]", "[CURR_SUBQ]"),
        formatter=_long_distance,
    ),
    "intra_iteration": RelationSpec(
        special_tokens=("[CURR_SUBQ]", "[CURR_TITLE]", "[CURR_INFO]"),
        formatter=_intra_iteration,
    ),
}


class RelationRegressionDataset(torch.utils.data.Dataset):
    """Continuous P_i/U_i regression dataset for one trajectory relation."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        relation: str,
        target: str,
        max_length: int = 1024,
    ) -> None:
        if relation not in RELATIONS:
            raise ValueError(f"unknown relation: {relation}")
        if target not in {"utility", "performance"}:
            raise ValueError(f"unknown target: {target}")
        if target not in dataframe.columns:
            raise ValueError(f"dataframe is missing target column: {target}")

        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.relation = relation
        self.target = target
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = RELATIONS[self.relation].formatter(row)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        return {
            "id": idx,
            "input_ids": input_ids,
            "attention_mask": torch.ones(input_ids.size(0), dtype=torch.long),
            "labels": torch.tensor(float(row[self.target]), dtype=torch.float32),
        }


def build_tokenizer(relation: str, model_id: str = DEFAULT_MODEL_ID):
    if relation not in RELATIONS:
        raise ValueError(f"unknown relation: {relation}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(RELATIONS[relation].special_tokens)}
    )
    return tokenizer


def build_regression_model(
    relation: str,
    tokenizer,
    model_id: str = DEFAULT_MODEL_ID,
    *,
    dropout_rate: float | None = None,
):
    if relation not in RELATIONS:
        raise ValueError(f"unknown relation: {relation}")

    kwargs = {"num_labels": 1, "problem_type": "regression"}
    if dropout_rate is not None:
        kwargs.update(
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )

    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    model.resize_token_embeddings(len(tokenizer))
    return model


__all__ = [
    "DEFAULT_MODEL_ID",
    "RELATIONS",
    "RelationRegressionDataset",
    "build_tokenizer",
    "build_regression_model",
]

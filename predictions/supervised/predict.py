"""Run a trained supervised trajectory-relation regressor and emit qid-aligned predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding

from .relations import (
    DEFAULT_MODEL_ID,
    RELATIONS,
    RelationRegressionDataset,
    build_regression_model,
    build_tokenizer,
)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "prediction_results"


def _strip_ddp_prefix(state_dict: dict) -> dict:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def predict(
    *,
    relation: str,
    target: str,
    data_csv: Path,
    checkpoint: Path,
    output_csv: Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 32,
    max_length: int = 1024,
) -> pd.DataFrame:
    if relation not in RELATIONS:
        raise ValueError(f"unknown relation: {relation}")
    if target not in {"utility", "performance"}:
        raise ValueError(f"unknown target: {target}")

    df = pd.read_csv(data_csv)
    if "qid" not in df.columns:
        raise ValueError("dataframe is missing qid")

    tokenizer = build_tokenizer(relation, model_id)
    model = build_regression_model(relation, tokenizer, model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(_strip_ddp_prefix(state_dict))
    model.to(device)
    model.eval()

    dataset = RelationRegressionDataset(df, tokenizer, relation, target, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )

    preds: list[float] = []
    labels: list[float] = []
    ids: list[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_preds = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits.squeeze(-1)
            preds.extend(batch_preds.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
            ids.extend(batch["id"].cpu().tolist())

    output = pd.DataFrame(
        {
            "qid": df.iloc[ids]["qid"].to_numpy(),
            "pred": preds,
            "label": labels,
        }
    )

    if output_csv is None:
        output_csv = DEFAULT_OUTPUT_ROOT / f"{relation}_{target}.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation", choices=sorted(RELATIONS), required=True)
    parser.add_argument("--target", choices=["utility", "performance"], required=True)
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser


def main() -> None:
    args = _parser().parse_args()
    predict(
        relation=args.relation,
        target=args.target,
        data_csv=args.data_csv,
        checkpoint=args.checkpoint,
        output_csv=args.output,
        model_id=args.model_id,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()

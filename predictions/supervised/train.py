"""Train one supervised trajectory-relation regressor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim import AdamW
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


def evaluate(labels: list[float], preds: list[float]) -> dict[str, float]:
    y = np.asarray(labels)
    p = np.asarray(preds)
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)),
        "pearson": float(pearsonr(y, p)[0]) if len(y) > 1 else float("nan"),
        "spearman": float(spearmanr(y, p)[0]) if len(y) > 1 else float("nan"),
    }


def train(
    *,
    relation: str,
    target: str,
    train_csv: Path,
    validation_csv: Path,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 8,
    effective_batch_size: int = 256,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.0,
    dropout_rate: float = 0.0,
    max_length: int = 1024,
    loss: str = "mse",
    seed: int = 42,
) -> None:
    if relation not in RELATIONS:
        raise ValueError(f"unknown relation: {relation}")
    if target not in {"utility", "performance"}:
        raise ValueError(f"unknown target: {target}")
    if effective_batch_size % batch_size:
        raise ValueError("effective_batch_size must be divisible by batch_size")

    torch.manual_seed(seed)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(validation_csv)

    tokenizer = build_tokenizer(relation, model_id)
    model = build_regression_model(
        relation, tokenizer, model_id, dropout_rate=dropout_rate
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = RelationRegressionDataset(train_df, tokenizer, relation, target, max_length)
    val_ds = RelationRegressionDataset(val_df, tokenizer, relation, target, max_length)
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss() if loss == "mse" else torch.nn.HuberLoss(delta=1.0)
    accumulation_steps = effective_batch_size // batch_size

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "relation": relation,
        "target": target,
        "model_id": model_id,
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate,
        "max_length": max_length,
        "loss": loss,
        "seed": seed,
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"train epoch {epoch}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).float()
            preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
            batch_loss = criterion(preds, labels) / accumulation_steps
            batch_loss.backward()
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        labels_all: list[float] = []
        preds_all: list[float] = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"eval epoch {epoch}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device).float()
                preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
                labels_all.extend(labels.cpu().tolist())
                preds_all.extend(preds.cpu().tolist())

        metrics = evaluate(labels_all, preds_all)
        print(f"epoch={epoch} {metrics}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": metadata,
                "metrics": metrics,
            },
            output_dir / f"checkpoint-epoch_{epoch}.pt",
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation", choices=sorted(RELATIONS), required=True)
    parser.add_argument("--target", choices=["utility", "performance"], required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--validation-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--effective-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout-rate", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _parser().parse_args()
    train(**vars(args))


if __name__ == "__main__":
    main()

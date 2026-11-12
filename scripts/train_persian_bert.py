#!/usr/bin/env python3
"""Fine-tune ParsBERT for MASSIVE fa-IR intent classification.

By default, train/validation are loaded first. The official test split is not
loaded until after the best checkpoint has been selected by validation
Macro-F1. Use --validation-only for seed/normalization experiments; this mode
never loads or evaluates the test split.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from persian_intent.text_normalization import normalize_persian_text

DEFAULT_DATASET = "AmazonScience/massive"
DEFAULT_DATASET_CONFIG = "fa-IR"
# A fixed public snapshot with the PyTorch checkpoint and tokenizer files.
DEFAULT_MODEL = "HooshvareLab/bert-fa-base-uncased"
DEFAULT_MODEL_REVISION = "a04aa40c97bcdde570ae11986a534542c2995a62"
DEFAULT_DATASET_REVISION = "9b095d98489fd72fef8fcde2aee186df56b7d281"


class UtteranceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, normalize: bool = False):
        texts = frame["utt"].fillna("").astype(str).tolist()
        self.texts = [normalize_persian_text(x) for x in texts] if normalize else texts
        self.labels = frame["intent"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"text": self.texts[index], "label": self.labels[index]}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def package_versions() -> dict[str, str]:
    packages = ["torch", "transformers", "tokenizers", "numpy", "pandas", "scikit-learn", "datasets", "pyarrow"]
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def load_frame_from_parquet(path: Path, split_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{split_name} Parquet file not found: {path}")
    try:
        frame = pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError("Parquet input requires pyarrow: pip install pyarrow") from exc
    required = {"utt", "intent"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{split_name} Parquet is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["intent"]).copy()
    frame["utt"] = frame["utt"].fillna("").astype(str)
    frame["intent"] = frame["intent"].astype(int)
    return frame.reset_index(drop=True)


def load_splits(args: argparse.Namespace, class_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    parquet_mode = args.train_parquet is not None or args.validation_parquet is not None
    if parquet_mode:
        if args.train_parquet is None or args.validation_parquet is None:
            raise ValueError("When using Parquet, provide both --train-parquet and --validation-parquet.")
        train_df = load_frame_from_parquet(args.train_parquet, "train")
        val_df = load_frame_from_parquet(args.validation_parquet, "validation")
        source = "local_parquet"
    else:
        from datasets import load_dataset
        train_ds = load_dataset(
            args.dataset, args.dataset_config, split="train", revision=args.dataset_revision
        )
        val_ds = load_dataset(
            args.dataset, args.dataset_config, split="validation", revision=args.dataset_revision
        )
        names = train_ds.features["intent"].names
        if not names or len(names) != 60:
            raise ValueError("Expected 60 class names in MASSIVE intent feature metadata.")
        class_names[:] = list(names)
        train_df = train_ds.to_pandas()[["utt", "intent"]]
        val_df = val_ds.to_pandas()[["utt", "intent"]]
        source = "huggingface_datasets"
    for split_name, frame in [("train", train_df), ("validation", val_df)]:
        if frame["intent"].min() < 0 or frame["intent"].max() >= len(class_names):
            raise ValueError(f"{split_name} intent IDs fall outside the 0..{len(class_names)-1} mapping.")
    return train_df, val_df, source


def make_collate_fn(tokenizer, max_length: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [item["text"] for item in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return encoded
    return collate


@torch.no_grad()
def evaluate(model, loader, device, num_labels: int) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")
        logits = model(**batch).logits
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    labels_range = list(range(num_labels))
    metrics = {
        "n_examples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_all_labels": float(f1_score(y_true, y_pred, labels=labels_range, average="macro", zero_division=0)),
        "macro_f1_labels_present_in_split": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels_range, average="weighted", zero_division=0)),
    }
    return metrics, y_true, y_pred


def write_evaluation(output_dir: Path, split_name: str, y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict[str, float]:
    num_labels = len(class_names)
    metrics = {
        "n_examples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_all_labels": float(f1_score(y_true, y_pred, labels=list(range(num_labels)), average="macro", zero_division=0)),
        "macro_f1_labels_present_in_split": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(range(num_labels)), average="weighted", zero_division=0)),
    }
    report = classification_report(
        y_true, y_pred, labels=list(range(num_labels)), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    rows = []
    for label in class_names:
        item = report[label]
        rows.append({"intent": label, **{k: item[k] for k in ("precision", "recall", "f1-score", "support")}})
    pd.DataFrame(rows).to_csv(output_dir / f"per_class_metrics_{split_name}.csv", index=False, encoding="utf-8-sig")
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_labels)))
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output_dir / f"confusion_matrix_{split_name}.csv", encoding="utf-8-sig"
    )
    return metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--train-parquet", type=Path, default=None)
    parser.add_argument("--validation-parquet", type=Path, default=None)
    parser.add_argument("--test-parquet", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/persian-bert"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-variant", choices=["raw", "normalized"], default="raw")
    parser.add_argument("--validation-only", action="store_true", help="Do not load or evaluate the test split.")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8:
        parser.error("epochs and batch-size must be positive; max-length must be >= 8")
    if not args.validation_only and args.test_parquet is not None and not args.test_parquet.is_file():
        parser.error(f"Test Parquet file not found: {args.test_parquet}")

    labels_path = Path(__file__).resolve().parents[1] / "data" / "intent_labels.json"
    class_names = json.loads(labels_path.read_text(encoding="utf-8"))
    if len(class_names) != 60:
        raise ValueError(f"Expected 60 labels, found {len(class_names)} in {labels_path}")

    set_seed(args.seed)
    train_df, val_df, data_source = load_splits(args, class_names)
    print(f"Loaded rows: train={len(train_df)}, validation={len(val_df)}")
    print(f"Text variant: {args.text_variant}; data source: {data_source}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.model_revision,
        num_labels=len(class_names),
        id2label={i: name for i, name in enumerate(class_names)},
        label2id={name: i for i, name in enumerate(class_names)},
    )
    resolved_model_revision = getattr(model.config, "_commit_hash", None) or args.model_revision

    normalize = args.text_variant == "normalized"
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        UtteranceDataset(train_df, normalize=normalize), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, generator=generator,
    )
    val_loader = DataLoader(
        UtteranceDataset(val_df, normalize=normalize), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; labels: {len(class_names)}; model: {args.model}@{resolved_model_revision}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    best_dir = args.output_dir / "best_model"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics, _, _ = evaluate(model, val_loader, device, len(class_names))
        history.append({"epoch": epoch, "mean_training_loss": float(np.mean(losses)), "validation": val_metrics})
        print(f"Epoch {epoch}/{args.epochs} | loss={np.mean(losses):.4f} | val_accuracy={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1_all_labels']:.4f}")
        if val_metrics["macro_f1_all_labels"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1_all_labels"]
            best_epoch = epoch
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    final_val_metrics, val_true, val_pred = evaluate(best_model, val_loader, device, len(class_names))
    validation_metrics = write_evaluation(args.output_dir, "validation", val_true, val_pred, class_names)
    results: dict[str, Any] = {
        "model": args.model,
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": resolved_model_revision,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision if data_source == "huggingface_datasets" else None,
        "data_source": data_source,
        "text_variant": args.text_variant,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "best_epoch_selected_by_validation_macro_f1": best_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "device": str(device),
        "split_sizes": {"train": len(train_df), "validation": len(val_df)},
        "number_of_intent_labels": len(class_names),
        "class_names": class_names,
        "training_history": history,
        "validation": validation_metrics,
        "environment": package_versions(),
        "notes": [
            "The checkpoint was selected by validation Macro-F1.",
            "The test split is not loaded in --validation-only mode.",
            "CUDA kernels may retain minor nondeterminism even when random seeds are fixed.",
        ],
    }

    if not args.validation_only:
        if args.train_parquet is not None or args.validation_parquet is not None:
            if args.test_parquet is None:
                parser.error("Parquet mode requires --test-parquet unless --validation-only is set.")
            test_df = load_frame_from_parquet(args.test_parquet, "test")
        else:
            from datasets import load_dataset
            test_ds = load_dataset(
                args.dataset, args.dataset_config, split="test", revision=args.dataset_revision
            )
            test_df = test_ds.to_pandas()[["utt", "intent"]]
        test_loader = DataLoader(
            UtteranceDataset(test_df, normalize=normalize), batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn,
        )
        test_metrics, y_true, y_pred = evaluate(best_model, test_loader, device, len(class_names))
        results["split_sizes"]["test"] = len(test_df)
        results["test"] = write_evaluation(args.output_dir, "test", y_true, y_pred, class_names)
        results["test"]["n_examples"] = test_metrics["n_examples"]

    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\nFinal metrics:")
    print(json.dumps({"validation": results["validation"], "test": results.get("test")}, indent=2))
    print(f"Best epoch: {best_epoch}; output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
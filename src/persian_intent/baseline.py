"""Train and evaluate a TF-IDF + logistic-regression baseline on MASSIVE fa-IR.

The existing notebooks are preserved as exploratory experiments. This module
provides a small, reproducible intent-classification baseline using the official
MASSIVE train, validation, and test splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

DEFAULT_DATASET = "AmazonScience/massive"
DEFAULT_CONFIG = "fa-IR"


def build_pipeline(max_features: int = 80_000, max_iter: int = 1_000) -> Pipeline:
    """Build a word/character TF-IDF classifier.

    Word n-grams capture lexical phrases; character n-grams can help with
    spelling and morphology variation. These are baseline features, not a
    claim that this setup is optimal for Persian.
    """
    if max_features < 1:
        raise ValueError("max_features must be a positive integer")
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", LogisticRegression(max_iter=max_iter)),
        ]
    )


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], intent_names: Sequence[str]
) -> dict[str, Any]:
    """Return overall scores and a per-intent classification report."""
    labels = list(range(len(intent_names)))
    return {
        "n_examples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=list(intent_names),
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_split(
    model: Pipeline, split: Mapping[str, Sequence[Any]], intent_names: Sequence[str]
) -> dict[str, Any]:
    """Evaluate a fitted model on one MASSIVE split."""
    texts = split["utt"]
    labels = split["intent"]
    predictions = model.predict(texts)
    return classification_metrics(labels, predictions, intent_names)


def train_and_evaluate(
    dataset_name: str = DEFAULT_DATASET,
    config: str = DEFAULT_CONFIG,
    output_dir: Path | str = Path("artifacts/persian-intent-baseline"),
    max_features: int = 80_000,
    max_iter: int = 1_000,
) -> dict[str, Any]:
    """Load MASSIVE, fit on train only, evaluate val/test, and save artifacts."""
    # Import on demand so metrics and pipeline unit tests do not require a
    # network connection or download the Hugging Face dataset.
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, config)
    intent_feature = dataset["train"].features["intent"]
    intent_names = getattr(intent_feature, "names", None)
    if not intent_names:
        raise ValueError(
            "The dataset's 'intent' feature does not expose class names; "
            "please check the selected dataset/configuration."
        )

    model = build_pipeline(max_features=max_features, max_iter=max_iter)
    model.fit(dataset["train"]["utt"], dataset["train"]["intent"])

    results: dict[str, Any] = {
        "dataset": dataset_name,
        "config": config,
        "target": "intent",
        "training_examples": len(dataset["train"]),
        "validation": evaluate_split(model, dataset["validation"], intent_names),
        "test": evaluate_split(model, dataset["test"], intent_names),
        "notes": [
            "The classifier was fitted on the official training split only.",
            "Use validation results for model selection; reserve test results for final reporting.",
            "Scores are for this baseline and this dataset configuration, not evidence of deployment readiness.",
        ],
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": model,
            "intent_names": list(intent_names),
            "dataset": dataset_name,
            "config": config,
        },
        destination / "model.joblib",
    )
    (destination / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset ID")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Dataset configuration (default: fa-IR)")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/persian-intent-baseline")
    )
    parser.add_argument("--max-features", type=int, default=80_000)
    parser.add_argument("--max-iter", type=int, default=1_000)
    args = parser.parse_args()

    results = train_and_evaluate(
        dataset_name=args.dataset,
        config=args.config,
        output_dir=args.output_dir,
        max_features=args.max_features,
        max_iter=args.max_iter,
    )
    for split_name in ("validation", "test"):
        metrics = results[split_name]
        print(
            f"{split_name}: n={metrics['n_examples']} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f}"
        )
    print(f"Saved model and metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()

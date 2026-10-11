# Persian Intent Recognition and Slot Filling

This repository collects experiments on **Persian-language understanding** using the Persian (`fa-IR`) configuration of the [MASSIVE dataset](https://github.com/amazon-science/massive). It explores related but distinct tasks: classifying utterance intents, labeling slot values, and generating example utterances conditioned on an intent.

To make the experiments easier to inspect and reproduce, the repository now includes a small, clean **TF–IDF + logistic-regression intent-classification baseline** alongside the original exploratory notebooks. The notebooks are preserved rather than silently rewritten.

## Project scope

- **Intent classification:** predict the intent label from a Persian utterance.
- **Slot filling:** assign token-level labels to values expressed in an utterance.
- **Intent-conditioned generation:** explore generating utterances from intent descriptions.

These are different tasks and their metrics are not directly comparable.

## Selected experiments

| Path | Focus |
| --- | --- |
| `src/persian_intent/baseline.py` | Reproducible intent-classification baseline using word and character TF–IDF features with logistic regression |
| `Part1.ipynb` | Exploratory baseline/classifier experiments. Some cells target the dataset's `scenario` field, so check the target column before interpreting results as intent classification |
| `Part2.ipynb` | Slot-label preprocessing, an LSTM-CRF experiment, and a Persian BERT token-classification experiment |
| `Part3.ipynb` | Exploratory ParsT5 experiment that generates utterances conditioned on intent descriptions |

## Clean baseline: setup and run

The baseline uses the official MASSIVE training split for fitting, and reports separate validation and test metrics. It writes the trained model and detailed metrics under `artifacts/`, which is ignored by Git.

### Requirements

- Python 3.10+
- Internet access on first run to download MASSIVE and any required package/model artifacts

Create an environment and install the project:

~~~bash
python -m venv .venv
source .venv/bin/activate
# On Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev]"
~~~

Run the baseline:

~~~bash
python -m persian_intent.baseline
~~~

Optional arguments:

~~~bash
python -m persian_intent.baseline --max-features 80000 --max-iter 1000 --output-dir artifacts/persian-intent-baseline
~~~

The output directory contains:

- `model.joblib` — fitted scikit-learn pipeline plus dataset/label metadata
- `metrics.json` — validation and test accuracy, macro-F1, weighted-F1, and per-intent classification reports

Use the **validation** split for model selection. Treat the **test** score as final reporting only; repeated tuning on the test set makes it less informative.

## Tests

The unit tests use small in-memory examples and do not download MASSIVE:

~~~bash
pytest
~~~

## Original notebooks

The three notebooks document exploratory work with additional model families. They may install packages inline and use different sample sizes, architectures, or evaluation procedures. For now, they are retained as historical experiments; the clean baseline is the recommended entry point for a reproducible, inspectable run.

To run a notebook, open it in Jupyter or Google Colab, inspect its data-loading and model-configuration cells, and execute it from top to bottom. Transformer fine-tuning may benefit from a GPU.

## Evaluation and limitations

This repository is a work in progress, not a finalized benchmark. In particular:

- The notebook experiments do not yet share one standardized evaluation protocol.
- Slot-filling requires careful alignment of word-level labels with subword tokens, and evaluation should exclude padding/special-token positions.
- Intent classification, token-level slot filling, and text generation require different metrics.
- The generation notebook evaluates outputs on a small exploratory subset and uses a metric configuration that should be reviewed before drawing conclusions.
- Results on MASSIVE alone do not establish robustness in real deployment settings.

The clean baseline reports standard classification metrics but should still be compared with other models under controlled conditions before any performance claims are made.

## Next steps

- Add controlled model comparisons using identical official data splits.
- Validate normalization and slot-label alignment, including BIO/BIOES conventions if used.
- Add a clear benchmark protocol, random seeds, and per-class error analysis.
- Document the exact environment and model revisions used for the original notebook experiments.

## Technologies

Python · scikit-learn · Hugging Face Datasets · Transformers · PyTorch · Persian NLP · Intent Classification · Sequence Labelling · Slot Filling

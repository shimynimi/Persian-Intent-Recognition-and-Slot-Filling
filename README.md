# Persian Intent Recognition and Slot Filling

This repository contains exploratory NLP experiments for **Persian (fa-IR)** using the [MASSIVE dataset](https://github.com/amazon-science/massive). The project explores how a Persian-language assistant can identify a user's intent and extract slot values from an utterance.

## Task overview

- **Intent recognition:** classify an utterance by the action or request it expresses.
- **Slot filling:** label spans or tokens that represent useful values, such as a time or other entity.
- **Language focus:** Persian (fa-IR).

## Experiments

The work is currently organized as three Jupyter notebooks:

| Notebook | Focus |
| --- | --- |
| Part1.ipynb | Dataset loading, a TF–IDF/logistic-regression baseline, and a BERT-based classification experiment |
| Part2.ipynb | Preprocessing and sequence-tagging experiments, including an LSTM-style model and Persian BERT token classification |
| Part3.ipynb | An exploratory sequence-to-sequence experiment using ParsT5, with text-generation evaluation metrics |

Model families and evaluation procedures differ between notebooks. These experiments should not be interpreted as a controlled benchmark comparison until the data splits, evaluation subsets, label alignment, and metrics have been standardized.

## Dataset

The notebooks load the Persian configuration of the MASSIVE dataset through Hugging Face Datasets:

~~~python
from datasets import load_dataset

dataset = load_dataset("AmazonScience/massive", "fa-IR")
~~~

Please consult the dataset's original documentation for its intended use, schema, and licensing terms.

## Environment

The notebooks use Python and libraries from the Hugging Face and PyTorch ecosystems. Depending on the notebook, dependencies include:

- datasets
- transformers
- torch
- scikit-learn
- seqeval
- evaluate
- accelerate
- pandas, numpy, matplotlib, and seaborn

Several notebook cells install packages inline. Exact dependency versions and hardware requirements have not yet been consolidated into a pinned environment file.

## Running the experiments

1. Clone or download this repository.
2. Open the notebook you want to explore in Jupyter or Google Colab.
3. Install the dependencies required by that notebook.
4. Run the notebook from top to bottom and review the label mapping and data split before training.
5. When evaluating a model, use the full held-out test set and report metrics appropriate to the task.

Transformer fine-tuning can require significant compute; a GPU may be helpful.

## Evaluation notes

This is an exploratory project rather than a finalized benchmark. The existing notebooks use different model architectures, sample sizes, and evaluation approaches. Before drawing conclusions across models, the following should be checked and made consistent:

- Intent classification versus token-level slot-filling metrics
- Alignment of word-level slot labels with subword tokens
- Full validation/test-set evaluation rather than small demonstration subsets
- Macro- and micro-averaged scores, per-label performance, and class imbalance
- Reproducible random seeds and documented model/configuration choices

For that reason, this README does not present a single headline accuracy as the performance of the overall system.

## Suggested next steps

- Consolidate shared dataset and preprocessing code into a small Python package.
- Keep intent classification and slot filling as separate, clearly evaluated tasks.
- Add a pinned requirements.txt or environment file.
- Add tests for label alignment and dataset preprocessing.
- Log configurations and metrics in a consistent format.
- Document the exact experimental contribution of each notebook.

## Technologies

Python · Hugging Face Datasets · Transformers · PyTorch · Persian NLP · Intent Classification · Sequence Labelling · Slot Filling · Evaluation

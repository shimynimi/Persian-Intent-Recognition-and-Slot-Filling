# Persian Intent Recognition, Slot Filling, and Intent-Conditioned Generation

This repository contains experiments on Persian-language understanding using the `fa-IR` configuration of the [MASSIVE dataset](https://huggingface.co/datasets/AmazonScience/massive). It covers three related but distinct tasks: utterance-level intent classification, token-level slot filling, and intent-conditioned utterance generation. Their metrics are task-specific and must not be compared directly.

## Repository map

| Path | Purpose | Status |
| --- | --- | --- |
| `src/persian_intent/baseline.py` | Word/character TF–IDF + Logistic Regression | Reproducible intent baseline |
| `scripts/train_persian_bert.py` | Fine-tune Persian BERT for utterance-level intent classification | Reproduction script; see environment caveat below |
| `src/persian_intent/text_normalization.py` | Conservative Persian text normalization used in the ablation | Unit-tested utility |
| `data/intent_labels.json` | Official MASSIVE intent label order (IDs 0–59) | Label metadata |
| `Part1.ipynb` | Early classification experiments | Exploratory; some cells classify `scenario`, not `intent` |
| `Part2.ipynb` | Slot-label preprocessing, BiLSTM-CRF, and Persian BERT token classification | Exploratory; alignment/evaluation need more validation |
| `Part3.ipynb` | ParsT5 intent-conditioned utterance generation | Exploratory; generation metrics need more validation |
| `results/` | Aggregate metrics and class-wise comparisons | Historical experiment outputs; no model weights or row-level utterances |

The BERT intent classifier is separate from the token-classification/slot-filling experiment in `Part2.ipynb`.

## Dataset

- Dataset: [AmazonScience/MASSIVE](https://huggingface.co/datasets/AmazonScience/massive)
- Configuration: `fa-IR`
- Prediction target: `intent` (60 labels), not `scenario` (18 broader domains)
- Official splits: 11,514 train, 2,033 validation, and 2,974 test utterances
- Dataset revision used by the current reproduction script: `9b095d98489fd72fef8fcde2aee186df56b7d281`

MASSIVE contains utterance-level intent and slot annotations across multiple languages. See the [dataset card](https://huggingface.co/datasets/AmazonScience/massive) and [dataset paper](https://arxiv.org/abs/2204.08582).

## 1. TF–IDF + Logistic Regression baseline

The baseline is fitted on the official training split and evaluated separately on validation and test.

- Word TF–IDF: n-grams 1–2
- Character TF–IDF: n-grams 3–5
- `min_df=2`, up to 80,000 features per vectorizer, `sublinear_tf=True`
- Logistic Regression: `max_iter=1000`

### Run

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -e "[dev]"
python -m persian_intent.baseline
```

The baseline writes `model.joblib` and `metrics.json` under `artifacts/persian-intent-baseline/` (ignored by Git). Use validation for model selection; do not repeatedly tune against test.

## 2. Fine-tuned Persian BERT intent classifier

The classifier uses [`HooshvareLab/bert-fa-base-uncased`](https://huggingface.co/HooshvareLab/bert-fa-base-uncased), a Persian pretrained BERT model intended for downstream fine-tuning. The training script pins a model snapshot (`a04aa40c97bcdde570ae11986a534542c2995a62`) and the MASSIVE dataset revision above.

### Install optional dependencies

```bash
python -m pip install -e ".[dev,bert]"
```

A CUDA GPU is recommended. Alternatively, install the same optional packages from `requirements-bert.txt` after installing the project. The script records the Python and library versions, requested and resolved model revisions, training configuration, and validation/test metrics in its output `metrics.json`.

### Run using the Hugging Face dataset

```bash
python scripts/train_persian_bert.py \
  --seed 42 \
  --text-variant raw \
  --output-dir artifacts/persian-bert-raw-seed42
```

The script loads Train and Validation first, selects the checkpoint by validation Macro-F1, and only then loads/evaluates the official Test split. It writes the best model checkpoint, per-class metrics, a named confusion matrix, and `metrics.json` under the output directory.

To run a validation-only seed/normalization experiment without loading Test:

```bash
python scripts/train_persian_bert.py \
  --seed 17 --text-variant normalized --validation-only \
  --output-dir artifacts/persian-bert-normalized-seed17
```

Supported text variants:

- `raw`: original utterances
- `normalized`: Unicode NFC, Arabic Yeh/Kaf canonicalization, removal of tatweel and Arabic diacritics, and whitespace collapsing. ZWNJ and punctuation are preserved.

### Run using downloaded Parquet files

The script can also read the three local files. Install the BERT extras, then run:

```bash
python scripts/train_persian_bert.py \
  --train-parquet data/massive-train.parquet \
  --validation-parquet data/massive-validation.parquet \
  --test-parquet data/massive-test.parquet \
  --seed 42 --text-variant raw \
  --output-dir artifacts/persian-bert-raw-seed42
```

The local Parquet files must expose integer `intent` IDs in the official MASSIVE order. `data/intent_labels.json` contains that order. Do not commit downloaded dataset files or model checkpoints unless there is a specific reason to distribute them.

### Reproducibility caveat

The reported historical BERT test run was completed in Colab. Its exact package versions and resolved model commit were not captured in the original session. The newly added script pins a model snapshot for future runs and records environment versions at runtime, but this cannot retroactively prove that the historical run used the same snapshot or dependency versions. The published historical scores below should therefore be treated as results from the recorded experiment, not as a bit-for-bit reproduction guarantee.

## 3. Reported results

The following single-run comparison uses the same 2,974 official test utterances. BERT selected its checkpoint by validation Macro-F1; test was evaluated after selection.

| Test metric | TF–IDF + Logistic Regression | Fine-tuned Persian BERT |
| --- | ---: | ---: |
| Accuracy | 82.38% | **85.78%** |
| Macro-F1 (all 60 labels) | 75.26% | **78.57%** |
| Macro-F1 (labels represented in test) | 76.54% | **79.90%** |
| Weighted-F1 | 82.59% | **85.46%** |

BERT improved test Accuracy by 3.40 percentage points and all-label Macro-F1 by 3.31 percentage points. Paired test-set predictions yielded 222 BERT-only correct cases and 121 Logistic-Regression-only correct cases; the exact two-sided McNemar test gave `p = 5.40e-8`. A paired bootstrap 95% interval for the Accuracy difference was 2.22–4.57 percentage points. These paired statistics describe the two fitted models on this test set; they do not measure training-seed variability.

### Exact-text overlap diagnostic

There are no shared record IDs between official splits, but exact raw utterance strings do overlap: 210 Test rows (177 distinct texts) occur in Train, and 8 overlapping Test rows have a conflicting intent label among matching Train texts.

On the 2,764 Test rows whose exact raw text was not seen in Train:

| Model | Accuracy |
| --- | ---: |
| TF–IDF + Logistic Regression | 81.33% |
| Fine-tuned Persian BERT | **85.06%** |

This is a diagnostic subset, not a replacement for the official Test score.

### Class-wise trade-offs

The aggregate improvement does not mean BERT improves every intent. For example, on `general_quirky`, BERT has higher Precision (66.67% vs. 44.44%) but lower Recall (43.79% vs. 66.27%), with similar F1 (52.86% vs. 53.21%). The complete 60-label comparison is in [`results/classwise_model_comparison.csv`](results/classwise_model_comparison.csv).

## 4. Validation-only normalization/robustness experiment

A separate ablation trained three seeds (17, 42, 123) for each text variant using Train and Validation only. **The Test split was not loaded or evaluated in this experiment.**

| Validation metric (mean ± sample SD across 3 seeds) | Raw text | Normalized text |
| --- | ---: | ---: |
| Accuracy | 85.54% ± 0.15 pp | **85.78% ± 0.34 pp** |
| Macro-F1 (all 60 labels) | 80.14% ± 0.34 pp | **80.20% ± 0.19 pp** |
| Weighted-F1 | 85.34% ± 0.18 pp | **85.52% ± 0.45 pp** |

The normalized variant improved mean Validation Macro-F1 by only 0.062 percentage points. Per-seed differences were mixed (normalized text was higher for seeds 17 and 123, but lower for seed 42), so this is a small preliminary effect, not evidence that normalization is reliably better. The normalized variant has **not** been evaluated on the official Test split.

## Tests

```bash
python -m pip install -e ".[dev]"
pytest -q
```

The unit tests cover the lightweight baseline and Persian text-normalization behavior. The BERT training script requires the optional `bert` dependencies and a model/data download for a full run; GitHub CI does not perform a multi-epoch GPU training job.

## Limitations and next steps

- The single-run BERT Test result has not been repeated across independent training seeds.
- The normalization ablation is validation-only; its small observed advantage is not confirmed on Test.
- Exact package versions for the historical Colab runs were not captured; future runs log versions and resolved model revision.
- The official Test split has exact raw-text overlap with Train; official and unseen-text subset results are both reported.
- The exploratory slot-filling notebook still needs careful word/subword alignment validation and sentence-preserving sequence metrics.
- The generation notebook's small-subset metrics and semantic validity need further review.
- Results on MASSIVE alone do not establish robustness in real-world deployment.

## Citation

- MASSIVE: [FitzGerald et al. (2022/2023)](https://arxiv.org/abs/2204.08582). Dataset card: [AmazonScience/MASSIVE](https://huggingface.co/datasets/AmazonScience/massive).
- ParsBERT: [Farahani et al. (2020)](https://arxiv.org/abs/2005.12515). Model: [HooshvareLab/bert-fa-base-uncased](https://huggingface.co/HooshvareLab/bert-fa-base-uncased).

## Technologies

Python · scikit-learn · PyTorch · Hugging Face Transformers · Hugging Face Datasets · Persian NLP · Intent Classification · Sequence Labelling · Slot Filling · Text Generation
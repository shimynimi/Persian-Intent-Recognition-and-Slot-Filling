# Experimental result artifacts

These files are compact aggregate outputs from completed local/Colab experiments. They do not contain training data, per-example utterance text, or model weights.

- `intent_test_comparison.json`: official-test metrics and paired comparison statistics for TF–IDF + Logistic Regression vs. one fine-tuned Persian BERT run.
- `classwise_model_comparison.csv`: class-wise precision, recall, F1, and support for both models (including a zero-support label where applicable).
- `normalization_validation_only.csv`: three-seed mean and sample standard deviation for raw vs. normalized text on Validation only.
- `normalization_validation_metadata.json`: seeds, normalization definition, training setup, and the explicit `test_used: false` flag.

The historical BERT run's exact dependency versions and resolved model commit were not captured. The new training script records these values on future runs.

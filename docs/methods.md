# Methods

[← README](../README.md)

## Systems

All seven systems train on the same training split. The hyperparameters, the epoch, the system, and the cutoff are all chosen on development macro-F1.

| System | What it is |
| --- | --- |
| `majority` | Always predicts the most common training section, which gives the floor. |
| `word` | Word unigram and bigram TF-IDF (min_df 2, at most 50,000 features, sublinear tf) with saga logistic regression. C comes from {0.25, 1, 4, 16}. |
| `char` | Character 2 to 4 grams inside word boundaries (at most 80,000 features), with the same classifier and C grid. |
| `char_balanced` | The winning `char` features and C with `class_weight="balanced"`, so rare sections count for more in the loss. |
| `probe` | [AfroXLMR-mini](https://huggingface.co/Davlan/afro-xlmr-mini) (Alabi et al., 2022), an XLM-R model (Conneau et al., 2020) adapted to 17 African languages including Kinyarwanda. The encoder is frozen and only the classification head trains, at lr 1e-3. |
| `full` | A full fine-tune of AfroXLMR-mini with encoder lr 2e-5, head lr 1e-3, AdamW, weight decay 0.01, gradient clipping at 1.0, batch 8, 3 epochs, and 128 subword tokens. |
| `full_weighted` | `full` with inverse-frequency class weights in the cross-entropy. |

Each encoder run saves the epoch with the highest development macro-F1. The linear models read the whole article, and the encoder reads the first 128 subword tokens, which is roughly the headline and the opening.

[notebooks/02_baselines.ipynb](../notebooks/02_baselines.ipynb) trains the majority and linear systems, and [notebooks/03_afroxlmr.ipynb](../notebooks/03_afroxlmr.ipynb) trains the encoder runs.

## Selection and refusal cutoff

[notebooks/04_evaluation.ipynb](../notebooks/04_evaluation.ipynb) picks the system with the highest development macro-F1 and writes it to `models/deployed.json` before it reads the test split. Then it scores every system on the test split once.

The cutoff comes from the selected system's development predictions. It is the top-section score that gives the highest accuracy among filed articles while still filing at least 80 percent of development (`inkuru.metrics.choose_threshold`). The scores are softmax outputs and are not calibrated probabilities.

## Own work and existing resources

The KinNews articles and labels, the pretrained AfroXLMR-mini weights, and the libraries (scikit-learn, PyTorch, Hugging Face Transformers, Flask, and Matplotlib) are used as published.

The project implements the data preparation (label check, deduplication, leakage filter, and development split), the seven training and selection runs, the refusal cutoff, the evaluation and error analysis, the Flask desk with its word marking, the JavaScript port of the deployed model with its parity check, and the Colab notebook.

Sources are in [references.md](references.md), and each design choice is in [DECISIONS.md](../DECISIONS.md).

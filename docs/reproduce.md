# Reproduce

[← README](../README.md)

The training happens in five notebooks. Run them in order from Jupyter or VS Code with `requirements.txt` installed, or open each one in Colab with its button; each notebook's first cell sets up either environment.

| Notebook | What it does | Writes |
| --- | --- | --- |
| [01_data_preparation](../notebooks/01_data_preparation.ipynb) | Reads the KinNews CSVs, removes duplicates and the train and test overlap, and holds out development | `data/raw/*.jsonl`, `data/kin_labels.json`, `results/data_profile.json` |
| [02_baselines](../notebooks/02_baselines.ipynb) | Majority, word and character TF-IDF logistic regression, and the class-balanced character model | `models/baselines/`, `results/*_dev.jsonl`, `results/baseline_grid.json` |
| [03_afroxlmr](../notebooks/03_afroxlmr.ipynb) | The AfroXLMR-mini probe, full fine-tune, and class-weighted fine-tune | `models/runs/`, `results/*_train.json` |
| [04_evaluation](../notebooks/04_evaluation.ipynb) | Chooses the system and the cutoff on development, then scores the test split once | `models/deployed.json`, `results/*_test.jsonl`, `results/metrics.json` |
| [05_error_analysis](../notebooks/05_error_analysis.ipynb) | Figures, confusions, confident mistakes, and the desk examples | `results/figures/`, `results/error_analysis.json`, `data/ui_examples.json` |

Notebook 01 needs the KinNews CSVs in `data/source/KINNEWS/raw/` ([data/README.md](../data/README.md)); without them, start from notebook 02 with the committed splits. Notebook 03 has a `RETRAIN` switch and notebook 04 a `RESCORE` switch. Both are off in the committed notebooks, which show the saved runs; turn them on to train and score again. Rerunning notebooks 01, 02, 04, and 05 reproduces the committed files byte for byte.

The encoder runs take the longest: on the Mac used for this project (PyTorch MPS, with `PYTORCH_ENABLE_MPS_FALLBACK=1`), each full fine-tune took about 20 minutes and the probe about 6, and a Colab T4 is faster. The repository includes only the deployed linear model; notebook 02 rebuilds the other linear models, and the encoder checkpoints (about 465 MB each) stay out.

## Tests

```bash
PYTHONPATH=src python -m unittest discover tests
```

The tests cover the label alias, the cutoff rule, how the desk handles empty input and example prefill, the deployed model end to end with its marked words, the experiments page, the browser copy's match with the deployed model, and the absence of repeated text across the three committed splits.

## Repository layout

```
app/
  app.py                    Flask desk
  requirements.txt          the desk alone (no torch)
  templates/, static/       desk pages and styles
notebooks/
  01_data_preparation.ipynb CSV -> deduplicated JSONL splits
  02_baselines.ipynb        majority and TF-IDF logistic regression
  03_afroxlmr.ipynb         AfroXLMR-mini probe and fine-tunes
  04_evaluation.ipynb       development selection, then one test scoring
  05_error_analysis.ipynb   figures, error notes, desk examples
  colab_app.ipynb           the desk with a public link
src/inkuru/
  paths.py                  every path, the seed, the encoder name
  labels.py                 the 14 sections, Kinyarwanda names, blurbs
  data.py                   splits, prediction records, the deduplication key
  metrics.py                accuracy, macro P/R/F1, the refusal cutoff
  encoder.py                AfroXLMR-mini dataset, batching, and prediction
  predict.py                load the deployed system, file one article
web/                        static Hugging Face Space: pages, ngram.js, desk.js, model/
data/
  README.md                 data card
  raw/                      prepared train/dev/test JSONL (committed)
  kin_labels.json           Kinyarwanda section names
  ui_examples.json          desk example articles
  source/                   KinNews CSVs (local, not committed)
models/
  deployed.json             selected system and cutoff
  baselines/                the deployed linear model (char_balanced.joblib)
  runs/                     encoder checkpoints (local, not committed)
results/                    metrics, predictions, error notes, parity, figures
docs/                       this write-up
tests/test_core.py          unit and end-to-end tests
requirements.txt            everything, including training
```

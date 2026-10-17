<div align="center">

# Ikinyamakuru

[![Live app](https://img.shields.io/badge/Live_app-Hugging_Face_Space-ffd21e?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Gkea/Ikinyamakuru)
[![Open in Colab](https://img.shields.io/badge/Open_in-Colab-f9ab00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Keaane/Ikinyamakuru/blob/main/notebooks/colab_app.ipynb)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square&logo=python&logoColor=white)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

Ikinyamakuru files a Kinyarwanda news article under one of 14 sections and leaves it for an editor when its score is low. The name is Kinyarwanda for a newspaper.

[Overview](#overview) • [Findings](#findings) • [Try it](#try-it) • [Getting started](#getting-started) • [Documentation](#documentation)

</div>

> [!TIP]
> The [live app](https://huggingface.co/spaces/Gkea/Ikinyamakuru) runs the model in your browser. Paste an article or pick one of the examples.

## Overview

This is a summative project for Machine Learning Techniques I at the African Leadership University, in the text classification track. A section editor pastes a story and the desk proposes one of the 14 KinNews sections: politics, sport, economy, health, entertainment, history, technology, tourism, culture, fashion, religion, environment, education, and relationship. If the top score is under a cutoff chosen on development data, the desk shows its best guess and leaves the filing to the editor.

| Dataset | Language | Articles | Used for |
| --- | --- | ---: | --- |
| [KinNews](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus) (Niyongabo et al., 2020) | Kinyarwanda, Rwanda | 21,268 raw, 11,091 after removing duplicates | train 6,864, development 763, test 3,464 |

The project compares seven systems: a majority baseline, logistic regression on word and on character TF-IDF features (with and without class balancing), and AfroXLMR-mini as a frozen probe, a full fine-tune, and a class-weighted fine-tune. Every choice is made on development macro-F1, and the official test split is scored once.

## Findings

- Over half of the official training file is repeated text. 9,384 of its 17,014 rows are exact duplicates, some copied 40 or 80 times, and 156 of the repeated articles carry more than one section label.
- The class-balanced character model, `char_balanced`, has the best development macro-F1 and is the deployed system. On the test split it reaches accuracy 0.734 and macro-F1 0.583.
- Character n-grams beat word n-grams on development macro-F1 (0.622 against 0.545). Kinyarwanda words carry many prefixes and suffixes, and a character window can still match the shared stem.
- Balancing the classes raised test macro-F1 from 0.456 to 0.583. Tourism, fashion, and environment have between 31 and 34 training articles each.
- AfroXLMR-mini reached 0.488 development macro-F1 at best. Its score was still rising at the third and last epoch, and it reads only the first 128 tokens of articles that average about 350 words.
- With the cutoff, the desk files 73.8 percent of test articles, and 84.7 percent of those are correct.

| System | Dev macro-F1 | Test accuracy | Test macro-F1 |
| --- | ---: | ---: | ---: |
| majority | 0.027 | 0.091 | 0.012 |
| word | 0.545 | 0.690 | 0.428 |
| char | 0.622 | 0.697 | 0.456 |
| char_balanced (deployed) | 0.692 | 0.734 | 0.583 |
| probe | 0.239 | 0.520 | 0.244 |
| full | 0.477 | 0.665 | 0.392 |
| full_weighted | 0.488 | 0.674 | 0.435 |

The test file has a different mix of sections from the training file. Politics is 24 percent of training and 9 percent of test, so every system scores lower on test than on development. [docs/results.md](docs/results.md) has precision, recall, per-section scores, the figures, and the error analysis.

## Try it

- The [Hugging Face Space](https://huggingface.co/spaces/Gkea/Ikinyamakuru) runs the deployed model in JavaScript and gives the same answers as the Python desk on all 4,227 development and test articles.
- [notebooks/colab_app.ipynb](notebooks/colab_app.ipynb) starts the Flask desk on Colab with a temporary public link.
- The training notebooks, [01](notebooks/01_data_preparation.ipynb) to [05](notebooks/05_error_analysis.ipynb), prepare the data, train and evaluate the seven systems, and draw the figures. Each one opens in Colab.
- `PYTHONPATH=src python app/app.py` runs the Flask desk on your own machine.
- Demo video: to be added.

## Getting started

You need Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python app/app.py                   # the desk, http://127.0.0.1:7860
PYTHONPATH=src python -m unittest discover tests
```

`app/requirements.txt` is enough if you only want the desk. To train, run the notebooks in `notebooks/` in order; [docs/reproduce.md](docs/reproduce.md) lists what each one does and writes.

## Project structure

```
Ikinyamakuru/
├── notebooks/    01 to 05: data preparation, training, evaluation, error analysis; colab_app: the desk with a public link
├── src/inkuru/   shared code for the notebooks and the desk: paths, labels, data, metrics, prediction
├── app/          the Flask desk
├── web/          the browser app on the Hugging Face Space
├── models/       the deployed model and its cutoff
├── results/      metrics, per-article predictions, error notes, and figures
├── data/         prepared splits and the data card
├── docs/         the full write-up
└── tests/
```

## Documentation

| Page | Contents |
| --- | --- |
| [Data](data/README.md) | source, download, preparation, split sizes, known limits, licences |
| [Methods](docs/methods.md) | the seven systems, training settings, selection, and the refusal cutoff |
| [Results](docs/results.md) | full scores, figures, findings, and error analysis |
| [Deployment](docs/deployment.md) | the browser app and its parity check, Colab, and the desk on your machine |
| [Reproduce](docs/reproduce.md) | the five training notebooks and the repository layout |
| [References](docs/references.md) | papers, data, models, and tools |
| [Design decisions](DECISIONS.md) | each choice and the reason for it |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | licences for the data, the model, and the libraries |

The code is released under the [MIT licence](LICENSE). The KinNews articles remain the copyright of their publishers.

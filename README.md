<div align="center">

# Ikinyamakuru

[![Live app](https://img.shields.io/badge/Live_app-Hugging_Face_Space-ffd21e?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/Gkea/Ikinyamakuru)
[![Model](https://img.shields.io/badge/Model-ikinyamakuru-ffd21e?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/Gkea/ikinyamakuru)
[![Open in Colab](https://img.shields.io/badge/Open_in-Colab-f9ab00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/Keaane/Ikinyamakuru/blob/main/notebooks/colab_app.ipynb)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square&logo=python&logoColor=white)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

Ikinyamakuru files a Kinyarwanda news article under one of 14 sections and leaves it for an editor when its score is low. The name is Kinyarwanda for a newspaper.

[Try it](#try-it) • [Results](#results) • [Getting started](#getting-started) • [Documentation](#documentation)

</div>

## About

This is a summative project for Machine Learning Techniques I at the African Leadership University, in the text classification track. It trains seven systems on [KinNews](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus) (Niyongabo et al., 2020), from a majority baseline to TF-IDF logistic regression and AfroXLMR-mini. Development macro-F1 picks the system, and the official test split is scored once. The selected model is deployed on Hugging Face.

## Try it

- [Live app](https://huggingface.co/spaces/Gkea/Ikinyamakuru) on a Hugging Face Space. Paste an article or pick an example, and the model runs in your browser.
- [Model](https://huggingface.co/Gkea/ikinyamakuru) on the Hugging Face Hub, with a model card and loading code.
- [Colab notebook](https://colab.research.google.com/github/Keaane/Ikinyamakuru/blob/main/notebooks/colab_app.ipynb) that starts the Flask desk with a temporary public link.
- Demo video: (https://www.awesomescreenshot.com/video/57558104?key=ece4998bad2bd8413c922ab69ba23a56)

## Results

The class-balanced character model, `char_balanced`, has the best development macro-F1 and is the deployed system. On the test split it reaches accuracy 0.734 and macro-F1 0.583. Below its cutoff it leaves the article for an editor, and it files 73.8 percent of test articles with 84.7 percent of those correct.

| System | Dev macro-F1 | Test macro-F1 |
| --- | ---: | ---: |
| majority | 0.027 | 0.012 |
| word | 0.545 | 0.428 |
| char | 0.622 | 0.456 |
| char_balanced (deployed) | 0.692 | 0.583 |
| probe | 0.239 | 0.244 |
| full | 0.477 | 0.392 |
| full_weighted | 0.488 | 0.435 |

The saga solver behind the three linear systems stops at 220 iterations, before converging for `char_balanced`. A later experiment ([notebooks/06_convergence.ipynb](notebooks/06_convergence.ipynb)) fits them to convergence and scores the test split a second time: the same development rules then choose a model that scores higher on development and lower on test. `char_balanced` stays deployed, a decision made after seeing both test scores.

[docs/results.md](docs/results.md) has the full scores, the findings, the error analysis, and [the second test scoring](docs/results.md#a-second-test-scoring).

## Getting started

You need Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt    # the desk alone, no torch
PYTHONPATH=src python app/app.py       # the desk, http://127.0.0.1:7860
```

To train, install `requirements.txt` (it adds torch, Transformers, and the notebook tools) and run the notebooks in `notebooks/` in order. [docs/reproduce.md](docs/reproduce.md) lists what each one writes, the tests, and the repository layout.

## Documentation

| Page | Contents |
| --- | --- |
| [Data](data/README.md) | source, preparation, duplicates, split sizes, and known limits |
| [Methods](docs/methods.md) | the seven systems, training settings, selection, and the cutoff |
| [Results](docs/results.md) | full scores, findings, figures, and error analysis |
| [Deployment](docs/deployment.md) | the Space, the model on the Hub, Colab, and the desk on your machine |
| [Reproduce](docs/reproduce.md) | the training notebooks, the tests, and the repository layout |
| [References](docs/references.md) | papers, data, models, and tools |
| [Design decisions](DECISIONS.md) | each choice and the reason for it |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | licences for the data, the model, and the libraries |

The code is released under the [MIT licence](LICENSE). The KinNews articles remain the copyright of their publishers.

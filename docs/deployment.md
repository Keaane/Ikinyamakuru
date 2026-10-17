# Deployment

[← README](../README.md)

The live app is a free static Space at https://huggingface.co/spaces/Gkea/Ikinyamakuru. Paste a Kinyarwanda article and press File. The desk proposes one of the 14 sections, shows the next two, and marks the words that raised the section. When the top score is under 0.5743 it leaves the piece for an editor.

## How the browser app works

Hugging Face charges for Docker and Gradio Spaces on CPU, while static Spaces are free and always on. The deployed model is a linear model over character n-grams with 10 MB of weights, so it runs in the browser in plain JavaScript, and the text a user pastes stays on their machine.

The vocabulary, IDF weights, float64 coefficients, and intercepts of `models/baselines/char_balanced.joblib` are exported to `web/model/`. [web/ngram.js](../web/ngram.js) re-implements scikit-learn's `char_wb` TF-IDF (2 to 4 grams, sublinear tf, l2 norm), the multinomial logistic regression, and the word marking from `src/inkuru/predict.py`. Where Python and JavaScript treat whitespace, lowercasing, or word characters differently, the file spells out Python's rules.

The port was checked on all 4,227 development and test articles and 18 edge cases against the Flask desk's own `Router.route`. The edge cases cover Python-only whitespace, the Greek final sigma, combining accents, emoji, and empty input. All 4,245 inputs get the same section, filing decision, top three, marked words, and empty-input error, and the largest probability difference is 1e-15 ([results/web_parity.json](../results/web_parity.json)).

`web/index.html` and `web/lab.html` are rendered from the Flask templates, so the Space and the local desk look the same, and [web/desk.js](../web/desk.js) draws the same answer block as the template. The examples fill the box with a full article without filing it, and the editor presses File.

## Model on the Hub

The deployed model is published at https://huggingface.co/Gkea/ikinyamakuru. The repository has the scikit-learn pipeline (`char_balanced.joblib`), the same model as plain arrays in `weights/` for use without pickle, `config.json` with the cutoff and the section names, and a model card with loading code and the per-section test scores.

## Colab

[notebooks/colab_app.ipynb](../notebooks/colab_app.ipynb) ([open in Colab](https://colab.research.google.com/github/Keaane/Ikinyamakuru/blob/main/notebooks/colab_app.ipynb)) has two parts. The first clones this repository, installs `app/requirements.txt`, starts `app/app.py`, and prints a temporary public link through a Cloudflare quick tunnel, which needs no account. The link works while the runtime runs. The second part links the five training notebooks, which also open in Colab; notebook 03 trains AfroXLMR-mini on a free T4 GPU.

## Run the desk on your machine

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r app/requirements.txt
PYTHONPATH=src .venv/bin/python app/app.py
```

Open http://127.0.0.1:7860. `HOST` and `PORT` change the defaults. `app/requirements.txt` pins scikit-learn 1.9.1 and NumPy 2.x, which the committed model was pickled with, and leaves out torch. `/lab` shows the experiment table and `/health` reports whether a model is deployed.

The model learned from whole articles. A headline alone gets a weaker score and is held for an editor more often, so paste the full article when you have it.


# Design decisions

## Problem and data

| Choice | Why |
| --- | --- |
| Classify Kinyarwanda news into the 14 KinNews sections | Newsrooms file every story under a section. KinNews is public and human labelled, and Rwandan outlets wrote the articles in Kinyarwanda. |
| Use the raw KinNews release | The cleaned release deletes stopwords and punctuation, and the apostrophe is part of Kinyarwanda spelling (*n’*, *y’*). |
| Keep the official train and test split | Results stay comparable with Niyongabo et al. (2020), and the test articles are the authors' choice. |
| Remove exact duplicates inside each file, keeping the first copy | 9,384 of the 17,014 training rows are repeats, some 40 or 80 times. Left in, a memorised article would count many times, and a repeat could land in both training and development. The key is the lowercased text with whitespace collapsed. |
| Drop the 3 training articles that also appear in the test file | A story that is scored should never be trained on. |
| Keep the first copy's label when repeats disagree | 156 repeated articles carry more than one label, mostly entertainment and sport. The labels are left as published, and the error analysis reports the noise. |
| Hold out a stratified 10 percent of the remaining training articles for development, seed 42 | KinNews has no validation split. Stratifying keeps the sections that have only 3 or 4 development articles. |
| Commit the prepared splits and leave out the CSVs | Every later notebook reads the splits. The CSVs come from the authors' release ([data/README.md](data/README.md)). |

## Models

| Choice | Why |
| --- | --- |
| A majority baseline | It shows what the class prior alone scores under macro-F1. |
| Word unigram and bigram TF-IDF with logistic regression | This is the usual strong baseline for news topics, and the strongest models in the KinNews paper are TF-IDF models. |
| Character 2 to 4 grams inside word boundaries | Kinyarwanda attaches many prefixes and suffixes to a stem, so character windows match inflected forms that whole words keep apart. |
| C from {0.25, 1, 4, 16}, chosen on development macro-F1 | One small grid per feature set. C = 16 won for both, which is the top of the grid, so larger values are untested. |
| `class_weight="balanced"` on the winning character model with the same C | This tests class weights on their own, apart from the feature search. Tourism, fashion, and environment have 31 to 34 training articles. |
| AfroXLMR-mini as the pretrained encoder | Alabi et al. (2022) adapted XLM-R to 17 African languages including Kinyarwanda, and the mini model is small enough to fine-tune on a laptop. |
| A frozen probe before the fine-tune | It checks whether the pretrained encoder already separates the sections. |
| Full fine-tune with encoder lr 2e-5, head lr 1e-3, batch 8, 3 epochs, and 128 tokens | These are standard settings that fit a laptop. 128 tokens cover the headline and the opening. The development score was still rising at epoch 3, so more epochs could help. |
| A class-weighted fine-tune | It asks the encoder the same question `char_balanced` asks the linear model. |
| Keep the epoch with the best development macro-F1 | Model selection stays on development data. |

## Evaluation

| Choice | Why |
| --- | --- |
| Macro-F1 as the selection metric | The test file has a different mix of sections from training, and accuracy would reward filing almost everything as politics. |
| Report accuracy, macro precision, macro recall, and per-section scores as well | The brief asks for all four, and the per-section scores show where macro-F1 comes from. |
| Choose the system before reading the test split, then score it once | Notebook 04 writes `models/deployed.json` before it reads any test prediction. |
| Refusal cutoff with the highest accuracy among filed development articles while filing at least 80 percent | A wrong section would publish a story in the wrong place, so low scores go to an editor. Filing 80 percent keeps the desk useful. |
| One seed | Time was limited. A second seed would come first before reading small gaps as real. |

## Desk and deployment

| Choice | Why |
| --- | --- |
| The desk proposes a section and a person publishes | That is how a newsroom works. Below the cutoff the page still shows the best guess. |
| Mark the words that raised the section | The marks come from the linear model's own weights, so an editor can see why it chose. |
| A free static Hugging Face Space with the model in JavaScript | Hugging Face charges for Docker and Gradio Spaces on CPU, while static Spaces are free and always on. The deployed linear model is 10 MB and ports exactly. |
| Check the port against Python on every development and test article and on edge cases | The browser copy has to answer like the Flask desk. All 4,245 inputs get the same section, decision, top three, and marked words ([results/web_parity.json](results/web_parity.json)). |
| Publish the deployed model on the Hugging Face Hub, with a pickle-free copy of its weights | Anyone can load the model without cloning the repository. A joblib file can run code when it loads, so the plain arrays offer a safer way to use it. |
| A Colab notebook | It runs the Flask desk with a temporary public link and the whole pipeline on a free GPU, with nothing to install. |
| Pin scikit-learn 1.9.1 and NumPy 2.x | The committed model was pickled with those versions. |
| A design based on The Pudding's essays | Cream paper, serif reading text, and highlighter marks keep the attention on the article and on the words the model used. |

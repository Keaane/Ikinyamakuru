# Data

[← README](../README.md)

KinNews (Niyongabo, Qu, Kreutzer, and Huang, 2020) has 21,268 Kinyarwanda articles from 20 Rwandan sources, 15 news websites and 5 newspapers. Native-speaker annotators labelled them into 14 sections, and the authors split them into 17,014 training and 4,254 test articles. The articles were written by Rwandan newsrooms in Kinyarwanda.

These are the 14 sections with the dataset's Kinyarwanda names: politics (*politiki*), sport (*imikino*), economy (*ubukungu*), health (*ubuzima*), entertainment (*imyidagaduro*), history (*amateka*), technology (*ikoranabuhanga*), tourism (*ubukerarugendo*), culture (*umuco*), fashion (*imideri*), religion (*iyobokamana*), environment (*ibidukikije*), education (*uburezi*), and relationship (*urukundo*).

## Get the raw files

Download the raw KINNEWS release from the Google Drive folder linked in the [KINNEWS-and-KIRNEWS-Corpus](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus) README and put the two files here:

```
data/source/KINNEWS/raw/train.csv
data/source/KINNEWS/raw/test.csv
```

Only notebook 01 reads them. Every later notebook starts from the committed splits.

## Processed files

| File | Contents |
| --- | --- |
| `raw/train.jsonl`, `raw/dev.jsonl`, `raw/test.jsonl` | One article per line with `id`, `split`, `label`, `kin_label`, `title`, `content`, `text` (the title, a line break, and the body), `url`, and `source_row` |
| `kin_labels.json` | The most common Kinyarwanda name of each section in the dataset |
| `ui_examples.json` | The eight example articles on the desk |

`results/data_profile.json` records the counts below.

## Preparation

[notebooks/01_data_preparation.ipynb](../notebooks/01_data_preparation.ipynb) keeps the publishers' wording, including stopwords and punctuation. It runs these steps in order:

1. Replace null bytes with spaces and join the title and body with a line break.
2. Map the CSV's `politic` label to `politics`, then check every row's numeric index against its section name.
3. Remove exact duplicates inside each official file, keeping the first copy. The key is the lowercased text with whitespace collapsed. This removes 9,384 of the 17,014 training rows, some of which appear 40 or 80 times, and 790 test rows.
4. Drop the 3 training articles that also appear in the test file. `tests/test_core.py` checks that no text appears in two of the committed splits.
5. Hold out a stratified 10 percent of the remaining training articles as development, with seed 42. The test file stays the official test file and is never used to choose anything.

## Split sizes

| Section | Train | Dev | Test |
| --- | ---: | ---: | ---: |
| politics | 1,628 | 181 | 315 |
| sport | 878 | 98 | 744 |
| economy | 1,000 | 111 | 174 |
| health | 978 | 109 | 272 |
| entertainment | 994 | 110 | 774 |
| history | 196 | 22 | 11 |
| technology | 136 | 15 | 103 |
| tourism | 31 | 3 | 76 |
| culture | 69 | 8 | 138 |
| fashion | 34 | 4 | 91 |
| religion | 401 | 44 | 256 |
| environment | 34 | 4 | 100 |
| education | 104 | 12 | 70 |
| relationship | 381 | 42 | 340 |
| total | 6,864 | 763 | 3,464 |

Articles average about 350 words. The test file has a different mix from the training file. Politics is 24 percent of training and 9 percent of test, while tourism, fashion, and environment each have two to three times as many test articles as training articles. Macro-F1, the unweighted mean over the 14 sections, is the selection metric for that reason.

## Known limits

Some labels are noisy. 156 repeated training articles carry more than one section label across their copies, most often entertainment and sport, and the first copy's label is kept. Some test articles are labelled against their content too, such as football stories filed under entertainment.

The 14 sections were merged from the publishers' own categories, so some stories fit two of them. A beauty pageant could be entertainment or fashion.

Tourism, fashion, and environment have 31 to 34 training articles and 3 or 4 development articles, so their development scores move when one article changes. History has 11 test articles.

All the articles are edited news, which is cleaner than social media posts or transcribed speech.

## Licences

The KINNEWS-and-KIRNEWS-Corpus repository uses the MIT licence. Its release notes state that the copyright of the news articles belongs to the original news sources. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Citation

Niyongabo, R. A., Qu, H., Kreutzer, J., and Huang, L. (2020). KINNEWS and KIRNEWS: Benchmarking cross-lingual text classification for Kinyarwanda and Kirundi. *Proceedings of COLING 2020*, 5507-5521.

# Results and error analysis

[← README](../README.md)

## Scores

Development macro-F1 selects `char_balanced`. The test split was scored once, after that choice was written to `models/deployed.json`.

| System | Dev macro-F1 | Test accuracy | Test macro-P | Test macro-R | Test macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0.027 | 0.091 | 0.006 | 0.071 | 0.012 |
| word | 0.545 | 0.690 | 0.540 | 0.476 | 0.428 |
| char | 0.622 | 0.697 | 0.708 | 0.500 | 0.456 |
| char_balanced (deployed) | 0.692 | 0.734 | 0.648 | 0.594 | 0.583 |
| probe | 0.239 | 0.520 | 0.252 | 0.283 | 0.244 |
| full | 0.477 | 0.665 | 0.412 | 0.461 | 0.392 |
| full_weighted | 0.488 | 0.674 | 0.549 | 0.491 | 0.435 |

The cutoff is 0.5743. On development it files 80.1 percent of articles and 86.3 percent of those are correct. On test it files 73.8 percent, and 84.7 percent of those are correct.

[notebooks/04_evaluation.ipynb](../notebooks/04_evaluation.ipynb) writes these numbers to `results/metrics.json`. The per-article predictions for every system and split are in `results/*_dev.jsonl` and `results/*_test.jsonl`.

![Development and test macro-F1 for every system](../results/figures/macro_f1.png)

## Findings

Over half of the official training file is repeated text. 9,384 of its 17,014 rows are exact duplicates, some copied 40 or 80 times, and 156 of the repeated articles carry more than one section label. Notebook 01 removes them before any training ([data/README.md](../data/README.md)).

Character n-grams beat word n-grams, with development macro-F1 of 0.622 against 0.545. Kinyarwanda builds words from many prefixes and suffixes, and a character window can match the shared stem of two inflected forms that a whitespace split treats as different words.

Balanced class weights help the rare sections most. Development macro-F1 rises from 0.622 to 0.692 and test macro-F1 from 0.456 to 0.583. The weights cannot add articles to sections that have only 31 to 34 training examples.

The frozen probe scores 0.239, so AfroXLMR-mini's pretraining alone does not separate the sections. Full fine-tuning (0.477) and class weights (0.488) close part of the gap and stay below the character model. The encoder sees only 128 tokens of a roughly 350-word article, and in all three encoder runs the development score was still rising at the third and last epoch (full: 0.383, 0.452, 0.477), so longer training and longer inputs are both open.

Every system drops from development to test because the test file has a different mix of sections ([data/README.md](../data/README.md#split-sizes)). The same shift is why the cutoff files fewer test articles (73.8 percent) than development articles (80.1 percent).

## Error analysis

[notebooks/05_error_analysis.ipynb](../notebooks/05_error_analysis.ipynb) draws the figures below and writes `results/error_analysis.json`, which lists the top confusions, the confident mistakes, and the refusals.

![Test F1 by section for the deployed system](../results/figures/per_class_f1.png)

![Test confusion matrix, log counts](../results/figures/confusion.png)

Sport has the highest test F1 (0.94), followed by entertainment (0.81), relationship (0.80), religion (0.73), and technology (0.72).

Education is the weakest section at F1 0.07, and 38 of its 70 test articles are filed as relationship. History reaches 0.33 on 11 test articles. Tourism reaches 0.37 with recall 0.25 and is most often filed as economy, and culture reaches 0.45 with recall 0.30 and often goes to politics.

The most common confusions are health filed as politics (50), entertainment as politics (47), entertainment as health (40), culture as politics (39), education as relationship (38), and religion as politics (35). Politics is the largest training section and pulls in stories from many of its neighbours.

Many confident mistakes sit on the boundary between publisher sections. Beauty-pageant stories (Miss Supranational, Miss Universe) labelled entertainment are filed as fashion with a score near 1.00. Others come from label noise: football stories about APR FC, Rayon Sports, and the national team carry the entertainment label in the test file, and the model files them as sport. The 156 repeated training articles with more than one label show the same entertainment and sport confusion.

![Confidence of the top section on test, correct against wrong](../results/figures/confidence.png)

Correct filings get higher scores than wrong ones on average, but the two distributions overlap, so the cutoff holds back some correct filings along with the errors.

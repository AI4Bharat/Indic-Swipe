---
layout: default
title: Performance
nav_order: 5
---

# Performance
{: .no_toc }

We define the final accuracy of the model as the percentage of input traces that are correctly processed by the model to generate the Indic word. When the beam size k is greater than 1, the model is said to make a correct prediction if at least one of the k predictions is accurate. Apart from the final accuracies, we provide the intermediate accuracies observed after each module in our architectural pipeline. The results below are reported with a beam size (k) of 3.

## Indic-to-Indic Decoding

The table below shows the results achieved by IndicSwipe on the Indic-to-Indic Decoding task across 7 Indic languages.

| Language | Dataset Size | CTC Accuracy (%) | Final Accuracy (%) | 
| :------- | :-------- | | :----- | :----- |
| Hindi  | 32415 | 68.93 | 86.75 |
| Tamil | 24094 | 68.16 | 95.73 |
| Bangla | 15320 | 75.06 | 88.74 |
| Telugu | 28996 | 74.94 | 94.86 |
| Kannada | 26551 | 66.78 | 89.06 |
| Gujrati | 30024 | 61.21 | 82.13 |
| Malayalam | 36258 | 56.98 | 95.53 |

## English-to-Indic Decoding

The table below shows the results achieved by IndicSwipe on the English-to-Indic Decoding task across 7 Indic languages.

| Language | Dataset Size | CTC Accuracy(%) | Transliteration Generation(%) | Final Accuracy(%) | 
| :------- | :-------- | | :----- | :----- | :----- |
| Hindi  | 29074 | 98.45 | 77.56 | 89.12  |
| Tamil | 21523 | 98.02 | 65.47 | 86.96 |
| Bangla | 17332 | 98.60 | 43.40 |  81.13 |
| Telugu | 12733 | 96.64 |38.19 | 78.21 |
| Kannada | 4877 | 96.95 | 37.60 | 72.00 |
| Gujrati | 8776 | 98.78 | 38.12 | 72.65 |
| Malayalam | 10097 | 98.21 | 31.61 | 70.00 |

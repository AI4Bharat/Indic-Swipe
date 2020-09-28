---
layout: default
title: Model Architecture
nav_order: 3
has_children: true
permalink: /docs/Model Architecture
---

# Model Architecture

A detailed description of the model architecture can be found here which addresses both the Indic-to-indic and English-to-indic decoding tasks.


[CTC Path Decoding](https://emilbiju.github.io/just-the-docs/docs/Model%20Architecture/ctc_path_decoding/){: .btn .btn-purple }
[Transliteration Generation](https://emilbiju.github.io/just-the-docs/docs/Model%20Architecture/transliteration/){: .btn .btn-blue }
[Transliteration Correction](https://emilbiju.github.io/just-the-docs/docs/Model%20Architecture/transliteration_correction/){: .btn .btn-green }

## Implementation Details

### English-to-Indic Decoding Task

The transformer encoder in the CTC Gesture Path Decoder uses multi- headed attention with 5 heads, hidden layer size of 128, dropout rate of 0.05 and ReLU activation in the Attention and Feed-Forward layers. The hidden layer sizes of all Bidirectional LSTM layers in this module are fixed to 256. This module is trained using the CTC loss function with a learning rate of 0.01 over 20 epochs. The Transliteration Generation module uses a single Unidirectional GRU layer with a hidden size of 512. This module is trained over 10 epochs using the Categorical Cross-Entropy loss function and Beam Search decoding with a beam size of 3 is performed on the GRU outputs. The 3 predictions are passed as independent samples to the Contrastive Transliteration Correction module for generating 3 suggestions for the final word, which may not all be unique. The Contrastive Transliteration Correction module has hidden layers of size 64 and 1. It is trained using the Sparse Categorical Cross-Entropy loss function. The Adam optimizer is used for training all the three modules.

### Indic-to-Indic Decoding Task
In this case, we make a few architectural changes to the model. Since the CTC Gesture Path Decoder directly predicts an Indic character sequence, we remove the Transliteration Generation module from the pipeline. Besides, we reduce the inference time and model complexity by removing the two Bidirectional LSTM layers of the CTC Gesture Path Decoder. We cannot afford to do this in English-to-Indic decoding as we require more accurate predictions to prevent compounding of errors as it passes through the Transliteration Generation module. The model parameters for the rest of the architecture remain the same.

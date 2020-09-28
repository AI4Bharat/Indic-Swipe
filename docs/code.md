---
layout: default
title: Code and Model Weights
nav_order: 4
---

# Code and Model Weights
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---
## Indic-to-Indic Decoding Task

The code that is used to develop the neural architecture pipeline for each language is made available in the GitHub repository and can be accessed from the links below. The pipeline for Indic-to-Indic Decoding consists of the CTC Path Decoding and Spelling Correction modules. We also provide the weights of the trained models for each language.

| Language | Code links| CTC Decoder Weights | Spell Correction Weights|
| :------- | :-------- | :-------- | :-------- |
| Hindi | [code](https://github.com/iitmnlp/indic-swipe/blob/master/indic-to-indic-decoding/Indic_to_Indic_hindi.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/CTC-weights/Hindi_CTC.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/ELMo-Correction/Hindi_ELMo_Correction.h5) |
| Tamil | [code](https://github.com/iitmnlp/indic-swipe/blob/master/indic-to-indic-decoding/Indic_to_Indic_tamil.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/CTC-weights.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/ELMo-Correction/Tamil_ELMo_Correction.h5) |
| Bangla | [code](https://github.com/iitmnlp/indic-swipe/blob/master/indic-to-indic-decoding/Indic_to_Indic_bangla.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/CTC-weights/Bangla_CTC.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/ELMo-Correction/Bangla_ELMo_Correction.h5) |
| Telugu | [code](https://github.com/iitmnlp/indic-swipe/blob/master/indic-to-indic-decoding/Indic_to_Indic_telugu.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/CTC-weights/Telugu_CTC.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/ELMo-Correction/Telugu_ELMo_Correction.h5) |
| Kannada | [code](https://github.com/iitmnlp/indic-swipe/blob/master/indic-to-indic-decoding/Indic_to_Indic_kannada.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/CTC-weights/Kannada_CTC.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/ELMo-Correction/Kannada_ELMo_Correction.h5) |
| Gujarati | [code](https://github.com/iitmnlp/indic-swipe/blob/master/indic-to-indic-decoding/Indic_to_Indic_gujarati.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/CTC-weights/Gujarati_CTC.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/ELMo-Correction/Gujarati_ELMo_Correction.h5) |
| Malayalam | [code](https://github.com/iitmnlp/indic-swipe/blob/master/indic-to-indic-decoding/Indic_to_Indic_malayalam.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/CTC-weights/Malayalam_CTC.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/indic-to-indic-weights/ELMo-Correction/Malayalam_ELMo_Correction.h5) |

## English-to-Indic Decoding Task

The code that is used to develop the neural architecture pipeline for each language is made available in the GitHub repository and can be accessed from the links below. The pipeline for English-to-Indic Decoding consists of the CTC Path Decoding, Transliteration Generation and Transliteration Correction modules. We also provide the weights of the trained models for each language.

| Language | Code links| CTC Decoder Weights| Transliteration Weights | Spell Correction Weights|
| :------- | :-------- | :-------- | :-------- | :-------- |
| Hindi | [code](https://github.com/iitmnlp/indic-swipe/blob/master/Indic-Indic%20Decoding/Indic_to_Indic_hindi.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/CTC-weights/ctc_weights_stored_eng_hindi.dms) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/translit-weights/nmt_hin_weights_3_epochs.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/ELMo-Correction/ELMo_weights_3_beam_4_epochs_hindi.h5) |
| Tamil | [code](https://github.com/iitmnlp/indic-swipe/blob/master/Indic-Indic%20Decoding/Indic_to_Indic_tamil.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/CTC-weights/ctc_weights_stored_eng_tamil.dms) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/translit-weights/nmt_tam_weights_4_epochs.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/ELMo-Correction/ELMo_weights_3_beam_4_epochs_tamil.h5) |
| Bangla | [code](https://github.com/iitmnlp/indic-swipe/blob/master/Indic-Indic%20Decoding/Indic_to_Indic_bangla.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/CTC-weights/ctc_weights_stored_eng_bengali.dms) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/translit-weights/nmt_ben_weights_4_epochs.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/ELMo-Correction/ELMo_weights_3_beam_4_epochs_bengali.h5) |
| Telugu | [code](https://github.com/iitmnlp/indic-swipe/blob/master/Indic-Indic%20Decoding/Indic_to_Indic_telugu.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/CTC-weights/ctc_weights_stored_eng_telugu.dms) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/translit-weights/nmt_tel_weights_4_epochs.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/ELMo-Correction/ELMo_weights_3_beam_4_epochs_telugu.h5) |
| Kannada | [code](https://github.com/iitmnlp/indic-swipe/blob/master/Indic-Indic%20Decoding/Indic_to_Indic_kannada.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/CTC-weights/ctc_weights_stored_eng_kannada.dms) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/translit-weights/nmt_kan_weights_5_epochs.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/ELMo-Correction/ELMo_weights_3_beam_3_epochs_kannada.h5) |
| Gujarati | [code](https://github.com/iitmnlp/indic-swipe/blob/master/Indic-Indic%20Decoding/Indic_to_Indic_gujarati.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/CTC-weights/ctc_weights_stored_eng_gujarati.dms) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/translit-weights/nmt_guj_weights_5_epochs.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/ELMo-Correction/ELMo_weights_3_beam_4_epochs_gujarati.h5) |
| Malayalam | [code](https://github.com/iitmnlp/indic-swipe/blob/master/Indic-Indic%20Decoding/Indic_to_Indic_malayalam.py) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/CTC-weights/ctc_weights_stored_eng_malayalam.dms) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/translit-weights/nmt_mal_weights_4_epochs.h5) | [download](https://github.com/iitmnlp/indic-swipe/blob/master/model-weights/english-to-indic-weights/ELMo-Correction/ELMo_weights_3_beam_10_epochs_malayalam.h5) |

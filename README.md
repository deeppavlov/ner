# Neural Networks for Named Entity Recognition

In this repo you can find several neural network architectures for named entity recognition from the paper "_Application of a Hybrid Bi-LSTM-CRF model to the task of Russian Named Entity Recognition_" https://arxiv.org/pdf/1709.09686.pdf.

NER class from ner/network.py provides methods for construction, training and inference neural networks for Named Entity Recognition.

An example of implementation of Convolutional Neural Network for Russian Named Entity Recognition is provided in example.ipynb. In this example notebook a pre-trained model is used. The model was trained on three datatasets:

- Gareev corpus [1] (obtainable by request to authors)
- FactRuEval 2016 [2]
- Persons-1000 [3]

Remark: at training stage the corpora were lemmatized and lowercased.
So text must be tokenized and lemmatized and lowercased before feeding it into the model.

The F1 measure for presented model along with other published solution provided in the table below:

| Models                | Gareev’s dataset | Persons-1000 | FactRuEval 2016 |
|---------------------- |:----------------:|:------------:|:---------------:|
| Gareev et al. [1]     | 75.05            |              |                 |
| Malykh et al. [4]     | 62.49            |              |                 |
| Trofimov  [5]         | 95.57            |              |                 |
| Rubaylo et al. [6]    |                  |              | 78.13           |
| Sysoev et al. [7]     |                  |              | 74.67           |
| Ivanitsky et al.  [7] |                  |              | **87.88**       |
| Mozharova et al.  [8] |                  | 97.21        |                 |
| Our (Bi-LSTM+CRF)     | **87.17**        | **99.26**    | 82.10           ||


[1] - Rinat Gareev, Maksim Tkachenko, Valery Solovyev, Andrey Simanovsky, Vladimir Ivanov: Introducing Baselines for Russian Named Entity Recognition. Computational Linguistics and Intelligent Text Processing, 329 -- 342 (2013).

[2] - https://github.com/dialogue-evaluation/factRuEval-2016

[3] - http://ai-center.botik.ru/Airec/index.php/ru/collections/28-persons-1000

[4] -  Reproducing Russian NER Baseline Quality without Additional Data. In proceedings of the 3rd Internationa
l Workshop on Concept
Discovery in Unstructured Data, Moscow, Russia, 54 – 59 (201
6)

[5] - Rubaylo A. V., Kosenko M. Y.: Software utilities for natural language information
retrievial. Almanac of modern science and education, Volume 12 (114), 87 – 92.(2016)

[6] - Sysoev A. A., Andrianov I. A.: Named Entity Recognition in Russian: the Power of Wiki-Based Approach. dialog-21.ru

[7] - Ivanitskiy Roman, Alexander Shipilo, Liubov Kovriguina: Russian Named Entities Recognition and Classification Using Distributed Word and Phrase Representations. In SIMBig, 150 – 156. (2016).

[8] - Mozharova V., Loukachevitch N.: Two-stage approach in Russian named entity recognition. In Intelligence, Social Media and Web (ISMW FRUCT), 2016 International FRUCT Conference, 1 – 6 (2016)

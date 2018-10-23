# This repository is outdated please move to https://github.com/deepmipt/DeepPavlov

# Neural Networks for Named Entity Recognition

In this repo you can find several neural network architectures for named entity recognition from the paper "_Application of a Hybrid Bi-LSTM-CRF model to the task of Russian Named Entity Recognition_" https://arxiv.org/pdf/1709.09686.pdf, which is inspired by LSTM+CRF architecture from https://arxiv.org/pdf/1603.01360.pdf.

NER class from ner/network.py provides methods for construction, training and inference neural networks for Named Entity Recognition.

We provide pre-trained CNN model for Russian Named Entity Recognition.
The model was trained on three datatasets:

- Gareev corpus [1] (obtainable by request to authors)
- FactRuEval 2016 [2]
- NE3 (extended Persons-1000) [3, 4]

The pre-trained model can recognize such entities as:

- Persons (PER)
- Locations (LOC)
- Organizations (ORG)

An example of usage of the pre-trained model is provided in [example.ipynb](https://github.com/deepmipt/ner/blob/master/example.ipynb).

Remark: at training stage the corpora were lemmatized and lowercased.
So text must be tokenized and lemmatized and lowercased before feeding it into the model.

The F1 measure for presented model along with other published solution provided in the table below:

| Models                | Gareev’s dataset | Persons-1000 | FactRuEval 2016 |
|---------------------- |:----------------:|:------------:|:---------------:|
| Gareev et al. [1]     | 75.05            |              |                 |
| Malykh et al. [5]     | 62.49            |              |                 |
| Trofimov  [6]         |                  | 95.57        |                 |
| Rubaylo et al. [7]    |                  |              | 78.13           |
| Sysoev et al. [8]     |                  |              | 74.67           |
| Ivanitsky et al.  [9] |                  |              | **87.88**       |
| Mozharova et al.  [10] |                  | 97.21        |                 |
| Our (Bi-LSTM+CRF)     | **87.17**        | **99.26**    | 82.10           ||

### Usage

#### Installing
The toolkit is implemented in Python 3 and requires a number of packages. To install all needed packages use:
```
$ pip3 install -r requirements.txt
```

or

```
$ pip3 install git+https://github.com/deepmipt/ner
```

Warning: there is no GPU version of TensorFlow specified in the requirements file

#### Command-Line Interface
The simplest way to use pre-trained Russian NER model is via command line interface:

    $ echo "На конспирологическом саммите в США глава Федерального Бюро Расследований сделал невероятное заявление" | ./ner.py

    На O
    конспирологическом O
    саммите O
    в O
    США B-LOC
    глава O
    Федерального B-ORG
    Бюро I-ORG
    Расследований I-ORG
    сделал O
    невероятное O
    заявление O

And for interactive usage simply type:

    $ ./ner.py

#### Usage as module

```
>>> import ner
>>> extractor = ner.Extractor()
>>> for m in extractor("На конспирологическом саммите в США глава Федерального Бюро Расследований сделал невероятное заявление"):
...     print(m)
Match(tokens=[Token(span=(32, 35), text='США')], span=Span(start=32, end=35), type='LOC')
Match(tokens=[Token(span=(42, 54), text='Федерального'), Token(span=(55, 59), text='Бюро'), Token(span=(60, 73), text='Расследований')], span=Span(start=42, end=73), type='ORG')
```

### Training

To see how to train the network and what format of data is required see [training_example.ipynb](https://github.com/deepmipt/ner/blob/master/training_example.ipynb) jupyter notebook.

### Literature

[1] - Rinat Gareev, Maksim Tkachenko, Valery Solovyev, Andrey Simanovsky, Vladimir Ivanov: Introducing Baselines for Russian Named Entity Recognition. Computational Linguistics and Intelligent Text Processing, 329 -- 342 (2013).

[2] - https://github.com/dialogue-evaluation/factRuEval-2016

[3] - http://ai-center.botik.ru/Airec/index.php/ru/collections/28-persons-1000

[4] - http://labinform.ru/pub/named_entities/descr_ne.htm

[5] -  Reproducing Russian NER Baseline Quality without Additional Data. In proceedings of the 3rd International Workshop on ConceptDiscovery in Unstructured Data, Moscow, Russia, 54 – 59 (2016)

[6] - Rubaylo A. V., Kosenko M. Y.: Software utilities for natural language information
retrievial. Almanac of modern science and education, Volume 12 (114), 87 – 92.(2016)

[7] - Sysoev A. A., Andrianov I. A.: Named Entity Recognition in Russian: the Power of Wiki-Based Approach. dialog-21.ru

[8] - Ivanitskiy Roman, Alexander Shipilo, Liubov Kovriguina: Russian Named Entities Recognition and Classification Using Distributed Word and Phrase Representations. In SIMBig, 150 – 156. (2016).

[9] - Mozharova V., Loukachevitch N.: Two-stage approach in Russian named entity recognition. In Intelligence, Social Media and Web (ISMW FRUCT), 2016 International FRUCT Conference, 1 – 6 (2016)

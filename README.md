# Neural Networks for Named Entity Recognition

In this repo you can find several neural network architectures for named entity recognition.
NER class from ner/network.py provides methods for construction, training and inference neural networks for Named Entity
Recognition.

The model folder contains files for Convolutional Neural Network with Conditional Random Fields on the top. This
model was trained on three datatasets:

- Gareev corpus [1] (obtainable by request to authors)
- FactRuEval 2016 [2]
- Persons-1000 [3]

An example of usage of this model is provided in example.ipynb.
Remark: at training stage corpus was lemmatized.
So text must be tokenized and lemmatized before feeding it into the model.

The results obtained by this model provided in table below:

    Total precision:  95.97%; recall:  95.62%; FB1:  95.80

	LOC: precision:   97.78%; recall:  98.13%; F1:  97.96 3013

	ORG: precision:   90.36%; recall:  89.69%; F1:  90.02 3340

	PER: precision:   98.92%; recall:  98.35%; F1:  98.63 4524

[1] - Rinat Gareev, Maksim Tkachenko, Valery Solovyev, Andrey Simanovsky, Vladimir Ivanov: Introducing Baselines for Russian Named Entity Recognition. Computational Linguistics and Intelligent Text Processing, 329 -- 342 (2013).

[2] - https://github.com/dialogue-evaluation/factRuEval-2016

[3] - http://ai-center.botik.ru/Airec/index.php/ru/collections/28-persons-1000
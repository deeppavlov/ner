import pkg_resources
import os.path
import json
import itertools

import pymorphy2

from ner.corpus import Corpus
from ner.extractor.match import Match
from ner.extractor.span import Span
from ner.network import NER
from ner.tokenizer import Tokenizer
from ner.utils import lemmatize, download_untar


class Extractor:
    def __init__(self,
                 model_path=None,
                 tokenizer=None,
                 model_url='http://lnsigo.mipt.ru/export/models/ner/ner_model_total_rus.tar.gz'):
        self.model_path = (
            model_path
            or pkg_resources.resource_filename(__name__, "../model")
        )
        self.model_url = model_url
        self._lazy_download()

        with open(self._get_path('params.json')) as f:
            self.network_params = json.load(f)

        self.corpus = Corpus(dicts_filepath=self._get_path('dict.txt'))
        self.network = NER(
            self.corpus,
            verbouse=False,
            pretrained_model_filepath=self._get_path('ner_model'),
            **self.network_params,
        )

        self.tokenizer = tokenizer or Tokenizer()
        self._morph = pymorphy2.MorphAnalyzer()

    def _lazy_download(self):
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.listdir(self.model_path):
            download_untar(self.model_url, self.model_path)

    def _get_path(self, filename):
        return os.path.join(self.model_path, filename)

    def __call__(self, text):
        tokens = list(self.tokenizer(text))
        tokens_lemmas = lemmatize([t.text for t in tokens], self._morph)
        tags = self.network.predict_for_token_batch([tokens_lemmas])[0]

        previous_tag = null_tag = 'O'
        previous_tokens = []

        for token, current_tag in zip(
                itertools.chain(tokens, [None]),
                itertools.chain(tags, [null_tag])
        ):
            if current_tag.startswith('I'):
                previous_tokens.append(token)
            elif previous_tag != null_tag:
                yield Match(
                    previous_tokens,
                    Span(
                        previous_tokens[0].span[0],
                        previous_tokens[-1].span[1],
                    ),
                    previous_tag[-3:]
                )
            if current_tag.startswith('B'):
                previous_tokens = [token]
            previous_tag = current_tag

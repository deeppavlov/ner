import re

from ner.tokenizer.token import Token


class Tokenizer:
    def __init__(self, token_pattern=r"[\w]+|[‑–—“”€№…’\"#$%&\'()+,-./:;<>?]"):
        self.token_pattern = token_pattern

    def __call__(self, text):
        for match in re.finditer(self.token_pattern, text):
            yield Token(match.span(), match.group())

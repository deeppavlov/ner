import pymorphy2
import re


def lemmatize(words):
    morph = pymorphy2.MorphAnalyzer()
    if isinstance(words, list):
        words_lemma = list()
        for word in words:
            p = morph.parse(word)[0]
            words_lemma.append(p.normal_form)
        return words_lemma
    else:
        p = morph.parse(words)[0]
        return p.normal_form


def tokenize(s):
    return re.findall(r"[\w]+|[‑–—“”€№…’\"#$%&\'()+,-./:;<>?]", s)


def is_end_of_sentence(prev_token, current_token):
    is_capital = current_token[0].isupper()
    is_punctuation = prev_token in ('!', '?', '.')
    return is_capital and is_punctuation


def split_sentences(tokens, tags=None):
    prev_token = ' '
    utterances = list()
    utterances_tags = list()
    if tags is not None:
        tmp_tokens = list()
        tmp_tags = list()
        for n, (token, tag) in enumerate(zip(tokens, tags)):
            if is_end_of_sentence(prev_token, token) and len(tmp_tokens) > 0:
                utterances.append(tmp_tokens)
                utterances_tags.append(tmp_tags)
                tmp_tokens = list()
                tmp_tags = list()
            else:
                tmp_tokens.append(token)
                tmp_tags.append(tag)
        if len(tmp_tokens) > 0:
            utterances.append(tmp_tokens)
            utterances_tags.append(tmp_tags)
        return list(zip(utterances, utterances_tags))
    else:
        tmp_tokens = list()
        for n, token in enumerate(tokens):
            if is_end_of_sentence(prev_token, token) and len(tmp_tokens) > 0:
                utterances.append(tmp_tokens)
                tmp_tokens = list()
            else:
                tmp_tokens.append(token)
        if len(tmp_tokens) > 0:
            utterances.append(tmp_tokens)
        return utterances

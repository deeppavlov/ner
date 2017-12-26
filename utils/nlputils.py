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
            prev_token = token
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


def get_list_of_us_geo_objects(geo_file='/home/mikhail/Data/us.csv', countries_file='/home/mikhail/Data/countries.txt'):
    states = set()
    cities = set()
    states_short = set()
    with open(geo_file) as f:
        for line in f:
            items = line.split(',')
            if len(items) == 10:
                cities.add(items[2])
                states.add(items[3])
                states_short.add(items[4])

    countries = set()
    with open(countries_file) as f:
        for line in f:
            countries.add(line.strip())
    return states, cities, states_short, countries


def get_list_of_countries(country_file='/home/mikhail/Data/countries.txt'):
    with open(country_file) as f:
        country_list = list()
        for line in f:
            country_list.append(line.strip())
    return country_list

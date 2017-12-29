"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import requests
from tqdm import tqdm
import tarfile
import hashlib
import sys
import pymorphy2
import re


# ------------------------ NLP utils ------------------------------

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


# -------------------------- File Utils ----------------------------------


def download(dest_file_path, source_url):
    r""""Simple http file downloader"""
    print('Downloading from {} to {}'.format(source_url, dest_file_path))
    sys.stdout.flush()
    datapath = os.path.dirname(dest_file_path)
    os.makedirs(datapath, mode=0o755, exist_ok=True)

    dest_file_path = os.path.abspath(dest_file_path)

    r = requests.get(source_url, stream=True)
    total_length = int(r.headers.get('content-length', 0))

    with open(dest_file_path, 'wb') as f:
        pbar = tqdm(total=total_length, unit='B', unit_scale=True)
        for chunk in r.iter_content(chunk_size=32 * 1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)


def untar(file_path, extract_folder=None):
    r"""Simple tar archive extractor

    Args:
        file_path: path to the tar file to be extracted
        extract_folder: folder to which the files will be extracted

    """
    if extract_folder is None:
        extract_folder = os.path.dirname(file_path)
    tar = tarfile.open(file_path)
    tar.extractall(extract_folder)
    tar.close()


def download_untar(url, download_path, extract_path=None):
    r"""Download an archive from http link, extract it and then delete the archive"""
    file_name = url.split('/')[-1]
    if extract_path is None:
        extract_path = download_path
    tar_file_path = os.path.join(download_path, file_name)
    download(tar_file_path, url)
    sys.stdout.flush()
    print('Extracting {} archive into {}'.format(tar_file_path, extract_path))
    untar(tar_file_path, extract_path)
    os.remove(tar_file_path)


def md5_hashsum(file_names):
    # Check hashsum for file_names
    hash_md5 = hashlib.md5()
    for file_name in file_names:
        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()

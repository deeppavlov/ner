from collections import Counter
from collections import defaultdict
import random
import numpy as np
import os
import ftplib

DATA_PATH = '/tmp/ner_gareev'
SEED = 42
SPECIAL_TOKENS = ['<PAD>', '<UNK>']
SPECIAL_TAGS = ['<PAD>']
DOC_START_STRING = '-DOCSTART- -X- -X- O'
np.random.seed(SEED)
random.seed(SEED)


def is_end_of_sentence(prev_token, current_token):
    is_capital = current_token[0].isupper()
    is_punctuation = prev_token in ('!', '?', '.')
    return is_capital and is_punctuation


def ftp_gareev_loader():
    server = 'share.ipavlov.mipt.ru'
    username = 'anonymous'
    password = ''
    directory = '/datasets/gareev/'
    filematch = '*.iob'
    ftp = ftplib.FTP(server)
    ftp.login(username, password)
    ftp.cwd(directory)
    tmp_tags = []
    tmp_tokens = []
    prev_token = '\n'
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    tmp_file_path = '/tmp/ner_gareev/ner_tmp_file.txt'
    for file_name in ftp.nlst(filematch):
        with open(tmp_file_path, 'wb') as tmp_f:
            ftp.retrbinary('RETR ' + file_name, tmp_f.write)
        with open(tmp_file_path) as tmp_f:
            lines_list = tmp_f.readlines()

        for line in lines_list:
            if len(line) > 2:
                token, tag = line.split()
                if not is_end_of_sentence(prev_token, token):
                    tmp_tags.append(tag)
                    tmp_tokens.append(token)
                elif len(tmp_tokens) > 0:
                    yield tmp_tokens, tmp_tags
                    tmp_tags = [tag]
                    tmp_tokens = [token]
                else:
                    tmp_tags = []
                    tmp_tokens = []
                prev_token = token
    os.remove(tmp_file_path)


def dataset_slicer(x_y_list,
                   train_part=0.6,
                   valid_part=0.2,
                   test_part=0.2,
                   shuffle=True):
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    assert np.abs(train_part + valid_part + test_part - 1) < 1e-6
    n_samples = len(x_y_list)

    if shuffle:
        random.shuffle(x_y_list)

    slices = [0] + [int(part * n_samples) for part in (train_part, valid_part, test_part)]
    slices = np.cumsum(slices)
    for n, name in enumerate(['train', 'valid', 'test']):
        start = slices[n]
        stop = slices[n + 1]
        with open(os.path.join(DATA_PATH, name + '.txt'), 'w') as f:
            for tokens, tags in x_y_list[start: stop]:
                for token, tag in zip(tokens, tags):
                    f.write(token + ' ' + tag + '\n')
                f.write('\n')


# Gareev preprocessed files reader
# No doc information preserved
def data_reader_gareev(data_path=None, data_type=None):
    if data_path is None:
        # Maybe download
        if not os.path.exists(os.path.join(DATA_PATH, 'train.txt')):
            xy_list = list(ftp_gareev_loader())
            dataset_slicer(xy_list)
        data_path = DATA_PATH
    data = dict()
    if data_type is None:
        data_types = ['train', 'test', 'valid']
    else:
        data_types = [data_type]
    for key in data_types:
        path = os.path.join(os.path.join(data_path, key + '.txt'))
        x = []
        y = []
        with open(path) as f:
            tokens = []
            tags = []
            for line in f:
                if len(line) > 1 and DOC_START_STRING not in line:
                    items = line.split()
                    tokens.append(items[0])
                    tags.append(items[-1])
                elif len(tokens) > 0:
                    x.append(tokens)
                    y.append(tags)
                    tokens = []
                    tags = []
        data[key] = (x, y)
    return data


# CoNLL-2003 preprocessed files reader
# No doc information preserved
def data_reader(data_path='data/', data_type=None):
    data = dict()
    if data_type is not None:
        data_types = ['train', 'test', 'valid']
    else:
        data_types = [data_type]
    for key in data_types:
        path = os.path.join(data_path + key + '.txt')
        x = []
        y = []
        with open(path) as f:
            tokens = []
            tags = []
            for line in f:
                if len(line) > 1 and DOC_START_STRING not in line:
                    items = line.split()
                    tokens.append(items[0])
                    tags.append(items[-1])
                elif len(tokens) > 0:
                    x.append(tokens)
                    y.append(tags)
                    tokens = []
                    tags = []
        data[key] = (x, y)
    return data


# Dictionary class. Each instance holds tags or tokens or characters and provides
# dictionary like functionality like indices to tokens and tokens to indices.
class Vocabulary:
    def __init__(self, tokens=None, default_token='<UNK>', is_tags=False):
        if is_tags:
            special_tokens = SPECIAL_TAGS
            self._t2i = dict()
        else:
            special_tokens = SPECIAL_TOKENS
            if default_token not in special_tokens:
                raise Exception('SPECIAL_TOKENS must contain <UNK> token!')
            # We set default ind to position of <UNK> in SPECIAL_TOKENS
            # because the tokens will be added to dict in the same order as
            # in SPECIAL_TOKENS
            default_ind = special_tokens.index('<UNK>')
            self._t2i = defaultdict(lambda: default_ind)
        self._i2t = []
        self.frequencies = Counter()

        self.counter = 0
        for token in special_tokens:
            self._t2i[token] = self.counter
            self.frequencies[token] += 0
            self._i2t.append(token)
            self.counter += 1
        if tokens is not None:
            self.update_dict(tokens)

    def update_dict(self, tokens):
        for token in tokens:
            if token not in self._t2i:
                self._t2i[token] = self.counter
                self._i2t.append(token)
                self.counter += 1
            self.frequencies[token] += 1

    def idx2tok(self, idx):
        return self._i2t[idx]

    def idxs2toks(self, idxs, filter_paddings=False):
        toks = []
        for idx in idxs:
            if not filter_paddings or idx != self.tok2idx('<PAD>'):
                toks.append(self._i2t[idx])
        return toks

    def tok2idx(self, tok):
        return self._t2i[tok]

    def toks2idxs(self, toks):
        return [self._t2i[tok] for tok in toks]

    def batch_toks2batch_idxs(self, b_toks):
        max_len = max(len(toks) for toks in b_toks)
        # Create array filled with paddings
        batch = np.ones([len(b_toks), max_len]) * self.tok2idx('<PAD>')
        for n, tokens in enumerate(b_toks):
            idxs = self.toks2idxs(tokens)
            batch[n, :len(idxs)] = idxs
        return batch

    def batch_idxs2batch_toks(self, b_idxs, filter_paddings=False):
        return [self.idxs2toks(idxs, filter_paddings) for idxs in b_idxs]

    def is_pad(self, x_t):
        assert type(x_t) == np.ndarray
        return x_t == self.tok2idx('<PAD>')

    def __getitem__(self, key):
        return self._t2i[key]

    def __len__(self):
        return self.counter

    def __contains__(self, item):
        return item in self._t2i


class Corpus:
    def __init__(self, reader=None, embeddings_file_path=None):
        if reader is None:
            self.dataset = data_reader()
        else:
            self.dataset = reader()
        self.token_dict = Vocabulary(self.get_tokens())
        self.tag_dict = Vocabulary(self.get_tags(), is_tags=True)
        self.char_dict = Vocabulary(self.get_characters())
        if embeddings_file_path is not None:
            self.embeddings = self.load_embeddings(embeddings_file_path)
        else:
            self.embeddings = None

    # All tokens for dictionary building
    def get_tokens(self, data_type='train'):
        utterances = self.dataset[data_type][0]
        for utterance in utterances:
            for token in utterance:
                yield token

    # All tags for dictionary building
    def get_tags(self, data_type='train'):
        utterances = self.dataset[data_type][1]
        for utterance in utterances:
            for tag in utterance:
                yield tag

    def get_characters(self, data_type='train'):
        utterances = self.dataset[data_type][0]
        for utterance in utterances:
            for token in utterance:
                for character in token:
                    yield character

    def load_embeddings(self, file_path):
        # Embeddins must be in fastText format
        print('Loading embeddins...')
        pre_trained_embeddins_dict = dict()
        with open(file_path) as f:
            _ = f.readline()
            for line in f:
                token, *embedding = line.split()
                embedding = np.array([float(val_str) for val_str in embedding])
                if token in self.token_dict:
                    pre_trained_embeddins_dict[token] = embedding
        print('Readed')
        pre_trained_std = np.std(list(pre_trained_embeddins_dict.values()))
        embeddings = pre_trained_std * np.random.randn(len(self.token_dict), len(embedding))
        for idx in range(len(self.token_dict)):
            token = self.token_dict.idx2tok(idx)
            if token in pre_trained_embeddins_dict:
                embeddings[idx] = pre_trained_embeddins_dict[token]
        return embeddings

    def tokens_to_x_xc(self, tokens):
        n_tokens = len(tokens)
        tok_idxs = self.token_dict.toks2idxs(tokens)
        char_idxs = []
        max_char_len = 0
        for token in tokens:
            char_idxs.append(self.char_dict.toks2idxs(token))
            max_char_len = max(max_char_len, len(token))
        toks = np.zeros([1, n_tokens], dtype=np.int32)
        chars = np.zeros([1, n_tokens, max_char_len], dtype=np.int32)
        toks[0, :] = tok_idxs
        for n, char_line in enumerate(char_idxs):
            chars[0, n, :len(char_line)] = char_line
        return toks, chars

    def batch_generator(self,
                        batch_size,
                        dataset_type='train',
                        shuffle=True,
                        allow_smaller_last_batch=True,
                        provide_char=True):
        utterances, tags = self.dataset[dataset_type]
        n_samples = len(utterances)
        if shuffle:
            order = np.random.permutation(n_samples)
        else:
            order = np.arange(n_samples)
        n_batches = n_samples // batch_size
        if allow_smaller_last_batch and n_samples % batch_size:
            n_batches += 1
        for k in range(n_batches):
            batch_start = k * batch_size
            batch_end = min((k + 1) * batch_size, n_samples)
            current_batch_size = batch_end - batch_start
            x_token_list = []
            x_char_list = []
            y_list = []
            max_len_token = 0
            max_len_char = 0
            # TODO: REFACTOR
            for idx in order[batch_start: batch_end]:
                current_char_list = []
                for token in utterances[idx]:
                    current_char_list.append(self.char_dict.toks2idxs(token))
                    max_len_char = max(max_len_char, len(token))
                x_char_list.append(current_char_list)
                x_token_list.append(self.token_dict.toks2idxs(utterances[idx]))
                y_list.append(self.tag_dict.toks2idxs(tags[idx]))
                max_len_token = max(max_len_token, len(tags[idx]))
            x_token = np.ones([current_batch_size, max_len_token], dtype=np.int32) * self.token_dict['<PAD>']
            x_char = np.ones([current_batch_size, max_len_token, max_len_char], dtype=np.int32) * self.char_dict['<PAD>']
            y = np.ones([current_batch_size, max_len_token], dtype=np.int32) * self.tag_dict['<PAD>']
            for n in range(current_batch_size):
                utt_len = len(x_token_list[n])
                x_token[n, :utt_len] = x_token_list[n]
                y[n, :utt_len] = y_list[n]
                for k, ch in enumerate(x_char_list[n]):
                    char_len = len(ch)
                    x_char[n, k, :char_len] = ch
            if provide_char:
                yield (x_token, x_char), y
            else:
                yield x_token, y


if __name__ == '__main__':

    # Create Gareev corpus
    corp = Corpus(data_reader_gareev)
    s = 'С . - ПЕТЕРБУРГ , 23 июн - РИА Новости . Группа компаний \" Связной \" ,'
    print(corp.tokens_to_x_xc(s.split()))
    # Check batching
    batch_size = 2
    (x, xc), y = corp.batch_generator(batch_size, dataset_type='test').__next__()


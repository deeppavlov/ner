from collections import Counter
from collections import defaultdict
import numpy as np
import os

SEED = 42
SPECIAL_TOKENS = ['<PAD>', '<UNK>']
SPECIAL_TAGS = ['<PAD>']
DOC_START_STRING = '-DOCSTART- -X- -X- O'
np.random.seed(SEED)


# Gareev preprocessed files reader
# No doc information preserved
def data_reader_gareev(data_path='./data', data_type=None):
    data = dict()
    if data_type is not None:
        ['train', 'test', 'valid']
    for key in ['train', 'test', 'valid']:
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
        ['train', 'test', 'valid']
    for key in ['train', 'test', 'valid']:
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
    # data = data_reader()
    corpus = Corpus()
    corpus.load_embeddings('/home/arcady/Data/nlp/embeddings_lenta.vec')
    batch = corpus.batch_generator(32, dataset_type='test').__next__()
    print(batch)
    print(corpus)

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
from collections import defaultdict
import numpy as np
import tensorflow as tf
from .layers import character_embedding_network
from .layers import embedding_layer
from .layers import stacked_convolutions
from .layers import highway_convolutional_network
from .layers import stacked_rnn
from tensorflow.contrib.layers import xavier_initializer


from .corpus import Corpus
from .evaluation import precision_recall_f1


SEED = 42
MODEL_PATH = 'model/'
MODEL_FILE_NAME = 'ner_model.ckpt'


class NER:
    def __init__(self,
                 corpus,
                 n_filters=(128, 256),
                 filter_width=3,
                 token_embeddings_dim=128,
                 char_embeddings_dim=50,
                 use_char_embeddins=True,
                 pretrained_model_filepath=None,
                 embeddings_dropout=False,
                 dense_dropout=False,
                 use_batch_norm=False,
                 logging=False,
                 use_crf=False,
                 net_type='cnn',
                 char_filter_width=5,
                 verbouse=True,
                 use_capitalization=False,
                 concat_embeddings=False,
                 cell_type=None,
                 two_dense_layers=True,
                 use_focal_loss=False,
                 l2_beta=0,
                 gpu=0):
        tf.reset_default_graph()

        n_tags = len(corpus.tag_dict)
        n_tokens = len(corpus.token_dict)
        n_chars = len(corpus.char_dict)
        embeddings_onethego = not concat_embeddings and \
                              corpus.embeddings is not None and \
                              not isinstance(corpus.embeddings, dict)

        # Create placeholders
        if embeddings_onethego:
            x_word = tf.placeholder(dtype=tf.float32, shape=[None, None, corpus.embeddings.vector_size], name='x_word')
        else:
            x_word = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_word')
        if concat_embeddings:
            x_emb = tf.placeholder(dtype=tf.float32, shape=[None, None, corpus.embeddings.vector_size], name='x_word')
        x_char = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='x_char')
        y_true = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_tag')
        mask = tf.placeholder(dtype=tf.float32, shape=[None, None], name='mask')
        x_capi = tf.placeholder(dtype=tf.float32, shape=[None, None], name='x_capi')

        # Auxiliary placeholders
        learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        dropout_ph = tf.placeholder_with_default(1.0, shape=[])
        training_ph = tf.placeholder_with_default(False, shape=[])
        learning_rate_decay_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate_decay')

        # Embeddings
        if not embeddings_onethego:
            with tf.variable_scope('Embeddings'):
                w_emb = embedding_layer(x_word, n_tokens=n_tokens, token_embedding_dim=token_embeddings_dim)
                if use_char_embeddins:
                    c_emb = character_embedding_network(x_char,
                                                        n_characters=n_chars,
                                                        char_embedding_dim=char_embeddings_dim,
                                                        filter_width=char_filter_width)
                    emb = tf.concat([w_emb, c_emb], axis=-1)
                else:
                    emb = w_emb
        else:
            emb = x_word

        if concat_embeddings:
            emb = tf.concat([emb, x_emb], axis=2)

        if use_capitalization:
            cap = tf.expand_dims(x_capi, 2)
            emb = tf.concat([emb, cap], axis=2)

        # Dropout for embeddings
        if embeddings_dropout:
            emb = tf.layers.dropout(emb, dropout_ph, training=training_ph)

        if 'cnn' in net_type.lower():
            # Convolutional network
            with tf.variable_scope('ConvNet'):
                units = stacked_convolutions(emb,
                                             n_filters=n_filters,
                                             filter_width=filter_width,
                                             use_batch_norm=use_batch_norm,
                                             training_ph=training_ph)
        elif 'rnn' in net_type.lower():
            if cell_type is None or cell_type not in {'lstm', 'gru'}:
                raise RuntimeError('You must specify the type of the cell! It could be either "lstm" or "gru"')
            with tf.variable_scope('RecurrentNet'):
                units = stacked_rnn(emb, n_filters, cell_type=cell_type)

        elif 'cnn_highway' in net_type.lower():
                units = highway_convolutional_network(emb,
                                                      n_filters=n_filters,
                                                      filter_width=filter_width,
                                                      use_batch_norm=use_batch_norm,
                                                      training_ph=training_ph)
        else:
            raise KeyError('There is no such type of network: {}'.format(net_type))

        # Classifier
        with tf.variable_scope('Classifier'):
            logits = tf.layers.dense(units, n_tags, kernel_initializer=xavier_initializer())

        if use_crf:
            sequence_lengths = tf.reduce_sum(mask, axis=1)
            log_likelihood, trainsition_params = tf.contrib.crf.crf_log_likelihood(logits,
                                                                                   y_true,
                                                                                   sequence_lengths)
            loss_tensor = -log_likelihood
            print(loss_tensor)
            if use_focal_loss:
                gamma = 5
                probs = tf.nn.softmax(logits)
                ground_truth_labels = tf.one_hot(y_true, n_tags)
                ground_truth_prob = tf.reduce_sum(ground_truth_labels * probs, axis=2)
                print(ground_truth_prob)
                loss_tensor *= (1 - ground_truth_prob) ** gamma
            predictions = None
        else:
            ground_truth_labels = tf.one_hot(y_true, n_tags)
            loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_labels, logits=logits)
            loss_tensor = loss_tensor * mask
            predictions = tf.argmax(logits, axis=-1)

        loss = tf.reduce_mean(loss_tensor)

        # Initialize session
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(gpu)

        sess = tf.Session(config=config)

        self._use_crf = use_crf
        self.summary = tf.summary.merge_all()

        self._learning_rate_decay_ph = learning_rate_decay_ph
        self._x_w = x_word
        self._x_c = x_char
        self._y_true = y_true
        self._y_pred = predictions
        if concat_embeddings:
            self._x_emb = x_emb
        if use_crf:
            self._logits = logits
            self._trainsition_params = trainsition_params
            self._sequence_lengths = sequence_lengths
        self._learning_rate_ph = learning_rate_ph
        self._dropout = dropout_ph

        self._loss = loss

        self._sess = sess
        self.corpus = corpus

        self._loss_tensor = loss_tensor
        self._use_dropout = True if embeddings_dropout or dense_dropout else None

        self._training_ph = training_ph
        self._logging = logging

        # Get training op
        self._train_op = self.get_train_op(loss, learning_rate_ph, lr_decay_rate=learning_rate_decay_ph)
        self._embeddings_onethego = embeddings_onethego
        self.verbose = verbouse
        sess.run(tf.global_variables_initializer())
        self._mask = mask
        if use_capitalization:
            self._x_capi = x_capi
        self._use_capitalization = use_capitalization
        self._concat_embeddings = concat_embeddings
        if pretrained_model_filepath is not None:
            self.load(pretrained_model_filepath)

    def save(self, model_file_path=None, add_recurrent_layer_prefix=False):
        var_dict = {(['', 'RecurrentNet/'][var.name.startswith('RNN')] + var.name[:-2]): var for var in tf.trainable_variables()}
        if model_file_path is None:
            if not os.path.exists(MODEL_PATH):
                os.mkdir(MODEL_PATH)
            model_file_path = os.path.join(MODEL_PATH, MODEL_FILE_NAME)
        saver = tf.train.Saver(var_dict)
        saver.save(self._sess, model_file_path)

    def load(self, model_file_path):
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(self._sess, model_file_path)

    def train_on_batch(self, x_word, x_char, y_tag):
        feed_dict = {self._x_w: x_word, self._x_c: x_char, self._y_true: y_tag}
        self._sess.run(self._train_op, feed_dict=feed_dict)

    @staticmethod
    def print_number_of_parameters():
        print('Number of parameters: ')
        vars = tf.trainable_variables()
        blocks = defaultdict(int)
        for var in vars:
            # Get the top level scope name of variable
            block_name = var.name.split('/')[0]
            number_of_parameters = np.prod(var.get_shape().as_list())
            blocks[block_name] += number_of_parameters
        for block_name in blocks:
            print(block_name, blocks[block_name])
        total_num_parameters = np.sum(list(blocks.values()))
        print('Total number of parameters equal {}'.format(total_num_parameters))

    def fit(self,
            batch_gen=None,
            batch_size=32,
            learning_rate=1e-3,
            epochs=1,
            dropout_rate=0.5,
            learning_rate_decay=1,
            save_path=None):
        best_score = 0
        try:
            for epoch in range(epochs):
                if self.verbose:
                    print('Epoch {}'.format(epoch))
                if batch_gen is None:
                    batch_generator = self.corpus.batch_generator(batch_size, dataset_type='train')
                for x, y in batch_generator:
                    feed_dict = self._fill_feed_dict(x,
                                                     y,
                                                     learning_rate,
                                                     dropout_rate=dropout_rate,
                                                     training=True,
                                                     learning_rate_decay=learning_rate_decay)
                    if self._logging:
                        summary, _ = self._sess.run([self.summary, self._train_op], feed_dict=feed_dict)
                        self.train_writer.add_summary(summary)

                    self._sess.run(self._train_op, feed_dict=feed_dict)
                scores = self.eval_conll('valid', print_results=True, short_report=False)
                if best_score < scores['__total__']['f1'] and save_path is not None:
                    print('Model saved')
                    self.save(save_path)
        except KeyboardInterrupt:
            self.eval_conll('test', print_results=True, short_report=False)

        if self.verbose:
            self.eval_conll(dataset_type='train', short_report=False)
            self.eval_conll(dataset_type='valid', short_report=False)
            results = self.eval_conll(dataset_type='test', short_report=False)
        else:
            results = self.eval_conll(dataset_type='test', short_report=True)
        return results

    def predict(self, x):
        feed_dict = self._fill_feed_dict(x, training=False)
        if self._use_crf:
            y_pred = []
            logits, trans_params, sequence_lengths = self._sess.run([self._logits,
                                                                     self._trainsition_params,
                                                                     self._sequence_lengths
                                                                     ],
                                                                    feed_dict=feed_dict)

            # iterate over the sentences because no batching in viterbi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:int(sequence_length)]  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                y_pred += [viterbi_seq]
        else:
            y_pred = self._sess.run(self._y_pred, feed_dict=feed_dict)
        return self.corpus.tag_dict.batch_idxs2batch_toks(y_pred, filter_paddings=True)

    def eval_conll(self, dataset_type='test', print_results=True, short_report=True):
        y_true_list = list()
        y_pred_list = list()
        print('Eval on {}:'.format(dataset_type))
        import time
        start_time = time.time()

        for x, y_gt in self.corpus.batch_generator(batch_size=32, dataset_type=dataset_type):
            y_pred = self.predict(x)
            y_gt = self.corpus.tag_dict.batch_idxs2batch_toks(y_gt, filter_paddings=True)
            for tags_pred, tags_gt in zip(y_pred, y_gt):
                for tag_predicted, tag_ground_truth in zip(tags_pred, tags_gt):
                    y_true_list.append(tag_ground_truth)
                    y_pred_list.append(tag_predicted)
                y_true_list.append('O')
                y_pred_list.append('O')
        self.eval_conll_script(y_true_list, y_pred_list)
        print("--- %s seconds ---" % (time.time() - start_time))
        return precision_recall_f1(y_true_list,
                                   y_pred_list,
                                   print_results,
                                   short_report)

    def _fill_feed_dict(self,
                        x,
                        y_t=None,
                        learning_rate=None,
                        training=False,
                        dropout_rate=1,
                        learning_rate_decay=1):

        feed_dict = dict()
        if self._embeddings_onethego:
            feed_dict[self._x_w] = x['emb']
        else:
            feed_dict[self._x_w] = x['token']
        feed_dict[self._x_c] = x['char']
        feed_dict[self._mask] = x['mask']
        feed_dict[self._training_ph] = training
        if y_t is not None:
            feed_dict[self._y_true] = y_t

        # Optional arguments
        if self._use_capitalization:
            feed_dict[self._x_capi] = x['capitalization']

        if self._concat_embeddings:
            feed_dict[self._x_emb] = x['emb']

        # Learning rate
        if learning_rate is not None:
            feed_dict[self._learning_rate_ph] = learning_rate
            feed_dict[self._learning_rate_decay_ph] = learning_rate_decay

        # Dropout
        if self._use_dropout is not None and training:
            feed_dict[self._dropout] = dropout_rate
        else:
            feed_dict[self._dropout] = 1.0
        return feed_dict

    def eval_loss(self, data_type='test', batch_size=32):
        # TODO: fixup
        num_tokens = 0
        loss = 0
        for x, y_t in self.corpus.batch_generator(batch_size=batch_size, dataset_type=data_type):
            feed_dict = self._fill_feed_dict(x, y_t, training=False)
            loss += np.sum(self._sess.run(self._loss_tensor, feed_dict=feed_dict))
            num_tokens += np.sum(self.corpus.token_dict.is_pad(x_w))
        return loss / num_tokens

    @staticmethod
    def get_trainable_variables(trainable_scope_names=None):
        vars = tf.trainable_variables()
        if trainable_scope_names is not None:
            vars_to_train = list()
            for scope_name in trainable_scope_names:
                for var in vars:
                    if var.name.startswith(scope_name):
                        vars_to_train.append(var)
            return vars_to_train
        else:
            return vars

    def get_train_op(self, loss, learning_rate, learnable_scopes=None, lr_decay_rate=None):
        global_step = tf.Variable(0, trainable=False)
        try:
            n_training_samples = len(self.corpus.dataset['train'])
        except TypeError:
            n_training_samples = 1024
        batch_size = tf.shape(self._x_w)[0]
        decay_steps = tf.cast(n_training_samples / batch_size, tf.int32)
        if lr_decay_rate is not None:
            learning_rate = tf.train.exponential_decay(learning_rate,
                                                       global_step,
                                                       decay_steps=decay_steps,
                                                       decay_rate=lr_decay_rate,
                                                       staircase=True)
            self._learning_rate_decayed = learning_rate
        variables = self.get_trainable_variables(learnable_scopes)

        # For batch norm it is necessary to update running averages
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, var_list=variables)
        return train_op

    def predict_for_token_batch(self, tokens_batch):
        batch_x, _ = self.corpus.tokens_batch_to_numpy_batch(tokens_batch)
        # Prediction indices
        predictions_batch = self.predict(batch_x)
        predictions_batch_no_pad = list()
        for n, predicted_tags in enumerate(predictions_batch):
            predictions_batch_no_pad.append(predicted_tags[: len(tokens_batch[n])])
        return predictions_batch_no_pad

    def eval_conll_script(self,
                          y_true_list,
                          y_pred_list,
                          output_filepath='output.txt',
                          report_file_path='conll_evaluation.txt',
                          dataset_type='test'):
        with open(output_filepath, 'w') as f:
            for tag_predicted, tag_ground_truth in zip(y_pred_list, y_true_list):
                f.write(' '.join(['word'] + ['pur'] * 4 + [tag_ground_truth] + [tag_predicted]) + '\n')

        conll_evaluation_script = os.path.join('.', 'conlleval')
        shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script, output_filepath, report_file_path)
        os.system(shell_command)
        print(dataset_type.capitalize() + ' set')
        with open(report_file_path) as f:
            for line in f:
                print(line)


if __name__ == '__main__':
    corp = Corpus(dicts_filepath='dict.txt')

    parameters = {'n_conv_layers': 2,
                  'n_filters': 100,
                  'filter_width': 5,
                  'token_embeddings_dim': 100,
                  'char_embeddings_dim': 25,
                  'use_batch_norm': False,
                  'use_crf': True}

    # Creating a convolutional NER model
    ner = NER(corp, **parameters)

    # Training the model
    ner.fit(epochs=10,
            batch_size=8,
            learning_rate=1e-2,
            dropout_rate=0.5)

    # Creating new predict_for_token_batch model and restoring pre-trained weights
    path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
    model_path = os.path.join(path, MODEL_PATH, MODEL_FILE_NAME)
    ner_ = NER(corp, pretrained_model_filepath=model_path, **parameters)
    # Evaluate loaded model
    print('Success')

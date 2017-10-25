import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from .layers import character_embedding_network
from .layers import embedding_layer
from .layers import stacked_convolutions
from tensorflow.contrib.layers import xavier_initializer

from .corpus import Corpus
from .evaluation import precision_recall_f1


SEED = 42
MODEL_PATH = 'model/'
MODEL_FILE_NAME = 'ner_model.ckpt'


class NER:
    def __init__(self,
                 corpus,
                 n_conv_layers=3,
                 n_filters=100,
                 filter_width=3,
                 token_embeddings_dim=100,
                 char_embeddings_dim=25,
                 pretrained_model_filepath=None,
                 embeddings_dropout=False,
                 dense_dropout=False,
                 use_batch_norm=False,
                 logging=False):
        tf.reset_default_graph()

        n_tags = len(corpus.tag_dict)
        n_tokens = len(corpus.token_dict)
        n_chars = len(corpus.char_dict)
        embeddings_onethego = corpus.embeddings is not None and not isinstance(corpus.embeddings, dict)

        # Create placeholders
        if embeddings_onethego:
            x_word = tf.placeholder(dtype=tf.float32, shape=[None, None, corpus.embeddings.vector_size], name='x_word')
        else:
            x_word = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_word')
        x_char = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='x_char')
        y_true = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_tag')
        # Auxiliary placeholders
        learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        dropout_ph = tf.placeholder_with_default(1.0, shape=[])
        training_ph = tf.placeholder_with_default(False, shape=[])
        learning_rate_decay_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate_decay')

        # Embeddings
        if not embeddings_onethego:
            with tf.variable_scope('Embeddings'):
                w_emb = embedding_layer(x_word, n_tokens=n_tokens, token_embedding_dim=token_embeddings_dim)
                c_emb = character_embedding_network(x_char, n_characters=n_chars, char_embedding_dim=char_embeddings_dim)
                emb = tf.concat([w_emb, c_emb], axis=-1)
        else:
            emb = x_word

        # First convolutional network
        with tf.variable_scope('ConvNet'):
            units = stacked_convolutions(emb,
                                         n_filters=n_filters,
                                         n_layers=n_conv_layers,
                                         filter_width=filter_width,
                                         use_batch_norm=use_batch_norm,
                                         training_ph=training_ph)

        # Classifier
        with tf.variable_scope('Classifier'):
            units = tf.layers.dense(units, n_filters, kernel_initializer=xavier_initializer())
            logits = tf.layers.dense(units, n_tags, kernel_initializer=xavier_initializer())
            predictions = tf.argmax(logits, axis=-1)

        # Loss with masking
        ground_truth_labels = tf.one_hot(y_true, n_tags)
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_labels, logits=logits)
        loss_tensor = loss_tensor * tf.cast(tf.not_equal(x_word, corpus.token_dict.tok2idx('<PAD>')), tf.float32)
        loss = tf.reduce_mean(loss_tensor)

        # Initialize session
        sess = tf.Session()

        self.print_number_of_parameters()
        if logging:
            self.train_writer = tf.summary.FileWriter('summary', sess.graph)

        self.summary = tf.summary.merge_all()
        self._learning_rate_decay_ph = learning_rate_decay_ph
        self._x_w = x_word
        self._x_c = x_char
        self._y_true = y_true
        self._y_pred = predictions
        self._loss = loss
        self._sess = sess
        self.corpus = corpus
        self._learning_rate_ph = learning_rate_ph
        self._dropout = dropout_ph
        self._loss_tensor = loss_tensor
        self._use_dropout = True if embeddings_dropout or dense_dropout else None
        if pretrained_model_filepath is not None:
            self.load(pretrained_model_filepath)
        self._training_ph = training_ph
        self._logging = logging
        # Get training op
        self._train_op = self.get_train_op(loss, learning_rate_ph, lr_decay_rate=learning_rate_decay_ph)
        self._embeddings_onethego = embeddings_onethego
        sess.run(tf.global_variables_initializer())

    def save(self, model_file_path=None):
        if model_file_path is None:
            if not os.path.exists(MODEL_PATH):
                os.mkdir(MODEL_PATH)
            model_file_path = os.path.join(MODEL_PATH, MODEL_FILE_NAME)
        saver = tf.train.Saver()
        saver.save(self._sess, model_file_path)

    def load(self, model_file_path):
        saver = tf.train.Saver()
        saver.restore(self._sess, model_file_path)

    def train_on_batch(self, x_word, x_char, y_tag):
        feed_dict = {self._x_w: x_word, self._x_c: x_char, self._y_true: y_tag}
        self._sess.run(self._train_op, feed_dict=feed_dict)

    def print_number_of_parameters(self):
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

    def fit(self, batch_gen=None, batch_size=32, learning_rate=1e-3, epochs=1, dropout_rate=0.5, learning_rate_decay=1):
        for epoch in range(epochs):
            count = 0
            print('Epoch {}'.format(epoch))
            if batch_gen is None:
                batch_generator = self.corpus.batch_generator(batch_size, dataset_type='train')
            for (x_word, x_char), y_tag in batch_generator:
                feed_dict = self._fill_feed_dict(x_word,
                                                 x_char,
                                                 y_tag,
                                                 learning_rate,
                                                 dropout_rate=dropout_rate,
                                                 training=True,
                                                 learning_rate_decay=learning_rate_decay)
                if self._logging:
                    summary, _ = self._sess.run([self.summary, self._train_op], feed_dict=feed_dict)
                    self.train_writer.add_summary(summary)

                self._sess.run(self._train_op, feed_dict=feed_dict)
                count += len(x_word)

            # DEBUG
            print('Learning rate: ', self._sess.run(self._learning_rate_decayed, feed_dict))

            self.eval_conll('valid', print_results=True, short_report=True)
            self.save()
        self.eval_conll(dataset_type='train', short_report=False)
        self.eval_conll(dataset_type='valid', short_report=False)
        self.eval_conll(dataset_type='test', short_report=False)

    def predict(self, x_word, x_char):
        feed_dict = self._fill_feed_dict(x_word, x_char, training=False)
        y_pred = self._sess.run(self._y_pred, feed_dict=feed_dict)
        return self.corpus.tag_dict.batch_idxs2batch_toks(y_pred, filter_paddings=True)

    def eval_conll(self, dataset_type='test', print_results=True, short_report=True):
        y_true_list = list()
        y_pred_list = list()
        print('Eval on {}:'.format(dataset_type))
        for (x_word, x_char), y_gt in self.corpus.batch_generator(batch_size=32, dataset_type=dataset_type):
            y_pred = self.predict(x_word, x_char)
            y_gt = self.corpus.tag_dict.batch_idxs2batch_toks(y_gt, filter_paddings=True)
            for tags_pred, tags_gt in zip(y_pred, y_gt):
                for tag_predicted, tag_ground_truth in zip(tags_pred, tags_gt):
                    y_true_list.append(tag_ground_truth)
                    y_pred_list.append(tag_predicted)
                y_true_list.append('O')
                y_pred_list.append('O')
        return precision_recall_f1(y_true_list, y_pred_list, print_results, short_report)

    def _fill_feed_dict(self,
                        x_w,
                        x_c,
                        y_t=None,
                        learning_rate=None,
                        training=False,
                        dropout_rate=1,
                        learning_rate_decay=1):
        feed_dict = dict()
        feed_dict[self._x_w] = x_w
        feed_dict[self._x_c] = x_c
        feed_dict[self._training_ph] = training
        if y_t is not None:
            feed_dict[self._y_true] = y_t
        if learning_rate is not None:
            feed_dict[self._learning_rate_ph] = learning_rate
            feed_dict[self._learning_rate_decay_ph] = learning_rate_decay
        if self._use_dropout is not None and training:
            feed_dict[self._dropout] = dropout_rate
        else:
            feed_dict[self._dropout] = 1.0
        return feed_dict

    def eval_loss(self, data_type='test', batch_size=32):
        num_tokens = 0
        loss = 0
        for (x_w, x_c), y_t in self.corpus.batch_generator(batch_size=batch_size, dataset_type=data_type):
            feed_dict = self._fill_feed_dict(x_w, x_c, y_t, training=False)
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
        n_training_samples = len(self.corpus.dataset['train'])
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
        (batch_tok, batch_char), _ = self.corpus.tokens_batch_to_numpy_batch(tokens_batch)
        # Prediction indices
        pred_idxs = self._sess.run(self._y_pred, feed_dict={self._x_w: batch_tok, self._x_c: batch_char})
        predictions_batch = self.corpus.tag_dict.batch_idxs2batch_toks(pred_idxs, filter_paddings=True)
        predictions_batch_no_pad = list()
        for n, predicted_tags in enumerate(predictions_batch):
            predictions_batch_no_pad.append(predicted_tags[: len(tokens_batch[n])])
        return predictions_batch_no_pad


if __name__ == '__main__':
    corp = Corpus(dicts_filepath='dict.txt')
    parameters = {'n_conv_layers': 2,
                  'n_filters': 100,
                  'filter_width': 5,
                  'token_embeddings_dim': 100,
                  'char_embeddings_dim': 25,
                  'use_batch_norm': False}

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

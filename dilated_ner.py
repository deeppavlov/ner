import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
from corpus import Corpus
from corpus import data_reader_gareev
import os
from corpus import DATA_PATH

MODEL_PATH = DATA_PATH
MODEL_FILE_NAME = 'ner_model.ckpt'


class DilatedNER:
    def __init__(self,
                 corpus,
                 token_emb_dim=100,
                 char_emb_dim=25,
                 char_n_filters=25,
                 char_filter_width=3,
                 n_layers_per_block=4,
                 n_blocks=1,
                 dilated_filter_width=3,
                 embeddings_dropout=False,
                 dense_dropout=False,
                 pretrained_model_filepath=None):
        tf.reset_default_graph()
        n_tags = len(corpus.tag_dict)
        n_tokens = len(corpus.token_dict)
        n_chars = len(corpus.char_dict)
        # Placeholders
        x_w = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_word')
        x_c = tf.placeholder(dtype=tf.int32, shape=[None, None, None], name='x_char')
        y_t = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_tag')
        lr = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')

        # Load embeddings
        with tf.variable_scope('Embeddings'):
            if corpus.embeddings is not None:
                w_embeddings = corpus.embeddings.astype(np.float32)
                w_embeddings = tf.Variable(w_embeddings, name='Token_Emb_Dict', trainable=False)

            else:
                w_embeddings = 0.1 * np.random.randn(n_tokens, token_emb_dim).astype(np.float32)
                w_embeddings = tf.Variable(w_embeddings, name='Token_Emb_Dict')
                self._w_emb = w_embeddings
            c_embeddings = 0.1 * np.random.randn(n_chars, char_emb_dim).astype(np.float32)

            c_embeddings = tf.Variable(c_embeddings, name='Char_Emb_Dict')
            # Word embedding layer
            w_emb = tf.nn.embedding_lookup(w_embeddings, x_w)

        dropout_ph = tf.placeholder_with_default(1.0, [], name='dropout_ph')

        if embeddings_dropout:
            w_emb = tf.nn.dropout(w_emb, dropout_ph)

        # Character embedding network
        with tf.variable_scope('Char_Emb_Network'):
            # Character embedding layer
            c_emb = tf.nn.embedding_lookup(c_embeddings, x_c)

            # Character embedding network
            char_conv = tf.layers.conv2d(c_emb, char_n_filters, (1, char_filter_width), padding='same', name='char_conv')
            char_emb = tf.reduce_max(char_conv, axis=2)

        wc_features = tf.concat([char_emb, w_emb], axis=-1)

        # Cutdown dimensionality of the network via projection
        units = tf.layers.dense(wc_features, 50, kernel_initializer=xavier_initializer())
        n_filters_dilated = char_n_filters + token_emb_dim

        for n_block in range(n_blocks):
            reuse_layer = n_block > 0

            for n_layer in range(n_layers_per_block):
                units = tf.layers.conv1d(units,
                                         n_filters_dilated,
                                         dilated_filter_width,
                                         padding='same',
                                         # dilation_rate=2 ** n_layer,
                                         name='Layer_' + str(n_layer),
                                         reuse=reuse_layer,
                                         activation=None,
                                         kernel_initializer=xavier_initializer())
                units = tf.nn.relu(units)
        if dense_dropout:
            units = tf.nn.dropout(units, dropout_ph)
        with tf.variable_scope('Dense_head'):
            units = tf.layers.dense(units,
                                    n_filters_dilated,
                                    kernel_initializer=xavier_initializer(),
                                    name='Hidden',
                                    activation=tf.nn.relu)
            if dense_dropout:
                tf.summary.histogram('Dense_no_drop', units)
                units = tf.nn.dropout(units, dropout_ph)
                tf.summary.histogram('Dense_drop', units)
            logits = tf.layers.dense(units, n_tags, kernel_initializer=xavier_initializer(), name='Output')
        # predictions = tf.argmax(logits, axis=-1)
        ground_truth_labels = tf.one_hot(y_t, n_tags)
        loss_tensor = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_labels, logits=logits)
        loss_tensor = loss_tensor * tf.cast(tf.not_equal(x_w, corpus.token_dict.tok2idx('<PAD>')), tf.float32)

        predictions = tf.argmax(logits, axis=-1)
        loss = tf.reduce_mean(loss_tensor)

        global_step = tf.Variable(0, trainable=False)
        lr_schedule = tf.train.exponential_decay(lr, global_step, decay_steps=1024, decay_rate=0.5, staircase=True)
        train_op = tf.train.AdamOptimizer(lr_schedule).minimize(loss, global_step=global_step)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        self.print_number_of_parameters()

        self.train_writer = tf.summary.FileWriter('summary', sess.graph)

        self.summary = tf.summary.merge_all()
        self._x_w = x_w
        self._x_c = x_c
        self._y_t = y_t
        self._y_pred = predictions
        self._loss = loss
        self._train_op = train_op
        self._sess = sess
        self.corpus = corpus
        self._lr = lr
        self._lr_schedule = lr_schedule
        self._loss_tensor = loss_tensor
        self._dropout = dropout_ph
        self._use_dropout = True if embeddings_dropout or dense_dropout else None
        if pretrained_model_filepath is not None:
            self.load(pretrained_model_filepath)

        #DEBUG
        self.global_step = global_step

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

    def get_l2_loss(self):
        vars = tf.trainable_variables()
        l2_loss = 0
        for var in vars:
            if 'kernel' in var.name:
                l2_loss += tf.nn.l2_loss(var)
        return l2_loss

    def train_on_batch(self, x_word, x_char, y_tag):
        feed_dict = {self._x_w: x_word, self._x_c: x_char, self._y_t: y_tag}
        self._sess.run(self._train_op, feed_dict=feed_dict)

    def print_number_of_parameters(self):
        print('Number of parameters: ')
        vars = tf.trainable_variables()
        blocks = ['Token_Emb_Dict', 'Char_Emb_Dict', 'Char_Emb_Network', 'Layer', 'Dense']
        for block_name in blocks:
            par_count = 0
            for var in vars:
                if block_name in var.name:
                    par_count += np.prod(var.get_shape().as_list())
            print(block_name + ':', par_count)

    def fit(self, batch_gen=None, batch_size=32, learning_rate=1e-3, epochs=1, dropout_rate=0.5):
        for epoch in range(epochs):

            count = 0
            self.eval_conll(dataset_type='train')
            self.eval_conll(dataset_type='valid')
            self.eval_conll(dataset_type='test')
            if batch_gen is None:
                batch_generator = self.corpus.batch_generator(batch_size, dataset_type='train')
            for (x_word, x_char), y_tag in batch_generator:
                feed_dict = self._fill_feed_dict(x_word, x_char, y_tag, learning_rate, dropout_rate=dropout_rate)
                summary, _ = self._sess.run([self.summary, self._train_op], feed_dict=feed_dict)
                self.train_writer.add_summary(summary)
                count += len(x_word)
            self.save()

    def eval_conll(self,
                   output_filepath='output.txt',
                   report_file_path='conll_evaluation.txt',
                   dataset_type='test'):
        with open(output_filepath, 'w') as f:
            for (x_word, x_char), y_gt in self.corpus.batch_generator(batch_size=32, dataset_type=dataset_type):
                feed_dict = self._fill_feed_dict(x_word, x_char, y_gt, eval=True)
                y_pred = self._sess.run(self._y_pred, feed_dict=feed_dict)

                x_word = self.corpus.token_dict.batch_idxs2batch_toks(x_word, filter_paddings=True)
                y_gt = self.corpus.tag_dict.batch_idxs2batch_toks(y_gt, filter_paddings=True)
                y_pred = self.corpus.tag_dict.batch_idxs2batch_toks(y_pred, filter_paddings=True)
                for utterance, tags_pred, tags_gt in zip(x_word, y_pred, y_gt):
                    for word, tag_predicted, tag_ground_truth in zip(utterance, tags_pred, tags_gt):
                        f.write(' '.join([word] + ['pur'] * 4 + [tag_ground_truth] + [tag_predicted]) + '\n')

        conll_evaluation_script = os.path.join('.', 'conlleval')
        shell_command = 'perl {0} < {1} > {2}'.format(conll_evaluation_script, output_filepath, report_file_path)
        os.system(shell_command)
        print(dataset_type.capitalize() + ' set')
        with open(report_file_path) as f:
            for line in f:
                print(line)

    def _fill_feed_dict(self, x_w, x_c, y_t, learning_rate=None, eval=False, dropout_rate=1):
        feed_dict = dict()
        feed_dict[self._x_w] = x_w
        feed_dict[self._x_c] = x_c
        feed_dict[self._y_t] = y_t
        if learning_rate is not None:
            feed_dict[self._lr] = learning_rate
        if self._use_dropout is not None and not eval:
            feed_dict[self._dropout] = dropout_rate
        else:
            feed_dict[self._dropout] = 1.0
        return feed_dict

    def eval_loss(self, data_type='test', batch_size=32):
        num_tokens = 0
        loss = 0
        for (x_w, x_c), y_t in self.corpus.batch_generator(batch_size=batch_size, dataset_type=data_type):
            feed_dict = self._fill_feed_dict(x_w, x_c, y_t, eval=True)
            loss += np.sum(self._sess.run(self._loss_tensor, feed_dict=feed_dict))
            num_tokens += np.sum(self.corpus.token_dict.is_pad(x_w))
        return loss / num_tokens


if __name__ == '__main__':
    n_layers_per_block = 3
    n_blocks = 1
    dilated_filter_width = 5
    embeddings_dropout = True
    dense_dropout = True
    corpus = Corpus(data_reader_gareev)

    # Creating a convolutional NER model
    ner = DilatedNER(corpus,
                     n_layers_per_block=n_layers_per_block,
                     n_blocks=n_blocks,
                     dilated_filter_width=dilated_filter_width,
                     embeddings_dropout=embeddings_dropout,
                     dense_dropout=dense_dropout)
    # Training the model
    ner.fit(epochs=10,
            batch_size=8,
            learning_rate=1e-3,
            dropout_rate=0.5)

    # Creating new model and restoring pre-trained weights
    model_path = os.path.join(MODEL_PATH, MODEL_FILE_NAME)
    ner_ = DilatedNER(corpus,
                      n_layers_per_block=n_layers_per_block,
                      n_blocks=n_blocks,
                      dilated_filter_width=dilated_filter_width,
                      embeddings_dropout=embeddings_dropout,
                      dense_dropout=dense_dropout,
                      pretrained_model_filepath=model_path)
    # Evaluate loaded model
    ner_.eval_conll()
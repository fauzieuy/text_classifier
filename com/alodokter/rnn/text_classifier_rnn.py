import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

class TextClassifierRNN:
    def length(self, sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            embedded_words = tf.contrib.layers.embed_sequence(self.input_x, vocab_size=vocab_size, embed_dim=embedding_size)
            inputs_series = tf.unstack(embedded_words, axis=1)

        # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
        cell = tf.contrib.rnn.GRUCell(num_units=embedding_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)

        # initial_state = cell.zero_state(self.batch_size, tf.float32)
        # Create an unrolled Recurrent Neural Networks to length of sequence_length
        # and passes inputs_series as inputs for each unit.
        # _, encoding = tf.contrib.rnn.static_rnn(
        #                         cell,
        #                         inputs_series,
        #                         sequence_length=self.length(inputs_series),
        #                         dtype=tf.float32)
        _, encoding = tf.contrib.rnn.static_rnn(
                            cell,
                            inputs_series,
                            dtype=tf.float32)

        with tf.name_scope("output"):
            # self.scores = tf.layers.dense(encoding, num_classes, activation=None, name="scores")
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda)
            self.scores = tf.contrib.layers.fully_connected(
                                    encoding,
                                    num_classes,
                                    weights_initializer = tf.contrib.layers.xavier_initializer(),
                                    weights_regularizer = regularizer,
                                    scope="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            loss = tf.losses.softmax_cross_entropy(logits=self.scores, onehot_labels=self.input_y)
            self.loss = loss + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            # self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
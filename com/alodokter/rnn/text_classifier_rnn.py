import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

class TextClassifierRNN(object):
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

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            # [batch_size, sequence_length, embedding_size]
            embedded_words = tf.contrib.layers.embed_sequence(self.input_x, vocab_size=vocab_size, embed_dim=embedding_size)
            inputs_series = tf.unstack(embedded_words, axis=1)

        # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
        cell = tf.contrib.rnn.GRUCell(num_units=embedding_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)

        _, state = tf.contrib.rnn.static_rnn(cell, inputs_series, dtype=tf.float32)
        with tf.name_scope("output"):
            # self.scores = tf.layers.dense(encoding, num_classes, activation=None, name="scores")
            regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda)
            self.scores = tf.contrib.layers.fully_connected(
                                    state,
                                    num_classes,
                                    weights_initializer = tf.contrib.layers.xavier_initializer(),
                                    weights_regularizer = regularizer,
                                    biases_regularizer = regularizer,
                                    scope="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            loss = tf.losses.softmax_cross_entropy(logits=self.scores, onehot_labels=self.input_y)
            self.loss = loss + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # # Create an unrolled Recurrent Neural Networks to length of sequence_length
        # # and passes inputs_series as inputs for each unit.
        # _, state = tf.contrib.rnn.static_rnn(
        #                         cell,
        #                         inputs_series,
        #                         sequence_length=self.length(inputs_series),
        #                         dtype=tf.float32)
        #
        # # Keeping track of l2 regularization loss (optional)
        # l2_loss = tf.constant(0.0)
        #
        # # Final (unnormalized) scores and predictions
        # with tf.name_scope("output"):
        #     W = tf.get_variable(
        #             "W",
        #             shape=[embedding_size, num_classes],
        #             initializer=tf.contrib.layers.xavier_initializer())
        #     b = tf.get_variable(
        #             "b",
        #             initializer=tf.constant(0.1, shape=[num_classes]))
        #     l2_loss += tf.nn.l2_loss(W)
        #     l2_loss += tf.nn.l2_loss(b)
        #     self.scores = tf.nn.xw_plus_b(state, W, b, name="scores")
        #     self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
        # num_layers = 3
        # cell = tf.contrib.rnn.GRUCell(num_units=embedding_size)
        # cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
        # cells = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
        #
        # # Create an unrolled Recurrent Neural Networks to length of sequence_length
        # # and passes inputs_series as inputs for each unit.
        # outputs, state = tf.contrib.rnn.static_rnn(
        #                         cells,
        #                         inputs_series,
        #                         sequence_length=self.length(inputs_series),
        #                         dtype=tf.float32)
        #
        # # Final (unnormalized) scores and predictions
        # with tf.name_scope("output"):
        #     W = tf.get_variable(
        #             "W",
        #             shape=[embedding_size, num_classes],
        #             initializer=tf.contrib.layers.xavier_initializer())
        #     b = tf.get_variable(
        #             "b",
        #             initializer=tf.constant(0.1, shape=[num_classes]))
        #     l2_loss += tf.nn.l2_loss(W)
        #     l2_loss += tf.nn.l2_loss(b)
        #     self.scores = tf.nn.xw_plus_b(outputs[-1], W, b, name="scores")
        #     self.predictions = tf.argmax(self.scores, 1, name="predictions")
        #
        # # CalculateMean cross-entropy loss
        # with tf.name_scope("loss"):
        #     losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
        #     self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

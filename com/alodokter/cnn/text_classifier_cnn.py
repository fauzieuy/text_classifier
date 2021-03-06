import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

class TextClassifierCNN:
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/gpu:0'), tf.name_scope("embedding"):
            # [batch_size, sequence_length, embedding_size]
            # self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            # self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            embedded_words = tf.contrib.layers.embed_sequence(self.input_x, vocab_size=vocab_size, embed_dim=embedding_size)
            inputs_series = tf.unstack(embedded_words, axis=1)
            self.embedded_words_expanded = tf.expand_dims(embedded_words, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_words_expanded,
                    W, strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                # ksize ==> Nout = (Nin + 2 * Npadding - Nfilter) + 1
                # see http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
                # Narrow vs. Wide convolution
                pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                        "b",
                        initializer=tf.constant(0.1, shape=[num_classes]))
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # with tf.variable_scope("output"):
        #     regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda)
        #     self.scores = tf.layers.dense(
        #                             self.h_drop,
        #                             num_classes,
        #                             activation = tf.nn.relu,
        #                             kernel_initializer = tf.contrib.layers.xavier_initializer(),
        #                             bias_initializer = tf.zeros_initializer(),
        #                             kernel_regularizer = regularizer,
        #                             bias_regularizer = regularizer,
        #                             name="scores")
        #     self.predictions = tf.argmax(self.scores, 1, name="predictions")
        #
        #     # regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda)
        #     # regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda)
        #     # self.scores = tf.contrib.layers.fully_connected(
        #     #                         self.h_drop,
        #     #                         num_classes,
        #     #                         weights_initializer = tf.contrib.layers.xavier_initializer(),
        #     #                         weights_regularizer = regularizer,
        #     #                         biases_regularizer = regularizer,
        #     #                         scope="scores")
        #     # self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            loss = tf.losses.softmax_cross_entropy(logits=self.scores, onehot_labels=self.input_y)
            self.loss = loss + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

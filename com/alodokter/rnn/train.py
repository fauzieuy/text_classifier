import tensorflow as tf
import numpy as np
import os
import time
import datetime
from com.alodokter.rnn import data_helpers
from com.alodokter.rnn.text_classifier_rnn import TextClassifierRNN
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split

# Parameters
# =========================================================================================================
# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("corpus_path", "corpus/interest/", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 32, "Dimensionality of character embedding (default: 32)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning Rate (default: 0.001)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================
# Load data
# x_text is list of question
# ['question-1', 'question-2',...,'question-n']
#
# y is one-hot-encoding class
# array([[ 0.,  0.,  1., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        ...,
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.],
#        [ 0.,  0.,  0., ...,  0.,  0.,  0.]])
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.corpus_path)

# Build vocabulary
# x is
# array([[   1,    2,    1, ...,    0,    0,    0],
#        [   5,    6,    7, ...,    0,    0,    0],
#        [   1,  125,    2, ...,    0,    0,    0],
#        ...,
#        [ 252,  143,  250, ...,    0,    0,    0],
#        [ 200,  201,    2, ...,    0,    0,    0],
#        [6398,    2,  144, ...,    0,    0,    0]])
# max_document_length = max([len(x.split(' ')) for x in x_text])
max_document_length = np.ceil(np.mean([len(x.split(' ')) for x in x_text])).astype(int)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
vocab_size = len(vocab_processor.vocabulary_)

# # Randomly shuffle data
# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_shuffled = x[shuffle_indices]
# y_shuffled = y[shuffle_indices]
#
# # Split train/test set
# # TODO: This is very crude, should use cross-validation
# dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
# x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
# y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
# print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
# print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# cross-validation
x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=FLAGS.dev_sample_percentage, random_state=5)

# Training
# ==================================================
with tf.Graph().as_default():
    """
    The allow_soft_placement setting allows TensorFlow to fall back on a device with a certain operation implemented when the preferred device doesn’t exist. For example, if our code places an operation on a GPU and we run the code on a machine without GPU, not using allow_soft_placement would result in an error. If log_device_placement is set, TensorFlow log on which devices (CPU or GPU) it places operations. That’s useful for debugging. FLAGS are command-line arguments to our program.
    http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
    """
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        rnn = TextClassifierRNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=vocab_size,
                    embedding_size=FLAGS.embedding_dim,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # # https://www.tensorflow.org/versions/r0.12/api_docs/python/train/decaying_the_learning_rate
        # starter_learning_rate = FLAGS.learning_rate
        # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(rnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", rnn.loss)
        acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                rnn.input_x: x_batch,
                rnn.input_y: y_batch,
                rnn.batch_size: len(x_batch),
                rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                                                    [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy],
                                                    feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                rnn.input_x: x_batch,
                rnn.input_y: y_batch,
                rnn.batch_size: len(x_batch),
                rnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                                                [global_step, dev_summary_op, rnn.loss, rnn.accuracy],
                                                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches and shuffle the data on every epoch
        batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

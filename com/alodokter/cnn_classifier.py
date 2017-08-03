import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import learn
from com.alodokter import data_helpers

# ==================================================
tf.flags.DEFINE_string("checkpoint_dir", "runs/1501761743/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

class TextClassifier:
    def __init__(self):
        data_helpers.setup_one_hot_encoder_class('corpus/interest/')

        # Map data into vocabulary
        vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

        # Prediction
        # ==================================================
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                                allow_soft_placement=FLAGS.allow_soft_placement,
                                log_device_placement=FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)

                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("input_x").outputs[0]
                self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]

    def predict(self, text):
        x = np.array(list(self.vocab_processor.transform([text])))
        output = self.sess.run(self.predictions, {self.input_x: x, self.dropout_keep_prob: 1.0})
        return data_helpers.get_class_name(output[0])

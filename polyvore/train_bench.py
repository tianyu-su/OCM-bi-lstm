# Copyright 2017 Xintong Han. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import configuration
import polyvore_model_bi as polyvore_model

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("inception_checkpoint_file", "",
                       "Path to a pretrained inception_v3 model.")
tf.flags.DEFINE_string("train_dir", "",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_boolean("train_inception", False,
                        "Whether to train inception submodel variables.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")

# update 2021.4.2
tf.flags.DEFINE_float("emb_loss_factor", 1.0, "weight of vse")
tf.flags.DEFINE_integer("batch_size", 10, "batch size of training steps.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    assert FLAGS.input_file_pattern, "--input_file_pattern is required"
    assert FLAGS.train_dir, "--train_dir is required"

    model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern
    model_config.inception_checkpoint_file = FLAGS.inception_checkpoint_file

    # update 2021.4.2
    model_config.emb_loss_factor = FLAGS.emb_loss_factor
    model_config.batch_size = FLAGS.batch_size

    training_config = configuration.TrainingConfig()
    now_trining_split = "default"
    input_split_which = FLAGS.input_file_pattern.split("/")[2]
    if "nondisjoint" == input_split_which:
        now_trining_split = "nondisjoint"
        training_config.num_examples_per_epoch = 51488
        FLAGS.number_of_steps = FLAGS.number_of_steps * 3

    if "disjoint" == input_split_which:
        now_trining_split = "disjoint"
        training_config.num_examples_per_epoch = 16644

    tf.logging.info("==> train dataset =====>  %s", now_trining_split)
    tf.logging.info("==> train dataset set: %d", training_config.num_examples_per_epoch)
    tf.logging.info("==> train batch size: %d", model_config.batch_size)
    tf.logging.info("==> train total step : %d", FLAGS.number_of_steps)
    tf.logging.info("==> coef vse: %f", model_config.emb_loss_factor)

    # Create training directory.
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = polyvore_model.PolyvoreModel(
            model_config, mode="train", train_inception=FLAGS.train_inception)
        model.build()
        learning_rate = tf.constant(training_config.initial_learning_rate)

        learning_rate_decay_fn = None
        if training_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                     model_config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              training_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=training_config.learning_rate_decay_factor,
                    staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn

        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=model.total_loss,
            global_step=model.global_step,
            learning_rate=learning_rate,
            optimizer=training_config.optimizer,
            clip_gradients=training_config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)
        # update 2021.4.3
        # g.finalize()

    # Run training.
    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        init_fn=model.init_fn,
        saver=saver,
        session_config=sess_config,
    )


if __name__ == "__main__":
    tf.app.run()

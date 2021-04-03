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

"""Fill in blank evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import pdb

import tensorflow as tf
import numpy as np
import pickle as pkl

import configuration
import polyvore_model_bi as polyvore_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("json_file", "",
                       "Json file containing questions and answers.")
tf.flags.DEFINE_string("feature_file", "", "pkl files containing the features")
tf.flags.DEFINE_string("rnn_type", "lstm", "Type of RNN.")
tf.flags.DEFINE_string("result_file", "", "File to store the results.")
tf.flags.DEFINE_integer("direction", 2, "2: bidirectional; 1: forward only;"
                                        "-1: backward only; 0: Average pooling no RNN.")


def run_question_inference(sess, question, test_ids, test_feat,
                           test_rnn_feat, num_lstm_units):
    question_ids = []
    answer_ids = []
    for q in question["question"]:
        try:
            question_ids.append(test_ids.index(q))
        except:
            return [], []

    for a in question["answers"]:
        try:
            answer_ids.append(test_ids.index(a))
        except:
            return [], []

    blank_posi = question["blank_position"]

    # Average pooling of the VSE embeddings
    question_emb = np.reshape(np.mean(test_feat[question_ids], 0), [1, -1])
    q_emb = question_emb / np.linalg.norm(question_emb, axis=1)[:, np.newaxis]
    a_emb = (test_feat[answer_ids] /
             np.linalg.norm(test_feat[answer_ids], axis=1)[:, np.newaxis])
    vse_score = (np.dot(q_emb, np.transpose(a_emb)) + 1) / 2  # scale to [0,1]
    vse_score = vse_score  # / np.sum(vse_score) # normalize to sum to 1.

    if FLAGS.direction == 0:
        # Only use VSE
        predicted_answer = np.argsort(-vse_score)[0]
        return vse_score, predicted_answer

    if FLAGS.rnn_type == "lstm":
        # LSTM has two states.
        zero_state = np.zeros([1, 2 * num_lstm_units])
    else:
        zero_state = np.zeros([1, num_lstm_units])

    # Blank is the last item.
    if blank_posi == len(question_ids) + 1:
        if FLAGS.direction == -1:
            return [], []
        # Only do forward rnn
        input_feed = np.reshape(test_rnn_feat[question_ids[0]], [1, -1])
        # Run first step with all zeros initial state.
        [lstm_state, lstm_output] = sess.run(
            fetches=["lstm/f_state:0", "f_logits/f_logits/BiasAdd:0"],
            feed_dict={"lstm/f_input_feed:0": input_feed,
                       "lstm/f_state_feed:0": zero_state})

        for step in range(len(question_ids) - 1):
            input_feed = np.reshape(test_rnn_feat[question_ids[step + 1]], [1, -1])
            [lstm_state, lstm_output] = sess.run(
                fetches=["lstm/f_state:0", "f_logits/f_logits/BiasAdd:0"],
                feed_dict={"lstm/f_input_feed:0": input_feed,
                           "lstm/f_state_feed:0": lstm_state})

        # Search in answers
        rnn_score = np.exp(np.dot(lstm_output,
                                  np.transpose(test_rnn_feat[answer_ids])))
        rnn_score = rnn_score / np.sum(rnn_score)

    # Blank is the frist item
    elif blank_posi == 1:
        if FLAGS.direction == 1:
            return [], []
        # only do backward rnn
        input_feed = np.reshape(test_rnn_feat[question_ids[-1]], [1, -1])
        # Run first step with all zeros initial state.
        [lstm_state, lstm_output] = sess.run(
            fetches=["lstm/b_state:0", "b_logits/b_logits/BiasAdd:0"],
            feed_dict={"lstm/b_input_feed:0": input_feed,
                       "lstm/b_state_feed:0": zero_state})

        for step in range(len(question_ids) - 1):
            input_feed = np.reshape(test_rnn_feat[question_ids[-step - 2]], [1, -1])
            [lstm_state, lstm_output] = sess.run(
                fetches=["lstm/b_state:0", "b_logits/b_logits/BiasAdd:0"],
                feed_dict={"lstm/b_input_feed:0": input_feed,
                           "lstm/b_state_feed:0": lstm_state})
        rnn_score = np.exp(np.dot(lstm_output,
                                  np.transpose(test_rnn_feat[answer_ids])))
        rnn_score = rnn_score / np.sum(rnn_score)

    # Blank is in the middle.
    else:
        # Do bidirectional rnn.
        # Forward:
        input_feed = np.reshape(test_rnn_feat[question_ids[0]], [1, -1])
        # Run first step with all zeros initial state.
        [lstm_state, lstm_output] = sess.run(
            fetches=["lstm/f_state:0", "f_logits/f_logits/BiasAdd:0"],
            feed_dict={"lstm/f_input_feed:0": input_feed,
                       "lstm/f_state_feed:0": zero_state})

        for step in range(blank_posi - 2):
            input_feed = np.reshape(test_rnn_feat[question_ids[step + 1]], [1, -1])
            [lstm_state, lstm_output] = sess.run(
                fetches=["lstm/f_state:0", "f_logits/f_logits/BiasAdd:0"],
                feed_dict={"lstm/f_input_feed:0": input_feed,
                           "lstm/f_state_feed:0": lstm_state})

        # Search in answers.
        f_softmax = np.exp(np.dot(lstm_output,
                                  np.transpose(test_rnn_feat[answer_ids])))
        # Backward:
        input_feed = np.reshape(test_rnn_feat[question_ids[-1]], [1, -1])
        # Run first step with all zeros initial state.
        [lstm_state, lstm_output] = sess.run(
            fetches=["lstm/b_state:0", "b_logits/b_logits/BiasAdd:0"],
            feed_dict={"lstm/b_input_feed:0": input_feed,
                       "lstm/b_state_feed:0": zero_state})

        for step in range(len(question_ids) - blank_posi):
            input_feed = np.reshape(test_rnn_feat[question_ids[-step - 2]], [1, -1])
            [lstm_state, lstm_output] = sess.run(
                fetches=["lstm/b_state:0", "b_logits/b_logits/BiasAdd:0"],
                feed_dict={"lstm/b_input_feed:0": input_feed,
                           "lstm/b_state_feed:0": lstm_state})

        b_softmax = np.exp(np.dot(lstm_output,
                                  np.transpose(test_rnn_feat[answer_ids])))
        if FLAGS.direction == 2:
            rnn_score = (f_softmax / np.sum(f_softmax) +
                         b_softmax / np.sum(b_softmax))
            rnn_score /= 2
        elif FLAGS.direction == 1:
            rnn_score = f_softmax / np.sum(f_softmax)
        else:
            rnn_score = b_softmax / np.sum(b_softmax)

    predicted_answer = np.argsort(-rnn_score)[0]
    return rnn_score, predicted_answer

def MRR_HR(array, HR_ks=(1, 3, 5)):
    """
    higher score, high ranking position
    :param array:  [bs,list], where index=0 is true
    """
    r_sort_idx = np.argsort(-array, axis=1)
    right_idx = np.where(r_sort_idx == 0)[1]
    HR_Ks = [right_idx < kk for kk in HR_ks]
    MRR = np.average(1. / (right_idx + 1))
    HR = np.average(HR_Ks, axis=1)
    return MRR, HR

def main(_):
    HR_ks = (1,5,10,20,30,40,50,60,70,80,90,100,150,200)

    # Build the inference graph.
    # top_k = 4  # Print the top_k accuracy.
    # true_pred = np.zeros(top_k)
    # Load pre-computed image features.
    with open(FLAGS.feature_file, "rb") as f:
        test_data = pkl.load(f)
    test_ids = test_data.keys()
    test_feat = np.zeros((len(test_ids),
                          len(test_data[test_ids[0]]["image_feat"])))
    test_rnn_feat = np.zeros((len(test_ids),
                              len(test_data[test_ids[0]]["image_rnn_feat"])))
    for i, test_id in enumerate(test_ids):
        # Image feature in visual-semantic embedding space.
        test_feat[i] = test_data[test_id]["image_feat"]
        # Image feature in the RNN space.
        test_rnn_feat[i] = test_data[test_id]["image_rnn_feat"]

    g = tf.Graph()
    with g.as_default():
        model_config = configuration.ModelConfig()
        model_config.rnn_type = FLAGS.rnn_type
        model = polyvore_model.PolyvoreModel(model_config, mode="inference")
        model.build()
        saver = tf.train.Saver()

        g.finalize()
        with tf.Session() as sess:
            saver.restore(sess, FLAGS.checkpoint_path)
            questions = json.load(open(FLAGS.json_file))

            all_pred = []
            set_ids = []
            all_scores = []
            for question in questions:
                score, pred = run_question_inference(sess, question, test_ids,
                                                     test_feat, test_rnn_feat,
                                                     model_config.num_lstm_units)
                if pred != []:
                    all_pred.append(pred)
                    all_scores.append(score)
                    set_ids.append(question["question"][0].split("_")[0])
                    # 0 is the correct answer, iterate over top_k.
                    # for i in range(top_k):
                    #     if 0 in pred[:i + 1]:
                    #         true_pred[i] += 1

            res_metrics = MRR_HR(np.concatenate(all_scores, axis=0), HR_ks=HR_ks)

            print("MRR:{:.3f} \t".format(res_metrics[0]))
            for idx, kk in enumerate(HR_ks):
                print("HR@{}:{:.3f} \t".format(kk, res_metrics[1][idx]))

            # # Print all top-k accuracy.
            # for i in range(top_k):
            #     print("Top %d Accuracy: " % (i + 1))
            #     print("%d correct answers in %d valid questions." %
            #           (true_pred[i], len(all_pred)))
            #     print("Accuracy: %f" % (true_pred[i] / len(all_pred)))
            #
            # s = np.empty((len(all_scores),), dtype=np.object)
            # for i in range(len(all_scores)):
            #     s[i] = all_scores[i]

            with open(FLAGS.result_file, "wb") as f:
                # pkl.dump({"set_ids": set_ids, "pred": all_pred, "score": s}, f)
                pkl.dump({"res_metrics": res_metrics}, f)


if __name__ == "__main__":
    tf.app.run()

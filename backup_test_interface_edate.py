# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run BERT on SQuAD 1.1 and SQuAD 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf
from tqdm import tqdm, trange
import codecs
import sys
sys.path.append("../TEX/")
from runCMedTEX import recognize

# os.environ["CUDA_VISIBLE_DEVICES"]= '0'

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", './bert_model/zh_base/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", './bert_model/zh_base/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", './tmp/finance_base_edate/',
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
    "predict_file", './data/finance/dev.json',
    "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
    "init_checkpoint", './bert_model/zh_base/bert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")#384

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 200,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")#64

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 5, "Total batch size for training.") #32

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 1,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
    "version_2_with_negative", False,
    "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
    "null_score_diff_threshold", 0.0,
    "If null_score - best_non_null is greater than the threshold predict null.")


class SquadExample(object):
  """A single training/test example for simple sequence classification.

     For examples without an answer, the start and end position are -1.
  """

  def __init__(self,
               qas_id,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_position=None,
               end_position=None,
               is_impossible=False):
    self.qas_id = qas_id
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "qas_id: %s" % (tokenization.printable_text(self.qas_id))
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_position: %d" % (self.start_position)
    if self.start_position:
      s += ", end_position: %d" % (self.end_position)
    if self.start_position:
      s += ", is_impossible: %r" % (self.is_impossible)
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               start_position=None,
               end_position=None,
               is_impossible=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_position = start_position
    self.end_position = end_position
    self.is_impossible = is_impossible


def read_insts_examples(insts_data):

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
          return True
        return False

    examples = []
    for paragraph_text, event, qas_id, _ in tqdm(insts_data):
        # paragraph_text = paragraph["context"]
        #doc_tokens = [item for item in paragraph_text]
        doc_tokens = []
        char_to_word_offset = []

        for c in paragraph_text:
          if not is_whitespace(c):
              doc_tokens.append(c)
          char_to_word_offset.append(len(doc_tokens) - 1)

        # for qa in paragraph["qas"]:
        #   qas_id = qa["id"]
        #   question_text = qa["question"]
        question_text = u"事件\""+event+u"\"发生的日期是？"
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False

        example = SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible)
        examples.append(example)

    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 output_fn):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000

  for (example_index, example) in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)
    #print(query_tokens)
    #exit()

    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
      orig_to_tok_index.append(len(all_doc_tokens))
      sub_tokens = tokenizer.tokenize(token)
      for sub_token in sub_tokens:
        tok_to_orig_index.append(i)
        all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
      tok_start_position = -1
      tok_end_position = -1
    if is_training and not example.is_impossible:
      tok_start_position = orig_to_tok_index[example.start_position]
      if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
      else:
        tok_end_position = len(all_doc_tokens) - 1
      (tok_start_position, tok_end_position) = _improve_answer_span(
          all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
          example.orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
      length = len(all_doc_tokens) - start_offset
      if length > max_tokens_for_doc:
        length = max_tokens_for_doc
      doc_spans.append(_DocSpan(start=start_offset, length=length))
      if start_offset + length == len(all_doc_tokens):
        break
      start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
      tokens = []
      token_to_orig_map = {}
      token_is_max_context = {}
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)

      for i in range(doc_span.length):
        split_token_index = doc_span.start + i
        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                               split_token_index)
        token_is_max_context[len(tokens)] = is_max_context
        tokens.append(all_doc_tokens[split_token_index])
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      start_position = None
      end_position = None
      if is_training and not example.is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
          out_of_span = True
        if out_of_span:
          start_position = 0
          end_position = 0
        else:
          doc_offset = len(query_tokens) + 2
          start_position = tok_start_position - doc_start + doc_offset
          end_position = tok_end_position - doc_start + doc_offset

      if is_training and example.is_impossible:
        start_position = 0
        end_position = 0

      '''
      if example_index < 20:
        tf.logging.info("*** Example ***")
        tf.logging.info("unique_id: %s" % (unique_id))
        tf.logging.info("example_index: %s" % (example_index))
        tf.logging.info("doc_span_index: %s" % (doc_span_index))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("token_to_orig_map: %s" % " ".join(
            ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
        tf.logging.info("token_is_max_context: %s" % " ".join([
            "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
        ]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info(
            "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        if is_training and example.is_impossible:
          tf.logging.info("impossible example")
        if is_training and not example.is_impossible:
          answer_text = " ".join(tokens[start_position:(end_position + 1)])
          tf.logging.info("start_position: %d" % (start_position))
          tf.logging.info("end_position: %d" % (end_position))
          tf.logging.info(
              "answer: %s" % (tokenization.printable_text(answer_text)))
      '''
      #exit()

      feature = InputFeatures(
          unique_id=unique_id,
          example_index=example_index,
          doc_span_index=doc_span_index,
          tokens=tokens,
          token_to_orig_map=token_to_orig_map,
          token_is_max_context=token_is_max_context,
          input_ids=input_ids,
          input_mask=input_mask,
          segment_ids=segment_ids,
          start_position=start_position,
          end_position=end_position,
          is_impossible=example.is_impossible)

      # Run callback
      output_fn(feature)

      unique_id += 1


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
  """Returns tokenized answer spans that better match the annotated answer."""

  # The SQuAD annotations are character based. We first project them to
  # whitespace-tokenized words. But then after WordPiece tokenization, we can
  # often find a "better match". For example:
  #
  #   Question: What year was John Smith born?
  #   Context: The leader was John Smith (1895-1943).
  #   Answer: 1895
  #
  # The original whitespace-tokenized answer will be "(1895-1943).". However
  # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
  # the exact answer, 1895.
  #
  # However, this is not always possible. Consider the following:
  #
  #   Question: What country is the top exporter of electornics?
  #   Context: The Japanese electronics industry is the lagest in the world.
  #   Answer: Japan
  #
  # In this case, the annotator chose "Japan" as a character sub-span of
  # the word "Japanese". Since our WordPiece tokenizer does not split
  # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
  # in SQuAD, but does happen.
  # tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
  tok_answer_text = "".join(tokenizer.tokenize(orig_answer_text))

  for new_start in range(input_start, input_end + 1):
    for new_end in range(input_end, new_start - 1, -1):
      # text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
      text_span = "".join(doc_tokens[new_start:(new_end + 1)])
      if text_span == tok_answer_text:
        return (new_start, new_end)

  return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""

  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/squad/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/squad/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  return (start_logits, end_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    '''
    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
    '''

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    '''
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
    '''

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]

      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)

      total_loss = (start_loss + end_loss) / 2.0

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
      }
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file):
  """Write final predictions to the json file and log-odds of null if needed."""
  # tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
  # tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

  example_index_to_features = collections.defaultdict(list)
  for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

  unique_id_to_result = {}
  for result in all_results:
    unique_id_to_result[result.unique_id] = result

  _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
      "PrelimPrediction",
      ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

  all_predictions = collections.OrderedDict()
  # all_predictions = []
  all_nbest_json = collections.OrderedDict()
  scores_diff_json = collections.OrderedDict()
  #print('all_examples length: ', len(all_examples))
  #num = 0
  for (example_index, example) in enumerate(all_examples):
    #if int(example.qas_id) == 9:
        #print('example.qas_id', example.qas_id)
        #print(example)
    #print(example.orig_answer_text)
    #exit()
    features = example_index_to_features[example_index]

    prelim_predictions = []
    # keep track of the minimum score of null start+end of position 0
    score_null = 1000000  # large and positive
    min_null_feature_index = 0  # the paragraph slice with min mull score
    null_start_logit = 0  # the start logit at the slice with min null score
    null_end_logit = 0  # the end logit at the slice with min null score
    for (feature_index, feature) in enumerate(features):
      result = unique_id_to_result[feature.unique_id]
      start_indexes = _get_best_indexes(result.start_logits, n_best_size)
      end_indexes = _get_best_indexes(result.end_logits, n_best_size)
      # if we could have irrelevant answers, get the min score of irrelevant
      if FLAGS.version_2_with_negative:
        feature_null_score = result.start_logits[0] + result.end_logits[0]
        if feature_null_score < score_null:
          score_null = feature_null_score
          min_null_feature_index = feature_index
          null_start_logit = result.start_logits[0]
          null_end_logit = result.end_logits[0]
      for start_index in start_indexes:
        for end_index in end_indexes:
          # We could hypothetically create invalid predictions, e.g., predict
          # that the start of the span is in the question. We throw out all
          # invalid predictions.
          if start_index >= len(feature.tokens):
            continue
          if end_index >= len(feature.tokens):
            continue
          if start_index not in feature.token_to_orig_map:
            continue
          if end_index not in feature.token_to_orig_map:
            continue
          if not feature.token_is_max_context.get(start_index, False):
            continue
          if end_index < start_index:
            continue
          length = end_index - start_index + 1
          if length > max_answer_length:
            continue
          prelim_predictions.append(
              _PrelimPrediction(
                  feature_index=feature_index,
                  start_index=start_index,
                  end_index=end_index,
                  start_logit=result.start_logits[start_index],
                  end_logit=result.end_logits[end_index]))
    #print('prelim_predictions length: ', len(prelim_predictions))

    if FLAGS.version_2_with_negative:
      prelim_predictions.append(
          _PrelimPrediction(
              feature_index=min_null_feature_index,
              start_index=0,
              end_index=0,
              start_logit=null_start_logit,
              end_logit=null_end_logit))
    prelim_predictions = sorted(
        prelim_predictions,
        key=lambda x: (x.start_logit + x.end_logit),
        reverse=True)

    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index"])

    seen_predictions = {}
    nbest = []
    for pred in prelim_predictions:
      if len(nbest) >= n_best_size:
        break
      feature = features[pred.feature_index]
      if pred.start_index > 0:  # this is a non-null prediction
        tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
        orig_doc_start = feature.token_to_orig_map[pred.start_index]
        orig_doc_end = feature.token_to_orig_map[pred.end_index]
        orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
        tok_text = " ".join(tok_tokens)

        # De-tokenize WordPieces that have been split off.
        tok_text = tok_text.replace(" ##", "")
        tok_text = tok_text.replace("##", "")

        # Clean whitespace
        tok_text = tok_text.strip()
        tok_text = " ".join(tok_text.split())
        orig_text = " ".join(orig_tokens)

        final_text = get_final_text(tok_text, orig_text, do_lower_case)
        if final_text in seen_predictions:
          continue

        seen_predictions[final_text] = True
      else:
        final_text = ""
        seen_predictions[final_text] = True
        orig_doc_start = 0
        orig_doc_end = 0

      nbest.append(
          _NbestPrediction(
              text=final_text,
              start_logit=pred.start_logit,
              end_logit=pred.end_logit, start_index=orig_doc_start, end_index=orig_doc_end))

    # if we didn't inlude the empty option in the n-best, inlcude it
    if FLAGS.version_2_with_negative:
      if "" not in seen_predictions:
        nbest.append(
            _NbestPrediction(
                text="", start_logit=null_start_logit,
                end_logit=null_end_logit, start_index=0, end_index=0))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
      nbest.append(
          _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=0, end_index=0))
    #num+=1
    #print('n-best!!', num)
    assert len(nbest) >= 1

    total_scores = []
    best_non_null_entry = None
    for entry in nbest:
      total_scores.append(entry.start_logit + entry.end_logit)
      if not best_non_null_entry:
        if entry.text:
          best_non_null_entry = entry

    probs = _compute_softmax(total_scores)

    nbest_json = []
    for (i, entry) in enumerate(nbest):
      output = collections.OrderedDict()
      output["text"] = entry.text
      output["probability"] = probs[i]
      output["start_logit"] = entry.start_logit
      output["end_logit"] = entry.end_logit
      output["start_index"] = entry.start_index
      output["end_index"] = entry.end_index
      nbest_json.append(output)

    assert len(nbest_json) >= 1
    #num+=1
    #print('nbest_json!!', num)
    
    if not FLAGS.version_2_with_negative:
      all_predictions[example.qas_id] = ''.join(nbest_json[0]["text"].split())
      # all_predictions[example.qas_id] = [''.join(nbest_json[0]["text"].split()), example.question_text, ''.join(example.doc_tokens), nbest_json[0]["start_index"], nbest_json[0]["end_index"]]
      '''
      edate=[{
            "anchorOffset": nbest_json[0]["start_index"],
            "extentOffset": nbest_json[0]["end_index"],
            "sourse": "text",
            "type": "开始时间",
            "value": ''.join(nbest_json[0]["text"].split())}]
      all_predictions.append({"context": ''.join(example.doc_tokens), "event": example.question_text[3:-8], "event_id": example.qas_id, "edate": edate})
      '''
    else:
      # predict "" iff the null score - the score of best non-null > threshold
      score_diff = score_null - best_non_null_entry.start_logit - (
          best_non_null_entry.end_logit)
      scores_diff_json[example.qas_id] = score_diff
      if score_diff > FLAGS.null_score_diff_threshold:
        # all_predictions[example.qas_id] = ""
        all_predictions.append("")
      else:
        # all_predictions[example.qas_id] = best_non_null_entry.text
        all_predictions.append(best_non_null_entry.text)

    all_nbest_json[example.qas_id] = nbest_json
  print('all_predictions length: ', len(all_predictions))
  with tf.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")

  #with tf.gfile.GFile(output_nbest_file, "w") as writer:
  #  writer.write(json.dumps(all_nbest_json, ensure_ascii=False, indent=4) + "\n")

  #if FLAGS.version_2_with_negative:
  #  with tf.gfile.GFile(output_null_log_odds_file, "w") as writer:
  #    writer.write(json.dumps(scores_diff_json, ensure_ascii=False, indent=4) + "\n")
  
  return all_predictions


def get_final_text(pred_text, orig_text, do_lower_case):
  """Project the tokenized prediction back to the original text."""

  def _strip_spaces(text):
    ns_chars = []
    ns_to_s_map = collections.OrderedDict()
    for (i, c) in enumerate(text):
      if c == " ":
        continue
      ns_to_s_map[len(ns_chars)] = i
      ns_chars.append(c)
    ns_text = "".join(ns_chars)
    return (ns_text, ns_to_s_map)

  # We first tokenize `orig_text`, strip whitespace from the result
  # and `pred_text`, and check if they are the same length. If they are
  # NOT the same length, the heuristic has failed. If they are the same
  # length, we assume the characters are one-to-one aligned.
  tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

  tok_text = " ".join(tokenizer.tokenize(orig_text))

  start_position = tok_text.find(pred_text)
  if start_position == -1:
    if FLAGS.verbose_logging:
      tf.logging.info(
          "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
    return orig_text
  end_position = start_position + len(pred_text) - 1

  (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
  (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

  if len(orig_ns_text) != len(tok_ns_text):
    if FLAGS.verbose_logging:
      tf.logging.info("Length not equal after stripping spaces: '%s' vs '%s'",
                      orig_ns_text, tok_ns_text)
    return orig_text

  # We then project the characters in `pred_text` back to `orig_text` using
  # the character-to-character alignment.
  tok_s_to_ns_map = {}
  for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
    tok_s_to_ns_map[tok_index] = i

  orig_start_position = None
  if start_position in tok_s_to_ns_map:
    ns_start_position = tok_s_to_ns_map[start_position]
    if ns_start_position in orig_ns_to_s_map:
      orig_start_position = orig_ns_to_s_map[ns_start_position]

  if orig_start_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map start position")
    return orig_text

  orig_end_position = None
  if end_position in tok_s_to_ns_map:
    ns_end_position = tok_s_to_ns_map[end_position]
    if ns_end_position in orig_ns_to_s_map:
      orig_end_position = orig_ns_to_s_map[ns_end_position]

  if orig_end_position is None:
    if FLAGS.verbose_logging:
      tf.logging.info("Couldn't map end position")
    return orig_text

  output_text = orig_text[orig_start_position:(orig_end_position + 1)]
  return output_text


def _get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def _compute_softmax(scores):
  """Compute softmax probability over raw logits."""
  if not scores:
    return []

  max_score = None
  for score in scores:
    if max_score is None or score > max_score:
      max_score = score

  exp_scores = []
  total_sum = 0.0
  for score in scores:
    x = math.exp(score - max_score)
    exp_scores.append(x)
    total_sum += x

  probs = []
  for score in exp_scores:
    probs.append(score / total_sum)
  return probs


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      impossible = 0
      if feature.is_impossible:
        impossible = 1
      features["is_impossible"] = create_int_feature([impossible])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `do_train` or `do_predict` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_file:
      raise ValueError(
          "If `do_train` is True, then `train_file` must be specified.")
  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def test_interface(insts):
  #tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.set_verbosity(tf.logging.WARN)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))


  num_train_steps = None
  num_warmup_steps = None

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_predict:
    # eval_examples = read_squad_examples(
    #     input_file=FLAGS.predict_file, is_training=False)
    eval_examples = read_insts_examples(insts)
    eval_writer = FeatureWriter(
        filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
        is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        max_query_length=FLAGS.max_query_length,
        is_training=False,
        output_fn=append_feature)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", len(eval_examples))
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_input_fn = input_fn_builder(
        input_file=eval_writer.filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    all_results = []
    for result in estimator.predict(
        predict_input_fn, yield_single_examples=True):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      unique_id = int(result["unique_ids"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits = [float(x) for x in result["end_logits"].flat]
      all_results.append(
          RawResult(
              unique_id=unique_id,
              start_logits=start_logits,
              end_logits=end_logits))
    tf.logging.info("Processing all example: %d" % (len(all_results)))

    output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
    output_nbest_file = os.path.join(FLAGS.output_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(FLAGS.output_dir, "null_odds.json")

    all_predictions = write_predictions(eval_examples, eval_features, all_results,
                      10, FLAGS.max_answer_length,
                      FLAGS.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file)


    #print (all_predictions)
    #return json.dumps(all_predictions, ensure_ascii=False)
    results = []
    print(all_predictions)
    for context, event, event_id, date in insts:
        #print(context, event, event_id, date)
        try:
            pred_sub = all_predictions[str(event_id)]
        except:
            pred_sub = ''
            #continue
        if pred_sub == '': print("error")
        pred = sub_s_e(pred_sub)
        edate = []
        anchorOffset = context.find(pred_sub)
        anchorOffset = anchorOffset if anchorOffset >= 0 else 0
        extentOffset = anchorOffset + len(pred_sub)

        # "edatenum": "2018-12-16\/2018-12-26", "type": "时间段"
        #print(str(date)+" "+pred)
        recognize_res = recognize(str(date)+" "+pred)
        print("convert_res", recognize_res)
        if len(recognize_res) == 0:
            continue
        edatenum = recognize_res[-1]["norm"]
        edate_type = "开始时间" if recognize_res[-1]["type"] == 'DATE' else "时间段"
        print(pred, '    ', edatenum)
        edate.append({
            "anchorOffset": anchorOffset,
            "extentOffset": extentOffset - 1,
            "sourse": "text",
            # "type": "开始时间",
            "type": edate_type,
            "edatenum": edatenum,
            "sedate": pred_sub,
            "value": pred})
        results.append({"context": context, "event": event, "event_id": event_id, "edate": edate})
    return json.dumps(results, ensure_ascii=False)


def sub_s_e(value):
    # 去除时间标注中多余的 截止、截至、上午、下午 等以及符号
    signs = ['截止', '截至', '上午', '下午', '晚上', '傍晚', '晚间', '午间', '中午', '早晨', '时', '从', '自', '起',
             '讯据', '报', '年中', '到', '讯', '止', '起息', '初', '电', '度', '底', '晚', '在',
             ',', '，', '。',
             ';', '；', ':', '：', "'", '"', '‘', '’', '“', '”', '!', '！',
             '?', '？', ' ', '\t', '  ', '\t\t', '\n']
    for sign in signs:
        if value.startswith(sign):
            value = value[len(sign):]
        if value.endswith(sign):
            value = value[:-len(sign)]

    return value


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")

  context1 = "近日，多家媒体报道，8月1日上午，在建国门的万达总部，两名万达高管被朝阳警方带走。据悉，多位万达员工目睹了二人被带进警车过程。不止一位万达员工向《中国企业家》证实了此事。涉案的金姓、尹218姓高管分别担任万达集团中区营销总经理、副总经理职务。有媒体报道，他们被带走的原因是涉嫌利用职务便利谋取私利。但截至发稿，万达集团并未回应此事。《中国企业家》多次致电万达集团审计举报电话18泻耸担苑绞贾彰挥薪犹不过，据此前中国证券报报道，此次案件或与安徽的项目有关，项目公司以销售去化有困难为由串通集团高管放宽审批权，将房子以极低的折扣卖给了外部供应商，从中赚取差价。此次共牵涉区域公司、集团公司20余人，集团高管个人非法所得金额高达千万。其中还涉及到一位在万达任职17年的老臣万达集团总裁助理高斌，目前此人已出逃美国。据一位不愿具名的万达员工透露，今年4月，万达审计部接到投诉后，在内部开展了近三个月的调查，“万达审计部的很多反腐案例都源于举报”。“一个地产项目的销售总监存在利用职务之便以权谋私的可能性很大。”一名不愿具名的房地产业内人士告诉本刊，这被地产圈的从业者视为“无足挂齿，再正常不过的事情”。看似正常的事，王健林却高度重视，要求从严处理。有消息称，目前万达集团已将主要涉案嫌犯交由司法部门，其余基层违规员工被开除，并重新修订了营销管理制度，禁止项目包销，严控电商。一个令人又怕又敬的部门业内流传一种说法，当万达集团的审计人员经王健林授权到地方公司查账时，地方负责人就要立即交出账本，足见万达审计之权威。这在房地产界无人不知。此次万达两名高管被朝阳警方带走时，一些房地产人士调侃道“习惯了”，他们认为万达高管被审计部门调查出问题是常态，且和其他一些房地产企业相比，万达违法违纪的数量其实相当少。这完全归因于万达内部令人又怕又敬的审计部门。从万达集团官网公开信息，万达审计部成立于2001年，其成员由财务、工程、预算、土水电各专业人才组成。该部门在万达内部被员工们称为“审计中心”，是王健林非常重视的一个部门，“老王（王健林）特别看重这个部门，这是帮他守住公司的关键”，一位不具名前万达员工告诉记者。这个部门被王健林看重，员工提及亦有所惧怕、颤抖三分。“我们对审计部门的主要感受是怕，我们希望永远不要和他们那边（审计中心）的人有交集。因为当他们找到你的时候，基本上没有什么好事儿。很多人对审计中心真的真的是怕。”上述不具名前万达员工补充道。令员工惧怕"
  event1 = "8月1日上午，在建国门的万达总部，两名万达高管被朝阳警方带走"
  event_id1 = "0"

  context2 = "2018-10-19商用新车网转载浏览：301【行业动态】当地时间10月16日，玲珑轮胎与欧洲复兴开发银行（以下简称欧开行）业务投资会谈在伦敦举行，欧开行常务副行长JurgenRigterink先生、分管风控副行长BetsyNelson女士、分..当地时间10月16日，玲珑轮胎与欧洲复兴开发银行（以下简称欧开行）业务投资会谈在伦敦举行，欧开行常务副行长JurgenRigterink先生、分管风控副行长BetsyNelson女士、分管制造业和工商业副行长Jean-MarcPeterschmitt先生分别出席会议，与玲珑轮胎董事长王锋针对玲珑轮胎塞尔维亚项目建设进行合作洽谈。王锋和欧开行常务副行长JurgenRigterink先生王锋和欧开行分管风控副行长BetsyNelson女士王锋和欧开行分管制造业和工商业副行长Jean-MarcPeterschmitt先生欧洲复兴开发银行是欧洲地区最重要的开发性金融机构之一，成立于1991年，总部在英国伦敦。作为一家政策性银行，欧开行的投资不仅看重企业带来的经济效益，更注重其社会效益。玲珑轮胎多年来在文化交流、社会责任、环境保护、安全生产等方面的优异表现，以及海外建厂的成功经验，使欧开行对此次会谈非常重视。在国家“一带一路”倡议的大背景下，玲珑轮胎以“5+3”发展战略为指导，在国内建设了招远、德州、柳州三个生产基地，第四个生产基地湖北荆门项目正在建设中。海外投资建设了泰国生产基地，为当地提供了3000多个就业岗位，并积极为当地社会的灾难救助、社会福利等慈善活动做出贡献，加强与泰国的文化融合，树立了海外建厂的典范。作为海外第二个工厂项目，塞尔维亚项目计划投资9.9亿美元，建设年产能1,362万条的高性能子午线轮胎生产工厂，建成后达产年将实现销售收入6亿美元。项目本身不仅能够为塞尔维亚和兹雷尼亚宁地区增加税收收入、提供1200个就业岗位，还可以持续提升当地工人的劳动技能，并带动当地与轮胎制造相关的上下游产业的发展，经济价值和社会价值巨大。在本次双方会谈中，玲珑轮胎董事长王锋对塞尔维亚项目的建设规划、投资估算、财务效益，以及在员工福利和培训投入等方面进行介绍。双方就当前的风险管控、如何以泰国经验来管理塞尔维亚工厂等方面进行深入探讨。欧开行表示，经过前期的调研评估，玲珑轮胎各方面的表现符合我们的投资要求，期待双方的合作。双方业务投资会谈“我们希望欧开行可以参与到本项目中，与我们共同推动塞尔维亚和兹雷尼亚宁地区的经济发展，提高当地人民生活水平。”玲珑轮胎董事长王锋对未来的合作同样满怀期望。免责声明：本文转载自互联网，来源：商用新车网，本站只做展示，并不代表本站赞同或附和文章观点，内容如有不当，请通过客服热线通知我们删除。原创投稿：中国汽车网诚征高质量原创稿件，投稿邮箱：2384395662@qq.com中国汽车网公众号了解更多最新商用车资讯请关注商用车之家（公众号）WAP版欢迎扫描左方的二维码可把这条新闻装进口袋接着读"
  event2 = "作为海外第二个工厂项目，塞尔维亚项目计划投资9.9亿美元"
  event_id2 = "1"

  context3 = "赣锋锂业(002460)2018年三季报点评：锂价下行拖累业绩，现金流相对稳健text: 股票代码：002460股票名称：赣锋锂业行业：有色冶炼加工相关概念：融资融券,新能源汽车...报告类型：公司研究研报机构：华创证券研报作者：任志强2018年三季报点评：锂价下行拖累业绩，现金流相对稳健　　事项：　　赣锋锂业29号晚发布三季报，公司Q3单季度实现营业收入12.62亿元，同比增加3.69%；实现归母净利润2.69亿元，同比减少31.98%。前三季度实现营业收入35.94亿元，同比增加26.44%；实现归母净利润11.07亿元，同比增加10.26%。公司全年业绩指引归母净利润区间14.69亿元~17.63亿元，同比增长0%~+20%。　　评论：　　锂价下行拖累业绩，现金流相对稳健。公司Q3归母净利润同比增速2017年Q1以来首次出现负增长，前三季度销售毛利率呈现下滑趋势，分别为46.08%、42.26%、28.27%，Q3毛利率创10个月以来的低点。公司主要产品电池级氢氧化锂自年初下滑近16%，目前报价12.5万元/吨，暂未止跌；电池级碳酸锂年初以来下滑近52%，目前报价7.85万元/吨，有企稳迹象。从现金流看，Q3销售商品、提供劳务收到的现金环比增加1亿元至9.75亿元，经营活动现金净流入1.45亿元，融资活动与投资活动分别净流出1.78亿、2.61亿元，Q3现金及等价物净流出2.73亿元，相当比较稳健。　　产能规模国内居于榜首，2018年底将形成碳酸锂产能约5万吨、氢氧化锂产能2.8万吨。公司现有碳酸锂设计产能2.3万吨，有效产能1.85万吨;宁都年产1.75万吨电池级碳酸锂项目预计将于2018年4季度建成投产;原有马洪厂区6000吨碳酸锂生产线改扩建至1.5万吨电池级碳酸锂项目预期2018年完成。氢氧化锂方面，公司现有电池级氢氧化锂产能2.8万吨，其中新余的年产2万吨单水氢氧化锂项目在今年2季度顺利投产。根据产能释放节奏，我们预计2018-2020年公司锂产品产销规模分别在4.6万吨、6.86万吨、7.87万吨。　　多次签订重大销售合同，产品进入新能源汽车高端供应链。氢氧化锂产品提前锁定下游客户，顺利进入高端供应链。2018年8月14日公司与LG化学签订供货合同，约定自2019年至2022年共四年向LG化学销售氢氧化锂4.76万吨，年均1.19万吨。紧接着2018年9月18日，双方签订《供货合同之补充合同》，约定自2019年1月1日起至2025年12月31日，公司增加向LG化学销售氢氧化锂和碳酸锂产品共计4.5万吨。2018年9月21日，公司与特斯拉签订氢氧化锂销售合同，采购数量约为公司该产品当年总产能的20%。　　盈利预测、估值及投资评级。假设赣锋2018-2020年锂产品产销规模分别在4.6万吨、6.86万吨、7.87万吨。锂产品（考虑氢氧化锂溢价）均价（含税价）分别12万元/吨、10.8万元/吨、9.5万元/吨，我们预计公司18-20年实现归母净利润15/20/21亿元（前值为22.65/24.32/22.06亿元），EPS分别为1.11/1.49//1.56元，对应收盘价22.64元PE分别为20/15/14X，维持“推荐”评级。　　风险提示：碳酸锂价格持续下行；新能源汽车发展增速不及预期。"
  event3 = "赣锋锂业29号晚发布三季报，公司Q3单季度实现营业收入12.62亿元，同比增加3.69%"
  event_id3 = "2"

  #insts = [(context1, event1, event_id1,"2019年3月6日"), (context2, event2, event_id2,"2019年3月6日"), (context3, event3, event_id3,"2019年3月6日")]
  context4 = u"新疆天山水泥股份有限公司非公开发行股票发范世琦新浪微博行情况报告及上市公告书摘要2小时前小编：adddw分类：新闻动态信息来源：http://www.83213.com/阅读()12个月3王士亮4，000，00012个月4中国银河投资管理有限公司5，000，00012个月5太平洋资产管理有限责任公司10，000，00012个月6叶祥尧2，936，53212个月合计76，911，544―注：叶祥尧不低于本次发行价格的申购数量为400万股，由于受募集资金规模限制，其最终获配数量为2，936，532股。（二）各发行对象的基本情况1．中国中材股份有限公司的基本情况（1）基本情况公司名称：中国中材股份有限公司注册地址：北京市西城区西直门北顺城街11号法定代表人：谭仲明注册资本：357，146．4万元公司类型：上市股份有限公司经营范围：许可经营项目：对外派遣实施承包境外建材工业安装工程所需劳务人员（有效期至2012年10月17日）。一般经营项目：无机非金属材料的研究、开发、生产、销售；无机非金属材料应用制品的设计、生产、销售；工程总承包；工程咨询、设计；进出口业务；建筑工程和矿山机械的租赁及配件的销售；与上述业务相关的技术咨询、技术服务。（2）与公司的关联关系本次发行前，中材股份持有本公司113，196，483股股份，持股比例为36．28％，为本公司的控股股东；本次发行后，中材股份持有本公司163，171，495股股份，持股比例为41．95％，仍为本公司的控股股东，与发行人构成关联关系。（3）本次发行认购情况认购股数：49，975，012股；限售期安排：自本次发行上市之日起，36个月内不得上市交易或转让。（4）发行对象及其关联方与公司最近一年重大交易情况ａ．中材股份为天山股份借款提供担保截至2009年12月31日，中材股份为天山股份向有关银行申请的两笔贷款提供连带责任保证，具体情况如下：贷款银行贷款金额（万元）贷款期限中国建设银行乌鲁木齐黄河路支行11，000．002007．10．31―2013．3．31乌鲁木齐农信社20，000．002008．3．4―2015．12．24截止目前，该两笔借款及其担保仍处于继续状态。ｂ．天山股份向中材股份借款2008年4月8日，天山股份召开第三届董事会第二十次会议，审议通过了《关于向中国中材股份有限公司借款的议案》，同意天山股份向中材股份借款34，000万元。2008年8月12日，天山股份召开第三届董事会第二十四次会议，审议通过了《关于向中国中材股份有限公司及其关联方借款的议案》，同意向中材股份新增借款20，000万元。天山股份向中材股份的54，000万元借款明细如下：合同编号借款用途借款金额（万元）借款利率起始日终止日中材股份合同2008－03－03号补充流动资金20，000．00一年期利率上浮10％2008－1－292009－1－29中材股份合同2008－04－01号补充流动资金3，000．00一年期利率上浮10％2008－4－32009－4－3中材股份合同2008－04－01号补充流动资金3，800．00一年期利率上浮10％2008－5－52009－5－5中材股份合同2008－09－09号补充流动资金1，200．00一年期利率上浮10％2008－8－132009－8－13中材股份合同2008－09－09号补充流动资金6，000．00一年期利率上浮10％2008－8－282009－8－28中材股份合同2008－09－10号补充流动资金20，000．00一年期利率上浮15％2008－9－192009－9－19合计12首页上一页123456下一页尾页继续阅读：公司发行股份此文由666分类目录编辑，未经允许不得转载！：首页>本站事项>新闻动态»新疆天山水泥股份有限公司非公开发行股票发范世琦新浪微博行情况报告及上市公告书摘要()分享到：上一篇国网分布式光伏云鲁豫有约董思阳网助力“互联网+”光伏创新发展下一篇山东寿光处分37名失信党员部分被开除党籍自由鸟王博相关推荐评论暂无评论"
  event4 = u"新疆天山水泥股份有限公司非公开发行股票发范世琦新浪微博行情况报告及上市公告书摘要2小时前小编：adddw分类：新闻动态信息来源：http://www.83213.com/阅读()12个月3王士亮4，000，00012个月4中国银河投资管理有限公司5，000，00012个月5太平洋资产管理有限责任公司10，000，00012个月6叶祥尧2，936，53212个月合计76，911，544―注：叶祥尧不低于本次发行价格的申购数量为400万股，由于受募集资金规模限制，其最终获配数量为2，936，532股"
  data4 = u"2012/10/17  0:00:00"
  event_id4 = "3"
  
  context5 = u"裁判要旨：执行法院在拍卖土地使用权时，未将该土地上的建筑物一并拍卖处分，造成房地分离的现状，违反法律的强制性规定，该拍卖行为应为无效。案情介绍：一、新疆高院在执行新疆天山水泥股份有限公司（下称“天山公司”）申请执行新疆交通物资供应公司（下称“物资公司”）一案中，查封了被执行人物资公司名下位于乌鲁木齐市土地证号为乌国用（2000）字第0001798号的土地（下称“上述土地”），并委托新疆嘉盛拍卖有限公司将该土地进行拍卖。李辉竞得该宗土地，新疆高院作出（2008）新执字第4－2号执行裁定书（下称“4-2号裁定”），确认该宗土地使用权归买受人李辉所有。二、被执行人物资公司就上述拍卖行为，向新疆高院提出执行异议，认为：上述土地属划拨用地，且未将土地使用权与地上建筑物等一并处理，故请求撤销新疆高院（2008）新执字第4－2号执行裁定书及对该宗土地的拍卖。新疆高院作出（2011）新执二监字第10号执行裁定（下称“10号裁定”）：撤销4-2号裁定，撤销对上述土地使用权的评估拍卖行为。三、李辉向最高法院申请复议，请求：维持4-2号裁定，撤销10号裁定。最高法院裁定：驳回李辉的复议申请，维持10号裁定。四、新疆高院退还李辉拍卖款，天山公司、物资公司与新疆北方机械化筑路工程处三方协议将上述土地过户到新疆北方机械化筑路工程处名下。申请执行人天山公司向新疆高院申请终结本次执行程序，新疆高院裁定准予终结。裁判要点及思路：《物权法》第一百四十六条规定：“建设用地使用权转让、互换、出资或者赠与的，附着于该土地上的建筑物、构筑物及其附属设施一并处分。”执行法院在拍卖土地使用权时，未将土地上建筑物等一并拍卖处分，该执行行为违反了法律规定。执行法院以执行行为违反法律规定为由，裁定4-2号执行裁定及撤销上述土地使用权的评估拍卖，并无不当。实务要点总结：前事不忘，后事之师，我们总结该案的实务要点如下，以供实务参考。同时也提请当事人注意房地分离时，单独处理房或地是否应遵循房地一体的处理原则结合最高法院的裁定文书，在执行实务中，应重点关注以下内容：一、当事人需注意执行法院在拍卖土地使用权时，未将该土地上的建筑物一并拍卖处分的，因造成房地分离的现状违反了法律的强制性规定，故该拍卖行为应为无效。此时，当事人可以向法院申请裁定撤销该拍卖裁定和裁定该拍卖行为无效。二、因房地分离，即便当事人通过生效判决实现债权的方式取得诉争房产并办理了房产证，但因生效法律文书中未涉及争议房产项下的土地使用权，且房屋产权转让并不导致土地使用权一并转让的法律后果，所以在该当事人未依法领取土地使用权证的情况下，又将诉争房产出卖给他人的转让行为因违反行政法规的禁止性规定而无效。三、此外，关于国有划拨土地转让的问题，以划拨方式取得土地使用权的，转让房地产时，应当按照国务院规定，报有批准权的人民政府审批。有批准权的人民政府准予转让的，应当由受让方办理土地使用权出让手续，并依照国家有关规定缴纳土地使用权出让金。所以，当事人在转让国有划拨土地时，应注意国有划拨土地的转让须经过有批准权的人民政府予以审批，未经批准的，转让合同无效。相关法律：《物权法》第一百四十六条建设用地使用权转让、互换、出资或者赠与的，附着于该土地上的建筑物、构筑物及其附属设施一并处分。第一百九十条抵押权设立后抵押财产出租的，该租赁关系不得对抗已登记的抵押权人。《中华人民共和国城市房地产管理法》第三十八条下列房地产，不得转让：（一）以出让方式取得土地使用权的，不符合本法第三十九条规定的条件的；（二）司法机关和行政机关依法裁定、决定查封或者以其他形式限制房地产权利的；（三）依法收回土地使用权的；（四）共有房地产，未经其他共有人书面同意的；（五）权属有争议的；（六）未依法登记领取权属证书的；（七）法律、行政法规规定禁止转让的其他情形。第三十九条以出让方式取得土地使用权的，转让房地产时，应当符合下列条件：（一）按照出让合同约定已经支付全部土地使用权出让金，并取得土地使用权证书；（二）按照出让合同约定进行投资开发，属于房屋建设工程的，完成开发投资总额的百分之二十五以上，属于成片开发土地的，形成工业用地或者其他建设用地条件。转让房地产时房屋已经建成的，还应当持有房屋所有权证书。第四十条第一款以划拨方式取得土地使用权的，转让房地产时，应当按照国务院规定，报有批准权的人民政府审批。有批准权的人民政府准予转让的，应当由受让方办理土地使用权出让手续，并依照国家有关规定缴纳土地使用权出让金。《最高人民法院关于审理涉及国有土地使用权合同纠纷案件适用法律问题的解释》第十一条土地使用权人未经有批准权的人民政府批准，与受让方订立合同转让划拨土地使用权的，应当认定合同无效。但起诉前经有批准权的人民政府批准办理土地使用权出让手续的，应当认定合同有效。以下为该案在最高法院审理阶段关于该事项分析的“本院认为”部分关于拍卖土地使用权时应按房随地走的物权变动原则一并处理地上房屋，否则该拍卖行为应为无效的详细论述和分析。本院认为，“根据《中华人民共和国物权法》第一百四十六条规定：‘建设用地使用权转让、互换、出资或者赠与的，附着于该土地上的建筑物、构筑物及其附属设施一并处分。’人民法院强制拍卖建设用地使用权时，应当严格遵循该条规定。本案执行法院在拍卖上述土地使用权时，未将土地上建筑物等一并拍卖处分，该执行行为违反了法律规定。新疆高院以执行行为违反法律规定为由，裁定撤销（2008）新执字第4-2号执行裁定及撤销上述土地使用权的评估拍卖，并无不当。如果申请复议人认为执行行为给其造成了损失，可通过相关法律途径寻求救济。综上，裁定：驳回李辉的复议申请；维持新疆维吾尔自治区高级人民法院（2011）新执二监字第10号执行裁定。”案件来源：最高人民法院：《新疆天山水泥股份有限公司与新疆交通物资供应公司买卖合同纠纷执行案复议裁定书》【(2011)执复字第16号】延伸阅读：有关拍卖土地使用权时应按房随地走的物权变动原则一并处理地上房屋，否则该拍卖行为应为无效的问题，以下是我们在写作中检索到与该问题相关的案例及裁判观点，以供读者参考。1、当事人虽通过生效判决实现债权的方式取得诉争房产并办理了房产证，但该法律文书中未涉及争议房产项下的土地使用权，房屋产权转让并不导致土地使用权一并转让的法律后果，在该当事人未依法领取土地使用权证的情况下，又将诉争房产出卖给他人的转让行为因违反行政法规的禁止性规定而无效。案例一：《上海某公司与南京某公司房屋买卖合同纠纷上诉案》【江苏省无锡市中级人民法院（2010）锡民终字第0974号】本院认为，“根据《中华人民共和国城市房地产管理法》第三十八条、第三十九条的规定，权属有争议的房地产或者未依法领取权属证书的房地产，不得转让；土地为出让性质的，应当依法交纳相应的土地出让金，并取得土地使用权证书方能转让房地产。南京某公司虽然通过生效判决实现债权的方式取得诉争房产并办理了房产证，但该法律文书中未涉及争议房产项下的土地使用权，房屋产权转让并不导致土地使用权一并转让的法律后果，南京某公司至今未提供其申请办理土地使用权过户审批及支付土地出让金的相关证据。在未依法领取土地使用权证的情况下，南京某公司将诉争房产出卖给上海某公司的转让行为因违反行政法规的禁止性规定而无效。本院对上海某公司要求依无效合同取得无锡市人民中路199号无锡机电大厦裙房一至四层确认归其所有的上诉请求，依法不予支持。原审法院认定事实清楚，适用法律正确，应予维持。据此，依照《民事诉讼法》第一百五十三条第一款第(一)项之规定，判决如下：驳回上诉，维持原判。”2、执行法院在拍卖土地使用权时，未将该土地上的建筑物一并拍卖处分，造成房地分离的现状，违反法律的强制性规定，拍卖行为应为无效。案例二：《赞皇县仁德房地产开发有限公司、中国农业银行股份有限公司石家庄新区支行与河北日升昌科工贸有限公司、石家庄开发区华鹏实业有限公司等金融借款合同纠纷、申请承认与执行法院判决、仲裁裁决案件执行裁定书》【河北省石家庄市中级人民法院（2015）石执复字第00050号】本院认为，“《物权法》第一百四十六条规定：‘建设用地使用权转让、互换、出资或者赠与的，附着于该土地上的建筑物、构筑物及其附属设施一并处分。’据此，人民法院在强制拍卖土地使用权时，应当严格遵循此‘房地一体’的原则。本案中，执行法院在拍卖土地使用权时，未将该土地上的建筑物一并拍卖处分，造成房地分离的现状，违反法律的强制性规定，拍卖行为应为无效。在当事人提出异议后，执行法院以拍卖行为违反法律规定为由，裁定撤销该拍卖行为并无不当。”3、以划拨方式取得土地使用权的，转让房地产时，应当按照国务院规定，报有批准权的人民政府审批。有批准权的人民政府准予转让的，应当由受让方办理土地使用权出让手续，并依照国家有关规定缴纳土地使用权出让金。所以，国有划拨土地的转让应当经过有批准权的人民政府予以审批，未经批准的，转让合同无效。案例三：《南通三九焊接机器制造有限公司与南通三九焊接设备有限公司房屋买卖合同纠纷二审民事判决书》【江苏省高级人民法院(2016)苏民终997号】本院认为，“涉案土地使用权证载明土地使用权类型为划拨土地而非出让土地。我国划拨土地使用权制度是把土地作为一种重要的社会公共政策资源，实现特定的社会公共利益。划拨土地系无偿取得，但土地使用权人对划拨土地使用权的处分受限。根据《中华人民共和国城市房地产管理法》第四十条第一款规定，以划拨方式取得土地使用权的，转让房地产时，应当按照国务院规定，报有批准权的人民政府审批。有批准权的人民政府准予转让的，应当由受让方办理土地使用权出让手续，并依照国家有关规定缴纳土地使用权出让金。《最高人民法院关于审理涉及国有土地使用权合同纠纷案件适用法律问题的解释》第十一条规定，土地使用权人未经有批准权的人民政府批准，与受让方订立合同转让划拨土地使用权的，应当认定合同无效。但起诉前经有批准权的人民政府批准办理土地使用权出让手续的，应当认定合同有效。综合上述规定，国有划拨土地的转让应当经过有批准权的人民政府予以审批，未经批准的，转让合同无效。”"
  event5 = u"案情介绍：一、新疆高院在执行新疆天山水泥股份有限公司（下称“天山公司”）申请执行新疆交通物资供应公司（下称“物资公司”）一案中，查封了被执行人物资公司名下位于乌鲁木齐市土地证号为乌国用（2000）字第0001798号的土地（下称“上述土地”），并委托新疆嘉盛拍卖有限公司将该土地进行拍卖"
  data5 = u"2018/4/11  0:00:00"
  event_id5 = "4"
  insts = [(context2, event2, event_id2, data4), (context3, event3, event_id3, data4),(context4, event4, event_id4, data4), (context5, event5, event_id5, data5)]
  
  label_res_str = test_interface(insts)
  print(label_res_str)
  label_res = json.loads(label_res_str)
  
  import datetime
  import calendar

  def check_day(time):
      _, last_day_num = calendar.monthrange(int(time[:4]), int(time[5:7]))
      if last_day_num < int(time[-2:]):
            time = time[:-2] + str(last_day_num)
      return time

  def transform_time(time):
      if "|" in time:
          time = time.split("|")[0].strip()
      if "/" in time:
          time = time.split("/")[0].strip()
      if ':' in time and time[6] == ':':
          time = time[7:]
      if "-" not in time:
          time = time + "-12-31"
      try:
          if len(time) > 7:
              time = check_day(time)
              print('check_day:', time)
              time = datetime.datetime.strptime(time, '%Y-%m-%d').date().isoformat()
          else:
              time = datetime.datetime.strptime(time, '%Y-%m').date().isoformat()
      except Exception as e:
          print(e)
          time = ''
      return time
  '''
  time1 = '2012-10-17'
  print('time1:', time1)
  print(transform_time(time1))
  time2 = '2018-08-10'
  print('time2:', time2)
  print(transform_time(time2))
  '''
  for itm in label_res:
      e_id = itm["event_id"]
      context = itm["context"]
      event = itm["event"]
      edates = itm["edate"][0]["edatenum"]
      print('original:', edates)
      edates = transform_time(edates)
      print(edates)



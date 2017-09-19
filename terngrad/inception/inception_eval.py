# Copyright 2016 Google Inc. All Rights Reserved.
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
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time
import re

import numpy as np
import tensorflow as tf
import inception.bingrad_common as bingrad_common

from inception import image_processing
from inception import inception_model as inception


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/dataset_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/dataset_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 60,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 50000,
                            """Number of examples to run. Note that the eval. """
                            """This may be changed according to dataset """
                            """Cifar10 test dataset contains 10000 examples."""
                            """ImageNet validation dataset contains 50000 examples.""")
tf.app.flags.DEFINE_string('subset', 'validation',
                           """Either 'validation', 'test' or 'train'.""")

tf.app.flags.DEFINE_string('device', '/gpu:0',
                           """Device to run eval.""")

tf.app.flags.DEFINE_integer('tower', -1,
                           """Recover model from the specified tower/gpu (-1 for cpu).""")
tf.app.flags.DEFINE_bool('restore_avg_var', True,
                           """Recover variables from moving average version.""")

def _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op, last_eval_step):
  """Runs Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_1_op: Top 1 op.
    top_5_op: Top 5 op.
    summary_op: Summary op.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.gpu_options.allow_growth = True
  final_step = False
  with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      if global_step == last_eval_step:
        print('step=%s is already evaluated. Skip.' % global_step)
        return last_eval_step

      if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        # Restores from checkpoint with relative path.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                         ckpt.model_checkpoint_path))
      print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
      if int(global_step) + 1 >= FLAGS.max_steps:
        final_step = True
    else:
      print('No checkpoint file found')
      return last_eval_step

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      # Counts the number of correct predictions.
      count_top_1 = 0.0
      count_top_5 = 0.0
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0

      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
      start_time = time.time()
      while step < num_iter and not coord.should_stop():
        top_1, top_5 = sess.run([top_1_op, top_5_op])
        count_top_1 += np.sum(top_1)
        count_top_5 += np.sum(top_5)
        step += 1
        if step % 100 == 0:
          duration = time.time() - start_time
          sec_per_batch = duration / 100.0
          examples_per_sec = FLAGS.batch_size / sec_per_batch
          print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                'sec/batch)' % (datetime.now(), step, num_iter,
                                examples_per_sec, sec_per_batch))
          start_time = time.time()

      # Compute precision @ 1.
      precision_at_1 = count_top_1 / total_sample_count
      recall_at_5 = count_top_5 / total_sample_count
      print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' %
            (datetime.now(), precision_at_1, recall_at_5, total_sample_count))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Accuracy @ 1', simple_value=precision_at_1)
      summary.value.add(tag='Accuracy @ 5', simple_value=recall_at_5)
      summary_writer.add_summary(summary, global_step)

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
    if final_step:
      print('The final step is evaluated. Exit!')
      exit()

    return global_step


def evaluate(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default(), tf.device(FLAGS.device):
  #with tf.Graph().as_default():
    # Get images and labels from the dataset.
    images, labels = image_processing.inputs(dataset)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    if FLAGS.dataset_name == 'imagenet':
      num_classes = dataset.num_classes() + 1
    else:
      num_classes = dataset.num_classes()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _ = inception.inference(images, num_classes, net=FLAGS.net)

    # Calculate predictions.
    top_1_op = tf.nn.in_top_k(logits, labels, 1)
    top_5_op = tf.nn.in_top_k(logits, labels, 5)

    if FLAGS.restore_avg_var:
      # Restore the moving average version of the learned variables for eval.
      variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)
      if FLAGS.tower>=0:
        var_dic = {}
        for _name, _var in variables_to_restore.items():
          _var_name = '%s_%d' % (inception.TOWER_NAME, FLAGS.tower) + '/' + _name
          var_dic[_var_name] = _var
        saver = tf.train.Saver(var_dic)
    else:
      saver = tf.train.Saver()
      if FLAGS.tower>=0:
        var_dic = {}
        _vars = tf.global_variables()
        for _var in _vars:
          _var_name = '%s_%d' % (inception.TOWER_NAME, FLAGS.tower) + '/' + _var.op.name
          var_dic[_var_name] = _var
        saver = tf.train.Saver(var_dic)

    # Build the summary operation based on the TF collection of Summaries.
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summary_op = tf.summary.merge(summaries)

    last_eval_step = -1
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
      graph=tf.get_default_graph())
    while True:
      last_eval_step = _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op, last_eval_step)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

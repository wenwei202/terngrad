# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AlexNet expressed in TensorFlow-Slim.

  Usage:

  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the batch_norm moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      # Force all Variables to reside on the CPU.
      with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
        logits, endpoints = slim.inception.inception_v3(
            images,
            dropout_keep_prob=0.8,
            num_classes=num_classes,
            is_training=for_training,
            restore_logits=restore_logits,
            scope=scope)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes


def alexnet(inputs,
                 dropout_keep_prob=0.5,
                 num_classes=1000,
                 is_training=True,
                 restore_logits=True,
                 scope=''):
  """AlexNet from https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    dropout_keep_prob: dropout keep_prob.
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: Optional scope for name_scope.

  Returns:
    a list containing 'logits', 'aux_logits' Tensors.
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  with tf.name_scope(scope, 'alexnet', [inputs]):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      with scopes.arg_scope([ops.conv2d, ops.fc],
                            weight_decay=0.0005, stddev=0.01, bias=0.1, seed=1):
        with scopes.arg_scope([ops.conv2d],
                              stride=1, padding='SAME'):
          with scopes.arg_scope([ops.max_pool],
                                stride=2, padding='VALID'):
            # 224 x 224 x 3
            end_points['conv1_1'] = ops.conv2d(inputs, 48, [11, 11], stride=4, bias=0.0, scope='conv1_1')
            end_points['conv1_2'] = ops.conv2d(inputs, 48, [11, 11], stride=4, bias=0.0, scope='conv1_2')
            end_points['lrn1_1'] = ops.lrn(end_points['conv1_1'], scope='lrn1_1')
            end_points['lrn1_2'] = ops.lrn(end_points['conv1_2'], scope='lrn1_2')
            end_points['pool1_1'] = ops.max_pool(end_points['lrn1_1'], [3, 3], scope='pool1_1')
            end_points['pool1_2'] = ops.max_pool(end_points['lrn1_2'], [3, 3], scope='pool1_2')

            # 27 x 27 x 48 x 2
            end_points['conv2_1'] = ops.conv2d(end_points['pool1_1'], 128, [5, 5], scope='conv2_1')
            end_points['conv2_2'] = ops.conv2d(end_points['pool1_2'], 128, [5, 5], scope='conv2_2')
            end_points['lrn2_1'] = ops.lrn(end_points['conv2_1'], scope='lrn2_1')
            end_points['lrn2_2'] = ops.lrn(end_points['conv2_2'], scope='lrn2_2')
            end_points['pool2_1'] = ops.max_pool(end_points['lrn2_1'], [3, 3], scope='pool2_1')
            end_points['pool2_2'] = ops.max_pool(end_points['lrn2_2'], [3, 3], scope='pool2_2')
            end_points['pool2'] = tf.concat([end_points['pool2_1'],end_points['pool2_2']],3)

            # 13 x 13 x 256
            end_points['conv3_1'] = ops.conv2d(end_points['pool2'], 192, [3, 3], bias=0.0, scope='conv3_1')
            end_points['conv3_2'] = ops.conv2d(end_points['pool2'], 192, [3, 3], bias=0.0, scope='conv3_2')

            # 13 x 13 x 192 x 2
            end_points['conv4_1'] = ops.conv2d(end_points['conv3_1'], 192, [3, 3], scope='conv4_1')
            end_points['conv4_2'] = ops.conv2d(end_points['conv3_2'], 192, [3, 3], scope='conv4_2')

            # 13 x 13 x 192 x 2
            end_points['conv5_1'] = ops.conv2d(end_points['conv4_1'], 128, [3, 3], scope='conv5_1')
            end_points['conv5_2'] = ops.conv2d(end_points['conv4_2'], 128, [3, 3], scope='conv5_2')
            end_points['pool5_1'] = ops.max_pool(end_points['conv5_1'], [3, 3], scope='pool5_1')
            end_points['pool5_2'] = ops.max_pool(end_points['conv5_2'], [3, 3], scope='pool5_2')
            end_points['pool5'] = tf.concat([end_points['pool5_1'], end_points['pool5_2']], 3)

            end_points['pool5'] = ops.flatten(end_points['pool5'], scope='flatten')
            end_points['fc6'] = ops.fc(end_points['pool5'], 4096, stddev=0.005, scope='fc6')
            end_points['dropout6'] = ops.dropout(end_points['fc6'], dropout_keep_prob, scope='dropout6')
            end_points['fc7'] = ops.fc(end_points['dropout6'], 4096, stddev=0.005, scope='fc7')
            net = ops.dropout(end_points['fc7'], dropout_keep_prob, scope='dropout7')

            # Final pooling and prediction
            with tf.variable_scope('logits'):
              # 4096
              logits = ops.fc(net, num_classes, activation=None, bias=0.0, scope='logits',
                              restore=restore_logits)
              # 1000
              end_points['logits'] = logits
              end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
  # There is no aux_logits for AlexNet
  end_points['aux_logits'] = tf.constant(0)
  return logits, end_points


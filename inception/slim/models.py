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
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes
from inception.slim import inception_utils
from inception.slim import inception_v1

slim = tf.contrib.slim

def alexnet(inputs,
                 dropout_keep_prob=0.5,
                 num_classes=1000,
                 is_training=True,
                 restore_logits=True,
                 seed=1,
                 weight_decay=0.0005,
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
  #print ("INFO: batch norm in alexnet is disabled")
  end_points = {}
  with tf.name_scope(scope, 'alexnet', [inputs]):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      with scopes.arg_scope([ops.conv2d, ops.fc],
                            weight_decay=weight_decay, stddev=0.01, bias=0.1,
                            #batch_norm_params=None,
                            weights_initializer=tf.truncated_normal_initializer):
        with scopes.arg_scope([ops.conv2d],
                              stride=1, padding='SAME'):
          with scopes.arg_scope([ops.max_pool],
                                stride=2, padding='VALID'):
            # 224 x 224 x 3
            end_points['conv1_1'] = ops.conv2d(inputs, 48, [11, 11], stride=4, bias=0.0, seed = seed +1, scope='conv1_1')
            end_points['conv1_2'] = ops.conv2d(inputs, 48, [11, 11], stride=4, bias=0.0, seed = seed +2, scope='conv1_2')
            end_points['lrn1_1'] = ops.lrn(end_points['conv1_1'], scope='lrn1_1')
            end_points['lrn1_2'] = ops.lrn(end_points['conv1_2'], scope='lrn1_2')
            end_points['pool1_1'] = ops.max_pool(end_points['lrn1_1'], [3, 3], scope='pool1_1')
            end_points['pool1_2'] = ops.max_pool(end_points['lrn1_2'], [3, 3], scope='pool1_2')

            # 27 x 27 x 48 x 2
            end_points['conv2_1'] = ops.conv2d(end_points['pool1_1'], 128, [5, 5], seed = seed +3, scope='conv2_1')
            end_points['conv2_2'] = ops.conv2d(end_points['pool1_2'], 128, [5, 5], seed = seed +4, scope='conv2_2')
            end_points['lrn2_1'] = ops.lrn(end_points['conv2_1'], scope='lrn2_1')
            end_points['lrn2_2'] = ops.lrn(end_points['conv2_2'], scope='lrn2_2')
            end_points['pool2_1'] = ops.max_pool(end_points['lrn2_1'], [3, 3], scope='pool2_1')
            end_points['pool2_2'] = ops.max_pool(end_points['lrn2_2'], [3, 3], scope='pool2_2')
            end_points['pool2'] = tf.concat([end_points['pool2_1'],end_points['pool2_2']],3)

            # 13 x 13 x 256
            end_points['conv3_1'] = ops.conv2d(end_points['pool2'], 192, [3, 3], bias=0.0, seed = seed +5, scope='conv3_1')
            end_points['conv3_2'] = ops.conv2d(end_points['pool2'], 192, [3, 3], bias=0.0, seed = seed +6, scope='conv3_2')

            # 13 x 13 x 192 x 2
            end_points['conv4_1'] = ops.conv2d(end_points['conv3_1'], 192, [3, 3], seed = seed +7, scope='conv4_1')
            end_points['conv4_2'] = ops.conv2d(end_points['conv3_2'], 192, [3, 3], seed = seed +8, scope='conv4_2')

            # 13 x 13 x 192 x 2
            end_points['conv5_1'] = ops.conv2d(end_points['conv4_1'], 128, [3, 3], seed = seed +9, scope='conv5_1')
            end_points['conv5_2'] = ops.conv2d(end_points['conv4_2'], 128, [3, 3], seed = seed +10, scope='conv5_2')
            end_points['pool5_1'] = ops.max_pool(end_points['conv5_1'], [3, 3], scope='pool5_1')
            end_points['pool5_2'] = ops.max_pool(end_points['conv5_2'], [3, 3], scope='pool5_2')
            end_points['pool5'] = tf.concat([end_points['pool5_1'], end_points['pool5_2']], 3)

            end_points['pool5'] = ops.flatten(end_points['pool5'], scope='flatten')
            end_points['fc6'] = ops.fc(end_points['pool5'], 4096, stddev=0.005, seed = seed +11, scope='fc6')
            end_points['dropout6'] = ops.dropout(end_points['fc6'], dropout_keep_prob, scope='dropout6')
            end_points['fc7'] = ops.fc(end_points['dropout6'], 4096, stddev=0.005, seed = seed +12, scope='fc7')
            net = ops.dropout(end_points['fc7'], dropout_keep_prob, scope='dropout7')

            # Final pooling and prediction
            with tf.variable_scope('logits'):
              # 4096
              logits = ops.fc(net, num_classes, activation=None, bias=0.0, seed = seed +13, scope='logits',
                              restore=restore_logits)
              # 1000
              end_points['logits'] = logits
              end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
  # There is no aux_logits for AlexNet
  end_points['aux_logits'] = tf.constant(0)
  return logits, end_points

#trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev,seed=1)

def _alexnet_v2_arg_scope(weight_decay):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      biases_initializer=tf.constant_initializer(0.1),
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc


def _alexnet_v2(inputs,
               dropout_keep_prob=0.5,
               num_classes=1000,
                is_training=True,
               restore_logits=True,
                seed=1,
               scope='alexnet_v2'):
  """AlexNet version 2.
  Described in: http://arxiv.org/pdf/1404.5997v2.pdf
  Parameters from:
  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/
  layers-imagenet-1gpu.cfg
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224. To use in fully
        convolutional mode, set spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope('alexnet_v2', 'alexnet_v2', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=[end_points_collection]):
      net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                        scope='conv1',
                        weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+1))
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
      net = slim.conv2d(net, 192, [5, 5], scope='conv2',
                        weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+2))
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
      net = slim.conv2d(net, 384, [3, 3], scope='conv3',
                        weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+3))
      net = slim.conv2d(net, 384, [3, 3], scope='conv4',
                        weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+4))
      net = slim.conv2d(net, 256, [3, 3], scope='conv5',
                        weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+5))
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')

      # Use conv2d instead of fully_connected layers.
      with slim.arg_scope([slim.conv2d],
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.005,seed=seed),
                          biases_initializer=tf.constant_initializer(0.1)):
        net = slim.conv2d(net, 4096, [5, 5], padding='VALID',
                          scope='fc6',
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.005,seed=seed+6))
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7',
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.005,seed=seed+7))
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          biases_initializer=tf.zeros_initializer(),
                          scope='fc8',
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.005,seed=seed+8))

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if True:
        net = tf.squeeze(net, [1, 2])
        #end_points[sc.name + '/fc8'] = net
      end_points['logits'] = net
      end_points['predictions'] = tf.nn.softmax(net, name='predictions')
      end_points['aux_logits'] = tf.constant(0)
      return net, end_points

_alexnet_v2.default_image_size = 224

def alexnet_v2(inputs,
               dropout_keep_prob=0.5,
               num_classes=1000,
                is_training=True,
               restore_logits=True,
               seed=1,
               weight_decay=0.0005,
               scope='alexnet_v2'):
  with slim.arg_scope(_alexnet_v2_arg_scope(weight_decay)):
    return _alexnet_v2(inputs,
               dropout_keep_prob=dropout_keep_prob,
               num_classes=num_classes,
               is_training=is_training,
               restore_logits=restore_logits,
               seed=seed,
               scope=scope)


def _vgg_arg_scope(weight_decay,
                   is_training):
  """Defines the VGG arg scope.
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      weights_initializer=tf.contrib.layers.xavier_initializer(),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.batch_norm], is_training=is_training):
      with slim.arg_scope([slim.conv2d], padding='SAME', normalizer_fn=slim.batch_norm) as arg_sc:
        return arg_sc

def _vgg_16(inputs,
            dropout_keep_prob=0.5,
           num_classes=1000,
           is_training=True,
           restore_logits=True,
           seed=1,
           scope='vgg_16'):
  """Oxford Net VGG 16-Layers version D Example.
  Note: To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    restore_logits: .
    scope: Optional scope for the variables.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope('vgg_16', 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      #net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.conv2d(inputs, 64, [3, 3], scope='conv1_1', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+1))
      net = slim.conv2d(net, 64, [3, 3], scope='conv1_2', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+2))

      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      #net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.conv2d(net, 128, [3, 3], scope='conv2_1', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+3))
      net = slim.conv2d(net, 128, [3, 3], scope='conv2_2', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+4))

      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      #net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.conv2d(net, 256, [3, 3], scope='conv3_1', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+5))
      net = slim.conv2d(net, 256, [3, 3], scope='conv3_2', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+6))
      net = slim.conv2d(net, 256, [3, 3], scope='conv3_3', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+7))

      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.conv2d(net, 512, [3, 3], scope='conv4_1', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+8))
      net = slim.conv2d(net, 512, [3, 3], scope='conv4_2', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+9))
      net = slim.conv2d(net, 512, [3, 3], scope='conv4_3', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+10))

      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.conv2d(net, 512, [3, 3], scope='conv5_1', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+11))
      net = slim.conv2d(net, 512, [3, 3], scope='conv5_2', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+12))
      net = slim.conv2d(net, 512, [3, 3], scope='conv5_3', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+13))

      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use fully_connected layers.
      net = slim.flatten(net)
      net = slim.fully_connected(net, 4096, scope='fc14', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+14))
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout14')
      net = slim.fully_connected(net, 4096, scope='fc15', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+15))
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout15')

      logits = slim.fully_connected(net, num_classes,
                                 activation_fn=None,
                                 scope='fc16', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+16))

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      end_points['logits'] = logits
      end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
      end_points['aux_logits'] = tf.constant(0)
      return logits, end_points
_vgg_16.default_image_size = 224

def vgg_16(inputs,
            dropout_keep_prob=0.5,
           num_classes=1000,
           is_training=True,
            restore_logits=True,
            weight_decay=0.0005,
           seed=1,
           scope='vgg_16'):
  with slim.arg_scope(_vgg_arg_scope(weight_decay, is_training)):
    return _vgg_16(inputs,
            dropout_keep_prob=dropout_keep_prob,
           num_classes=num_classes,
           is_training=is_training,
           restore_logits=restore_logits,
           seed=seed,
           scope=scope)


def _vgg_a(inputs,
            dropout_keep_prob=0.5,
           num_classes=1000,
           is_training=True,
            restore_logits=True,
           seed=1,
           scope='vgg_a'):
  """Oxford Net VGG 16-Layers version A Example.
  Note: To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    restore_logits: .
    scope: Optional scope for the variables.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope('vgg_a', 'vgg_a', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      #net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.conv2d(inputs, 64, [3, 3], scope='conv1', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+1))

      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      #net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.conv2d(net, 128, [3, 3], scope='conv2', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+2))

      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      #net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.conv2d(net, 256, [3, 3], scope='conv3_1', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+3))
      net = slim.conv2d(net, 256, [3, 3], scope='conv3_2', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+4))

      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      #net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.conv2d(net, 512, [3, 3], scope='conv4_1', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+5))
      net = slim.conv2d(net, 512, [3, 3], scope='conv4_2', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+6))

      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      #net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.conv2d(net, 512, [3, 3], scope='conv5_1', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+7))
      net = slim.conv2d(net, 512, [3, 3], scope='conv5_2', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+8))

      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use fully_connected layers.
      net = slim.flatten(net)
      net = slim.fully_connected(net, 4096, scope='fc14', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+9))
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout14')
      net = slim.fully_connected(net, 4096, scope='fc15', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+10))
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout15')

      logits = slim.fully_connected(net, num_classes,
                                 activation_fn=None,
                                 scope='fc16', weights_initializer=tf.contrib.layers.xavier_initializer(seed=seed+11))

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      end_points['logits'] = logits
      end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
      end_points['aux_logits'] = tf.constant(0)
      return logits, end_points
_vgg_a.default_image_size = 224

def vgg_a(inputs,
           dropout_keep_prob=0.5,
           num_classes=1000,
           is_training=True,
           restore_logits=True,
           seed=1,
           weight_decay=0.0005,
           scope='vgg_a'):
  with slim.arg_scope(_vgg_arg_scope(weight_decay, is_training)):
    return _vgg_a(inputs,
            dropout_keep_prob=dropout_keep_prob,
            num_classes=num_classes,
            is_training=is_training,
            restore_logits=restore_logits,
            seed=seed,
            scope=scope)


def cifar10_alexnet(inputs,
                 dropout_keep_prob=0.5,
                 num_classes=10,
                 is_training=True,
                 restore_logits=True,
                 weight_decay=0.004,
                 seed=1,
                 scope=''):
  """AlexNet on cifar10 from https://www.tensorflow.org/tutorials/deep_cnn

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
  print ("Warning: batch_norm_params is always None in cifar10_alexnet")
  end_points = {}
  with tf.name_scope(scope, 'cifar10_alexnet', [inputs]):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      with scopes.arg_scope([ops.conv2d, ops.fc],
                            bias=0.0, batch_norm_params=None, seed=seed):
        with scopes.arg_scope([ops.conv2d], stride=1, padding='SAME'):
          with scopes.arg_scope([ops.max_pool], stride=2, padding='SAME'):
            # 32 x 32 x 3
            end_points['conv1'] = ops.conv2d(inputs, 64, [5, 5], stride=1, stddev=0.05,
                                             weight_decay=0.0, seed=seed+1, scope='conv1')
            end_points['pool1'] = ops.max_pool(end_points['conv1'], [3, 3], scope='pool1')
            end_points['lrn1'] = ops.lrn(end_points['pool1'], depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, scope='lrn1')

            end_points['conv2'] = ops.conv2d(end_points['lrn1'], 64, [5, 5], stride=1, stddev=0.05, bias=0.1,
                                             weight_decay=0.0, seed=seed+2, scope='conv2')
            end_points['lrn2'] = ops.lrn(end_points['conv2'], depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, scope='lrn2')
            end_points['pool2'] = ops.max_pool(end_points['lrn2'], [3, 3], scope='pool2')


            end_points['pool2'] = ops.flatten(end_points['pool2'], scope='flatten')
            end_points['fc3'] = ops.fc(end_points['pool2'], 384, stddev=0.04, weight_decay=weight_decay, bias=0.1,
                                       seed = seed +3, scope='fc3')
            net = ops.fc(end_points['fc3'], 192, stddev=0.04, weight_decay=weight_decay, bias=0.1,
                         seed=seed + 4, scope='fc4')

            # Final pooling and prediction
            with tf.variable_scope('logits'):
              logits = ops.fc(net, num_classes, activation=None, stddev=1/192.0, weight_decay=0.0,
                              bias=0.0, scope='logits', seed = seed +5, restore=restore_logits)
              # 10
              end_points['logits'] = logits
              end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
  # There is no aux_logits for AlexNet
  end_points['aux_logits'] = tf.constant(0)
  return logits, end_points

def lenet(inputs,
                 dropout_keep_prob=1.0,
                 num_classes=10,
                 is_training=True,
                 restore_logits=True,
                 weight_decay=0.0005,
                 seed=1,
                 scope=''):
  """LeNet in Caffe https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet_train_test.prototxt

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
  print ("Warning: batch_norm_params is always None in lenet")
  end_points = {}
  with tf.name_scope(scope, 'lenet', [inputs]):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      with scopes.arg_scope([ops.conv2d, ops.fc],
                            bias=0.0, batch_norm_params=None, seed=seed):
        with scopes.arg_scope([ops.conv2d], stride=1, padding='SAME'):
          with scopes.arg_scope([ops.max_pool], stride=2, padding='SAME'):
            # 32 x 32 x 3
            end_points['conv1'] = ops.conv2d(inputs, 20, [5, 5], stride=1, stddev=0.05,
                                             weight_decay=weight_decay, seed=seed+1, scope='conv1')
            end_points['pool1'] = ops.max_pool(end_points['conv1'], [2, 2], scope='pool1')

            end_points['conv2'] = ops.conv2d(end_points['pool1'], 50, [5, 5], stride=1, stddev=0.05,
                                             weight_decay=weight_decay, seed=seed+2, scope='conv2')
            end_points['pool2'] = ops.max_pool(end_points['conv2'], [2, 2], scope='pool2')


            end_points['pool2'] = ops.flatten(end_points['pool2'], scope='flatten')
            net = ops.fc(end_points['pool2'], 500, stddev=0.048, weight_decay=weight_decay,
                                       seed = seed +3, scope='fc3')

            # Final pooling and prediction
            with tf.variable_scope('logits'):
              logits = ops.fc(net, num_classes, activation=None, stddev=0.0767, weight_decay=weight_decay,
                              scope='logits', seed = seed +5, restore=restore_logits)
              # 10
              end_points['logits'] = logits
              end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
  # There is no aux_logits for LeNet
  end_points['aux_logits'] = tf.constant(0)
  return logits, end_points

def googlenet(inputs,
           dropout_keep_prob=0.5,
           num_classes=1000,
           is_training=True,
           restore_logits=True,
           seed=1,
           weight_decay=0.00004,
           scope='googlenet'):
    with slim.arg_scope(inception_utils.inception_arg_scope(weight_decay=weight_decay)):
      logits, end_points = inception_v1.inception_v1(inputs,
          num_classes=num_classes,
          is_training=is_training,
          dropout_keep_prob=dropout_keep_prob,
          scope=scope)
      return logits, end_points
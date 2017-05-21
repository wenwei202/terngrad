# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_steps',370000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('net', 'alexnet',
                          """The net to train (inception_v3, alexnet, vgg_16, vgg_a).""")
tf.app.flags.DEFINE_integer('size_to_binarize', 1,
                            """The min number of parameters to enable binarizing.""")

def ternary_encoder(input_data):
  """Encoding and compressing the signs """
  a = tf.sign(input_data) # -1, 0, 1
  a = tf.add(a,1) # shift -1,0,1 to 0,1,2 (2'b00,2'b01,2'b10)
  a = tf.reshape(a,[-1])
  pad_size = 4 - tf.mod(tf.size(a), 4)
  pad = tf.range(0.0, pad_size)
  a = tf.concat([a, pad], 0)
  a_split1, a_split2, a_split3, a_split4 = tf.split(a,4) # assume the size is dividable by 4

  # encode 4 grads into 1 Byte
  sum_1 = tf.add(a_split1, a_split2*4)
  sum_2 = tf.add(a_split3*16, a_split4*64)
  sum_all = tf.add(sum_1, sum_2)
  encoded = tf.cast(sum_all, tf.uint8)
  return encoded

def ternary_decoder(encoded_data, scaler, shape):
  """Decoding the signs to float format """
  a = tf.cast(encoded_data, tf.int32)
  a_split1 = tf.mod(a,4)
  a_split2 = tf.to_int32(tf.mod(a/4,4))
  a_split3 = tf.to_int32(tf.mod(a/16,4))
  a_split4 = tf.to_int32(tf.mod(a/64,4))
  a = tf.concat([a_split1, a_split2, a_split3, a_split4], 0)
  real_size = tf.reduce_prod(shape)
  a = tf.to_float(a)
  a = tf.gather(a, tf.range(0,real_size))

  a = tf.reshape(a, shape)
  a = tf.subtract(a,1)
  decoded = a*scaler
  return decoded

def encode_to_ternary_gradients(grads_and_vars, get_shape=False):
  """Encode each gradient tensor."""
  with tf.name_scope('ternary_encoder'):
    gradients, variables = zip(*grads_and_vars)
    ternary_gradients = []
    gradient_shapes = []
    for gradient in gradients:
      if gradient is None:
        ternary_gradients.append(None)
        if get_shape:
          gradient_shapes.append(None)
        continue

      if get_shape:
        if isinstance(gradient, tf.IndexedSlices):
          gradient_shape = gradient.dense_shape
        else:
          gradient_shape = gradient.get_shape()
        gradient_shapes.append(gradient_shape)

      ternary_gradient = tf.cond(tf.size(gradient) < FLAGS.size_to_binarize,
                                 lambda: tf.bitcast(gradient, type=tf.uint8),
                                 lambda: ternary_encoder(gradient))
      ternary_gradients.append(ternary_gradient)

    if get_shape:
      return list(zip(ternary_gradients, variables)), gradient_shapes
    else:
      return list(zip(ternary_gradients, variables))

def decode_from_ternary_gradients(grads_and_vars, scalers, shapes):
  """Decode each gradient tensor."""
  with tf.name_scope('ternary_decoder'):
    gradients, variables = zip(*grads_and_vars)
    floating_gradients = []
    for gradient, variable, scaler, shape in zip(gradients, variables, scalers, shapes):
      if gradient is None:
        floating_gradients.append(None)
      # gradient is encoded, so we use variable to check its size
      # We also assume dtype of variable and gradient is the same
      floating_gradient = tf.cond(tf.size(variable) < FLAGS.size_to_binarize,
                                 lambda: tf.bitcast(gradient, variable.dtype),
                                 lambda: ternary_decoder(gradient, scaler, shape))
      floating_gradients.append(floating_gradient)

    return list(zip(floating_gradients, variables))

def clip_gradients_by_stddev(grads_and_vars, clip_factor = 2.5):
    """ Clip gradients to [-clip_factor*stddev, clip_factor*stddev]."""
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients = []
    for gradient in gradients:
        if gradient is None:
            clipped_gradients.append(None)
            continue

        mean_gradient = tf.reduce_mean(gradient)
        stddev_gradient = tf.sqrt(tf.reduce_mean(tf.square(gradient - mean_gradient)))
        #clipped_gradient = tf.clip_by_value(gradient, -clip_factor * stddev_gradient, clip_factor * stddev_gradient)
        clipped_gradient = tf.cond(tf.size(gradient) < FLAGS.size_to_binarize,
                               lambda: gradient,
                               lambda: tf.clip_by_value(gradient, -clip_factor * stddev_gradient, clip_factor * stddev_gradient))

        clipped_gradients.append(clipped_gradient)
    return list(zip(clipped_gradients, variables))

def clip_gradients_by_thresholds(grads_and_vars, thresholds):
    """ Clip gradients to [-threshold, threshold]."""
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients = []
    for gradient,threshold in zip(gradients,thresholds):
        if gradient is None:
            clipped_gradients.append(None)
            continue

        #clipped_gradient = tf.clip_by_value(gradient, -threshold, threshold)
        clipped_gradient = tf.cond(tf.size(gradient) < FLAGS.size_to_binarize,
                               lambda: gradient,
                               lambda: tf.clip_by_value(gradient, -threshold, threshold))
        clipped_gradients.append(clipped_gradient)
    return list(zip(clipped_gradients, variables))

def stochastical_binarize_gradients(grads_and_vars, scalers):
  """Stochastically binarize gradients."""
  gradients, variables = zip(*grads_and_vars)
  binarized_gradients = []
  for gradient, scaler in zip(gradients, scalers):
    if gradient is None:
      binarized_gradients.append(None)
      continue
    if isinstance(gradient, tf.IndexedSlices):
      gradient_shape = gradient.dense_shape
    else:
      gradient_shape = gradient.get_shape()

    zeros = tf.zeros(gradient_shape)
    abs_gradient = tf.abs(gradient)
    sign_gradient = tf.sign( gradient )
    rnd_sample = tf.random_uniform(gradient_shape,0,scaler)
    where_cond = tf.less(rnd_sample, abs_gradient)
    binarized_gradient = tf.cond(tf.size(gradient) < FLAGS.size_to_binarize,
                               lambda: gradient,
                               lambda: tf.where(where_cond, sign_gradient * scaler, zeros))

    binarized_gradients.append(binarized_gradient)
  return list(zip(binarized_gradients, variables))

def gradient_binarizing_scalers(grads_and_vars, clip_factor):
    """ Get the scalers."""
    gradients, variables = zip(*grads_and_vars)
    scalers = []
    for gradient in gradients:
        if gradient is None:
            scalers.append(None)
            continue

        if(clip_factor > 1.0e-5):
            mean_gradient = tf.reduce_mean(gradient)
            stddev_gradient = tf.sqrt(tf.reduce_mean(tf.square(gradient - mean_gradient)))
            scalers.append(clip_factor * stddev_gradient)
        else:
            scalers.append(tf.reduce_max(tf.abs(gradient)))

    return list(zip(scalers, variables))


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def average_gradients2(tower_grads):
  """This is identical to average_gradients() but returns pairs of (shared gradient, unshared variable) across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of Lists of pairs of (gradient, variable) where the gradient has been averaged
     across all towers and variable is the one in each tower.
  """
  res = []
  mean_grads = average_gradients(tower_grads)
  for grad_and_vars in tower_grads:
      _grads = []
      for _grad1, _grad2 in zip(mean_grads, grad_and_vars):
          _grads.append( (_grad1[0],_grad2[1]) )
      res.append(_grads)

  return res

def average_scalers(tower_scalers):
  """Calculate the average scalers for gradients across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_scalers: List of lists of (scaler, variable) tuples. The outer list
      is over individual scaler. The inner list is over the scaler
      calculation for each tower.
  Returns:
     List of pairs of scaler where the scaler has been averaged
     across all towers.
  """
  average_scalers = []
  for scale_and_vars in zip(*tower_scalers):
    # Note that each scale_and_vars looks like the following:
    #   ((scale0_gpu0, var0_gpu0), ... , (scale0_gpuN, var0_gpuN))
    scalers = []
    for s, _ in scale_and_vars:
      # Add 0 dimension to the scalers to represent the tower.
      expanded_s = tf.expand_dims(s, 0)

      # Append on a 'tower' dimension which we will average over below.
      scalers.append(expanded_s)

    # Average over the 'tower' dimension.
    scaler = tf.concat(scalers, 0)
    scaler = tf.reduce_mean(scaler, 0)

    average_scalers.append(scaler)
  return average_scalers

def max_scalers(tower_scalers):
  """Calculate the max scalers for gradients across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_scalers: List of lists of (scaler, variable) tuples. The outer list
      is over individual scaler. The inner list is over the scaler
      calculation for each tower.
  Returns:
     List of pairs of scaler where the scaler is the max one
     across all towers.
  """
  max_scalers = []
  for scale_and_vars in zip(*tower_scalers):
    # Note that each scale_and_vars looks like the following:
    #   ((scale0_gpu0, var0_gpu0), ... , (scale0_gpuN, var0_gpuN))
    scalers = []
    for s, _ in scale_and_vars:
      # Add 0 dimension to the scalers to represent the tower.
      expanded_s = tf.expand_dims(s, 0)

      # Append on a 'tower' dimension which we get the max over below.
      scalers.append(expanded_s)

    # Get the max over the 'tower' dimension.
    scaler = tf.concat(scalers, 0)
    scaler = tf.reduce_max(scaler, 0)

    max_scalers.append(scaler)
  return max_scalers
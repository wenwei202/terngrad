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

tf.app.flags.DEFINE_float('zero_threshold',0.0,
                            """Threshold to zero out gradients.""")

FLAGS = tf.app.flags.FLAGS

OLD_GRAD_COLLECTION = '_old_grad_'

def get_tensor_list_percentile(tlist, percentile):
  # concat all
  concat_t = None
  for t in tlist:
    if t is None:
      continue

    if concat_t is None:
      concat_t = tf.reshape(tf.abs(t), [-1])
    else:
      concat_t = tf.concat([concat_t, tf.reshape(tf.abs(t), [-1])], 0)

  return tf.contrib.distributions.percentile(concat_t, percentile)

def sparsify_gradients(grads_and_vars, percentile):
  """ Sparsify gradients."""
  gradients, variables = zip(*grads_and_vars)
  threshold = get_tensor_list_percentile(gradients, percentile)
  sparse_gradients = []
  for gradient in gradients:
    if gradient is None:
      sparse_gradients.append(None)
      continue
    abs_grad = tf.abs(gradient)
    where_cond = tf.less(abs_grad, threshold)
    sparse_gradient = tf.where(where_cond,
             tf.zeros(tf.shape(gradient)),
             gradient)
    sparse_gradients.append(sparse_gradient)
  return list(zip(sparse_gradients, variables))

def add_gradients(grads_and_vars1, grads_and_vars2):
  """ Subtract two gradients."""
  gradients1, variables = zip(*grads_and_vars1)
  gradients2, variables_copy = zip(*grads_and_vars2)
  assert (len(gradients1) == len(variables))
  assert (len(gradients1) == len(gradients2))
  assert (len(gradients2) == len(variables_copy))
  resgrads = []
  for grad1, grad2 in zip(gradients1, gradients2):
    if grad1 is None:
      assert grad2 is None
      resgrads.append(None)
      continue

    resgrads.append(tf.add(grad1, grad2))

  return list(zip(resgrads, variables))

def sub_gradients(grads_and_vars1, grads_and_vars2):
  """ Subtract two gradients."""
  gradients1, variables = zip(*grads_and_vars1)
  gradients2, variables_copy = zip(*grads_and_vars2)
  assert (len(gradients1) == len(variables))
  assert (len(gradients1) == len(gradients2))
  assert (len(gradients2) == len(variables_copy))
  resgrads = []
  for grad1, grad2 in zip(gradients1, gradients2):
    if grad1 is None:
      assert grad2 is None
      resgrads.append(None)
      continue

    #resgrads.append(tf.subtract(grad1, grad2))
    resgrad = tf.subtract(grad1, grad2)
    where_cond = tf.less(tf.abs(resgrad), FLAGS.zero_threshold)
    resgrad = tf.where(where_cond,
                   tf.zeros(tf.shape(resgrad)),
                    resgrad)
    resgrads.append(resgrad)

  return list(zip(resgrads, variables))

def assign_add_gradients(grads_and_vars1, grads_and_vars2):
  """ Add gradients."""
  gradients1, variables = zip(*grads_and_vars1) # gradients must be tf.Variable
  gradients2, variables_copy = zip(*grads_and_vars2)
  assert (len(gradients1) == len(variables))
  assert (len(gradients1) == len(gradients2))
  assert (len(gradients2) == len(variables_copy))
  assigned_ops = []
  for grad1, grad2 in zip(gradients1, gradients2):
    if grad1 is None:
      assert grad2 is None
      continue

    _op = tf.assign(grad1, tf.add(grad1, grad2))
    assigned_ops.append(_op)

  return tf.group(*assigned_ops)

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
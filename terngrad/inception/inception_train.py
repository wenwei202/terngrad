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
"""A library to train Inception using multiple GPU's with synchronous updates.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception
from inception.slim import slim
import inception.bingrad_common as bingrad_common

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/dataset_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train', 'validation' or 'test'.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_integer('num_nodes', -1,
                            """How many virtual nodes to use. One GPU can have multiple nodes""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_string('optimizer', 'momentum',
                          """The optimizer of SGD (momentum, adam, gd, rmsprop).""")
# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# With 8 Tesla K40's and a batch size = 256, the following setup achieves
# precision@1 = 73.5% after 100 hours and 100K steps (20 epochs).
# Learning rate decay factor selected from http://arxiv.org/abs/1404.5997.
tf.app.flags.DEFINE_float('initial_learning_rate', 0.01,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 20.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """The momentum value of optimizer.""")
tf.app.flags.DEFINE_string('learning_rate_decay_type','exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

# Configurations for BinGrad
tf.app.flags.DEFINE_integer('grad_bits', 32,
                            """The number of gradient bits.""")
tf.app.flags.DEFINE_float('clip_factor', 0.0,
                            """The factor of stddev to clip gradients.""")
tf.app.flags.DEFINE_integer('floating_grad_epoch', 0,
                            """Performing floating gradients every # epochs. 0 means bingrad is always used.""")
tf.app.flags.DEFINE_integer('save_tower', -1,
                            """Save the variables in a specific tower. -1 refers all towers""")
tf.app.flags.DEFINE_bool('use_encoding', False,
                            """If use encoder-decoder to communicate. Current implementation is NOT efficient.""")
tf.app.flags.DEFINE_bool('quantize_logits', False,
                            """If quantize the gradients in the last logits layer.""")
tf.app.flags.DEFINE_integer('save_iter', 5000,
                            """Save summaries and model checkpoint per iterations.""")

tf.app.flags.DEFINE_bool('benchmark_mode', False,
                            """benchmarking mode to test the scalability.""")

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
# RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.


def _tower_loss(images, labels, num_classes, scope, reuse_variables=None):
  """Calculate the total loss on a single tower running the ImageNet model.

  We perform 'batch splitting'. This means that we cut up a batch across
  multiple GPU's. For instance, if the batch size = 32 and num_nodes = 2,
  then each tower will operate on an batch of 16 images.

  Args:
    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,
                                       FLAGS.image_size, 3].
    labels: 1-D integer Tensor of [batch_size].
    num_classes: number of classes
    scope: unique prefix string identifying the ImageNet tower, e.g.
      'tower_0'.

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # When fine-tuning a model, we do not restore the logits but instead we
  # randomly initialize the logits. The number of classes in the output of the
  # logit is the number of classes in specified Dataset.
  restore_logits = not FLAGS.fine_tune

  # Build inference Graph.
  with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    logits = inception.inference(images, num_classes, net=FLAGS.net, for_training=True,
                                 restore_logits=restore_logits,
                                 scope=scope)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  split_batch_size = images.get_shape().as_list()[0]
  inception.loss(logits, labels, batch_size=split_batch_size, aux_logits=('inception_v3'==FLAGS.net))

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)

  #total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
  total_cross_entropy_loss = tf.add_n(losses, name='total_cross_entropy_loss')
  total_regularization_loss = tf.constant(0.0)
  # Calculate the total loss for the current tower.
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope)
  if regularization_losses:
    total_regularization_loss = tf.add_n(regularization_losses, name='total_regularization_loss')
  total_loss = tf.add(total_cross_entropy_loss,
                      total_regularization_loss,
                      name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  # loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  # loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss] + regularization_losses + [total_regularization_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on TensorBoard.
    loss_name = re.sub('%s_[0-9]*/' % inception.TOWER_NAME, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(loss_name +'_raw', l)
    #tf.summary.scalar(loss_name + '_avg', loss_averages.average(l))

  # with tf.control_dependencies([loss_averages_op]):
  #   total_loss = tf.identity(total_loss)
  return total_loss, total_cross_entropy_loss, total_regularization_loss


def _average_gradients(tower_grads):
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
    #grad = tf.concat(0, grads)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def _gradient_summary(grad_vars, name="", add_sparsity=False):
  """Helper to create summaries for gradiants and variables.

  Args:
    grad_vars: pairs of gradients and variables
  """
  for grad, var in grad_vars:
    if grad is not None:
      tf.summary.histogram(var.op.name + "/" + name +'/gradients', grad)
      if add_sparsity:
        tf.summary.scalar(var.op.name + "/" + name +'/sparsity', tf.nn.zero_fraction(grad))


def train(dataset):
  """Train on dataset for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    tf.set_random_seed(FLAGS.seed)
    if FLAGS.num_nodes > 0:
      num_nodes = FLAGS.num_nodes
    else:
      num_nodes = FLAGS.num_gpus
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_nodes.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (dataset.num_examples_per_epoch() /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    if ('fixed'==FLAGS.learning_rate_decay_type or 'adam' == FLAGS.optimizer):
      lr = FLAGS.initial_learning_rate
    elif 'exponential'==FLAGS.learning_rate_decay_type:
      lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step/num_nodes,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)
    elif 'polynomial'==FLAGS.learning_rate_decay_type:
      lr = tf.train.polynomial_decay(FLAGS.initial_learning_rate,
                                    global_step/num_nodes,
                                    FLAGS.max_steps,
                                    end_learning_rate=0.0,
                                    power=0.5)
    else:
      raise ValueError('Wrong learning_rate_decay_type!')

    # Create an optimizer that performs gradient descent.
    opt = None
    if ('gd' == FLAGS.optimizer):
        opt = tf.train.GradientDescentOptimizer(lr)
    elif ('momentum' == FLAGS.optimizer):
        opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum)
    elif ('adam' == FLAGS.optimizer):
        opt = tf.train.AdamOptimizer(lr)
    elif ('rmsprop' == FLAGS.optimizer):
        opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
              momentum=FLAGS.momentum,
              epsilon=RMSPROP_EPSILON)
    else:
        raise ValueError("Wrong optimizer!")

    # Get images and labels for ImageNet and split the batch across GPUs.
    assert FLAGS.batch_size % num_nodes == 0, (
        'Batch size must be divisible by number of nodes')

    # Override the number of preprocessing threads to account for the increased
    # number of GPU towers.
    num_preprocess_threads = FLAGS.num_preprocess_threads * num_nodes
    if FLAGS.benchmark_mode:
      images = tf.constant(0.5, shape=[FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
      labels = tf.random_uniform([FLAGS.batch_size], minval=0, maxval=dataset.num_classes()-1, dtype=tf.int32)
    else:
      images, labels = image_processing.distorted_inputs(
          dataset,
          num_preprocess_threads=num_preprocess_threads)


    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    if FLAGS.dataset_name == 'imagenet':
      num_classes = dataset.num_classes() + 1
    else:
      num_classes = dataset.num_classes()

    # Split the batch of images and labels for towers.
    images_splits = tf.split(images, num_nodes, 0)
    labels_splits = tf.split(labels, num_nodes, 0)

    # Calculate the gradients for each model tower.
    tower_grads = [] # gradients of cross entropy or total cost for each tower
    tower_floating_grads = []  # gradients of cross entropy or total cost for each tower
    tower_batchnorm_updates = []
    tower_scalers = []
    #tower_reg_grads = []
    reuse_variables = None
    tower_entropy_losses = []
    tower_reg_losses = []
    for i in range(num_nodes):
      with tf.device('/gpu:%d' % (i%FLAGS.num_gpus)):
        with tf.name_scope('%s_%d' % (inception.TOWER_NAME, i)) as scope:
          with tf.variable_scope('%s_%d' % (inception.TOWER_NAME, i)):
            # Force Variables to reside on the individual GPU.
            #with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
            with slim.arg_scope([slim.variables.variable], device='/gpu:%d' % (i%FLAGS.num_gpus)):
              # Calculate the loss for one tower of the ImageNet model. This
              # function constructs the entire ImageNet model but shares the
              # variables across all towers.
              loss, entropy_loss, reg_loss = _tower_loss(images_splits[i], labels_splits[i], num_classes,
                                 scope, reuse_variables)
            tower_entropy_losses.append(entropy_loss)
            tower_reg_losses.append(reg_loss)

            # Reuse variables for the next tower?
            reuse_variables = None

            # Retain the Batch Normalization updates operations.
            batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                                scope)
            batchnorm_updates = batchnorm_updates + \
                                tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

            tower_batchnorm_updates.append(batchnorm_updates)

            # Calculate the gradients for the batch of data on this ImageNet
            # tower.
            grads = opt.compute_gradients(loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope))

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)
            tower_floating_grads.append(grads)

            # Calculate the scalers of binary gradients
            if 1 == FLAGS.grad_bits:
              # Always calculate scalers whatever clip_factor is.
              # Returns max value when clip_factor==0.0
              scalers = bingrad_common.gradient_binarizing_scalers(grads, FLAGS.clip_factor)
              tower_scalers.append(scalers)

            # regularization gradients
            #if FLAGS.weight_decay:
            #  reg_grads = opt.compute_gradients(reg_loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope))
            #  tower_reg_grads.append(reg_grads)

    if 1 == FLAGS.grad_bits:
      # for grads in tower_grads:
      #   _gradient_summary(grads, 'floating')

      # We must calculate the mean of each scaler. Note that this is the
      # synchronization point across all towers @ CPU.
      # mean_scalers = bingrad_common.average_scalers(tower_scalers)
      mean_scalers = bingrad_common.max_scalers(tower_scalers)
      # for mscaler in mean_scalers:
      #   if mscaler is not None:
      #     tf.summary.scalar(mscaler.op.name + '/mean_scaler', mscaler)

      grad_shapes_for_deocder = []
      for i in range(num_nodes):
        with tf.device('/gpu:%d' % (i%FLAGS.num_gpus)):
          with tf.name_scope('binarizer_%d' % (i)) as scope:
            # Clip and binarize gradients
            # and keep track of the gradients across all towers.
            if FLAGS.quantize_logits:
              tower_grads[i][:] = bingrad_common.stochastical_binarize_gradients(
                tower_grads[i][:], mean_scalers[:])
            else:
              tower_grads[i][:-2] = bingrad_common.stochastical_binarize_gradients(
                  tower_grads[i][:-2], mean_scalers[:-2])

            _gradient_summary(tower_grads[i], 'binary', add_sparsity=True)

          if FLAGS.use_encoding:
            # encoding
            with tf.name_scope('encoder_%d' % (i)) as scope:
              if 0==i:
                tower_grads[i][:-2], grad_shapes_for_deocder = \
                  bingrad_common.encode_to_ternary_gradients(tower_grads[i][:-2], get_shape=True)
              else:
                tower_grads[i][:-2] = bingrad_common.encode_to_ternary_gradients(tower_grads[i][:-2], get_shape=False)

    # decoding @ CPU
    if (1 == FLAGS.grad_bits) and FLAGS.use_encoding:
      with tf.name_scope('decoder') as scope:
        for i in range(num_nodes):
          tower_grads[i][:-2] = bingrad_common.decode_from_ternary_gradients(
            tower_grads[i][:-2], mean_scalers[:-2], grad_shapes_for_deocder)

    # Switch between binarized and floating gradients
    if (FLAGS.floating_grad_epoch>0) and (1 == FLAGS.grad_bits):
      epoch_remainder = tf.mod( ( (global_step / num_nodes) * FLAGS.batch_size) / dataset.num_examples_per_epoch(),
             FLAGS.floating_grad_epoch)
      cond_op = tf.equal(tf.to_int32(tf.floor(epoch_remainder)), tf.to_int32(FLAGS.floating_grad_epoch-1))
      for i in range(num_nodes):
        with tf.name_scope('switcher_%d' % (i)) as scope:
          _, selected_variables = zip( *(tower_floating_grads[i]) )
          selected_gradients = []
          for j in range(len(tower_floating_grads[i])):
            selected_gradients.append( tf.cond(cond_op,
                                  lambda: tower_floating_grads[i][j][0],
                                  lambda: tower_grads[i][j][0]) )
          tower_grads[i] = list(zip(selected_gradients, selected_variables))


    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers @ CPU.
    if len(tower_grads)>1:
      tower_grads = bingrad_common.average_gradients2(tower_grads)


    # Add a summary to track the learning rate.
    tf.summary.scalar('learning_rate', lr)

    # Add histograms for gradients.
    # for grads in tower_grads:
    #   _gradient_summary(grads, 'final')

    # Apply the gradients to adjust the shared variables.
    # @ GPUs
    #apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    apply_gradient_op = []
    for i in range(num_nodes):
      with tf.device('/gpu:%d' % (i%FLAGS.num_gpus)):
        with tf.name_scope('grad_applier_%d' % (i)) as scope:
          # apply data loss SGD. global_step is incremented by num_nodes per iter
          apply_gradient_op.append(opt.apply_gradients(tower_grads[i],
                                          global_step=global_step))
          #if FLAGS.weight_decay:
          #  # apply regularization, global_step is omitted to avoid incrementation
          #  apply_gradient_op.append(opt.apply_gradients(tower_reg_grads[i]))

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    # Track the moving averages of all trainable variables.
    # Note that we maintain a "double-average" of the BatchNormalization
    # global statistics. This is more complicated then need be but we employ
    # this for backward-compatibility with our previous models.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY, global_step/num_nodes)

    # Another possiblility is to use tf.slim.get_variables().
    variables_to_average = (tf.trainable_variables() +
                            tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Group all updates to into a single train op.
    #batchnorm_updates_op = tf.group(*batchnorm_updates)
    #train_op = tf.group(apply_gradient_op, variables_averages_op,
    #                    batchnorm_updates_op)
    batchnorm_updates_op = tf.no_op()
    for tower_batchnorm_update in tower_batchnorm_updates:
      batchnorm_updates_op = tf.group(batchnorm_updates_op, *tower_batchnorm_update)
    apply_gradient_op = tf.group(*apply_gradient_op)
    train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

    # Create a saver.
    #saver = tf.train.Saver(tf.all_variables())
    if FLAGS.save_tower>=0:
      # Only save the variables in a tower
      save_pattern = ('(%s_%d)' % (inception.TOWER_NAME, FLAGS.save_tower)) + ".*" #+ ".*ExponentialMovingAverage"
      var_dic = {}
      _vars = tf.global_variables()
      for _var in _vars:
          if re.compile(save_pattern).match(_var.op.name):
              _var_name = re.sub('%s_[0-9]*/' % inception.TOWER_NAME, '', _var.op.name)
              var_dic[_var_name] = _var
      saver = tf.train.Saver(var_dic)
    else:
      saver = tf.train.Saver(tf.global_variables())

    # average loss summaries
    avg_entropy_loss = tf.reduce_mean(tower_entropy_losses)
    avg_reg_loss = tf.reduce_mean(tower_reg_losses)
    avg_total_loss = tf.add(avg_entropy_loss, avg_reg_loss)
    tf.summary.scalar('avg_entropy_loss', avg_entropy_loss)
    tf.summary.scalar('avg_reg_loss', avg_reg_loss)
    tf.summary.scalar('avg_total_loss', avg_total_loss)

    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True ############ Excepted GPU op may be placed CPU
    config.log_device_placement = FLAGS.log_device_placement
    sess = tf.Session(config=config)
    sess.run(init)

    trained_step = 0
    if FLAGS.pretrained_model_checkpoint_path:
      assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
      ckpt = tf.train.get_checkpoint_state(FLAGS.pretrained_model_checkpoint_path)
      if ckpt and ckpt.model_checkpoint_path:
        trained_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        trained_step = int(trained_step) + 1
        variables_to_restore = tf.get_collection(
            slim.variables.VARIABLES_TO_RESTORE)+ \
                               tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        restorer = tf.train.Saver(variables_to_restore)
        if os.path.isabs(ckpt.model_checkpoint_path):
          restorer.restore(sess, ckpt.model_checkpoint_path)
        else:
          restorer.restore(sess, os.path.join(FLAGS.pretrained_model_checkpoint_path,
                                         ckpt.model_checkpoint_path))
        print('%s: Pre-trained model restored from %s' %
              (datetime.now(), FLAGS.pretrained_model_checkpoint_path))
      else:
        print('%s: Restoring pre-trained model from %s failed!' %
              (datetime.now(), FLAGS.pretrained_model_checkpoint_path))
        exit()

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(
        FLAGS.train_dir,
        graph=tf.get_default_graph())

    for step in range(trained_step, FLAGS.max_steps):
      start_time = time.time()
      _, entropy_loss_value, reg_loss_value = sess.run([train_op, entropy_loss, reg_loss])
      duration = time.time() - start_time

      assert not np.isnan(entropy_loss_value), 'Model diverged with entropy_loss = NaN'

      if step % 10 == 0:
        examples_per_sec = FLAGS.batch_size / float(duration)
        format_str = ('%s: step %d, entropy_loss = %.2f, reg_loss = %.2f, total_loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.now(), step,
                            entropy_loss_value, reg_loss_value, entropy_loss_value+reg_loss_value,
                            examples_per_sec, duration))

      if step % FLAGS.save_iter == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % FLAGS.save_iter == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

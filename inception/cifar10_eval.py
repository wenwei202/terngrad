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
"""A binary to evaluate Inception on the flowers data set.

Note that using the supplied pre-trained inception checkpoint, the eval should
achieve:
  precision @ 1 = 0.7874 recall @ 5 = 0.9436 [50000 examples]

See the README.md for more details.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from inception import inception_eval
from inception.cifar10_data import Cifar10Data

FLAGS = tf.app.flags.FLAGS


def main(unused_argv=None):
  dataset = Cifar10Data(subset=FLAGS.subset)
  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  FLAGS.dataset_name = 'cifar10'
  FLAGS.num_examples = dataset.num_examples_per_epoch()
  inception_eval.evaluate(dataset)


if __name__ == '__main__':
  tf.app.run()

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
"""Small library that points to the cifar-10 data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from inception.dataset import Dataset
import os

FLAGS = tf.app.flags.FLAGS

class MnistData(Dataset):
  """mnist data set."""

  def __init__(self, subset):
    super(MnistData, self).__init__('mnist', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 10

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data set."""
    if self.subset == 'train':
      return 60000
    if self.subset == 'test':
      return 10000

  def download_message(self):
    """Instruction to download and extract the tarball from Flowers website."""

    print('Failed to find any mnist %s files'% self.subset)
    print('')
    print('If you have already downloaded and processed the data, then make '
          'sure to set --data_dir to point to the directory containing the '
          'location of the sharded TFRecords.\n')
    print('If you have not downloaded and prepared the mnist data in the '
          'TFRecord format, you will need to do this at least once. This '
          'process could take a while depending on the speed of your '
          'computer and network connection\n')
    print('Please see README.md for instructions on how to build '
          'the mnist dataset using download_and_convert_data.py. For example: \n')
    print ('cd ./slim\n')
    print ('python download_and_convert_data.py '
           '--dataset_name mnist --dataset_dir ~/dataset/mnist-data/\n')

  def available_subsets(self):
    """Returns the list of available subsets."""
    return ['train', 'test']
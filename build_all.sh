#!/bin/bash

set -x
set -e

bazel build inception/download_and_preprocess_imagenet

bazel build inception/mnist_train
bazel build inception/mnist_eval

bazel build inception/cifar10_train
bazel build inception/cifar10_eval

bazel build inception/imagenet_train
bazel build inception/imagenet_eval

bazel build inception/imagenet_distributed_train
bazel build inception/cifar10_distributed_train

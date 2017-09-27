#!/bin/bash
set -e
set -x

# The executable path. Must be the same across all nodes
WORKSPACE="~/github/users/wenwei202/terngrad/terngrad"

#WORKER_SCRIPT="./run_single_worker_alexnet.sh"
#PS_SCRIPT="./run_single_ps_imagenet.sh"
WORKER_SCRIPT="./run_single_worker_cifarnet.sh"
PS_SCRIPT="./run_single_ps_cifar10.sh"

PS_HOSTS=( \
  10.236.176.29:2222 \
)
WORKER_HOSTS=( \
  10.236.176.28:2224 \
  10.236.176.29:2226 \
)
WORKER_DEVICES=( \
  1 \
  2 \
)
DATA_DIR=( \
  ~/dataset/cifar10-data-shard-0-499 \
  ~/dataset/cifar10-data-shard-500-999 \
)



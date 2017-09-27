#!/bin/bash
set -e
set -x

########################## Example settup ###############################
# Parameter server is 10.236.176.29:2222
# Worker 0 is GPU 1 in 10.236.176.28:2224 using ~/dataset/cifar10-data-shard-0-499 as training data
# Worker 1 is GPU 2 in 10.236.176.29:2226 using ~/dataset/cifar10-data-shard-500-999 as training data
# The whole cifar10 dataset are split to cifar10-data-shard-0-499 and cifar10-data-shard-500-999 


######################## Workspace of TernGrad ##########################
# The path of executables (terngrad/terngrad). Must be the same across all nodes
WORKSPACE="~/github/users/wenwei202/terngrad/terngrad"




#################### Scripts to start workers and ps #####################
# The script to start worker
# Customize WORKER_SCRIPT for your own training
WORKER_SCRIPT="./run_single_worker_cifarnet.sh"
#WORKER_SCRIPT="./run_single_worker_alexnet.sh"

# The script to start ps (depending on dataset only)
# Select one from those below
PS_SCRIPT="./run_single_ps_cifar10.sh"
#PS_SCRIPT="./run_single_ps_imagenet.sh"



######################### Configurations of ps ###########################
# The list of hosts and ports of ps
# Multiple ps not tested yet
PS_HOSTS=( \
  10.236.176.29:2222 \
)



######################### Configurations of workers #######################
# The list of hosts and ports of workers
WORKER_HOSTS=( \
  10.236.176.28:2224 \
  10.236.176.29:2226 \
)
# GPU IDs in corresponding workers
WORKER_DEVICES=( \
  1 \
  2 \
)
# Paths of dataset shards in corresponding workers
DATA_DIR=( \
  ~/dataset/cifar10-data-shard-0-499 \
  ~/dataset/cifar10-data-shard-500-999 \
)

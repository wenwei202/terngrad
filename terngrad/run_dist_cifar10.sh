#!/bin/bash
set -x
set -e
PS=localhost
WORKER1=localhost
WORKER2=localhost

export CUDA_VISIBLE_DEVICES=1
bazel-bin/inception/cifar10_distributed_train \
--optimizer adam \
--initial_learning_rate 0.0002 \
--batch_size 64 \
--num_epochs_per_decay 200 \
--max_steps 300000 \
--seed 123 \
--weight_decay 0.004 \
--net cifar10_alexnet \
--image_size 24 \
--data_dir="$HOME/dataset/cifar10-data-shard-500-999" \
--job_name='worker' \
--task_id=1 \
--ps_hosts="$PS:2222" \
--worker_hosts="${WORKER1}:2224,${WORKER2}:2226" \
--train_dir=/tmp/cifar10_distributed_train &

export CUDA_VISIBLE_DEVICES=0
bazel-bin/inception/cifar10_distributed_train \
--optimizer adam \
--initial_learning_rate 0.0002 \
--batch_size 64 \
--num_epochs_per_decay 200 \
--max_steps 300000 \
--seed 123 \
--weight_decay 0.004 \
--net cifar10_alexnet \
--image_size 24 \
--data_dir="$HOME/dataset/cifar10-data-shard-0-499" \
--job_name='worker' \
--task_id=0 \
--ps_hosts="$PS:2222" \
--worker_hosts="${WORKER1}:2224,${WORKER2}:2226" \
--train_dir=/tmp/cifar10_distributed_train &

export CUDA_VISIBLE_DEVICES=1
bazel-bin/inception/cifar10_distributed_train \
--job_name='ps' \
--task_id=0 \
--ps_hosts="$PS:2222" \
--worker_hosts="${WORKER1}:2224,${WORKER2}:2226" &

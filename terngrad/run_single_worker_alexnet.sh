#!/bin/bash
set -e
set -x

if [ "$#" -ne 7 ]; then
	echo "Illegal number of parameters"
	echo "Usage: $0 PS_HOSTS WORKER_HOSTS JOB_NAME TASK_ID DEVICE DATA_DIR EXPERIMENT_ID"
	exit
fi

# cluster and task
PS_HOSTS=$1
WORKER_HOSTS=$2
JOB_NAME=$3
TASK_ID=$4
DEVICE=$5
DATA_DIR=$6 # dataset location
EXPERIMENT_ID=$7

if [ "${JOB_NAME}" != "worker" ]
then
    echo "JOB_NAME(${JOB_NAME}) is not worker"
    exit
fi

DATASET_NAME=imagenet # imagenet or cifar10
ROOT_WORKSPACE=${HOME}/tmp/ # the location to store tf.summary and logs
FINETUNED_MODEL_PATH=
OPTIMIZER=momentum
NET=alexnet
IMAGE_SIZE=224
GRAD_BITS=32
BASE_LR=0.02
CLIP_FACTOR=0.0 # 0.0 means no clipping
# when GRAD_BITS=1 and FLOATING_GRAD_EPOCH>0, switch to floating gradients every FLOATING_GRAD_EPOCH epoch and then switch back
FLOATING_GRAD_EPOCH=0 # 0 means no switching
WEIGHT_DECAY=0.0005 # default - alexnet/vgg_a/vgg_16:0.0005, inception_v3:0.00004, cifar10_alexnet:0.004
DROPOUT_KEEP_PROB=0.5 # The probability to keep in dropout
MOMENTUM=0.9
SIZE_TO_BINARIZE=9217 # the min size of variable to enable binarizing. 1 means binarizing all variables when GRAD_BITS=1
TRAIN_BATCH_SIZE=128 # batch size per node
NUM_EPOCHS_PER_DECAY=20 # per decay learning rate
MAX_STEPS=185000
SEED=123 # use ${RANDOM} if no duplicable results are required

TRAIN_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_training_data/
INFO_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_info/
if [ ! -d "${INFO_WORKSPACE}" ]; then
  echo "Creating ${INFO_WORKSPACE} ..."
  mkdir -p ${INFO_WORKSPACE}
fi
FOLDER_NAME=${EXPERIMENT_ID}_${JOB_NAME}_${TASK_ID}
TRAIN_DIR=${TRAIN_WORKSPACE}/${FOLDER_NAME}
if [ ! -d "$TRAIN_DIR" ]; then
  echo "Creating ${TRAIN_DIR} ..."
  mkdir -p ${TRAIN_DIR}
fi

export CUDA_VISIBLE_DEVICES=${DEVICE} # specify visible gpus to tensorflow
bazel-bin/inception/${DATASET_NAME}_distributed_train \
--seed ${SEED}  \
--pretrained_model_checkpoint_path "${FINETUNED_MODEL_PATH}" \
--num_epochs_per_decay ${NUM_EPOCHS_PER_DECAY} \
--initial_learning_rate ${BASE_LR} \
--grad_bits ${GRAD_BITS} \
--clip_factor ${CLIP_FACTOR} \
--floating_grad_epoch ${FLOATING_GRAD_EPOCH} \
--weight_decay ${WEIGHT_DECAY} \
--dropout_keep_prob ${DROPOUT_KEEP_PROB} \
--momentum ${MOMENTUM} \
--size_to_binarize ${SIZE_TO_BINARIZE} \
--optimizer ${OPTIMIZER} \
--net ${NET} \
--image_size ${IMAGE_SIZE} \
--batch_size ${TRAIN_BATCH_SIZE} \
--max_steps ${MAX_STEPS} \
--train_dir ${TRAIN_DIR} \
--job_name ${JOB_NAME} \
--task_id ${TASK_ID} \
--ps_hosts ${PS_HOSTS} \
--worker_hosts ${WORKER_HOSTS} \
--data_dir ${DATA_DIR} > ${INFO_WORKSPACE}/${FOLDER_NAME}.log 2>&1 &

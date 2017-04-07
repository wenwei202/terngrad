#!/bin/bash
set -e
set -x

DATASET_NAME=imagenet # imagenet or cifar10
ROOT_WORKSPACE=${HOME}/dataset/results/imagenet # the location to store tf.summary and logs
DATA_DIR=${HOME}/dataset/${DATASET_NAME}-data # dataset location
FINETUNED_MODEL_PATH=
NUM_GPUS=4
# export CUDA_VISIBLE_DEVICES=0,1,2,3 # specify visible gpus to tensorflow
OPTIMIZER=momentum
NET=vgg_a # vgg_a or vgg_16
IMAGE_SIZE=224
GRAD_BITS=32
BASE_LR=0.01
CLIP_FACTOR=0.0 # 0.0 means no clipping
# when GRAD_BITS=1 and FLOATING_GRAD_EPOCH>0, switch to floating gradients every FLOATING_GRAD_EPOCH epoch and then switch back
FLOATING_GRAD_EPOCH=0 # 0 means no switching
WEIGHT_DECAY=0.0005 # default - alexnet/vgg_a/vgg_16:0.0005, inception_v3:0.00004, cifar10_alexnet:0.004
DROPOUT_KEEP_PROB=0.5 # The probability to keep in dropout
MOMENTUM=0.9
SIZE_TO_BINARIZE=1 # the min size of variable to enable binarizing. 1 means binarizing all variables when GRAD_BITS=1
TRAIN_BATCH_SIZE=256 # total batch size
VAL_BATCH_SIZE=25 # set smaller to avoid OOM
NUM_EPOCHS_PER_DECAY=20 # per decay learning rate
MAX_STEPS=370000
VAL_TOWER=0 # -1 for cpu
EVAL_INTERVAL_SECS=900 # seconds to evaluate the accuracy
EVAL_DEVICE="/gpu:0" # specify the device to eval. e.g. "/gpu:1", "/cpu:0"
RESTORE_AVG_VAR=True # use the moving average parameters to eval?
SEED=123 # use ${RANDOM} if no duplicable results are required

if [ ! -d "$ROOT_WORKSPACE" ]; then
  echo "${ROOT_WORKSPACE} does not exsit!"
  exit
fi

TRAIN_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_training_data/
EVAL_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_eval_data/
INFO_WORKSPACE=${ROOT_WORKSPACE}/${DATASET_NAME}_info/
if [ ! -d "${INFO_WORKSPACE}" ]; then
  echo "Creating ${INFO_WORKSPACE} ..."
  mkdir -p ${INFO_WORKSPACE}
fi
current_time=$(date)
current_time=${current_time// /_}
current_time=${current_time//:/-}
FOLDER_NAME=${DATASET_NAME}_${NET}_${IMAGE_SIZE}_${OPTIMIZER}_${GRAD_BITS}_${BASE_LR}_${CLIP_FACTOR}_${FLOATING_GRAD_EPOCH}_${WEIGHT_DECAY}_${MOMENTUM}_${SIZE_TO_BINARIZE}_${TRAIN_BATCH_SIZE}_${NUM_GPUS}_${current_time}
TRAIN_DIR=${TRAIN_WORKSPACE}/${FOLDER_NAME}
EVAL_DIR=${EVAL_WORKSPACE}/${FOLDER_NAME}
if [ ! -d "$TRAIN_DIR" ]; then
  echo "Creating ${TRAIN_DIR} ..."
  mkdir -p ${TRAIN_DIR}
fi
if [ ! -d "$EVAL_DIR" ]; then
  echo "Creating ${EVAL_DIR} ..."
  mkdir -p ${EVAL_DIR}
fi

bazel-bin/inception/${DATASET_NAME}_eval \
--eval_interval_secs ${EVAL_INTERVAL_SECS} \
--device ${EVAL_DEVICE} \
--restore_avg_var ${RESTORE_AVG_VAR} \
--data_dir ${DATA_DIR} \
--net ${NET} \
--image_size ${IMAGE_SIZE} \
--batch_size ${VAL_BATCH_SIZE} \
--max_steps ${MAX_STEPS} \
--checkpoint_dir ${TRAIN_DIR} \
--tower ${VAL_TOWER} \
--eval_dir ${EVAL_DIR} >  ${INFO_WORKSPACE}/eval_${FOLDER_NAME}_info.txt 2>&1 &

bazel-bin/inception/${DATASET_NAME}_train \
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
--num_gpus ${NUM_GPUS} \
--batch_size ${TRAIN_BATCH_SIZE} \
--max_steps ${MAX_STEPS} \
--train_dir ${TRAIN_DIR} \
--data_dir ${DATA_DIR} > ${INFO_WORKSPACE}/training_${FOLDER_NAME}_info.txt 2>&1 &

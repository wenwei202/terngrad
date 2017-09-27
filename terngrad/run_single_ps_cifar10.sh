#!/bin/bash
. ~/.bashrc
set -e
set -x

if [ "$#" -ne 5 ]; then
	echo "Illegal number of parameters"
	echo "Usage: $0 PS_HOSTS WORKER_HOSTS JOB_NAME TASK_ID EXPERIMENT_ID"
	exit
fi

# cluster and task
PS_HOSTS=$1
WORKER_HOSTS=$2
JOB_NAME=$3
TASK_ID=$4
EXPERIMENT_ID=$5

if [ "${JOB_NAME}" != "ps" ]
then
    echo "JOB_NAME(${JOB_NAME}) is not ps"
    exit
fi

DATASET_NAME=cifar10 # imagenet or cifar10
INFO_WORKSPACE=${HOME}/tmp/${DATASET_NAME}_info/
if [ ! -d "${INFO_WORKSPACE}" ]; then
  echo "Creating ${INFO_WORKSPACE} ..."
  mkdir -p ${INFO_WORKSPACE}
fi
LOG_FILE=${INFO_WORKSPACE}/${EXPERIMENT_ID}_${JOB_NAME}_${TASK_ID}.log

bazel-bin/inception/${DATASET_NAME}_distributed_train \
--job_name ${JOB_NAME} \
--task_id ${TASK_ID} \
--ps_hosts ${PS_HOSTS} \
--worker_hosts ${WORKER_HOSTS}  > ${LOG_FILE} 2>&1 &

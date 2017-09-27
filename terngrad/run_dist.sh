#!/bin/bash
set -e
#set -x

# The executable path. Must be the same across all nodes
WORKSPACE="~/github/users/wenwei202/terngrad/terngrad"

#WORKER_SCRIPT="./run_single_worker_alexnet.sh"
#PS_SCRIPT="./run_single_ps_imagenet.sh"
WORKER_SCRIPT="./run_single_worker_cifarnet.sh"
PS_SCRIPT="./run_single_ps_cifar10.sh"

PS_HOSTS=( \
  localhost:2222 \
)
WORKER_HOSTS=( \
  localhost:2224 \
  localhost:2226 \
)
WORKER_DEVICES=( \
  0 \
  1 \
)
DATA_DIR=( \
  ~/dataset/cifar10-data-shard-0-499 \
  ~/dataset/cifar10-data-shard-500-999 \
)

WORKER_STRING=$(echo ${WORKER_HOSTS[*]} | sed 's/ /,/g') 
PS_STRING=$(echo ${PS_HOSTS[*]} | sed 's/ /,/g') 
EXPERIMENT_ID=$(date)
EXPERIMENT_ID=${EXPERIMENT_ID// /_}
EXPERIMENT_ID=${EXPERIMENT_ID//:/-}

PS_NUM=${#PS_HOSTS[@]}
WORKER_NUM=${#WORKER_HOSTS[@]}
DEVICE_NUM=${#WORKER_DEVICES[@]}
DATA_NUM=${#DATA_DIR[@]}
if [ ${WORKER_NUM} -ne ${DEVICE_NUM}  ]
then
  echo "The number of workers (${WORKER_NUM}) does not match the number of devices (${DEVICE_NUM})"
  exit
fi
if [ ${WORKER_NUM} -ne ${DATA_NUM}  ]
then
  echo "The number of workers (${WORKER_NUM}) does not match the number of data paths (${DATA_NUM})"
  exit
fi

echo "${PS_NUM} ps hosts: ${PS_STRING}"
echo "${WORKER_NUM} worker hosts: ${WORKER_STRING}"

# start workers
task_id=0
for HOST in ${WORKER_HOSTS[*]}; do
  worker=$(echo ${HOST} |cut -d':' -f1)
  ssh ${worker} "hostname; \
    cd ${WORKSPACE}; \
    pwd; \
    ${WORKER_SCRIPT} ${PS_STRING} ${WORKER_STRING} worker ${task_id} ${WORKER_DEVICES[$task_id]} ${DATA_DIR[$task_id]} ${EXPERIMENT_ID}"
  task_id=`expr $task_id + 1`
done

# start ps
task_id=0
for HOST in ${PS_HOSTS[*]}; do
  ps=$(echo ${HOST} |cut -d':' -f1)
  ssh ${ps} "hostname; \
             cd ${WORKSPACE}; \
             pwd; \
             ${PS_SCRIPT} ${PS_STRING} ${WORKER_STRING} ps ${task_id} ${EXPERIMENT_ID}"
  task_id=`expr $task_id + 1`
done


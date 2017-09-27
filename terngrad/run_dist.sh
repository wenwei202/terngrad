#!/bin/bash
set -e
#set -x

. ./config_dist.sh

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


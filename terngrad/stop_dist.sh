#!/bin/bash
set -e
set -x

. ./config_dist.sh 

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


# stop workers
for HOST in ${WORKER_HOSTS[*]}; do
  worker=$(echo ${HOST} |cut -d':' -f1)
  ssh ${worker} "hostname; \
    cd ${WORKSPACE}; \
    pwd; \
    ./kill_local.sh "
done

# stop ps
for HOST in ${PS_HOSTS[*]}; do
  ps=$(echo ${HOST} |cut -d':' -f1)
  ssh ${ps} "hostname; \
             cd ${WORKSPACE}; \
             pwd; \
             ./kill_local.sh "
done


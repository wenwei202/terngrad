#!/bin/bash
#set -e
#set -x

if [ "$#" -ne 3 ]; then
        echo "Illegal number of parameters"
        echo "Usage: $0 DATA_DIR WORKER_NUM WORKER_ID"
        exit
fi

DATA_DIR=$1
WORKER_NUM=$2
WORKER_ID=$3

if [ "${WORKER_ID}" -ge ${WORKER_NUM} ] || [ "${WORKER_ID}" -lt 0 ] ; then
        echo "WORKER_ID between [0,WORKER_NUM)"
        exit
fi

SPLIT_DIR=${DATA_DIR}/worker_${WORKER_ID}_of_${WORKER_NUM}
if [ ! -d "$SPLIT_DIR" ]; then
  export total_files=$( ls -l ${DATA_DIR}/train-* | wc -l )
  split_size=$( expr ${total_files} / ${WORKER_NUM}  )
  remainder=$( expr ${total_files} % ${WORKER_NUM}  )
  if [ "${remainder}" -ne 0 ]; then
    echo "Dataset cannot be evenly split"
    exit
  fi
  echo "Splitting to ${SPLIT_DIR} ..."
  mkdir ${SPLIT_DIR}
  cd ${SPLIT_DIR}
  files=$( ls -dl ${DATA_DIR}/train-*|head -n $( expr $( expr ${WORKER_ID} + 1 ) * ${split_size}  ) | tail -n ${split_size} | awk '{print $9}')

  for file in ${files}; do
   ln -s ${file};
  done

else
  echo "${SPLIT_DIR} exists."
fi


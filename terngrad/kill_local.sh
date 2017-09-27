#!/bin/bash
threadid=$( ps aux | grep python | grep distributed_train | grep ${USER} | awk '{print $2}')
if [[ "$threadid" =~ ^-?[0-9]+.*$ ]] ; 
then
  kill $threadid
else
  echo "Stopped already."
fi

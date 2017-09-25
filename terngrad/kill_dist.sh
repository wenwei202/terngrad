#!/bin/bash
ps aux | grep python | grep distributed_train | grep ${USER} | awk '{print $2}' | xargs kill

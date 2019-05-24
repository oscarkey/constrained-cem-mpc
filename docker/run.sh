#!/bin/bash

gpu=$1

DIR=/project

if hash nvidia-docker 2>/dev/null; then
  cmd=nvidia-docker
else
  cmd=docker
fi

NV_GPU="$gpu" ${cmd} run -d \
        --net host \
        --name ${2} \
        -v `pwd`/:$DIR:rw \
        -t constrained-cem-$USER \
        ${@:3}
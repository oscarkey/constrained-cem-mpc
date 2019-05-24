#!/usr/bin/env bash

set -e

if [ hash nvidia-docker 2>/dev/null ]; then
  CMD=nvidia-docker
else
  CMD=docker
fi

$CMD build -f docker/Dockerfile -t oscar-cem-$USER --build-arg UID=$UID .
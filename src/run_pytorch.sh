#!/bin/bash

die() { [ -n "$1" ] && echo -e "Error: $1\n" >&2; [ -z "$1" ]; exit;}

[ $# -ne 1 ] && die "Takes exactly one argument (python file to run)"

docker run --rm -it --init \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD/../:/app" \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -w "/app" \
  ufoym/deepo:pytorch python3 ./src/$1

#!/bin/bash

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

RAY_TEMP_DIR="${RAY_TMPDIR:-/tmp/ray}"
mkdir -p "$RAY_TEMP_DIR"

# Absence means Ray masks one GPU per TorchTrain worker. Set to 1 only for the
# legacy Shifter/NCCL workaround documented in start_interactive_ray.sh.
if [ "${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-}" = "0" ]; then
    unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES
fi

echo "starting ray worker node"
ray start --address "$1" --temp-dir="$RAY_TEMP_DIR"
sleep infinity

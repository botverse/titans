#!/bin/bash

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# If we're in a datacenter environment (you may want to customize this check)
if [[ "$(hostname)" == *"cluster"* ]] || [[ "$(hostname)" == *"node"* ]] || [[ "$(hostname)" == *"compute"* ]] || [[ "$(hostname)" == *"gpu"* ]]; then
    # Use all available GPUs in datacenter
    GPUS_TO_USE=$NUM_GPUS
else
    # Use single GPU for local development
    GPUS_TO_USE=1
fi

# Ensure we don't try to use more GPUs than available
if [ $GPUS_TO_USE -gt $NUM_GPUS ]; then
    GPUS_TO_USE=$NUM_GPUS
fi

# Run training
if [ $GPUS_TO_USE -gt 1 ]; then
    echo "Running in distributed mode with $GPUS_TO_USE GPUs"
    torchrun \
        --nproc_per_node=$GPUS_TO_USE \
        --master_port=29500 \
        distil.py
else
    echo "Running in single GPU mode"
    python distil.py
fi 
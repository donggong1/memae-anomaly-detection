#!/bin/bash
time python script_training.py \
    --ModelName MemAE \
    --ModelSetting Conv3DSpar \
    --Dataset UCSD_P2_256 \
    --MemDim 2000 \
    --EntropyLossWeight 0.0002 \
    --ShrinkThres 0.0025 \
    --BatchSize 10 \
    --Seed 1 \
    --SaveCheckInterval 1 \
    --IsTbLog True \
    --IsDeter True \
    --DataRoot ./datasets/processed/ \
    --ModelRoot ./results/ \
    --Suffix Non
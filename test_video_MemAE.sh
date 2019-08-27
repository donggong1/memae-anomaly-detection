#!/bin/bash
# testing MemAE on video dataset
sudo /home/dong/.conda/envs/py36pt040/bin/python script_testing.py \
    --ModelName MemAE \
    --ModelSetting Conv3DSpar \
    --Dataset UCSD_P2_256 \
    --MemDim 2000 \
    --EntropyLossWeight 0.0002 \
    --ShrinkThres 0.0025 \
    --ModelRoot /media/dong/Data1/proj_anomaly/github_testing_root/memae_trained_models/ \
    --DataRoot /media/dong/Data1/proj_anomaly/dataset/ \
    --OutRoot /media/dong/Data1/proj_anomaly/github_testing_root/results/ \
    --Suffix Non

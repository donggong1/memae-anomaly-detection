#!/bin/bash
# testing AE video
sudo /home/dong/.conda/envs/py36pt040/bin/python script_testing.py \
    --ModelName AE \
    --ModelSetting Conv3D \
    --Dataset UCSD_P2_256 \
    --ModelRoot /media/dong/Data1/proj_anomaly/github_testing_root/memae_trained_models/ \
    --DataRoot /media/dong/Data1/proj_anomaly/dataset/ \
    --OutRoot /media/dong/Data1/proj_anomaly/github_testing_root/results/ \
    --Suffix Non

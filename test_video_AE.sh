#!/bin/bash
# testing AE video
sudo /home/dong/.conda/envs/py36pt040/bin/python script_testing.py \
    --ModelName AE \
    --ModelSetting Conv3D \
    --Dataset UCSD_P2_256 \
    --ModelRoot ./memae_models/ \
    --DataRoot ./dataset/ \
    --OutRoot ./results/ \
    --Suffix Non

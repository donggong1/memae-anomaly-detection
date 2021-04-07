import matplotlib.pyplot as plt
import os
import utils

root_path = './'

data_name = 'UCSD_P2_256'
data_path = os.path.join(root_path, 'dataset', data_name)

res_root = os.path.join(root_path, 'video-proj_results')
model_setting = 'MemAE_Conv3DSpar_UCSD_P2_256_MemDim2000_EntW0.0002_ShrThres0.0025_Non'
res_path = os.path.join(res_root, 'res_'+model_setting)

auc = utils.eval_video(data_path, res_path, is_show=False)

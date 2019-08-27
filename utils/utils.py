from __future__ import absolute_import, print_function
import os
import numpy as np

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def crop_image(img, s):
    # s: cropping size
    if(s>0):
        if(len(img.shape)==3):
            # F(or C) x H x W
            return img[:, s:(-s), s:(-s)]
        elif(len(img.shape)==4):
            # F x C x H x W
            return img[:, :, s:(-s), s:(-s)]
    else:
        return img

def tensor2numpy(tensor_in):
    """Transfer pythrch tensor to numpy array"""
    nparray_out = (tensor_in.data).cpu().numpy()
    return nparray_out

def get_subdir_list(path, is_sort=True):
    subdir_list = [name for name in os.listdir(path) \
                  if os.path.isdir(os.path.join(path, name))]
    if(is_sort):
        subdir_list.sort()
    return subdir_list

def get_file_list(path, is_sort=True):
    file_list = [name for name in os.listdir(path) \
                  if os.path.isfile(os.path.join(path, name))]
    if(is_sort):
        file_list.sort()
    return file_list

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor_in):
        t_out = tensor_in.clone()
        for t, m, s in zip(t_out, self.mean, self.std):
            t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
        return t_out

def vframes2imgs(frames_in, step=1, batch_idx = 0):
    frames_np = tensor2numpy(frames_in)
    frames_shape = frames_np.shape
    if(len(frames_shape)==4):
        # for 2D convolution, N x F (C) x H x W
        frames_np = frames_np[batch_idx,:,:,:]
        if step==1:
            return frames_np
        elif step>1:
            num_frame = frames_np.shape[0]
            idx_list = range(1, num_frame, step)
            return frames_np[idx_list,:,:]
    elif(len(frames_shape)==5):
        # for 3D convolution, N x C x F x H x W
        frames_np = frames_np[batch_idx, :, :, :, :]
        frames_np = np.transpose(frames_np, (1,0,2,3))
        if step==1:
            # all frames
            return frames_np
        elif step>1:
            # select frames based on the step interval
            num_frame = frames_shape[2]
            idx_list = range(1, num_frame, step)
            return frames_np[idx_list, :, :, :]

def get_model_setting(opt):
    if(opt.ModelName == 'MemAE'):
        model_setting = opt.ModelName + '_' + opt.ModelSetting + '_' + opt.Dataset + '_MemDim' + str(opt.MemDim) \
                        + '_EntW' + str(opt.EntropyLossWeight) + '_ShrThres' + str(opt.ShrinkThres) \
                        + '_' + opt.Suffix
    elif(opt.ModelName == 'AE'):
        model_setting = opt.ModelName + '_' + opt.ModelSetting + '_' + opt.Dataset \
                        + '_' + opt.Suffix
    else:
        model_setting = ''
        print('Wrong Model Name.')
    return model_setting

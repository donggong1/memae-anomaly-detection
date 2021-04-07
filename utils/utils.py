from __future__ import absolute_import, print_function
import os
import numpy as np
import torch
import random

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
        elif(len(img.shape)==5):
            # N x F x C x H x W
            return img[:, :, :, s:(-s), s:(-s)]
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
        # NCFHW or CFHW
        t_out = tensor_in.clone()
        s = t_out.shape
        if(len(s)==5):
            channel_num = s[1]
            # TODO?: make efficient
            for i in range(channel_num):
                t_out[:, i, :, :, :] = t_out[:, i, :, :, :]*self.std[i] + self.mean[i]
        elif(len(s)==4):
            channel_num = s[0]
            for i in range(channel_num):
                t_out[i, :, :, :] = t_out[i, :, :, :]*self.std[i] + self.mean[i]
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

# NxCxFxHxW (Tensor) -> NxFxCxHxW (np)
def btv2btf(frames_in):
    frames_np = tensor2numpy(frames_in)
    frames_shape = frames_np.shape
    if(len(frames_shape)==5):
        # N x C x F x H x W
        frames_np = np.transpose(frames_np, (0,2,1,3,4))
    return frames_np

def get_model_setting(opt):
    if(opt.ModelName == 'MemAE'):
        model_setting = opt.ModelName + '_' + opt.ModelSetting + '_' + opt.Dataset + '_MemDim' + str(opt.MemDim) \
                        + '_EntW' + str(opt.EntropyLossWeight) + '_ShrThres' + str(opt.ShrinkThres) \
                        + '_Seed' + str(opt.Seed) + '_' + opt.Suffix
    elif(opt.ModelName == 'AE'):
        model_setting = opt.ModelName + '_' + opt.ModelSetting + '_' + opt.Dataset \
                        + '_' + opt.Suffix
    else:
        model_setting = ''
        print('Wrong Model Name.')
    return model_setting

def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def seed(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
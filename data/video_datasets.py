from __future__ import print_function, absolute_import
import torch
from torch.utils.data import Dataset
import os, os.path
import scipy.io as sio
from skimage import io
from torchvision import transforms
import numpy as np


# Video index files are organized in correlated sub folders.
# N x C x T x H x W
class VideoDataset(Dataset):
    def __init__(self, idx_root, frame_root, use_cuda=False, transform=None):
        # dir name
        self.idx_root = idx_root
        self.frame_root = frame_root

        # video_name_list, subdir names
        self.video_list = [name for name in os.listdir(self.idx_root) \
                              if os.path.isdir(os.path.join(self.idx_root, name))]
        self.video_list.sort()

        #
        self.idx_path_list = []
        for ite_vid in range(len(self.video_list)):
            video_name = self.video_list[ite_vid]
            # idx file name list
            idx_file_name_list = [name for name in os.listdir(os.path.join(self.idx_root, video_name)) \
                              if os.path.isfile(os.path.join(self.idx_root, video_name, name))]
            idx_file_name_list.sort()
            # idx file path list
            idx_file_list = [self.idx_root + '/' + video_name + '/' + file_name for file_name in idx_file_name_list]
            # merger lists
            self.idx_path_list = self.idx_path_list + idx_file_list
        self.idx_num = len(self.idx_path_list)
        self.use_cuda = use_cuda
        self.transform = transform

    def __len__(self):
        return self.idx_num

    def __getitem__(self, item):
        """ get a video clip with stacked frames indexed by the (idx) """
        idx_path = self.idx_path_list[item] # idx file path
        idx_data = sio.loadmat(idx_path)    # idx data
        v_name = idx_data['v_name'][0]  # video name
        frame_idx = idx_data['idx'][0, :]  # frame index list for a video clip

        v_dir = self.frame_root + v_name

        tmp_frame = io.imread(os.path.join(v_dir, ('%03d' % frame_idx[0]) + '.jpg'))
        tmp_frame_shape = tmp_frame.shape
        frame_cha_num = len(tmp_frame_shape)
        # h = tmp_frame_shape[0]
        # w = tmp_frame_shape[1]
        if frame_cha_num==3:
            c = tmp_frame_shape[2]
        elif frame_cha_num==2:
            c = 1
        # each sample is concatenation of the indexed frames
        if self.transform:
            if c==3:
                frames = torch.cat([self.transform(
                    io.imread(os.path.join(v_dir, ('%03d' % i) + '.jpg'))).unsqueeze(1) for i
                                    in frame_idx], 1)
            elif c==1:
                frames = torch.cat([self.transform(
                    np.expand_dims(io.imread(os.path.join(v_dir, ('%03d' % i) + '.jpg')), axis=2)).unsqueeze(1) for i
                                    in frame_idx], 1)
        else:
            tmp_frame_trans = transforms.ToTensor() # trans Tensor
            if c==3:
                frames = torch.cat([tmp_frame_trans(
                    io.imread(os.path.join(v_dir, ('%03d' % i) + '.jpg'))).unsqueeze(1) for i
                                    in frame_idx], 1)
            elif c==1:
                frames = torch.cat([tmp_frame_trans(
                    np.expand_dims(io.imread(os.path.join(v_dir, ('%03d' % i) + '.jpg'), axis=2))).unsqueeze(1) for i
                                    in frame_idx], 1)
        return item, frames



###
# All video index files are in one dir.
# N x C x T x H x W
class VideoDatasetOneDir(Dataset):
    def __init__(self, idx_dir, frame_root, is_testing=False, use_cuda=False, transform=None):
        self.idx_dir = idx_dir
        self.frame_root = frame_root
        self.idx_name_list = [name for name in os.listdir(self.idx_dir) \
                              if os.path.isfile(os.path.join(self.idx_dir, name))]
        self.idx_name_list.sort()
        self.use_cuda = use_cuda
        self.transform = transform
        self.is_testing = is_testing

    def __len__(self):
        return len(self.idx_name_list)

    def __getitem__(self, item):
        """ get a video clip with stacked frames indexed by the (idx) """
        idx_name = self.idx_name_list[item]
        idx_data = sio.loadmat(os.path.join(self.idx_dir, idx_name))
        v_name = idx_data['v_name'][0]     # video name
        frame_idx = idx_data['idx'][0,:]   # frame index list for a video clip
        v_dir = self.frame_root
        #
        tmp_frame = io.imread(os.path.join(v_dir, ('%03d' % frame_idx[0]) + '.jpg'))

        tmp_frame_shape = tmp_frame.shape
        frame_cha_num = len(tmp_frame_shape)
        h = tmp_frame_shape[0]
        w = tmp_frame_shape[1]
        if frame_cha_num == 3:
            c = tmp_frame_shape[2]
        elif frame_cha_num == 2:
            c = 1
        # each sample is concatenation of the indexed frames
        if self.transform:
            frames = torch.cat([self.transform(
                io.imread(os.path.join(v_dir, ('%03d' % i) + '.jpg')).reshape(h, w, c)).resize_(c, 1, h, w) for i
                                in frame_idx], 1)
        else:
            tmp_frame_trans = transforms.ToTensor() # trans Tensor
            frames = torch.cat([self.transform(
                io.imread(os.path.join(v_dir, ('%03d' % i) + '.jpg')).reshape(h, w, c)).resize_(c, 1, h, w) for i
                                in frame_idx], 1)

        return item, frames

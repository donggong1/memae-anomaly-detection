# baseline AE models for video data
from __future__ import absolute_import, print_function
import torch
from torch import nn

class AutoEncoderCov3D(nn.Module):
    def __init__(self, chnum_in):
        super(AutoEncoderCov3D, self).__init__()
        self.chnum_in = chnum_in # input channel number is 1;
        feature_num = 128
        feature_num_2 = 96
        feature_num_x2 = 256
        self.encoder = nn.Sequential(
            nn.Conv3d(self.chnum_in, feature_num_2, (3,3,3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_2, feature_num, (3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num, feature_num_x2, (3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(feature_num_x2, feature_num_x2, (3,3,3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(feature_num_x2, feature_num_x2, (3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_num_x2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_x2, feature_num, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(feature_num),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num, feature_num_2, (3,3,3), stride=(2,2,2), padding=(1,1,1), output_padding=(1,1,1)),
            nn.BatchNorm3d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(feature_num_2, self.chnum_in, (3,3,3), stride=(1,2,2), padding=(1,1,1), output_padding=(0,1,1))
        )

    def forward(self, x):
        f = self.encoder(x)
        out = self.decoder(f)
        return out



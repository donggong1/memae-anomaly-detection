import argparse
from .testing_options import str2bool


class TrainOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--UseCUDA', help='Use CUDA?', type=str2bool, nargs='?', default=True)
        parser.add_argument('--NumWorker', help='num of worker for dataloader', type=int, default=1)
        parser.add_argument('--Mode', help='script mode', choices=['train', 'eval'], default='train')
        parser.add_argument('--ModelName', help='AE/MemAE', type=str, default='MemAE')
        parser.add_argument('--ModelSetting',
                            help='Conv3D/Conv3DSpar',
                            type=str,
                            default='Conv3DSpar')
        parser.add_argument('--Seed', type=int, default=1)
        parser.add_argument('--IsDeter', type=str2bool, help='set False for efficiency', default=True)
        parser.add_argument('--IsTbLog', type=str2bool, default=True)
        parser.add_argument('--Dataset', help='Dataset', type=str, default='UCSD_P2_256')
        parser.add_argument('--ImgChnNum', help='image channel', type=int, default=1)
        parser.add_argument('--FrameNum', help='frame num for video clip', type=int, default=16)
        parser.add_argument('--BatchSize', help='training batchsize', type=int, default=14)
        parser.add_argument('--LR', help='learning rate', type=float, default=1e-4)
        parser.add_argument('--EpochNum', help='max epoch num', type=int, default=100) 
        parser.add_argument('--MemDim', help='Memory Dimention', type=int, default=2000)
        parser.add_argument('--EntropyLossWeight', help='EntropyLossWeight', type=float, default=0.0002)
        parser.add_argument('--ShrinkThres', help='ShrinkThres', type=float, default=0.0025)
        # ##
        parser.add_argument('--TextLogInterval', help='text log ite interval', type=int, default=1)
        parser.add_argument('--SnapInterval', help='snap saving ite interval', type=int, default=100)
        parser.add_argument('--TBImgLogInterval', help='text log ite interval', type=int, default=200)
        parser.add_argument('--SaveCheckInterval', help='checkpoint saving epoch interval', type=int, default=1)
        ##
        parser.add_argument('--DataRoot', help='DataPath', type=str, default='./dataset/')
        parser.add_argument('--ModelRoot', help='Path for saving model', type=str, default='./models/')
        ##
        parser.add_argument('--Suffix', help='Suffix', type=str, default='Non')

        self.initialized = True
        self.parser = parser
        return parser

    def print_options(self, opt):
        # This function is adapted from 'cycleGAN' project.
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        self.message = message

    def parse(self, is_print):
        parser = self.initialize()
        opt = parser.parse_args()
        if(is_print):
            self.print_options(opt)
        self.opt = opt
        return self.opt




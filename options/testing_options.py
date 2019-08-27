import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class TestOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--UseCUDA', help='Use CUDA?', type=str2bool, nargs='?', default=True)
        parser.add_argument('--ModelName', help='AE/MemAE', type=str, default='MemAE')
        parser.add_argument('--ModelSetting',
                            help='Conv3D/Conv3DSpar',
                            type=str,
                            default='Conv3DSpar')  # give the layer details later
        parser.add_argument('--Dataset', help='Dataset', type=str, default='UCSD_P2_256')
        parser.add_argument('--ImgChnNum', help='image channel', type=int, default=1)
        parser.add_argument('--FrameNum', help='frame num for VIDEO clip', type=int, default=16)
        parser.add_argument('--BatchSize', help='BatchSize', type=int, default=1)
        parser.add_argument('--MemDim', help='Memory Dimention', type=int, default=2000)
        parser.add_argument('--EntropyLossWeight', help='EntropyLossWeight', type=float, default=0.0002)
        parser.add_argument('--ShrinkThres', help='ShrinkThres', type=float, default=0.0025)
        ##
        parser.add_argument('--ModelRoot', help='Path and name for trained model.', type=str, default='./memae_models/')
        parser.add_argument('--DataRoot', help='DataPath', type=str, default='./dataset/')
        parser.add_argument('--OutRoot', help='Path for output', type=str, default='./results/')
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

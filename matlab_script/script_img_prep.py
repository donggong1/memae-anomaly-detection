# preprocessing UCSD data frames
import os
from PIL import Image

# data_root_path = '/data/root/path/'
in_path = os.path.join('datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped2/')
out_path = os.path.join('datasets/processed/UCSD_P2_256/')

opts = {"is_gray": True, "maxs": 320, "outsize": [256, 256], "img_type": 'tif'}


def trans_img2img(inpath, outpath, opts):
    """ preprocess images: trans images to images """
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    img_list = os.listdir(in_path)

    for img in img_list:
        print(img)


if not os.path.exists(out_path):
    os.mkdir(out_path)

sub_dir_list = {'Train', 'Test'}
file_num_list = [16, 12]


def main():
    trans_img2img(v_path, v_out_path, opts)


if __name__ == '__main__':
    main()

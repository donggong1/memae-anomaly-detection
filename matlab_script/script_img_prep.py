# preprocessing UCSD data frames
import os
from PIL import Image
import numpy as np
from scipy import io


def trans_tif2jpg(in_path, out_path, opts):
    """ preprocess images: trans tif images to jpg images """
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)

    img_list = [
        file for file in os.listdir(in_path) if file.endswith(opts["img_type"])
    ]
    for img in img_list:
        img_path = os.path.join(in_path, img)
        img_pil = Image.open(img_path).convert('L')
        if "outsize" in opts:
            img_pil = img_pil.resize(opts["outsize"], Image.ANTIALIAS)
        img_pil.save(os.path.join(out_path, img[:-4] + ".jpg"))

    # for img in img_list:
    #     print(img)


def trans_img2label(gt_in_path, outpath):
    idx = gt_in_path[-6:-3]
    gt_out_path = os.path.join(outpath, 'Test')
    if not os.path.exists(gt_out_path):
        os.makedirs(gt_out_path)
    # print(idx)
    file_list = [
        file for file in os.listdir(gt_in_path) if gt_in_path.endswith('.bmp')
    ]
    label = [0] * len(file_list)
    for i in range(len(file_list)):
        img_pil = Image.open(os.path.join(gt_in_path, file_list[i]))
        if np.asarray(img_pil).sum() < 1:
            label[i] = 0
        else:
            label[i] = 1
    io.savemat(gt_out_path + idx + '.mat', {"l": label})


def test_trans_tif2jpg():
    trans_tif2jpg('datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train/Train001',
                  'datasets/processed/UCSD_P2_256/Train/Train001', {
                      "is_gray": True,
                      "maxs": 320,
                      "outsize": [256, 256],
                      "img_type": 'tif'
                  })


def test_trans_img2label():
    pass


def main():

    sub_dir_list = ['Train', 'Test']
    # file_num_list = [16, 12]
    # data_root_path = '/data/root/path/'
    in_path = 'datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped2/'
    out_path = 'datasets/processed/UCSD_P2_256/'
    opts = {
        "is_gray": True,
        "maxs": 320,
        "outsize": [256, 256],
        "img_type": 'tif'
    }
    sub_dir_list = ['Train', 'Test']

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for dir in sub_dir_list:
        # for dirpath, dirnames, filenames in os.walk(os.path.join(in_path,
        #                                                          dir)):
        #     print(dirpath, dirnames, filenames)
        sub_dir = os.path.join(in_path, dir)
        for file in [
                path for path in os.listdir(sub_dir)
                if os.path.isdir(os.path.join(sub_dir, path))
        ]:
            print(file)
            if not file.endswith("_gt"):
                trans_tif2jpg(os.path.join(in_path, dir, file),
                              os.path.join(out_path, dir, file), opts)
            else:
                trans_img2label(os.path.join(in_path, dir, file),
                                os.path.join(out_path, 'Test_gt/'))


def index_gen():
    in_path = 'datasets/processed/UCSD_P2_256/'
    frame_file_type = 'jpg'
    clip_len = 16
    skip_step = 1
    # 15 clip range
    clip_rng = clip_len * skip_step - 1
    # 15
    overlap_shift = clip_len - 1
    sub_dir_list = ['Train', 'Test']
    for sub_dir_name in sub_dir_list:
        sub_in_path = os.path.join(in_path, sub_dir_name)
        idx_out_path = os.path.join(in_path, sub_dir_name + '_idx/')
        if not os.path.exists(idx_out_path):
            os.makedirs(idx_out_path)

        for dir in [
                path for path in os.listdir(sub_in_path)
                if os.path.isdir(os.path.join(sub_in_path, path))
        ]:
            frame_list = [
                os.path.join(dir, file)
                for file in os.listdir(os.path.join(sub_in_path, dir))
                if file.endswith(frame_file_type)
            ]
            frame_num = len(frame_list)
            s_list = range(1, len(frame_list), clip_rng + 1 - overlap_shift)

            e_list = [i + clip_rng for i in s_list]

            s_list = [i for i in s_list if i + clip_rng <= frame_num]

            e_list = [i for i in e_list if i <= frame_num]

            video_sub_dir_out_path = os.path.join(idx_out_path, dir)
            if not os.path.exists(video_sub_dir_out_path):
                os.makedirs(video_sub_dir_out_path)

            for j in range(0, len(s_list)):
                idx = range(s_list[j], e_list[j]+1, skip_step)
                io.savemat(
                    os.path.join(video_sub_dir_out_path,
                                 dir + "_i%03d.mat" % j), {
                                     'v_name': dir,
                                     'idx': idx
                    })


def test_index_gen():
    print(
        io.loadmat(
            "dataset/processed/UCSD_P2_256/Train_idx/Train001/Train001_i001.mat"
        ))
    print(
        io.loadmat(
            "dataset/processed/UCSD_P2_256/Train_idx/Train001/Train001_i002.mat"
        ))


if __name__ == '__main__':
    # main()
    index_gen()
    # test_index_gen()

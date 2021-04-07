from __future__ import absolute_import, print_function
import os
import scipy.io as sio
import numpy as np
import sklearn.metrics as skmetr
import utils
import matplotlib.pyplot as plt


def eval_video(data_path, res_path, is_show=False):
    gt_path = os.path.join(data_path, 'Test_gt/')

    ###
    video_list = utils.get_file_list(gt_path, is_sort=True)
    video_num = len(video_list)

    gt_labels_list = []
    res_prob_list = []
    res_prob_list_org = []

    ###
    for vid_ite in range(video_num):
        gt_file_name = video_list[vid_ite]

        p_idx = [pos for pos, char in enumerate(gt_file_name) if char == '.']
        video_name = gt_file_name[0:p_idx[0]]
        print('Eval: %d/%d-%s' % (vid_ite + 1, video_num, video_name))
        # res file name
        res_file_name = video_name + '.npy'
        # gt file and res file - path
        gt_file_path = os.path.join(gt_path, gt_file_name)
        res_file_path = os.path.join(res_path, res_file_name)
        #     print(gt_file_path)
        #     print(res_file_path)

        # read data
        gt_labels = sio.loadmat(gt_file_path)['l'][0]  # ground truth labels
        res_prob = np.load(res_file_path)  # estimated probability scores
        #     res_prob = np.log10(res_prob)-2*np.log10(255)

        res_prob_list_org = res_prob_list_org + list(res_prob)
        gt_labels_res = gt_labels[8:-7]

        # normalize regularity score
        res_prob_norm = res_prob - res_prob.min()
        res_prob_norm = 1 - res_prob_norm / res_prob_norm.max()

        ##
        gt_labels_list = gt_labels_list + list(1 - gt_labels_res + 1)
        res_prob_list = res_prob_list + list(res_prob_norm)

    fpr, tpr, thresholds = skmetr.roc_curve(np.array(gt_labels_list), np.array(res_prob_list), pos_label=2)
    auc = skmetr.auc(fpr, tpr)
    print(('auc:%f' % auc))

    # output_path = os.path.join(res_path,)
    output_path = res_path
    sio.savemat(os.path.join(output_path, video_name + '_gt_label.mat'),  {'gt_labels_list': np.double(gt_labels_res)}  )
    sio.savemat(os.path.join(output_path, video_name + '_est_label.mat'), {'est_labels_list': np.double(res_prob_list)} )
    acc_file = open(os.path.join(output_path, 'acc.txt'), 'w')
    acc_file.write( '{}\nAUC: {}\n'
              .format(data_path, auc ))
    acc_file.close()

    if(is_show):
        plt.figure()
        plt.plot(gt_labels_list)
        plt.plot(res_prob_list)

    return auc
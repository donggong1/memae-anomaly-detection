% preprocessing UCSD data frames
addpath('utils')

data_root_path = '/data/root/path/';
in_path = [data_root_path, 'datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped2/'];
out_path = [data_root_path, 'datasets/processed/UCSD_P2_256/'];

mkdirfunc(out_path)

sub_dir_list = {'Train', 'Test'};
file_num_list = [16, 12];


opts.is_gray = true;
opts.maxs = 320;
opts.outsize = [256, 256]; % output size
% opts.outsize = [112, 112]; 
opts.img_type = 'tif';

for subdir_idx = 1:length(sub_dir_list)
    % Train, Test
    subdir_file_num = file_num_list(subdir_idx);
    subdir_name = sub_dir_list{subdir_idx};
    subdir_in_path = [in_path, subdir_name, '/'];
    subdir_out_path = [out_path, subdir_name, '/'];
    for i = 1:subdir_file_num
        v_name = [subdir_name, num2str(i, '%03d')];
        v_path = [subdir_in_path, v_name, '/'];
        v_out_path = [subdir_out_path, v_name,  '/'];
        mkdirfunc(v_out_path);
        fprintf(v_path)
        fprintf(v_out_path)
        trans_img2img(v_path, v_out_path, opts);
    end
end

%% generate frame level gt labels only for Test
gt_in_path = [in_path, 'Test/'];
gt_out_path = [out_path, 'Test_gt/'];
mkdirfunc(gt_out_path);
for i = 1:file_num_list(2)
    % sub_gt_in_path = [gt_in_path, 'Test', num2str(i, '%03d'), '_gt/'];
    trans_img2label(gt_in_path, i, gt_out_path);
end



% generating index files for video clips
data_root_path = '/data/root/path/';
in_path = [data_root_path, 'datasets/processed/UCSD_P2_256/'];
%%
frame_file_type = 'jpg';
clip_len = 16; %10; % number of frames in a clip
overlap_rate = 0; % overlap
skip_step = 1; % 
clip_rng = clip_len*skip_step-1; % 
% overlap backward shift
overlap_shift = clip_len - 1; % full overlap (shift 1 one step), can be for testing.
% overlap_shift = clip_len/2; % overlap less (backword more) for less clips
% overlap_shift = 10;
% overlap_shift = 5;
sub_dir_list = {'Train', 'Test'};
%%
for sub_dir_idx = 1:length(sub_dir_list)
    sub_dir_name = sub_dir_list{sub_dir_idx};
    fprintf('%s\n', sub_dir_name);
    sub_in_path = [in_path, sub_dir_name, '/'];
    idx_out_path = [in_path, sub_dir_name, '_idx/'];
    mkdirfunc(idx_out_path);
    %%
    % subdir for (preprocessed) video seqs
    v_list = dir([sub_in_path, sub_dir_name, '*']);
    for i=1:length(v_list)
        v_name = v_list(i).name;
        fprintf('%s\n', v_name);
        frame_list = dir([sub_in_path, v_name, '/*.', frame_file_type]);
        frame_num = length(frame_list);
        s_list = 1:(clip_rng+1-overlap_shift):frame_num;
        e_list = s_list + clip_rng;
        idx_val = e_list<=frame_num;
        s_list = s_list(idx_val);
        e_list = e_list(idx_val);
        %% make sub-dir for the video index
        video_sub_dir_out_path = [idx_out_path, v_name, '/'];
        mkdirfunc(video_sub_dir_out_path)        
        for j = 1:length(s_list)
            clear idx
            idx = s_list(j):skip_step:e_list(j);
            save([video_sub_dir_out_path, v_name, '_i', num2str(j, '%03d'), '.mat'], 'v_name', 'idx');
        end
    end
end


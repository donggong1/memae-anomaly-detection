function trans_img2label(inpath, idx, outpath)
    % for UCSD data frames
    gt_in_path = [inpath, 'Test', num2str(idx, '%03d'), '_gt/'];
    fprintf('%s\n', gt_in_path);
    mkdirfunc(gt_in_path);
    file_list = dir([gt_in_path, '/*.bmp']);
    l = zeros(1, length(file_list));
    for j=1:length(file_list)
        name = file_list(j).name;
        file_path = [gt_in_path, name];
        img = imread(file_path);
%         img = rgb2gray(img);
        img = im2double(img);
        f = sum(img(:));
        if(f<1)
            l(j) = 0;
        else
            l(j) = 1;
        end
    end
    save([outpath, 'Test', num2str(idx, '%03d'), '.mat'], 'l');
return

function trans_img2img(inpath, outpath, opts)
% preprocess images: trans images to images
% 
mkdirfunc(outpath)

img_list = dir([inpath, '/', '*.', opts.img_type]);
img_name = img_list(1).name;
img = imread([inpath, '/', img_name]);
H = size(img, 1);
W = size(img, 2);

fprintf('%s\n', [inpath]);

img_list = dir([inpath, '/', '*.', opts.img_type]);
for i=1:length(img_list)
    img_name = img_list(i).name;
    img = imread([inpath, '/', img_name]);
    if(opts.is_gray && size(img,3)==3)
        img = rgb2gray(img);
    end
    if(~opts.is_gray && size(img,3)==1)
        img = repmat(img, 1,1,3);
    end
    
    if(~isempty(opts.outsize))
        img = imresize(img, opts.outsize);
    end
%     imwrite(img, [outpath, name_p, '/', name_p, '_', num2str(cnt, '%40d'), '.jpg']);
    imwrite(img, [outpath, num2str(i, '%03d'), '.jpg']);
end


return

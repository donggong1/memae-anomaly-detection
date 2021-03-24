function mkdirfunc(p)
    if(~exist(p, 'dir'))
        mkdir(p);
    end
return
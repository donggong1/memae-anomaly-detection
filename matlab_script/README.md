# matlab script
+ matlab script for preparing datasets

### Dataset folder structure
```
$video_dataset
  └──UCSD_P2_256
        ├──Test
        |    ├──Test001
        |    |    ├──001.jpg
        |    |    ├──002.jpg
        |    |    ├──003.jpg
        |    |    └──...
        |    ├──Test002
        |    |    ├──001.jpg
        |    |    ├──002.jpg
        |    |    └──...
        |    └──...
        ├──Test_idx
        |    ├──Test001
        |    |    ├──Test001_i001.mat (the \#1 video clip -- indices of the frames)
        |    |    ├──Test001_i002.mat (the \#2 video clip -- indices of the frames)
        |    |    ├──Test001_i003.mat
        |    |    └──...
        |    ├──Test002
        |    |    ├──Test002_i001.mat
        |    |    ├──Test002_i001.mat
        |    |    └──...
        |    └──...
        ├──Test_gt
        |    ├──Test001.mat
        |    └──...
        ├──Train
        |    └──... (similar to 'testing')
        └──Train_idx
             └──... (similar to 'testing_idx')
```
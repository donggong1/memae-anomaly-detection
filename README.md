# memae-anomaly-detection

<small>Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder (MemAE) for Unsupervised Anomaly Detection 

Dong Gong, Lingqiao Liu, Vuong Le, Budhaditya Saha, Moussa Reda Mansour, Svetha Venkatesh, Anton van den Hengel.  
In IEEE International Conference on Computer Vision (ICCV), 2019.
\[[Paper (v2)](https://arxiv.org/abs/1904.02639)\]\[[Project](https://donggong1.github.io/anomdec-memae.html)\]
</small>

## Requirements
+ Python 3.6
+ PyTorch 0.4.0 (tested with 0.4.0)
+ MATLAB (for data preparation)

## Usage
### Testing
1. Install this repository and the required packages.
2. Download pretrained models from \[[MODELS](https://drive.google.com/drive/folders/1N2DvmZwCKx_8bZWeueJNn9nsh3rQXdTg?usp=sharing)\]. Move them into `./memae_models`.
3. Prepare dataset.
   1) Download dataset.
   2) Move the dataset into `./dataset`. 
   3) The dataset folder should be organized in specific stuctures to fit the `dataloader` as the followings. This can be obtained by running the corresponding script in `./matlab_script`.
4. Run `.sh` files or `python script_testing.py`. 

### Training
Training code will be released later. (To be added)

### Dataset folder structure
```
$video_dataset
  └──UCSD_P2_256
        ├──testing
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
        ├──testing_idx
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
        ├──testing_gt
        |    ├──Test001.mat
        |    └──...
        ├──training
        |    └──... (similar to 'testing')
        └──training_idx
             └──... (similar to 'testing_idx')
```

```
$image_dataset
(To be added)
```

```
$kddcup_dataset
(To be added)
```

## Citation
If you use this code for your research, please cite our paper.

```
@inproceedings{gong2019memorizing,
  title={Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection},
  author={Gong, Dong and Liu, Lingqiao and Le, Vuong and Saha, Budhaditya and Mansour, Moussa Reda and Venkatesh, Svetha and Hengel, Anton van den},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

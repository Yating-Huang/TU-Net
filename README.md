# TU-Net: A Precise Network for Tongue Segmentation
by [Yating Huang](https://Yating-Huang.github.io/), Zhihui Lai, Wenjing Wang
## Summary:
### Intoduction:
  This repository is for our ICCPR paper ["TU-Net: A Precise Network for Tongue Segmentation"](https://dl.acm.org/doi/pdf/10.1145/3436369.3437428)
  
  
### Framework:
![](https://github.com/Yating-Huang/TU-Net/blob/main/TU-Net.png)

## Usage:
### Requirement:
Ubuntu 16.04+pycharm+python3.6+pytorch1.7.1  
### Preprocessing:
Clone the repository:
```
git clone https://github.com/Yating-Huang/TU-Net.git
cd TU-Net
```
## HOW TO RUN:
The only thing you should do is enter the dataset.py and correct the path of the datasets.
then run ~
example:
```
python main.py --action train&test --arch UNet --epoch 21 --batch_size 21 
```
## RESULTS
after train and test,3 folders will be created,they are "result","saved_model","saved_predict".

### saved_model folder:
After training,the saved model is in this folder.

### result folder:
in result folder,there are the logs and the line chart of metrics.such as:
![image](https://github.com/Yating-Huang/TU-Net/blob/main/result/plot/TUNet_2_tongue_50_iou&dice.jpg)

### saved_predict folder:
in this folder,there are the ouput predict of the saved model,such as:
![image](https://github.com/Yating-Huang/TU-Net/blob/main/saved_predict/TUNet/2/50/tongue/272.jpg)


### the datasets:
the Tongue dataset(dsb2018)
link：https://github.com/BioHit/TongeImageDataset

## Citation:
If you found Triple ANet helpful for your research, please cite our paper:
```
@inproceedings{10.1145/3436369.3437428,
author = {Huang, Yating and Lai, Zhihui and Wang, Wenjing},
title = {TU-Net: A Precise Network for Tongue Segmentation},
year = {2020},
isbn = {9781450387835},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3436369.3437428},
doi = {10.1145/3436369.3437428},
abstract = {Tongue diagnosis is a valuable clinical experience accumulated by Traditional Chinese
Medicine (TCM) through long-term research. In TCM, tongue body reflects the most sensitive
indicators of the physiological function and pathological changes, which can help
doctors to determine the strategy of syndrome differentiation. Therefore, tongue segmentation
is particularly important for intelligent tongue diagnosis. With the development of
Convolutional Neural Network (CNN) in image processing, some popular networks such
as U-shape Network (U-Net), Fully Convolutional Networks (FCN) and their variants
have been used in medical image segmentation. It is challenging to segment tongue
because the pixels of human lips, chin, and other parts in the tongue images are the
same as the tongue. In this paper, we propose an end-to-end network called Tongue
U-Net (TU-Net) which combines the classical U-Net structure with Squeeze-and-Excitation
(SE) block, Dense Atrous Convolution (DAC) block and Residual Multi-kernel Pooling
(RMP) block. The model is inspired from Squeeze-and-Excitation Networks (SENet) and
Context Encoder Network (CE-Net) that can capture more useful information. Applied
to a tongue dataset with 300 images, TU-Net performs better than the four segmentation
methods (FCN, U-Net, Attention U-Net and U2 -Net) in the evaluation of Dice coefficient,
Intersection over Union and Hausdorff distance.},
booktitle = {Proceedings of the 2020 9th International Conference on Computing and Pattern Recognition},
pages = {244–249},
numpages = {6},
keywords = {Tongue segmentation, traditional Chinese medicine, deep learning},
location = {Xiamen, China},
series = {ICCPR 2020}
}
```

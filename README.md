# TU-Net: A Precise Network for Tongue Segmentation
by [Yating Huang](https://Yating-Huang.github.io/), Zhihui Lai, Wenjing Wang
## Summary:
### Intoduction:
  This repository is for our ICCPR2020 paper ["TU-Net: A Precise Network for Tongue Segmentation"](https://dl.acm.org/doi/pdf/10.1145/3436369.3437428)
  
  
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
## How to run:
The only thing you should do is enter the dataset.py and correct the path of the datasets.
then run ~
example:
```
python main.py --action train&test --arch TUNet --epoch 2 --batch_size 2 
```
## Results
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
the Tongue dataset
link：https://github.com/BioHit/TongeImageDataset
## Citation:
If you found Triple ANet helpful for your research, please cite our paper:
```
@inproceedings{DBLP:conf/iccpr/HuangLW20,
  author    = {Yating Huang and
               Zhihui Lai and
               Wenjing Wang},
  title     = {TU-Net: {A} Precise Network for Tongue Segmentation},
  booktitle = {{ICCPR} 2020: 9th International Conference on Computing and Pattern
               Recognition, Xiamen, China, October 30 - Vovember 1, 2020},
  pages     = {244--249},
  publisher = {{ACM}},
  year      = {2020},
  url       = {https://doi.org/10.1145/3436369.3437428},
  doi       = {10.1145/3436369.3437428},
  timestamp = {Tue, 19 Jan 2021 15:37:23 +0100},
  biburl    = {https://dblp.org/rec/conf/iccpr/HuangLW20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
## Acknowledgement
Part of the code is revised from the [UNET-ZOO](https://github.com/Andy-zhujunwen/UNET-ZOO).

## Note
* The repository is being updated
* Contact: Yating Huang (huangyating2019@email.szu.edu.cn)

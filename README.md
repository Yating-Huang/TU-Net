# UNET-ZOO
including unet,unet++,attention-unet,r2unet,cenet,segnet ,fcn.

# ENVIRONMENT
Ubuntu 16.04+pycharm+python3.6+pytorch1.7.1  

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
![image](https://github.com/Andy-zhujunwen/UNET-ZOO/blob/master/linechart.png)

### saved_predict folder:
in this folder,there are the ouput predict of the saved model,such as:
![image](https://github.com/Andy-zhujunwen/UNET-ZOO/blob/master/eye.png)


### the datasets:
the Tongue dataset(dsb2018)
link：https://github.com/BioHit/TongeImageDataset


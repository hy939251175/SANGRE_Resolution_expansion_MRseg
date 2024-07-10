
import logging
import sys
import tempfile
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.pyplot as plt



import pandas as pd
import scipy
from scipy.io import loadmat, savemat
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import random
import scipy.io
from sklearn.model_selection import KFold

import torch
import torch.nn.functional as F
from torch import nn
from torch import initial_seed, float32, int64, Generator
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.utils import MetricReduction

import monai
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, compute_meandice, HausdorffDistanceMetric
from monai.losses import DiceLoss
from monai.visualize import plot_2d_or_3d_image

from monai.data import (
    create_test_image_2d,
    Dataset,
    list_data_collate,
    pad_list_data_collate,
    decollate_batch,
    ZipDataset)

from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscreted,
    SpatialCropd,
    RandSpatialCropd,
    Resized,
    RandZoomd,
    AsDiscrete,
    Compose,
    RandAffined,
    SqueezeDimd,
    LoadImaged,
    RandCropByPosNegLabeld,
    BorderPadd,
    ToTensord,
    RandRotate90d,
    ScaleIntensityd,
    EnsureTyped,
    EnsureType)
from lib.pvt_debug import PvtUNet
torch.use_deterministic_algorithms(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





#@title Running Parameters

k_folds = 5
no_classes = 7

csv_name = '/home/ying/Downloads/colab(1).csv'

no_epochs = 200
k_folds = 5
n_fr_dict = {'ah': 71,
             'aa': 105,
             'br': 71,
             'gc': 78,
             'mr': 67}
# In the order of aa, ah, br, gc and mr
coords = [[105, 30],[105, 25],[110, 20],[100, 16],[100, 15]]
size_parameters = [128, 128]
deviation_parameters = [20]
deviation_scale_parameters= [10, 20]
deviation_size_parameters = [100, 140]
reduced_size = deviation_size_parameters[0]/size_parameters[0]
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
hausdorff_metric = HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=None, directed=False, reduction=MetricReduction.MEAN, get_not_nans=False)
classes = [1,2,3,4,5,6]

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.use_deterministic_algorithms(False)



model = PvtUNet(in_channels=1,num_classes=7).cpu()
# model=UNet_n_classes(n_classes=7).cpu()


metric_values = list()
metrics_list = []
tes=['aa','ah','br','gc','mr']
df=pd.DataFrame(columns=['Volunteer_test','Frame','Class','Dice','HD','Model'])
for i in range(5):
    test_vol=tes[i]
    #if i == 1:
    #    sys.exit()
    #test_metric = -1
    # print("best_metric_model_{}.pth".format(saved_model_best[i]))
    model.load_state_dict(torch.load(f'./model_pth/best_metric_model_{i}.pth', map_location=torch.device('cpu')),strict=False) 
    model.to(device)
   
    model.eval()
    test_set = torch.load(f"./test_dataset/test_{i+1}_1.pt", map_location=torch.device('cpu'))
    testloaderCV = torch.utils.data.DataLoader(test_set, shuffle=False)
    metrics = []
    outputs = np.zeros([1,7,256,256])
    j = 0
    k=0
    op=3
    lp=[]
    iop=1
    for test_data in testloaderCV:
        k+=1
        test_images, test_labels = test_data['img'].to(device), test_data['seg'].to(device)
        #print(test_labels.size())
        test_outputs  = model(test_images)
        test=0
        for out in test_outputs:
            test+=out

        test_outputs = torch.argmax(test, dim=1)

        test_outputs = F.one_hot(test_outputs, num_classes = -1)
        test_outputs = torch.permute(test_outputs, (0, 3, 1, 2))
                 
        test_labels = F.one_hot(test_labels, num_classes = no_classes)
        test_labels = torch.permute(test_labels, (0, 1, 4, 2, 3))

        test_labels = torch.squeeze(test_labels, dim=1)
        dice_metric(y_pred=test_outputs, y=test_labels)
        pp=dice_metric(y_pred=test_outputs, y=test_labels)
        m=torch.mean(pp,0,True)
        
        
        pp_HD=hausdorff_metric(y_pred=test_outputs, y=test_labels)
        
        if test_data == 0:
            outputs = test_outputs.detach().cpu().numpy()
            print(str(np.shape(outputs)) + 'outputs 1')
        else:
            outputs = np.append(outputs, test_outputs.detach().cpu().numpy(), axis = 0)
        
                    
            result=[
                    [test_vol,k,'Head',float(m[0,1]),float(pp_HD[0,1]),'SNet'],
                    [test_vol,k,'Soft-palate',float(m[0,2]),float(pp_HD[0,2]),'SNet'],
                    [test_vol,k,'Jaw',float(m[0,3]),float(pp_HD[0,3]),'SNet'],
                    [test_vol,k,'Tongue',float(m[0,4]),float(pp_HD[0,4]),'SNet'],
                    [test_vol,k,'Vocal-Tract',float(m[0,5]),float(pp_HD[0,5]),'SNet'],
                    [test_vol,k,'Tooth-space',float(m[0,6]),float(pp_HD[0,6]),'SNet']
                    ]
            
            df1=pd.DataFrame(result,columns=df.columns)
            df=df.append(df1)
                

    print(str(np.shape(outputs)) + 'outputs at end')

    metric = dice_metric.aggregate().item()
    metric1 = dice_metric
    print(metric)

    path="./pred"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print(f"The new directory {path} is created!")
    np.save(path  + f'/Sub_{i+1}_outputs', outputs)
    dice_metric.reset()
    print(type(metric1))
    metric_values.append(metric)
    

del model
print(df.shape)
df.to_csv('./pred/test_PVTUnet_.csv',index=True)   

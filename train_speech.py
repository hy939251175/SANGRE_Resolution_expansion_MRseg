
import logging
import sys
import tempfile
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.io import loadmat, savemat
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import random
from sklearn.model_selection import KFold

import torch
import torch.nn.functional as F
from torch import initial_seed, float32, int64, Generator
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import nn

import monai
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, compute_meandice
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

from torch.optim.lr_scheduler import StepLR
from lib.net import SANGRENet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.use_deterministic_algorithms(False)

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from torch.nn.modules.loss import CrossEntropyLoss

#@title Running Parameters

csv_name = './updated_data (1).csv'

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



#@title Transformations
def transformations(coordinates):
   
    left_parameters = [coordinates[0], coordinates[1]]
    fn_keys = ['img', 'seg']  

    # Define transforms for image and segmentation, first is Rescale, second is RandomTranslation,
    # third is RandomCrop, fourth is PadRotateAndCrop
    #I MOOOST MAKE SURE THE CHANNEL STUFF IS CORRECT


    transform_1 = Compose([LoadImaged(keys=fn_keys, image_only=True),
                      AddChanneld(keys=fn_keys),
                       # SpatialCropd(fn_keys, roi_center = [left_parameters[0]+size_parameters[0]/2, left_parameters[1]+size_parameters[1]/2], roi_size = size_parameters),
                      ToTensord(keys=['img'], dtype=torch.float32),
                      ToTensord(keys=['seg'], dtype=torch.int64)])
   

    evaluation_transform = Compose([LoadImaged(keys=fn_keys, image_only=True),
                      AddChanneld(keys=fn_keys),
                      # SpatialCropd(fn_keys, roi_center = [left_parameters[0]+size_parameters[0]/2, left_parameters[1]+size_parameters[1]/2], roi_size = size_parameters),
                      ToTensord(keys=['img'], dtype=torch.float32),
                      ToTensord(keys=['seg'], dtype=torch.int64)])
   
    return transform_1, evaluation_transform



#@title Cross Validation Functions and File Creation

cross_val_matrix = [['aa', 'ah', 'br', 'gc', 'mr'],
                    ['ah', 'br', 'gc', 'mr', 'aa'],
                    ['br', 'gc', 'mr', 'aa', 'ah'],
                    ['gc', 'mr', 'aa', 'ah', 'br'],
                    ['mr', 'aa', 'ah', 'br', 'gc']]

def get_experiments_and_subjects(matrix, experiment_number):
    list1 = []
    list2 = matrix[experiment_number -1]
    list1.append(list2[0])
    list1.append(list2[1])
    list1.append(list2[2] + list2[3] + list2[4])
    list1.append(experiment_number)
    return list1



strings_for_RandSeedTransforms = []
def get_training_data(training_subjects, val_subject, test_subject, d1, s1, d2, s2, d3, s3, d4, s4, d5, s5):
    order = ['aa', 'ah', 'br', 'gc', 'mr']
    mega_list_image = [d1, d2, d3, d4, d5]
    mega_list_seg = [s1, s2, s3, s4, s5]
    string1 = training_subjects[0:2]
    string2 = training_subjects[2:4]
    string3 = training_subjects[4:6]
    string4 = val_subject[0:2]
    string5 = test_subject[0:2]
    strings_for_RandSeedTransforms.append(string1)
    strings_for_RandSeedTransforms.append(string2)
    strings_for_RandSeedTransforms.append(string3)
    strings_for_RandSeedTransforms.append(string4)
    strings_for_RandSeedTransforms.append(string5)
    subject_number_1 = order.index(string1)
    subject_number_2 = order.index(string2)
    subject_number_3 = order.index(string3)
    subject_val = order.index(val_subject)
    subject_test = order.index(test_subject)
    img_files_train_1 = mega_list_image[subject_number_1]
    img_files_train_2 = mega_list_image[subject_number_2]
    img_files_train_3 = mega_list_image[subject_number_3]  
    seg_files_train_1 = mega_list_seg[subject_number_1]
    seg_files_train_2 = mega_list_seg[subject_number_2]
    seg_files_train_3 = mega_list_seg[subject_number_3]  
    img_files_val = mega_list_image[subject_val]
    seg_files_val = mega_list_seg[subject_val]
    img_files_test = mega_list_image[subject_test]
    seg_files_test = mega_list_seg[subject_test]
   
    return img_files_train_1, img_files_train_2, img_files_train_3, img_files_val, img_files_test, seg_files_train_1, seg_files_train_2, seg_files_train_3, seg_files_val, seg_files_test  
   


information_list = get_experiments_and_subjects(cross_val_matrix, 1)
information_list2 = get_experiments_and_subjects(cross_val_matrix, 2)
information_list3 = get_experiments_and_subjects(cross_val_matrix, 3)
information_list4 = get_experiments_and_subjects(cross_val_matrix, 4)
information_list5 = get_experiments_and_subjects(cross_val_matrix, 5)


iterative_info_list  = [information_list, information_list2, information_list3, information_list4, information_list5]




#@title Random Initialisation Formula For Transforms

def random_seed_initialisation_transforms(files_1, files_2, files_3, cross_val_list, strings_list):


  subject_list = ['aa', 'ah', 'br', 'gc', 'mr']
  subject_number_1 = subject_list.index(strings_list[0]) + 1
  seed_1 = '{}0{}01'.format(cross_val_list[3], subject_number_1)

  subject_number_2 = subject_list.index(strings_list[1]) + 1
  seed_6 = '{}0{}01'.format(cross_val_list[3], subject_number_2)

  subject_number_3 = subject_list.index(strings_list[2]) + 1
  seed_11 = '{}0{}01'.format(cross_val_list[3], subject_number_3)

  subject_number_4 = subject_list.index(strings_list[3]) + 1
  subject_number_5 = subject_list.index(strings_list[4]) + 1

  t1, e1 = transformations(coords[subject_number_1 - 1])
  t1.set_random_state(seed=int(seed_1))
  train_1_ds = Dataset(data=files_1, transform=t1)

  t1, e1 = transformations(coords[subject_number_2 - 1])
  t1.set_random_state(seed=int(seed_6))
  train_2_ds = Dataset(data=files_2, transform=t1)
 
  t1, e1 = transformations(coords[subject_number_3 - 1])
  t1.set_random_state(seed=int(seed_11))
  train_3_ds = Dataset(data=files_3, transform=t1)

  t1, evaluation_transform = transformations(coords[subject_number_4 - 1])
  t1, test_transform = transformations(coords[subject_number_5 - 1])
 
  dataset = ConcatDataset([train_1_ds, train_2_ds, train_3_ds])
  strings_list.clear()

  return dataset, evaluation_transform, test_transform



#@title Random Initialisation Formula For Dataloader

def random_seed_initialisation_dataloaders(cross_val_list):

  subject_list = ['brgcmr', 'gcmraa', 'mraaah', 'aaahbr', 'ahbrgc']
  subject_number = subject_list.index(cross_val_list[2]) + 1
  seed_1 = '{}0{}050'.format(cross_val_list[3], subject_number)

  return seed_1
# Import required modules

# Getting Subject by Subject information from the mounted CSV
def create_files(data_sheet, info_list):
    data_info = pd.read_csv(data_sheet, header=None)
    data_info = data_info.to_numpy()
    aa_data = data_info[0:n_fr_dict['aa']]
    ah_data = data_info[n_fr_dict['aa']:n_fr_dict['aa'] + n_fr_dict['ah']]
    br_data = data_info[n_fr_dict['aa'] + n_fr_dict['ah']:n_fr_dict['aa'] + n_fr_dict['ah']+n_fr_dict['br']]
    gc_data = data_info[n_fr_dict['aa'] + n_fr_dict['ah']+n_fr_dict['br']:n_fr_dict['aa'] + n_fr_dict['ah']+n_fr_dict['br']+n_fr_dict['gc']]
    mr_data = data_info[n_fr_dict['aa'] + n_fr_dict['ah']+n_fr_dict['br']+n_fr_dict['gc']:n_fr_dict['aa'] + n_fr_dict['ah']+n_fr_dict['br']+n_fr_dict['gc']+n_fr_dict['mr']]

    img_files_train_1, img_files_train_2, img_files_train_3, img_files_val, img_files_test, seg_files_train_1, seg_files_train_2, seg_files_train_3, seg_files_val, seg_files_test = get_training_data(
    info_list[2],
    info_list[0],
    info_list[1],
    aa_data[:,0],
    aa_data[:,1],
    ah_data[:,0],
    ah_data[:,1],
    br_data[:,0],
    br_data[:,1],
    gc_data[:,0],
    gc_data[:,1],
    mr_data[:,0],
    mr_data[:,1])


    # Set up dictionary of images and segmentations
    training_files_1 = [{"img": img, "seg": seg} for img, seg in zip(img_files_train_1, seg_files_train_1)]
    training_files_2 = [{"img": img, "seg": seg} for img, seg in zip(img_files_train_2, seg_files_train_2)]
    training_files_3 = [{"img": img, "seg": seg} for img, seg in zip(img_files_train_3, seg_files_train_3)]
 
    evaluating_files = [{"img": img, "seg": seg} for img, seg in zip(img_files_val, seg_files_val)]
    test_files = [{"img": img, "seg": seg} for img, seg in zip(img_files_test, seg_files_test)]
   
    return training_files_1, training_files_2, training_files_3, evaluating_files, test_files






# Create Metric and Activation Functions
'''Change weighting for different focus on the classes'''
weighting = 1
def tversky_loss(true, logits, alpha, beta, eps=1e-7):
    counts = torch.bincount(torch.flatten(torch.squeeze(true)))
    weight_tensor = torch.sum(counts)**weighting/counts**weighting
    weight_tensor = weight_tensor/torch.mean(weight_tensor)
    weight_tensor = weight_tensor.unsqueeze(dim=0)
    weight_tensor = weight_tensor.unsqueeze(dim=2)
    weight_tensor = weight_tensor.unsqueeze(dim=3)
    num_classes = logits.shape[1]
    true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot * weight_tensor, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()
    return (1-tversky_loss)

def get_individual_classes(image, class_for_dice):
    array_for_seg = np.where(image == class_for_dice, 1, 0)
    return array_for_seg

def dice(a,b):
    union = np.count_nonzero(a) + np.count_nonzero(b)
    intersection = np.count_nonzero(a*b)
    
    return 2*intersection/union

ce_loss = CrossEntropyLoss()

# Create Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT_CODE = 1

LR=[0.0005]

for P4 in range(1):

            LR_=LR[P4]
            no_epochs = 150
            k_folds = 5
            test_set_directory=f'./test_dataset/'
            model_pth = f'./model_pth/'
      
            isExist = os.path.exists(model_pth)
            if not isExist:
                os.makedirs(model_pth)
                print(f"The new directory {model_pth} is created!")
                      
            
            isExist =  os.path.exists(test_set_directory)
            if not isExist:
                os.makedirs(test_set_directory)
                print(f"The new directory {test_set_directory} is created!")

            #@title Cross Validation Datasets
            # Create a Training Dataset
            training_files_11, training_files_12, training_files_13, evaluating_files1, test_files1 = create_files(csv_name, information_list)
            train_ds_1, evaluation_transform, test_transform = random_seed_initialisation_transforms(training_files_11,
                                                          training_files_12,
                                                          training_files_13,
                                                          information_list,
                                                          strings_for_RandSeedTransforms)
            val_ds_1 = monai.data.Dataset(data=evaluating_files1, transform = evaluation_transform)
            test_ds_1 = monai.data.Dataset(data=test_files1, transform = test_transform)
            torch.save(test_ds_1,  test_set_directory + 'test_1_{}.pt'.format(EXPERIMENT_CODE))
            cv_dataset_1 = ConcatDataset([train_ds_1, val_ds_1])
            
            
            training_files_21, training_files_22, training_files_23, evaluating_files2, test_files2 = create_files(csv_name, information_list2)
            train_ds_2, evaluation_transform, test_transform = random_seed_initialisation_transforms(training_files_21,
                                                          training_files_22,
                                                          training_files_23,
                                                          information_list2,
                                                          strings_for_RandSeedTransforms)
            val_ds_2 = monai.data.Dataset(data=evaluating_files2, transform = evaluation_transform)
            test_ds_2 = monai.data.Dataset(data=test_files2, transform = test_transform)
            torch.save(test_ds_2,  test_set_directory + 'test_2_{}.pt'.format(EXPERIMENT_CODE))
            cv_dataset_2 = ConcatDataset([train_ds_2, val_ds_2])
            
            training_files_31, training_files_32, training_files_33, evaluating_files3, test_files3 = create_files(csv_name, information_list3)
            train_ds_3, evaluation_transform, test_transform = random_seed_initialisation_transforms(training_files_31,
                                                          training_files_32,
                                                          training_files_33,
                                                          information_list3,
                                                          strings_for_RandSeedTransforms)
            val_ds_3 = monai.data.Dataset(data=evaluating_files3, transform = evaluation_transform)
            test_ds_3 = monai.data.Dataset(data=test_files3, transform = test_transform)
            torch.save(test_ds_3,  test_set_directory + 'test_3_{}.pt'.format(EXPERIMENT_CODE))
            cv_dataset_3 = ConcatDataset([train_ds_3, val_ds_3])
            
            training_files_41, training_files_42, training_files_43, evaluating_files4, test_files4 = create_files(csv_name, information_list4)
            train_ds_4, evaluation_transform, test_transform = random_seed_initialisation_transforms(training_files_41,
                                                          training_files_42,
                                                          training_files_43,
                                                          information_list4,
                                                          strings_for_RandSeedTransforms)
            val_ds_4 = monai.data.Dataset(data=evaluating_files4, transform = evaluation_transform)
            test_ds_4 = monai.data.Dataset(data=test_files4, transform = test_transform)
            torch.save(test_ds_4,  test_set_directory + 'test_4_{}.pt'.format(EXPERIMENT_CODE))
            cv_dataset_4 = ConcatDataset([train_ds_4, val_ds_4])
            
            training_files_51, training_files_52, training_files_53, evaluating_files5, test_files5 = create_files(csv_name, information_list5)
            train_ds_5, evaluation_transform, test_transform = random_seed_initialisation_transforms(training_files_51,
                                                          training_files_52,
                                                          training_files_53,
                                                          information_list5,
                                                          strings_for_RandSeedTransforms)
            val_ds_5 = monai.data.Dataset(data=evaluating_files5, transform = evaluation_transform)
            test_ds_5 = monai.data.Dataset(data=test_files5, transform = test_transform)
            torch.save(test_ds_5,  test_set_directory + 'test_5_{}.pt'.format(EXPERIMENT_CODE))
            cv_dataset_5 = ConcatDataset([train_ds_5, val_ds_5])
            

            
            # Define the K-fold Cross Validator
            kfold = KFold(n_splits=k_folds, shuffle=False)
            
            #Produce Final Dataset
            train_CV_dataset = [train_ds_1, train_ds_2, train_ds_3, train_ds_4, train_ds_5]
            eval_CV_dataset = [val_ds_1, val_ds_2, val_ds_3, val_ds_4, val_ds_5]
            cross_validation_dataset = ConcatDataset([cv_dataset_1, cv_dataset_2, cv_dataset_3, cv_dataset_4, cv_dataset_5])
            
            # Set-up Monai Config
            monai.config.print_config()
            logging.basicConfig(stream=sys.stdout, level=logging.INFO)
            
            # model=UNet_n_classes(n_classes=7).to(device)
            # model = AttU_Net(img_ch=1,output_ch=7).to(device)
            model = SANGRENet(in_channels=1,num_classes=7).to(device)

            from torchinfo import summary
            #    summary(model, (1, 3, 352, 352))
            from thop import profile
            import torch
            input = torch.randn(1, 1, 256, 256).to('cuda')
            macs, params = profile(model, inputs=(input,))
            print('macs:', macs / 1000000000)
            print('params:', params / 1000000)            
            # Define the Collate Function
            def collate(batch):
                img = torch.stack([item['img'] for item in batch])
                seg = torch.stack([item['seg'] for item in batch])
                return {'img' : img, 'seg' : seg}
            
            # Function to preserve DataLoader reproducibility
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2**32
                
                np.random.seed(worker_seed)
                random.seed(worker_seed)
                
                
 
            
            cross_val_matrix = [['aa', 'ah', 'br', 'gc', 'mr'],
                                ['ah', 'br', 'gc', 'mr', 'aa'],
                                ['br', 'gc', 'mr', 'aa', 'ah'],
                                ['gc', 'mr', 'aa', 'ah', 'br'],
                                ['mr', 'aa', 'ah', 'br', 'gc']]
            
            vol_dict = {'aa': 105,
                        'ah': 71,
                        'br': 71,
                        'gc': 78,
                        'mr': 67}     
            
            
             
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR_)  
            scheduler =torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=15000, power=1.0)
            no_classes = 7
 
            
            
            # K-fold Cross Validation model evaluation
            for fold in range(0,k_folds):
                
                # model.load_state_dict(torch.load(model_file_directory + 'original_model.pth'))  # Choose whatever GPU device number you want
                model.to(device)
            
                g = torch.Generator()
                g.manual_seed(int(random_seed_initialisation_dataloaders(iterative_info_list[fold])))
               
                # Print
                print(f'Fold {fold}')
                print('-----------------')
                
                trainloaderCV = torch.utils.data.DataLoader(
                                  train_CV_dataset[fold],
                                  batch_size =14,
                                  # num_workers=16,
                                  drop_last = False,

                                  shuffle=True)
                evalloaderCV = torch.utils.data.DataLoader(
                                  eval_CV_dataset[fold],
                                  batch_size = 1,
                           
                             
                                  collate_fn = collate)
                print('len',len(train_CV_dataset[fold]))
                

                # Run the training loop for defined number of epochs
                ce_loss = CrossEntropyLoss()
                loss_function=DiceLoss(include_background=True,to_onehot_y=True,softmax=True,reduction='mean')
                best_metrics = []
                val_interval = 2
                epoch_loss_values = list()
                writer = SummaryWriter()
                train_loss=[]
                val_loss=[]
                for epoch in range(no_epochs):
                    model.train()
                    epoch_loss = 0
                    step = 0
                    for batch_data in trainloaderCV:
                        
                        step += 1
                        inputs, labels = batch_data['img'].to(device),  batch_data['seg'].to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        out = 0
                        for ind in outputs:
                            out += ind
                     
                        loss=loss_function(out,labels)
                        
                        loss.backward()
                        optimizer.step()
                        # print(val_loss)
                        scheduler.step()
                        epoch_loss += loss.item()
                        epoch_len = len(train_CV_dataset[fold]) // trainloaderCV.batch_size
                        #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
                    epoch_loss /= step
                    epoch_loss_values.append(epoch_loss)
                    train_loss.append(epoch_loss)
                    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
                    
                    'Need to work on dice metric part, look up how dice metric works properly'
            
                    if (epoch + 1) % val_interval == 0:
                      model.eval()
                      with torch.no_grad():
                          val_images = None
                          val_labels = None
                          val_outputs = None
                          metrics = []
                          for val_data in evalloaderCV:
                              val_images, val_labels = val_data['img'].to(device), val_data['seg'].to(device)
                              val_outputs= model(val_images)

                              val_out = 0
                              for ind in val_outputs:
                                  val_out += ind
                              val_labels= val_labels.detach().cpu().numpy()
                              

                              val_outputs3 = torch.argmax(val_out, dim = 1)
                              val_outputs3 = val_outputs3.detach().cpu().numpy()
                            
                              metrics_pre = []
                              for j in range(1,7):

                                  a3 = get_individual_classes(val_outputs3, j)
                                  b = get_individual_classes(val_labels, j)

                                  metrics_pre.append(dice(a3,b))
                              metrics.append(sum(metrics_pre)/len(metrics_pre))
                              metrics_pre.clear()
                          if len(best_metrics) == 0:
                               print('Hello There')
                               best_metric_epoch=0
                               best_metrics.append(sum(metrics)/len(metrics))
                               torch.save(model.state_dict(), os.path.join(model_pth, "best_metric_model_{}{}.pth".format(fold, epoch)))
                          elif sum(metrics)/len(metrics) > best_metrics[0]:
                               best_metrics.clear()
                               best_metrics.append(sum(metrics)/len(metrics))
                               best_metric_epoch = epoch + 1
                               print("saved new best metric model")
                               torch.save(model.state_dict(), os.path.join(model_pth, "best_metric_model_{}.pth".format(fold)))
                          elif epoch >= no_epochs+1:
                               torch.save(model.state_dict(), os.path.join(model_pth, "final_metric_model_{}{}.pth".format(fold,epoch)))
                          else:
                                print('Old Model Better')
                          print('Best dice',best_metrics,'at epoch ---',best_metric_epoch)
                          logging.info('11')
                          val_loss.append(best_metrics)
                np.save(f'{model_pth}/fold_{fold}_loss.npy', train_loss)
                np.save(f'{model_pth}/fold_{fold}_val_loss.npy', val_loss)
                print('Finished fold')

   

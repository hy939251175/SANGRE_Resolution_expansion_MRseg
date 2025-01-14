# This repository is for the paper 'SANGRE: a Shallow Attention Network Guided by Resolution Expansion for MR Image Segmentation'

## Data
* ACDC - The data is available from [MICCAI2017 challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/miccai_results.html), or you can download the processed dataset from Google Drive of [MT-UNet](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view) and save the data to the same folder path as the code for training.
* Speech MRI Data - The data is available at [Zenodo](https://zenodo.org/records/10046815). The processed dataset in .npy form can be downloaded from [Google Drive](https://drive.google.com/file/d/1wT64P9YtIot7PrxMrnJRkXJ8T5sBSiWS/view?usp=sharing). Save the downloaded files to the same folder path as the code for training.

## Transformer encoder
*Pretrained weight of the encoder can be downloaded [here](https://github.com/whai362/PVT) and the paper is [here](https://link.springer.com/article/10.1007/s41095-022-0274-8). Once downloaded 'pvt_v2_b2.pth' put it into directory 'SANGRE'.

## Training
```
cd into SANGRE
```

For ACDC, run  ``` CUDA_VISIBLE_DEVICES=0 python train_ACDC.py ``` 

For Speech MRI, run ``` CUDA_VISIBLE_DEVICES=0 python train_speech.py ```

## Testing
```
cd into SANGRE
```

For ACDC, run  ``` CUDA_VISIBLE_DEVICES=0 python test_ACDC.py ``` 

For Speech MRI, run ``` CUDA_VISIBLE_DEVICES=0 python test_speech.py ```

## Acknowledgement

We appreciate the work of [G-CASCADE](https://github.com/SLDGroup/G-CASCADE) for providing the foundation of our framework. Also, thank our collegues Aggie and Oscar for providing the foundational framework for the speech MRI experiment.

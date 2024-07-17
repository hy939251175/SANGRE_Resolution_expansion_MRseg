# This repository is for the paper 'SANGRE: a Shallow Attention Network Guided by Resolution Expansion for MR Image Segmentation'

## Data preparation
* ACDC - The data is available from [MICCAI2017 challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/miccai_results.html), or you can download the processed dataset from Google Drive of [MT-UNet](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view) and save the data to the same folder path as the code for training.
* Speech MRI Data - The data is available at [Zenodo](https://zenodo.org/records/10046815). The processed dataset in .npy form can be downloaded from [Google Drive](https://drive.google.com/file/d/1wT64P9YtIot7PrxMrnJRkXJ8T5sBSiWS/view?usp=sharing). Save the downloaded files to the same folder path as the code for training.

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

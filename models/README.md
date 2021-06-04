## Models

The CANDLE, VGG and ResNet models requires tensorflow to run. The MT-WND and DIEN models requires Pytorch.

### Software environment

OS: Ubuntu Server 20.04 LTS (HVM)

Package manager: Anaconda 4.10

CUDA: 11.2 

To install environment to run CANDLE, VGG and ResNet models:
```conda env create -f environment1.yml```

To install environment for MT-WND and DIEN models:
```conda env create -f environment2.yml```

### Start characterization

To run characterization, first launch a desired AWS EC2 instance. Set up the software environment.

To characterize a desired model, run ```python model_name.py instance_name```

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

To run CANDLE Combo model, clone the [candle repo](https://github.com/ECP-CANDLE/Benchmarks) and copy ```candle_inf.py``` over.

```shell
cd /usr_git_dir # replace with custom path
git clone https://github.com/ECP-CANDLE/Benchmarks candle
cp SIMBO/models/candle_inf.py candle/Pilot1/Combo
cd candle/Pilot1/Combo
```
first download the trained model files: [saved.model.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.model.h5) and [saved.weights.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.weights.h5). Put them in current directory.

To run the benchmark, do ```python candle_inf.py --testcase <instance name>```

To run the ResNet and VGG models, download the trained model files: [resnet.h5](https://drive.google.com/file/d/1aCpICrCKuU7QFIG73jLXWwtTfWIVAFWL/view?usp=sharing) and [vgg.h5](https://drive.google.com/file/d/17F_GGAnKU23M5I4VEZUtU2TmPIqjp34E/view?usp=sharing)

No other actions are required for MT-WND and DIEN models.

To characterize a desired model, run ```python model_name.py instance_name```

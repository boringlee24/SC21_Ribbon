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

To run CANDLE Combo model, clone the [candle repo](https://github.com/ECP-CANDLE/Benchmarks) and copy ```Combo``` directory files over.

```shell
conda activate tf-gpu
cd /usr_git_dir # replace with custom path
git clone https://github.com/ECP-CANDLE/Benchmarks candle
cp SIMBO/models/Combo/* candle/Pilot1/Combo
cd candle/Pilot1/Combo
```
Download the trained model files: [saved.model.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.model.h5) and [saved.weights.h5](http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/saved.weights.h5). Put them in current directory.

To run the benchmark, do ```python candle_inf.py --testcase <instance name>```. The characterization results will reside in the ```logs``` directory

To run the ResNet and VGG models, navigate to SIMBO directory, download the trained model files: [resnet.h5](https://drive.google.com/file/d/1aCpICrCKuU7QFIG73jLXWwtTfWIVAFWL/view?usp=sharing) and [vgg.h5](https://drive.google.com/file/d/17F_GGAnKU23M5I4VEZUtU2TmPIqjp34E/view?usp=sharing), execute the scripts. Characterization results are saved to the ```logs``` directory

```shell
cd /usr_git_dir/SIMBO/models
# download the trained models
wget <model link>
python resnet_inf.py --testcase <instance name>
python vgg_inf.py --testcase <instance name>
```

To run the recommendation models, switch environment and clone the [deeprecsys](https://github.com/harvard-acc/DeepRecSys) repo. Switch to the new repo and run the script. Results are written to the ```log``` directory. 
``` shell
conda deactivate
conda activate pytorch-gpu
cd /usr_git_dir # replace with custom path
git clone https://github.com/harvard-acc/DeepRecSys
cp -r SIMBO/models/rec_inf/ DeepRecSys/models
cp SIMBO/models/utils.py DeepRecSys/utils
cd DeepRecSys/models/rec_inf
python experiment.py --testcase <instance name>
```


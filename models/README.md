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
cp Ribbon/models/Combo/* candle/Pilot1/Combo
cd candle/Pilot1/Combo
```
Download the trained model files: [saved.model.h5](https://drive.google.com/file/d/1Tfs5Jyi9iDS7rlutX9GzlDtBElT6ybfi/view?usp=sharing) and [saved.weights.h5](https://drive.google.com/file/d/1Yd81NjTPeEBpUgK9W_WWaHcAMWNeyasd/view?usp=sharing). These links do not work with Linux wget, try download directly from browser, and upload these files from local into the current directory.

To run the benchmark, do ```python candle_inf.py --testcase <instance name>```. The characterization results will reside in the ```logs``` directory

To run the ResNet and VGG models, navigate to RIBBON directory, download the trained model files: [resnet.h5](https://drive.google.com/file/d/1aCpICrCKuU7QFIG73jLXWwtTfWIVAFWL/view?usp=sharing) and [vgg.h5](https://drive.google.com/file/d/17F_GGAnKU23M5I4VEZUtU2TmPIqjp34E/view?usp=sharing). These links do not work with Linux wget, try download directly from browser, and upload these files from local into the current directory. Execute the scripts. Characterization results are saved to the ```logs``` directory

```shell
cd /usr_git_dir/Ribbon/models
# download the trained models
python resnet_inf.py --testcase <instance name>
python vgg_inf.py --testcase <instance name>
```

To run the recommendation models, switch environment and clone the [deeprecsys](https://github.com/harvard-acc/DeepRecSys) repo. Switch to the new repo and run the script. Results are written to the ```log``` directory. 
``` shell
conda deactivate
conda activate pytorch-gpu
cd /usr_git_dir # replace with custom path
git clone https://github.com/harvard-acc/DeepRecSys
cp -r Ribbon/models/rec_inf/ DeepRecSys/models
cp Ribbon/models/utils.py DeepRecSys/utils
cd DeepRecSys/models/rec_inf
python experiment.py --testcase <instance name>
```

If you want to replace an instance with your own hardware, e.g. for CANDLE model, replace the m5.2xlarge instance with ```<machine-type>```, please follow the following steps as this is not automated. Firstly you should remove the candle*.json logs in ```characterization/logs/m5.2xlarge``` directory, and copy over your own logs into that directory. Next, update the price you want to set for this hardware type at ```query/price.csv```, the price is in unit of $/hour. Then, navigate to ```query/``` directory and run the ```distributor.py``` script as commented, with "candle" as the mode option ([BO_functions](https://github.com/boringlee24/SIMBO/blob/main/BO/BO_functions.py#L83)) to update the ```query/lookups/candle.json``` file. The keys in the ```candle.json``` file corresponds to the three numbers X, Y and Z when running the ```distributor.py``` script, and the returns of this script correspond to the first and second item in the value list of ```candle.json``` (total cost and QoS satisfaction rate). After updating all values in the ```candle.json``` dictionary, go back to the ```BO/``` directory and re-run the main scripts the same way as the README file.

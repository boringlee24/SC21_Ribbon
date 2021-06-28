# SIMBO

SIMBO applies a Bayesian Optimization (BO) engine for heterogeneous instance serving of ML inference queries.

## Dependencies

With Python 3.7 ready, the other required packages can be installed with command
```shell
pip install -r requirements.txt
```

## Bayesian Optimization Engine Setup

SIMBO uses a modified public open-source BO library from [fmfn](https://github.com/fmfn/BayesianOptimization)

To setup the BO backend, clone the repo, copy the source file over and build the library

```shell
cd /<usr_git_dir> # replace with custom path
git clone https://github.com/fmfn/BayesianOptimization.git
cp SIMBO/bayesian_optimization.py BayesianOptimization/bayes_opt
cp SIMBO/util.py BayesianOptimization/bayes_opt
cd BayesianOptimization
python setup.py build
PYTHONPATH="$PYTHONPATH:/<usr_gir_dir>/BayesianOptimization/build/lib" # make sure python sees this library
export PYTHONPATH
cd /<usr_git_dir>/SIMBO
```
## Inference models

The source code for evaluated models are in the ```models``` directory. The characterization data of each model on various instances are in the ```characterization``` directory. To verify the characterization data, navigate to the ```models``` directory, follow the instructions to run the benchmarks, and compare the collected logs with data in ```characterization```.

Here are the links to each model implementation.

1. CANDLE (cancer distributed learning environment) Combo model: [link](https://github.com/ECP-CANDLE/Benchmarks/tree/master/Pilot1/Combo)
2. VGG model: [link](https://keras.io/api/applications/vgg/)
3. ResNet model: [link](https://keras.io/api/applications/resnet/)
4. MT-WND (multi-task wide and deep): [link](https://github.com/harvard-acc/DeepRecSys/blob/master/models/multi_task_wnd.py)
5. DIEN (deep interest evolution network): [link](https://github.com/harvard-acc/DeepRecSys/blob/master/models/dien.py)

## Start SIMBO

The characterization data is used to evaluate whether a certain configuration meets the target QoS. First extract the zipped file.

```shell
cd characterization
tar -xf logs.tar.gz
cd ../
```

Navigate to the BO directory, run SIMBO and all competing schemes

```shell
cd BO/
./all_scheme.sh
```

To visualize the comparison, run

```shell
cd visualize
python num_of_samples.py
python explore_cost.py
```

After running the visualization scripts, new figures will appear in the ```visualize``` directory. The ```num_of_samples.png``` picture shows the number of samples to find the optimal instance pool for all schemes, the ```explore_cost.png``` picture shows the total cost of exploration for all schemes.

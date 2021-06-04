# SIMBO

SIMBO applies a Bayesian Optimization (BO) engine for heterogeneous instance serving of ML inference queries.

## Dependencies

* Python 3.7
* Numpy 1.18
* Scipy 1.4
* Scikit-learn 0.24

## Bayesian Optimization Engine Setup

SIMBO uses a modified public open-source BO library from [fmfn](https://github.com/fmfn/BayesianOptimization)

To setup the BO backend, clone the repo, copy the source file over and build the library

```shell
cd /usr_git_dir # replace with custom path
git clone https://github.com/fmfn/BayesianOptimization.git
cp SIMBO/bayesian_optimization.py BayesianOptimization/bayes_opt
cd BayesianOptimization
python setup.py build
PYTHONPATH="$PYTHONPATH:/usr_gir_dir/BayesianOptimization/build/lib" # make sure python sees this library
export PYTHONPATH
cd /usr_git_dir/SIMBO
```

## Inference models

The source code for evaluated models are in the ```models``` directory. The characterization data of each model on various instances are in the ```characterization``` directory.

Here are the links to each model implementation.

1. CANDLE (cancer distributed learning environment) Combo model: [link](https://github.com/ECP-CANDLE/Benchmarks/tree/master/Pilot1/Combo)
2. VGG model: [link](https://keras.io/api/applications/vgg/)
3. ResNet model: [link](https://keras.io/api/applications/resnet/)
4. MT-WND (multi-task wide and deep): [link](https://github.com/harvard-acc/DeepRecSys/blob/master/models/multi_task_wnd.py)
5. DIEN (deep interest evolution network): [link](https://github.com/harvard-acc/DeepRecSys/blob/master/models/dien.py)

## Start SIMBO

```shell
python simbo.py
```

## Compare SIMBO against other schemes

```shell
python all_scheme.py
```
To visualize the comparison, run

```shell
python visualize.py
```

The ```result.pdf``` shows the visualized comparison.

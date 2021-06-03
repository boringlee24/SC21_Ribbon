# SIMBO

SIMBO applies a Bayesian Optimization (BO) engine for heterogeneous instance serving of ML inference queries.

## Dependencies

* Python3
* Numpy
* Scipy
* Scikit-learn

## Bayesian Optimization Engine Setup

SIMBO uses a modified public open-source BO library from [fmfn](https://github.com/fmfn/BayesianOptimization)

To setup the BO backend, clone the repo, copy the source file over and build the library

```shell
cd /usr_git_dir # replace with custom path
git clone https://github.com/fmfn/BayesianOptimization.git
cp SIMBO/bayesian_optimization.py BayesianOptimization/bayes_opt
cd BayesianOptimization
python setup.py build
Add build directory to python path
```


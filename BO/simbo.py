import pandas
import pdb
import matplotlib
import numpy as np
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import glob
import sys
#from matplotlib.ticker import MultipleLocator
import json
import os
import math
import random
import time
import BO_functions
from termcolor import colored
import subprocess
from BO_functions import BO
from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed
import multiprocessing

######################
# when doing prune, change the following
# acq = 'ei_prune'
# add optimier.explore_or_prune_XD
# in BO library set normalize_y to True, point Matern kernel to customized one
#####################

acq = 'ei_prune'
models = ['candle', 'resnet', 'vgg', 'mtwnd', 'dien']
with open('configs/homogeneous.json') as f:
    homo_key = json.load(f)
with open('configs/saving.json') as f:
    saving = json.load(f)

num_iter = 100 # run monte-carlo 
xi = 0.1

def check_step(optimizer, data, current_iter): 
    # check if data has reached a score
    score = optimizer._space.target[-1]
    if optimizer._space.target[-1] >= 0.5: 
        checks = [k for k,v in data.items() if v == 0]
        for check in checks:
            if score >= check:
                data[check] = current_iter
    return data

def inner_loop(iteration, scores):
    print(f'trial: {iteration}')
    BO_functions.model = model
    data = {}
    for j in scores:
        # each score is associated with the number of samples to reach it
        data[j] = 0
    current_iter = 0
    optimizer = BO(f=BO_functions.obj_function_3D, pbounds=pbounds, random_state=iteration, verbose=0)
    #optimizer.probe(params={"x": homo_base, "y": 0, "z": 0},lazy=True)
    optimizer.maximize(init_points=1, n_iter=0,acq=acq,xi=xi)
    data = check_step(optimizer, data, current_iter)
    violate_dict = BO_functions.get_violate_dict()
    optimizer.bo_prune(violate_dict)
    optimizer.maximize(init_points=1, n_iter=0,acq=acq,xi=xi)
    data = check_step(optimizer, data, current_iter)
    violate_dict = BO_functions.get_violate_dict()
    optimizer.bo_prune(violate_dict)

    while data[optimal] == 0: # run till it converges to oracle
        try: 
            optimizer.maximize(init_points=0, n_iter=1,acq=acq,xi=xi)
        except ValueError:
            for k,v in data.items():
                if v == 0:
                    data[k] = None
            print(f'value error occured on model {model}, BO trial {current_iter}')
            break

        current_iter += 1
        data = check_step(optimizer, data, current_iter)
        violate_dict = BO_functions.get_violate_dict()
        optimizer.bo_prune(violate_dict)
        if current_iter >= 200:
            # set unreached point to None
            for k,v in data.items():
                if v == 0:
                    data[k] = None
            break
    targets = optimizer._space.target.tolist()
    within_qos = [k*99*2 for k in targets if k <= 99/99/2] # QoS rate for configs that violates QoS
    qos_rate = len(within_qos)

    explored_points = np.round(optimizer._space._params).tolist()
    cost = np.mean([BO_functions.total_price(model, *p) for p in explored_points])

    return [data, qos_rate, cost]

usable_cores = os.sched_getaffinity(0)
#usable_cores = [1]#

for model in models:
    print(f'model: {model}')
    BO_functions.model = model
    xmax, ymax, zmax = BO_functions.max_instance(model)
    pbounds = {'x': (0, xmax), 'y': (0, ymax), 'z': (0, zmax)}

    homo_p = BO_functions.total_price(model, homo_key[model], 0, 0)
    saving_arr = np.array(saving[model][::-1])
    hetero_p = homo_p * (1 - saving_arr / 100)
    hetero_p = np.array([round(val,2) for val in hetero_p])
    scores = list(1/2 + (1-hetero_p / BO_functions.max_price(model)) / 2)

    optimal = max(scores)
    # record number of samples needed to reach the score
    summary = {}
    qos_rate = []
    cost = []
    for j in hetero_p:
        summary[j] = []
    results = Parallel(n_jobs=len(usable_cores))(delayed(inner_loop)(i,scores) for i in range(num_iter))
    for result in results:
        qos_rate.append(result[1])
        cost.append(result[2])

        for j in scores:
            ind = scores.index(j)
            summary[hetero_p[ind]].append(result[0][j])

    with open(f'../BO/result/{model}_simbo.json', 'w') as f: 
        json.dump(summary, f, indent=4)
    with open(f'../BO/result/qos_rate/{model}_simbo.json', 'w') as f: 
        json.dump(qos_rate, f, indent=4)
    with open(f'../BO/result/cost/{model}_simbo.json', 'w') as f: 
        json.dump(cost, f, indent=4)


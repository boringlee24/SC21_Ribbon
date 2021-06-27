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
from joblib import Parallel, delayed
import multiprocessing

models = ['candle', 'resnet', 'vgg', 'mtwnd', 'dien'] 
with open('configs/homogeneous.json') as f:
    homo_key = json.load(f)
with open('configs/saving.json') as f:
    saving = json.load(f)
num_iter = 100 # run monte-carlo

def check_step(gd, data): 
    # check if data has reached a score
    price = gd.best_price
    current_iter = gd.num_iter
    checks = [k for k,v in data.items() if v == 0]
    for check in checks:
        if price <= check:
            data[check] = current_iter
    return data

def inner_loop(iteration, prices):
    print(f'trial: {iteration}')
    BO_functions.model = model
    data = {} # num of iters needed to reach certain price
    for j in prices:
        # each price is associated with the number of samples to reach it
        data[j] = 0
    gd = BO_functions.Random_Prune(remain, seed=iteration)
    total_space = len(gd.remain)
    while data[optimal] == 0:
        gd.iterate()
        data = check_step(gd, data)
    if gd.num_iter + len(gd.remain) + gd.num_pruned != total_space: # minus one initialization point
        print('error with trial counting')
        sys.exit()
    targets = gd.rate_history
    within_qos = [k for k in targets if k <= 99]
    qos_rate = len(within_qos)# / len(targets) * 100
    explored_points = gd.config_history
    cost = np.mean([BO_functions.total_price(model, *p) for p in explored_points])
    return [data, qos_rate, cost]

usable_cores = os.sched_getaffinity(0)
#usable_cores = [1]#

for model in models:
    BO_functions.model = model
    xmax, ymax, zmax = BO_functions.max_instance(model)
    pbounds = {'x': (0, xmax), 'y': (0, ymax), 'z': (0, zmax)}
    remain = []
    for x in range(0, xmax+1):
        for y in range(0, ymax+1):
            for z in range(0, zmax+1):
                point = x, y, z
                remain.append(point)

    homo_p = BO_functions.total_price(model, homo_key[model], 0, 0)
    saving_arr = np.array(saving[model][::-1])
    hetero_p = homo_p * (1 - saving_arr / 100)
    prices = [round(val,2) for val in hetero_p]
    optimal = min(prices)
    # record number of samples needed to reach the score
    summary = {}
    qos_rate = []
    cost = []
    for j in prices:
        summary[j] = []
    results = Parallel(n_jobs=len(usable_cores))(delayed(inner_loop)(i,prices) for i in range(num_iter))

    for result in results:
        qos_rate.append(result[1])
        cost.append(result[2])

        for j in prices:
            ind = prices.index(j)
            summary[prices[ind]].append(result[0][j])

    with open(f'../BO/result/{model}_rand_plus.json', 'w') as f: 
        json.dump(summary, f, indent=4)
    with open(f'../BO/result/qos_rate/{model}_rand_plus.json', 'w') as f: 
        json.dump(qos_rate, f, indent=4)
    with open(f'../BO/result/cost/{model}_rand_plus.json', 'w') as f: 
        json.dump(cost, f, indent=4)



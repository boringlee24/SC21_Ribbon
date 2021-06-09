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

models = ['vgg'] #['candle', 'resnet', 'vgg'] TODO
homo_key = {'candle': '8, 0, 0', 'resnet': '8, 0, 0', 'vgg': '9, 0, 0'} 
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
    print(f'iteration: {iteration}')
    BO_functions.model = model
    data = {} # num of iters needed to reach certain price
    for j in prices:
        # each price is associated with the number of samples to reach it
        data[j] = 0
    gd = BO_functions.RSM_gradient(remain, seed=iteration)
    total_space = len(gd.remain)
    ccf_list = gd.gen_ccf()
    for point in ccf_list:
        point = tuple(point)
        gd.rank_iter(point)
        data = check_step(gd, data)
    gd.rank_ccf()
    starting_point = gd.get_sp()
#    pdb.set_trace()
    gd.initialize(starting_point)
    data = check_step(gd, data)
    while data[optimal] == 0:
        gd.iterate()
        data = check_step(gd, data)
    if gd.num_iter + len(gd.remain) != total_space: # minus one initialization point
        print('error with iteration counting')
        sys.exit()
    targets = gd.rate_history
    within_qos = [k for k in targets if k <= 99]
    qos_rate = len(within_qos)# / len(targets) * 100
    explored_points = gd.config_history
    cost = np.mean([BO_functions.total_price(model, *p) for p in explored_points])
    return [data, qos_rate, cost]

usable_cores = os.sched_getaffinity(0)
#usable_cores = [1] 

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

    path = f'data/{model}/99.json' 
    with open(path, 'r') as f:
        output = json.load(f)
    homo_p = output[homo_key[model]][0]
    hetero_p = np.unique(np.array([v[0] for k,v in output.items() if v[0] < homo_p]))
    prices = list(hetero_p)
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
            summary[hetero_p[ind]].append(result[0][j])

    with open(f'data/BO/data/monte_carlo/{model}_rsm_grad.json', 'w') as f:
        json.dump(summary, f, indent=4)
    with open(f'data/BO/data/monte_carlo/qos_rate/{model}_rsm_grad.json', 'w') as f: 
        json.dump(qos_rate, f, indent=4)
    with open(f'data/BO/data/monte_carlo/cost/{model}_rsm_grad.json', 'w') as f: 
        json.dump(cost, f, indent=4)



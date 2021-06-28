import pandas
import pdb
from datetime import datetime
import matplotlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import sys
from matplotlib.ticker import MultipleLocator
import json
import os

models = ['candle', 'resnet', 'vgg', 'mtwnd', 'dien']
MODELS = ['CANDLE', 'ResNet50', 'VGG19', 'MT-WND', 'DIEN']
methods = ['gradient', 'rand_plus', 'rsm', 'simbo']
names = ['Hill-Climb', 'RAND', 'RSM', 'SIMBO'] 
colors = ['deepskyblue', 'purple', 'green', 'orangered']

fig, axs = plt.subplots(1, 1, gridspec_kw={'hspace': 0, 'wspace': 0, 'bottom': 0.13, 'top': 0.83, 'right':0.995, 'left':0.123}, figsize=(7,2.3))
data = {}
bar_offset = [-1.5, -0.5, 0.5, 1.5]

def remove_none(a):
    return [val for val in a if val is not None]

x = np.arange(len(models))
width=0.2
for method in methods:
    data[method] = []
    ind = methods.index(method)
    for model in models:
        path = f'../result/cost/{model}_{method}.json'
        with open(path, 'r') as f:
            output = json.load(f)
        avg_cost = np.mean(output)

        path = f'../result/{model}_{method}.json'
        with open(path, 'r') as f:
            output = json.load(f)
        num_sample = np.mean(remove_none(list(output.values())[0]))
        total_cost = num_sample * avg_cost

        # get exclusive search cost
        path = f'../../query/lookups/{model}.json'
        with open(path, 'r') as f:
            output = json.load(f)
        all_cost = [x[0] for x in list(output.values())]
        all_cost = np.sum(all_cost)
        data[method].append(total_cost / all_cost * 100)

    axs.bar(x+width*bar_offset[ind], data[method], width=width, color=colors[ind], label=names[ind], edgecolor='black', zorder=3)
axs.set_xticks(x)
axs.set_xticklabels(MODELS, fontsize=14)
axs.legend(loc='lower left', fontsize=13, ncol=4, mode='expand', borderaxespad=0., bbox_to_anchor=(0., 1.02, 1., .102), borderpad=0.3,
           handletextpad=0.5, columnspacing=0.4, edgecolor='black', handlelength=1.5)
axs.set_ylim(1,30)
axs.tick_params(axis = 'y', which = 'major', labelsize = 13)
#axs.set_ylabel('Mean Sample\nPrice (Normalized)', fontsize=14)
axs.set_ylabel('Search Cost (%)', fontsize=14)
axs.set_yscale('log')
axs.grid(which='major', axis='y', ls='dotted', zorder=0)
plt.savefig('explore_cost.png')



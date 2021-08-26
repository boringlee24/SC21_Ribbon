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

data = {}
methods = ['gradient', 'rand_plus', 'rsm', 'ribbon']
names = ['Hill-Cl.', 'RAND', 'RSM', 'SIMBO']
colors = ['deepskyblue', 'purple', 'green', 'orangered']

with open('../../query/configs/saving.json') as f:
    x_stamps_line = json.load(f)
with open('../../query/configs/homogeneous.json') as f:
    homo_key = json.load(f)

def remove_none(a):
    return [val for val in a if val is not None]

def process_output(output, model):
    y = np.array([np.mean(remove_none(v)) for k,v in output.items()][::-1])
    return y

fig, axs = plt.subplots(1, 5, gridspec_kw={'hspace': 0.3, 'wspace': 0.3, 'top':0.89, 'bottom': 0.2, 'right':0.983, 'left':0.065}, figsize=(14,2.5))

for model in models:
    i = models.index(model)
    axs[i].set_title(MODELS[i], fontsize=14)
    for method in methods:
        j = methods.index(method)

        path = f'../result/{model}_{method}.json'
        with open(path, 'r') as f:
            output = json.load(f)
        outy = process_output(output, model)
        outx = x_stamps_line[model]
        if method == 'simbo':
            marker = 'd'
        else:
            marker = 'o'
        axs[i].plot(outx, outy, label=names[j], marker=marker, color=colors[j])
    axs[i].set_yscale('log')
    axs[i].set_xticks(x_stamps_line[model])
    axs[i].yaxis.set_major_locator(MultipleLocator(100))
    axs[i].tick_params(axis = 'both', which = 'major', labelsize = 13)
    axs[i].set_ylim(10, 1000)
    axs[i].set_yticks([10,100,1000])

handles, labels = axs[-1].get_legend_handles_labels()
axs[2].legend(loc='upper left', fontsize=11, ncol=2, borderaxespad=0.2, handletextpad=0.2, columnspacing=0.4, edgecolor='black', handlelength=1.5)
axs[-1].set_ylim(10, 180)
fig.text(0.5, 0.04, 'Cost Saving Compared to Homogeneous Configuration (%)', ha='center', va='center', fontsize=14)

axs[0].set_ylabel('Number of\nSamples (Log Scale)', fontsize=14)
plt.savefig('num_of_samples.png')


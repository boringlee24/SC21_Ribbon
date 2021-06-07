"""
#Trains a ResNet on the CIFAR10 dataset.

"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from keras import models, layers, optimizers
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import pdb
import sys
import argparse
from pathlib import Path
import time
import random
import json

parser = argparse.ArgumentParser(description='Tensorflow Cifar10 Training')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--model', metavar='MODEL', type=str, help='specific model name', default='resnet50')
parser.add_argument('--lr', metavar='LEARNING_RATE', type=float, help='learning rate', default=0.001)
parser.add_argument('--testcase', metavar='TC', type=str)
parser.add_argument('--batch_min', default=10, type=int, metavar='MIN')
parser.add_argument('--batch_max', default=501, type=int, metavar='MAX')

args = parser.parse_args()

# Training parameters
batch_size = args.batch_size  # orig paper trained all networks with batch_size=128
epochs = 10
data_augmentation = True
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

class GetInfBatchLat(keras.callbacks.Callback):
    def __init__(self, batch_size, testcase):
        super(GetInfBatchLat, self).__init__()
        self.lat_list = []
        self.s_time = 0
        self.testcase = testcase
        self.batch_size = batch_size
        
#    def on_predict_batch_begin(self, batch, logs=None):
#        self.s_time = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        if self.s_time != 0:
            lat_ms = round((time.time() - self.s_time) * 1000,2)
            self.lat_list.append(lat_ms)
        self.s_time = time.time()
    def on_predict_end(self, logs=None):
        # remove highest percentile latencies (warmup)
        outlier = np.percentile(self.lat_list,95)
        filtered = [x for x in self.lat_list if x <= outlier]
        if len(filtered) <= 100:
            self.lat_list = filtered
        else:
            self.lat_list = random.sample(filtered, 100)
        Path(f"logs/{self.testcase}").mkdir(parents=True, exist_ok=True)
        with open(f'logs/{self.testcase}/resnet_{self.batch_size}_1.json', 'w') as f:
            json.dump(self.lat_list, f, indent=4)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.models.load_model('resnet50.h5')

for batch_size in range(args.batch_min, args.batch_max, 10):
    print(f'testing Resnet on batch size {batch_size}')
    model.predict(x_train, batch_size=batch_size, verbose=1, callbacks=[GetInfBatchLat(batch_size, args.testcase)])



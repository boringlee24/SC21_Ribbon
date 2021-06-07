#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import collections
import logging
import os
import random
import threading

import numpy as np
import pandas as pd

from itertools import cycle, islice

import keras
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from keras.utils import get_custom_objects
from keras.utils.vis_utils import plot_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from scipy.stats.stats import pearsonr

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import combo

import NCI60
import combo
import candle
import pdb
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)

    random.seed(seed)

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        tf.set_random_seed(seed)
        # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # K.set_session(sess)

        # Uncommit when running on an optimized tensorflow where NUM_INTER_THREADS and
        # NUM_INTRA_THREADS env vars are set.
        # session_conf = tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
        #	intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS']))
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # K.set_session(sess)


def discretize(y, bins=5):
    percentiles = [100 / bins * (i + 1) for i in range(bins - 1)]
    thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    return classes


class ComboDataLoader(object):
    """Load merged drug response, drug descriptors and cell line essay data
    """

    def __init__(self, seed, val_split=0.2, shuffle=True,
                 cell_features=['expression'], drug_features=['descriptors'],
                 response_url=None, use_landmark_genes=False,
                 use_combo_score=False, use_mean_growth=False,
                 preprocess_rnaseq=None, exclude_cells=[], exclude_drugs=[],
                 feature_subsample=None, scaling='std', scramble=False,
                 cv_partition='overlapping', cv=0):
        """Initialize data merging drug response, drug descriptors and cell line essay.
           Shuffle and split training and validation set

        Parameters
        ----------
        seed: integer
            seed for random generation
        val_split : float, optional (default 0.2)
            fraction of data to use in validation
        cell_features: list of strings from 'expression', 'expression_5platform', 'mirna', 'proteome', 'all', 'categorical' (default ['expression'])
            use one or more cell line feature sets: gene expression, microRNA, proteome
            use 'all' for ['expression', 'mirna', 'proteome']
            use 'categorical' for one-hot encoded cell lines
        drug_features: list of strings from 'descriptors', 'latent', 'all', 'categorical', 'noise' (default ['descriptors'])
            use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder
            trained on NSC drugs, or both; use random features if set to noise
            use 'categorical' for one-hot encoded drugs
        shuffle : True or False, optional (default True)
            if True shuffles the merged data before splitting training and validation sets
        scramble: True or False, optional (default False)
            if True randomly shuffle dose response data as a control
        feature_subsample: None or integer (default None)
            number of feature columns to use from cellline expressions and drug descriptors
        use_landmark_genes: True or False
            only use LINCS1000 landmark genes
        use_combo_score: bool (default False)
            use combination score in place of percent growth (stored in 'GROWTH' column)
        use_mean_growth: bool (default False)
            use mean aggregation instead of min on percent growth
        scaling: None, 'std', 'minmax' or 'maxabs' (default 'std')
            type of feature scaling: 'maxabs' to [-1,1], 'maxabs' to [-1, 1], 'std' for standard normalization
        """

        self.cv_partition = cv_partition

        np.random.seed(seed)

        df = NCI60.load_combo_response(response_url=response_url, use_combo_score=use_combo_score, use_mean_growth=use_mean_growth, fraction=True, exclude_cells=exclude_cells, exclude_drugs=exclude_drugs)
        logger.info('Loaded {} unique (CL, D1, D2) response sets.'.format(df.shape[0]))

        if 'all' in cell_features:
            self.cell_features = ['expression', 'mirna', 'proteome']
        else:
            self.cell_features = cell_features

        if 'all' in drug_features:
            self.drug_features = ['descriptors', 'latent']
        else:
            self.drug_features = drug_features

        for fea in self.cell_features:
            if fea == 'expression' or fea == 'rnaseq':
                self.df_cell_expr = NCI60.load_cell_expression_rnaseq(ncols=feature_subsample, scaling=scaling, use_landmark_genes=use_landmark_genes, preprocess_rnaseq=preprocess_rnaseq)
                df = df.merge(self.df_cell_expr[['CELLNAME']], on='CELLNAME')
            elif fea == 'expression_u133p2':
                self.df_cell_expr = NCI60.load_cell_expression_u133p2(ncols=feature_subsample, scaling=scaling, use_landmark_genes=use_landmark_genes)
                df = df.merge(self.df_cell_expr[['CELLNAME']], on='CELLNAME')
            elif fea == 'expression_5platform':
                self.df_cell_expr = NCI60.load_cell_expression_5platform(ncols=feature_subsample, scaling=scaling, use_landmark_genes=use_landmark_genes)
                df = df.merge(self.df_cell_expr[['CELLNAME']], on='CELLNAME')
            elif fea == 'mirna':
                self.df_cell_mirna = NCI60.load_cell_mirna(ncols=feature_subsample, scaling=scaling)
                df = df.merge(self.df_cell_mirna[['CELLNAME']], on='CELLNAME')
            elif fea == 'proteome':
                self.df_cell_prot = NCI60.load_cell_proteome(ncols=feature_subsample, scaling=scaling)
                df = df.merge(self.df_cell_prot[['CELLNAME']], on='CELLNAME')
            elif fea == 'categorical':
                df_cell_ids = df[['CELLNAME']].drop_duplicates()
                cell_ids = df_cell_ids['CELLNAME'].map(lambda x: x.replace(':', '.'))
                df_cell_cat = pd.get_dummies(cell_ids)
                df_cell_cat.index = df_cell_ids['CELLNAME']
                self.df_cell_cat = df_cell_cat.reset_index()

        for fea in self.drug_features:
            if fea == 'descriptors':
                self.df_drug_desc = NCI60.load_drug_descriptors(ncols=feature_subsample, scaling=scaling)
                df = df[df['NSC1'].isin(self.df_drug_desc['NSC']) & df['NSC2'].isin(self.df_drug_desc['NSC'])]
            elif fea == 'latent':
                self.df_drug_auen = NCI60.load_drug_autoencoded_AG(ncols=feature_subsample, scaling=scaling)
                df = df[df['NSC1'].isin(self.df_drug_auen['NSC']) & df['NSC2'].isin(self.df_drug_auen['NSC'])]
            elif fea == 'categorical':
                df_drug_ids = df[['NSC1']].drop_duplicates()
                df_drug_ids.columns = ['NSC']
                drug_ids = df_drug_ids['NSC']
                df_drug_cat = pd.get_dummies(drug_ids)
                df_drug_cat.index = df_drug_ids['NSC']
                self.df_drug_cat = df_drug_cat.reset_index()
            elif fea == 'noise':
                ids1 = df[['NSC1']].drop_duplicates().rename(columns={'NSC1':'NSC'})
                ids2 = df[['NSC2']].drop_duplicates().rename(columns={'NSC2':'NSC'})
                df_drug_ids = pd.concat([ids1, ids2]).drop_duplicates()
                noise = np.random.normal(size=(df_drug_ids.shape[0], 500))
                df_rand = pd.DataFrame(noise, index=df_drug_ids['NSC'],
                                       columns=['RAND-{:03d}'.format(x) for x in range(500)])
                self.df_drug_rand = df_rand.reset_index()

        logger.info('Filtered down to {} rows with matching information.'.format(df.shape[0]))

        ids1 = df[['NSC1']].drop_duplicates().rename(columns={'NSC1':'NSC'})
        ids2 = df[['NSC2']].drop_duplicates().rename(columns={'NSC2':'NSC'})
        df_drug_ids = pd.concat([ids1, ids2]).drop_duplicates().reset_index(drop=True)

        n_drugs = df_drug_ids.shape[0]
        n_val_drugs = int(n_drugs * val_split)
        n_train_drugs = n_drugs - n_val_drugs

        logger.info('Unique cell lines: {}'.format(df['CELLNAME'].nunique()))
        logger.info('Unique drugs: {}'.format(n_drugs))
        # df.to_csv('filtered.growth.min.tsv', sep='\t', index=False, float_format='%.4g')
        # df.to_csv('filtered.score.max.tsv', sep='\t', index=False, float_format='%.4g')

        if shuffle:
            df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            df_drug_ids = df_drug_ids.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        self.df_response = df
        self.df_drug_ids = df_drug_ids

        self.train_drug_ids = df_drug_ids['NSC'][:n_train_drugs]
        self.val_drug_ids = df_drug_ids['NSC'][-n_val_drugs:]

        if scramble:
            growth = df[['GROWTH']]
            random_growth = growth.iloc[np.random.permutation(np.arange(growth.shape[0]))].reset_index()
            self.df_response[['GROWTH']] = random_growth['GROWTH']
            logger.warn('Randomly shuffled dose response growth values.')

        logger.info('Distribution of dose response:')
        logger.info(self.df_response[['GROWTH']].describe())

        self.total = df.shape[0]
        self.n_val = int(self.total * val_split)
        self.n_train = self.total - self.n_val
        logger.info('Rows in train: {}, val: {}'.format(self.n_train, self.n_val))

        self.cell_df_dict = {'expression': 'df_cell_expr',
                             'expression_5platform': 'df_cell_expr',
                             'expression_u133p2': 'df_cell_expr',
                             'rnaseq': 'df_cell_expr',
                             'mirna': 'df_cell_mirna',
                             'proteome': 'df_cell_prot',
                             'categorical': 'df_cell_cat'}

        self.drug_df_dict = {'descriptors': 'df_drug_desc',
                             'latent': 'df_drug_auen',
                             'categorical': 'df_drug_cat',
                             'noise': 'df_drug_rand'}

        self.input_features = collections.OrderedDict()
        self.feature_shapes = {}
        for fea in self.cell_features:
            feature_type = 'cell.' + fea
            feature_name = 'cell.' + fea
            df_cell = getattr(self, self.cell_df_dict[fea])
            self.input_features[feature_name] = feature_type
            self.feature_shapes[feature_type] = (df_cell.shape[1] - 1,)

        for drug in ['drug1', 'drug2']:
            for fea in self.drug_features:
                feature_type = 'drug.' + fea
                feature_name = drug + '.' + fea
                df_drug = getattr(self, self.drug_df_dict[fea])
                self.input_features[feature_name] = feature_type
                self.feature_shapes[feature_type] = (df_drug.shape[1] - 1,)

        logger.info('Input features shapes:')
        for k, v in self.input_features.items():
            logger.info('  {}: {}'.format(k, self.feature_shapes[v]))

        self.input_dim = sum([np.prod(self.feature_shapes[x]) for x in self.input_features.values()])
        logger.info('Total input dimensions: {}'.format(self.input_dim))

        if cv > 1:
            if cv_partition == 'disjoint':
                pass
            elif cv_partition == 'disjoint_cells':
                y = self.df_response['GROWTH'].values
                groups = self.df_response['CELLNAME'].values
                gkf = GroupKFold(n_splits=cv)
                splits = gkf.split(y, groups=groups)
                self.cv_train_indexes = []
                self.cv_val_indexes = []
                for index, (train_index, val_index) in enumerate(splits):
                    print(index, train_index)
                    self.cv_train_indexes.append(train_index)
                    self.cv_val_indexes.append(val_index)
            else:
                y = self.df_response['GROWTH'].values
                # kf = KFold(n_splits=cv)
                # splits = kf.split(y)
                skf = StratifiedKFold(n_splits=cv, random_state=seed)
                splits = skf.split(y, discretize(y, bins=cv))
                self.cv_train_indexes = []
                self.cv_val_indexes = []
                for index, (train_index, val_index) in enumerate(splits):
                    print(index, train_index)
                    self.cv_train_indexes.append(train_index)
                    self.cv_val_indexes.append(val_index)

    def load_data_all(self, switch_drugs=False):
        df_all = self.df_response
        y_all = df_all['GROWTH'].values
        x_all_list = []

        for fea in self.cell_features:
            df_cell = getattr(self, self.cell_df_dict[fea])
            df_x_all = pd.merge(df_all[['CELLNAME']], df_cell, on='CELLNAME', how='left')
            x_all_list.append(df_x_all.drop(['CELLNAME'], axis=1).values)

        # for fea in loader.cell_features:
        #     df_cell = getattr(loader, loader.cell_df_dict[fea])
        #     df_x_all = pd.merge(df_all[['CELLNAME']], df_cell, on='CELLNAME', how='left')
        #     df_x_all[:1000].to_csv('df.{}.1k.csv'.format(fea), index=False, float_format="%g")

        drugs = ['NSC1', 'NSC2']
        if switch_drugs:
            drugs = ['NSC2', 'NSC1']

        for drug in drugs:
            for fea in self.drug_features:
                df_drug = getattr(self, self.drug_df_dict[fea])
                df_x_all = pd.merge(df_all[[drug]], df_drug, left_on=drug, right_on='NSC', how='left')
                x_all_list.append(df_x_all.drop([drug, 'NSC'], axis=1).values)

        # for drug in drugs:
        #     for fea in loader.drug_features:
        #         df_drug = getattr(loader, loader.drug_df_dict[fea])
        #         df_x_all = pd.merge(df_all[[drug]], df_drug, left_on=drug, right_on='NSC', how='left')
        #         print(df_x_all.shape)
        #         df_x_all[:1000].drop([drug], axis=1).to_csv('df.{}.{}.1k.csv'.format(drug, fea), index=False, float_format="%g")

        # df_all[:1000].to_csv('df.growth.1k.csv', index=False, float_format="%g")

        return x_all_list, y_all, df_all

    def load_data_by_index(self, train_index, val_index):
        x_all_list, y_all, df_all = self.load_data_all()
        x_train_list = [x[train_index] for x in x_all_list]
        x_val_list = [x[val_index] for x in x_all_list]
        y_train = y_all[train_index]
        y_val = y_all[val_index]
        df_train = df_all.iloc[train_index, :]
        df_val = df_all.iloc[val_index, :]
        if self.cv_partition == 'disjoint':
            logger.info('Training drugs: {}'.format(set(df_train['NSC1'])))
            logger.info('Validation drugs: {}'.format(set(df_val['NSC1'])))
        elif self.cv_partition == 'disjoint_cells':
            logger.info('Training cells: {}'.format(set(df_train['CELLNAME'])))
            logger.info('Validation cells: {}'.format(set(df_val['CELLNAME'])))
        return x_train_list, y_train, x_val_list, y_val, df_train, df_val

    def load_data_cv(self, fold):
        train_index = self.cv_train_indexes[fold]
        val_index = self.cv_val_indexes[fold]
        # print('fold', fold)
        # print(train_index[:5])
        return self.load_data_by_index(train_index, val_index)

    def load_data(self):
        if self.cv_partition == 'disjoint':
            train_index = self.df_response[(self.df_response['NSC1'].isin(self.train_drug_ids)) & (self.df_response['NSC2'].isin(self.train_drug_ids))].index
            val_index = self.df_response[(self.df_response['NSC1'].isin(self.val_drug_ids)) & (self.df_response['NSC2'].isin(self.val_drug_ids))].index
        else:
            train_index = range(self.n_train)
            val_index = range(self.n_train, self.total)
        return self.load_data_by_index(train_index, val_index)

    def load_data_old(self):
        # bad performance (4x slow) possibly due to incontiguous data
        df_train = self.df_response.iloc[:self.n_train, :]
        df_val = self.df_response.iloc[self.n_train:, :]

        y_train = df_train['GROWTH'].values
        y_val = df_val['GROWTH'].values

        x_train_list = []
        x_val_list = []

        for fea in self.cell_features:
            df_cell = getattr(self, self.cell_df_dict[fea])
            df_x_train = pd.merge(df_train[['CELLNAME']], df_cell, on='CELLNAME', how='left')
            df_x_val = pd.merge(df_val[['CELLNAME']], df_cell, on='CELLNAME', how='left')
            x_train_list.append(df_x_train.drop(['CELLNAME'], axis=1).values)
            x_val_list.append(df_x_val.drop(['CELLNAME'], axis=1).values)

        for drug in ['NSC1', 'NSC2']:
            for fea in self.drug_features:
                df_drug = getattr(self, self.drug_df_dict[fea])
                df_x_train = pd.merge(df_train[[drug]], df_drug, left_on=drug, right_on='NSC', how='left')
                df_x_val = pd.merge(df_val[[drug]], df_drug, left_on=drug, right_on='NSC', how='left')
                x_train_list.append(df_x_train.drop([drug, 'NSC'], axis=1).values)
                x_val_list.append(df_x_val.drop([drug, 'NSC'], axis=1).values)

        return x_train_list, y_train, x_val_list, y_val, df_train, df_val

def initialize_parameters(default_model = 'combo_default_model.txt'):

    # Build benchmark object
    comboBmk = combo.BenchmarkCombo(combo.file_path, default_model, 'keras',
        prog='combo_baseline',
        desc = 'Build neural network based models to predict tumor response to drug pairs.')

    # Initialize parameters
    gParameters = candle.finalize_parameters(comboBmk)
    #combo.logger.info('Params: {}'.format(gParameters))

    return gParameters

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

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
        self.lat_list = random.sample(filtered, 100) 
        Path(f"logs/{self.testcase}").mkdir(parents=True, exist_ok=True)
        with open(f'logs/{self.testcase}/candle_{self.batch_size}_1.json', 'w') as f:
            json.dump(self.lat_list, f, indent=4)

def run(params):
    args = Struct(**params)
    set_seed(args.rng_seed)

    loader = ComboDataLoader(seed=args.rng_seed,
                             val_split=args.val_split,
                             cell_features=args.cell_features,
                             drug_features=args.drug_features,
                             use_mean_growth=args.use_mean_growth,
                             response_url=args.response_url,
                             use_landmark_genes=args.use_landmark_genes,
                             preprocess_rnaseq=args.preprocess_rnaseq,
                             exclude_cells=args.exclude_cells,
                             exclude_drugs=args.exclude_drugs,
                             use_combo_score=args.use_combo_score,
                             cv_partition=args.cv_partition, cv=args.cv)

    model = keras.models.load_model('saved.model.h5', compile=False)
    model.load_weights('saved.weights.h5')
    model.summary()
    x_val_list = loader.load_data()[0]

    for batch_size in range(args.batch_min, args.batch_max, 10):
        print(f'testing CANDLE on batch size {batch_size}')
        model.predict(x_val_list, batch_size=batch_size, verbose=1, callbacks=[GetInfBatchLat(batch_size, args.testcase)])

def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()


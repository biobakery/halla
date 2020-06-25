'''Generate synthetic dataset X and Y

Params:

Generates: a directory that contains 3 files:
- X, Y: a paired dataset
- A   : contains the associated features between the paired dataset
'''

import argparse
import sys
import numpy as np
import itertools
import pandas as pd
import os
from os.path import join, isdir
import shutil

def parse_argument(args):
    parser = argparse.ArgumentParser(
        description='HAllA synthetic data generator - produces a pair of datasets X & Y with specified association among their features'
    )
    parser.add_argument('-n', '--samples', help='# samples in both X and Y', default=50, type=int, required=False)
    parser.add_argument('-xf', '--xfeatures', help='# features in X', default=500, type=int, required=False)
    parser.add_argument('-yf', '--yfeatures', help='# features in Y', default=500, type=int, required=False)
    parser.add_argument('-b', '--blocks', help='# significant blocks; default = min(5, min(xfeatures, yfeatures)/2)',
                        default=None, type=int, required=False)
    parser.add_argument('-a', '--association', help='association type {line, parabola}; default: line',
                        default='line', choices=['line', 'parabola'], required=False)
    parser.add_argument('-d', '--distribution', help='Distribution: {normal, uniform}',
                        default='uniform', choices=['normal', 'uniform'], required=False)
    # TODO: add noise distribution?
    parser.add_argument('-nw', '--noise-within', dest='noise_within', help='noise within blocks [0 (no noise)..1 (complete noise)]',
                        default=0.25, type=float, required=False)
    parser.add_argument('-nb', '--noise-between', dest='noise_between', help='noise between associated blocks [0 (no noise)..1 (complete noise)]',
                        default=0.25, type=float, required=False)
    parser.add_argument('-o', '--output', help='the output directory', required=True)
    
    # check requirements
    params = parser.parse_args()
    # samples must be > 0
    if params.samples <= 0: raise ValueError('# samples must be > 0')
    # xfeatures and yfeatures must be > 0
    if params.xfeatures <= 0 or params.yfeatures <= 0: raise ValueError('# features must be > 0')
    # blocks must be 1 .. min(5, min(xfeatures, yfeatures)/2)
    if params.blocks is None:
        params.blocks = min(5, min(params.xfeatures, params.yfeatures)/2)
    if not (params.blocks > 0 and params.blocks <= min(params.xfeatures, params.yfeatures)):
        raise ValueError('# blocks is invalid; must be [1..min(xfeatures, yfeatures)]')
    # noises must be [0..1]
    if params.noise_between < 0 or params.noise_between > 1 or \
        params.noise_within < 0 or params.noise_within > 1:
        raise ValueError('Noise within/between must be [0..1]')
    return(params)

def run_data_generator(sample_num=50, features_num=(500, 500), block_num=5, association='line', dist='uniform',
                        noise_within=0.25, noise_between=0.25, noise_within_magnitude=0.5):
    '''Distribution functions
    '''
    def uniform_dist_func(mat_size, low=-1, high=1):
        return(np.random.uniform(low=low, high=high, size=mat_size))
    def normal_dist_func(mat_size, loc=0, scale=1):
        return(np.random.normal(loc=loc, scale=scale, size=mat_size))
    
    '''Utility functions
    '''
    def div_features_into_blocks(feat_num):
        # initialize
        blocks_size = [0] * block_num
        assoc = [[]] * block_num
        # obtain block size
        for _ in range(feat_num): blocks_size[np.random.choice(block_num)] += 1
        # assign feature indices to blocks
        start_idx = 0
        for i in range(block_num):
            assoc[i] = [i for i in range(start_idx, start_idx + blocks_size[i])]
            start_idx = start_idx + blocks_size[i]
        return(assoc)

    rand_dist_func = uniform_dist_func if dist == 'uniform' else normal_dist_func
    # initialize X, Y, common base for generating X and Y, A
    x_feat_num, y_feat_num = features_num
    X = np.zeros((x_feat_num, sample_num))
    Y = np.zeros((y_feat_num, sample_num))
    common_base = rand_dist_func((block_num, sample_num)) + (rand_dist_func((block_num, 1)) * noise_between)
    # TODO: should we orthogonalize common_base?
    A = np.zeros(features_num)

    # assign features to blocks
    x_assoc, y_assoc = div_features_into_blocks(x_feat_num), div_features_into_blocks(y_feat_num)
    for block_i in range(block_num):
        # derive X features from common base
        for feat_x in x_assoc[block_i]:
            X[feat_x] = common_base[block_i] + noise_within * normal_dist_func(sample_num, scale=noise_within_magnitude)
        sign_corr = np.random.choice([-1, 1], p=[0.4, 0.6]) # arbitrary probabilities
        # derive Y features from common base
        for feat_y in y_assoc[block_i]:
            added_noise = noise_within * normal_dist_func(sample_num, scale=noise_within_magnitude)
            if association == 'line':
                Y[feat_y] = sign_corr * common_base[block_i] + added_noise
            elif association == 'parabola':
                Y[feat_y] = sign_corr * common_base[block_i] * common_base[block_i] + added_noise
        # update A
        for i, j in itertools.product(x_assoc[block_i], y_assoc[block_i]):
            A[i][j] = 1
    return(X, Y, A)

def store_tables(X, Y, A, association, out_dir):
    '''Store generated tables X,Y,A into files under out_dir directory
    '''
    def create_dir(dir_name):
        # remove any existing directory with the same name
        if isdir(dir_name):
            try:
                shutil.rmtree(dir_name)
            except EnvironmentError:
                sys.exit('Unable to remove directory %s' % dir_name)
        # create a new directory
        try:
            os.mkdir(dir_name)
        except EnvironmentError:
            sys.exit('Unable to create directory %s' % dir_name)

    def create_df(table, col_pref, row_pref):
        return pd.DataFrame(
            data={ '%s%d' % (col_pref, j): table[:,j] for j in range(table.shape[1]) },
            index=['%s%d' % (row_pref, i) for i in range(table.shape[0])]
        )

    x_feat_num, sample_num = X.shape
    y_feat_num, _ = Y.shape

    # create directory
    create_dir(out_dir)
    
    # store df in files
    filename_format = '%s_%s_%s_%s.txt' % ('%s', association, '%s', '%d')
    dataset_format = filename_format % ('%s', '%d', sample_num)
    create_df(X, 'S', 'X').to_csv(join(out_dir, dataset_format % ('X', x_feat_num)), sep='\t', index=True)
    create_df(Y, 'S', 'Y').to_csv(join(out_dir, dataset_format % ('Y', y_feat_num)), sep='\t', index=True)
    create_df(A, 'Y', 'X').to_csv(join(out_dir, filename_format % ('A', x_feat_num, y_feat_num)), sep='\t', index=True)

if __name__ == "__main__":
    # parse arguments
    params = parse_argument(sys.argv)
    # generate datasets
    X, Y, A = run_data_generator(params.samples, (params.xfeatures, params.yfeatures), params.blocks,
                                    params.association, params.distribution, params.noise_within, params.noise_between)
    # store datasets
    store_tables(X, Y, A, params.association, out_dir=params.output)
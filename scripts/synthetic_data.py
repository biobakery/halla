'''Generate synthetic dataset X and Y

Generates: a directory that contains 3 files:
- X, Y: a paired dataset
- A   : contains the associated features between the paired dataset
'''

import argparse
import sys
import numpy as np
import itertools
import pandas as pd
from os.path import join
from scipy.stats import ortho_group
import math

from halla.utils.data import discretize_vector
from halla.utils.filesystem import reset_dir

def parse_argument(args):
    parser = argparse.ArgumentParser(
        description='HAllA synthetic data generator - produces a pair of datasets X & Y with specified association among their features'
    )
    parser.add_argument('-n', '--samples', help='# samples in both X and Y', default=50, type=int, required=False)
    parser.add_argument('-xf', '--xfeatures', help='# features in X', default=500, type=int, required=False)
    parser.add_argument('-yf', '--yfeatures', help='# features in Y', default=500, type=int, required=False)
    parser.add_argument('-b', '--blocks', help='# significant blocks; default = min(xfeatures, yfeatures, samples)/3',
                        default=None, type=int, required=False)
    parser.add_argument('-a', '--association', help='association type {line, parabola, log, sine, step, mixed, categorical}; default: line',
                        default='line', choices=['line', 'parabola', 'log', 'sine', 'step', 'mixed', 'categorical'], required=False)
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
        params.blocks = min(params.xfeatures, params.yfeatures, params.samples)//3
    if not (params.blocks > 0 and params.blocks <= min(params.xfeatures, params.yfeatures, params.samples)/3):
        raise ValueError('# blocks is invalid; must be [1..min(xfeatures, yfeatures, samples)/3]')
    # noises must be [0..1]
    if params.noise_between < 0 or params.noise_between > 1 or \
        params.noise_within < 0 or params.noise_within > 1:
        raise ValueError('Noise within/between must be [0..1]')
    return(params)

def run_data_generator(sample_num=50, features_num=(500, 500), block_num=5, association='line',
                        noise_within=0.25, noise_between=0.25, noise_within_std=0.7, noise_between_std=0.7):
    '''Generate synthetic data with the following steps:
    1) generate a base B [-1, 1] from uniform distribution
    2) derive base_X and base_Y from B with noise = between_noise
    3) derive features in X and Y from base_X and base_Y with noise = within_noise

    Available associations include:
    - line       : X = base_X + noise; Y = base_Y + noise
    - parabola   : X = base_X + noise; Y = base_Y * base_Y + noise
    - log        : base = abs(base); X = base_X + noise; Y = log(abs(base_Y)) + noise
    - sine       : base = base * 2 ; X = base_X + noise; Y = 2 * sin(pi * base_Y) + noise
    - step       : X = base_X + noise; Y = { div base_Y into 4 quantiles; p1 = 2.0, p2 = 1.0, p3 = 3.0, p4 = 0.0 } + noise
    - categorical: same as step; discretize all features into {[3..6]} bins
    - mixed      : same as step; discretize some features in X and Y into into {[3..6]} bins
    '''
    def create_base():
        '''Generate base matrix [block_num x sample_num] with rows independent to each other
        '''
        # generate orthogonal matrix from uniform distribution
        #   then pick block_num rows given block_num <= sample_num
        base = ortho_group.rvs(sample_num)[:block_num]
        # enlarge the range to around [-1, 1]
        mult = min(abs(1.0 / base.min()), abs(1.0 / base.max()))
        base = base * mult
        # test if rows are independent
        test = base @ base.T
        np.testing.assert_allclose(test, mult*mult*np.eye(block_num), atol=1e-10, err_msg='The rows in base are not orthonormal')
        return(base)

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
    
    def abs_if_necessary(a):
        if association == 'log': return(np.abs(a))
        return(a)

    # initialize all matrices
    x_feat_num, y_feat_num = features_num
    X, Y = np.zeros((x_feat_num, sample_num)), np.zeros((y_feat_num, sample_num))
    A = np.zeros(features_num)

    # step 1: generate base
    base = abs_if_necessary(create_base())
    if association == 'sine': base = 2 * base # for spreading out x
    
    # assign features in X and Y to blocks
    x_assoc, y_assoc = div_features_into_blocks(x_feat_num), div_features_into_blocks(y_feat_num)
    for block_i in range(block_num):
        # step 2.1: derive base_X from base given noise_between
        base_X = base[block_i] + noise_between * np.random.normal(scale=noise_between_std, size=1)
        # step 3.1: derive X from base_X given noise_within
        for feat_x in x_assoc[block_i]:
            X[feat_x] = base_X + noise_within * np.random.normal(scale=noise_within_std, size=sample_num)

        # determine positive or negative association if appropriate; arbitrary probs
        sign_corr = np.random.choice([-1, 1], p=[0.4, 0.6])
        
        # step 2.2: derive base_Y from base given noise_between
        base_Y = abs_if_necessary(base[block_i] + noise_between * np.random.normal(scale=noise_between_std, size=1))
        # step 3.2: derive Y from base_Y given noise_within
        for feat_y in y_assoc[block_i]:
            if association == 'line':
                Y[feat_y] = sign_corr * base_Y
            elif association == 'parabola':
                Y[feat_y] = sign_corr * base_Y * base_Y
            elif association == 'log':
                Y[feat_y] = np.log(base_Y)
            elif association == 'sine':
                Y[feat_y] = 2 * np.sin(math.pi * base_Y)
            elif association == 'step':
                # divide base_Y into 4 quantiles
                p1, p2, p3 = np.percentile(base_Y, 25), np.percentile(base_Y, 50), np.percentile(base_Y, 75)
                Y[feat_y] = [2.0 if val < p1 else \
                             1.0 if val < p2 else \
                             3.0 if val < p3 else 0.0 for val in base_Y]
            else:
                # default for now
                Y[feat_y] = sign_corr * base_Y
            Y[feat_y] = Y[feat_y] + noise_within * np.random.normal(scale=noise_within_std, size=sample_num)
        # update A
        for i, j in itertools.product(x_assoc[block_i], y_assoc[block_i]):
            A[i][j] = 1
    if association in ['categorical', 'mixed']:
        X_new = np.empty((x_feat_num, sample_num), dtype=object)
        Y_new = np.empty((y_feat_num, sample_num), dtype=object)
        # select features to be discretized
        if association == 'categorical':
            xdisc_feat_indices = [i for i in range(x_feat_num)]
            ydisc_feat_indices = [j for j in range(y_feat_num)]
        else:
            xdisc_feat_num, ydisc_feat_num = np.random.choice(x_feat_num), np.random.choice(y_feat_num)
            xdisc_feat_indices = np.random.choice(x_feat_num, xdisc_feat_num, replace=False)
            ydisc_feat_indices = np.random.choice(y_feat_num, ydisc_feat_num, replace=False)
        for i in range(x_feat_num):
            if i in xdisc_feat_indices:
                discretized = discretize_vector(X[i], func='quantile',
                                num_bins=min(np.random.choice(range(3,7)), sample_num//2))
                X_new[i] = [chr(int(val) + 65) for val in discretized]
            else:
                X_new[i] = X[i]
        for j in range(y_feat_num):
            if j in ydisc_feat_indices:
                discretized = discretize_vector(Y[j], func='quantile',
                                num_bins=min(np.random.choice(range(3,7)), sample_num//2))
                Y_new[j] = [chr(int(val) + 65) for val in discretized]
            else:
                Y_new[j] = Y[j]
        return(X_new, Y_new, A)
    return(X, Y, A)

def store_tables(X, Y, A, association, out_dir):
    '''Store generated tables X,Y,A into files under out_dir directory
    '''

    def create_df(table, col_pref, row_pref):
        '''Create pandas DataFrame from table given:
        - table   : a 2D numpy array
        - col_pref: the column prefix
        - row_pref: the row prefix 
        '''
        return pd.DataFrame(
            data={ '%s%d' % (col_pref, j): table[:,j] for j in range(table.shape[1]) },
            index=['%s%d' % (row_pref, i) for i in range(table.shape[0])]
        )

    x_feat_num, sample_num = X.shape
    y_feat_num, _ = Y.shape

    # create directory
    reset_dir(out_dir)
    
    # store df in files
    filename_format = '%s_%s_%s_%s.txt' % ('%s', association, '%s', '%d')
    dataset_format = filename_format % ('%s', '%d', sample_num)
    create_df(X, 'S', 'X').to_csv(join(out_dir, dataset_format % ('X', x_feat_num)), sep='\t', index=True)
    create_df(Y, 'S', 'Y').to_csv(join(out_dir, dataset_format % ('Y', y_feat_num)), sep='\t', index=True)
    create_df(A, 'Y', 'X').to_csv(join(out_dir, filename_format % ('A', x_feat_num, y_feat_num)), sep='\t', index=True)

def main():
    # parse arguments
    params = parse_argument(sys.argv)
    # generate datasets
    X, Y, A = run_data_generator(params.samples, (params.xfeatures, params.yfeatures), params.blocks,
                                    params.association, params.noise_within, params.noise_between)
    # store datasets
    store_tables(X, Y, A, params.association, out_dir=params.output)

if __name__ == "__main__":
    main()
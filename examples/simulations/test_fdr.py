'''Simulation to compare HAllA and AllA, specifically for n iterations,
1.1) create synthetic dataset
1.2) run HAllA and compute power & FDR
1.3) run AllA and compute power & FDR
2  ) store results in a csv file
'''
import argparse
import os
from os.path import dirname, abspath, join
import sys
import pandas as pd
import shutil
import time

from halla import HAllA, AllA
from halla.utils.stats import compute_result_power, compute_result_fdr
from halla.utils.filesystem import create_dir

def parse_argument(args):
    parser = argparse.ArgumentParser(
        description='Run simulations - generate synthetic dataset and compare HAllA vs AllA given n iterations'
    )
    parser.add_argument('-i', '--iter', help='# iterations', default=10, type=int, required=False)
    parser.add_argument('-n', '--samples', help='# samples in both X and Y', default=50, type=int, required=False)
    parser.add_argument('-xf', '--xfeatures', help='# features in X', default=500, type=int, required=False)
    parser.add_argument('-yf', '--yfeatures', help='# features in Y', default=500, type=int, required=False)
    parser.add_argument('-b', '--blocks', help='# significant blocks; default = min(xfeatures, yfeatures, samples)/3',
                        default=None, type=int, required=False)
    parser.add_argument('-a', '--association', help='association type {line, parabola, log, sine, step, mixed, categorical}; default: line',
                        default='line', choices=['line', 'parabola', 'log', 'sine', 'step', 'mixed', 'categorical'], required=False)
    parser.add_argument('-nw', '--noise-within', dest='noise_within', help='noise within blocks [0 (no noise)..1 (complete noise)]',
                        default=0.35, type=float, required=False)
    parser.add_argument('-nb', '--noise-between', dest='noise_between', help='noise between associated blocks [0 (no noise)..1 (complete noise)]',
                        default=0.35, type=float, required=False)
    parser.add_argument('-fdr', '--fdr_alpha', help='FDR alpha', default=0.05, type=float, required=False)
    parser.add_argument('-fnr', '--fnr_thresh', help='FNR threshold for HAllA', default=0.2, type=float, required=False)
    parser.add_argument('-m', '--metric', help='Similarity metric', default='pearson', choices=['pearson', 'spearman', 'nmi', 'dcor'], required=False)
    parser.add_argument('-o', '--output', help='Output file prefix', default='simul_out', required=False)
    parser.add_argument('--skip_alla', default=False, action='store_true')

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

def store_result(halla_power, halla_fdr, halla_dur, output_pref, alla_power=None, alla_fdr=None):
    if skip_alla:
        pd.DataFrame(data={
           'power': halla_power,
           'fdr'  : halla_fdr, 
           'duration': halla_dur,
        }).to_csv('%s.csv' % output_pref)
    else:
        pd.DataFrame(data={
           'type' : ['halla']*len(halla_power) + ['alla']*len(alla_power),
           'power': halla_power + alla_power,
           'fdr'  : halla_fdr + alla_fdr,
           'duration': halla_dur + [-1]*len(alla_power),
        }).to_csv('%s.csv' % output_pref)

if __name__ == '__main__':
    print('Running simulation...')
    params = parse_argument(sys.argv)
    print(params)

    num_iters = params.iter
    halla_fdr, halla_power = [], []
    alla_fdr, alla_power = [], []
    halla_dur = []

    create_dir('simulation')
    create_dir('simulation_out')

    x_feat_num, y_feat_num = params.xfeatures, params.yfeatures
    sample_num = params.samples
    association = params.association
    pdist_metric = params.metric
    noise_within, noise_between = params.noise_within, params.noise_between
    fdr_alpha = params.fdr_alpha
    fnr_thresh = params.fnr_thresh
    output_pref = params.output
    skip_alla = params.skip_alla
    
    dataset_dir = 'simulation/%s' % output_pref
    result_dir = 'simulation_out/%s' % output_pref

    start_time = time.time()

    for i in range(num_iters):
        print('Iteration', i+1)
        dataset_iter_path = '%s_%d' % (dataset_dir, i)
        # 1.1) create a synthetic dataset
        os.system('halladata -a %s -n %d -xf %d -yf %d -nw %f -nb %f  -o %s' % (
            association, sample_num, x_feat_num, y_feat_num, noise_within, noise_between,
            dataset_iter_path))

        X_file = join(dataset_iter_path, 'X_%s_%d_%d.txt' % (association, x_feat_num, sample_num))
        Y_file = join(dataset_iter_path, 'Y_%s_%d_%d.txt' % (association, y_feat_num, sample_num))
        A_file = join(dataset_iter_path, 'A_%s_%d_%d.txt' % (association, x_feat_num, y_feat_num))
        A = pd.read_table(A_file, index_col=0).to_numpy()

        iter_start_time = time.time()

        # 1.2) run HAllA
        test_halla = HAllA(discretize_func='quantile', discretize_num_bins=4,
                      pdist_metric=pdist_metric, fnr_thresh=fnr_thresh, fdr_alpha=fdr_alpha, verbose=False,
                      out_dir='%s_%d' % (result_dir, i))
        test_halla.load(X_file, Y_file)
        test_halla.run()

        iter_end_time = time.time()

        # 1.2) compute power and FDR
        halla_power.append(compute_result_power(test_halla.significant_blocks, A))
        halla_fdr.append(compute_result_fdr(test_halla.significant_blocks, A))
        halla_dur.append(iter_end_time - iter_start_time)

        if skip_alla: continue
        # 1.3) run AllA; avoid repeating the same computation
        test_alla = AllA(discretize_func='quantile', discretize_num_bins=4,
                      pdist_metric=pdist_metric, fdr_alpha=fdr_alpha, verbose=False,
                      out_dir='%s_%d_alla' % (result_dir, i))
        test_alla.logger = test_halla.logger
        test_alla.similarity_table = test_halla.similarity_table
        test_alla.pvalue_table     = test_halla.pvalue_table
        test_alla.fdr_reject_table = test_halla.fdr_reject_table
        test_alla.qvalue_table     = test_halla.qvalue_table
        test_alla.X, test_alla.Y   = test_halla.X, test_halla.Y
        test_alla._find_dense_associated_blocks()

        # 1.3) compute power and FDR
        alla_power.append(compute_result_power(test_alla.significant_blocks, A))
        alla_fdr.append(compute_result_fdr(test_alla.significant_blocks, A))

        print(halla_power, halla_fdr)
        print(alla_power, alla_fdr)

        if skip_alla:
            store_result(halla_power, halla_fdr, halla_dur, output_pref)
        else:
            store_result(halla_power, halla_fdr, halla_dur, output_pref, alla_power, alla_fdr)
    
    total_time = time.time() - start_time
    print('Total time is', total_time)
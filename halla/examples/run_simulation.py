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

sys.path.append(dirname(dirname(abspath(__file__))))
                
from tools import HAllA, AllA
from tools.utils.stats import compute_result_power, compute_result_fdr

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
                        default=0.55, type=float, required=False)
    parser.add_argument('-nb', '--noise-between', dest='noise_between', help='noise between associated blocks [0 (no noise)..1 (complete noise)]',
                        default=0.5, type=float, required=False)
    parser.add_argument('-fdr', '--fdr_alpha', help='FDR alpha', default=0.05, type=float, required=False)
    parser.add_argument('-m', '--metric', help='Similarity metric', default='pearson', choices=['pearson', 'spearman', 'nmi'], required=False)
    parser.add_argument('-o', '--output', help='Output file prefix', default='simul_out', required=False)

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

if __name__ == '__main__':
    print('Running simulation...')
    params = parse_argument(sys.argv)

    num_iters = params.iter
    halla_fdr, halla_power = [], []
    alla_fdr, alla_power = [], []

    x_feat_num, y_feat_num = params.xfeatures, params.yfeatures
    sample_num = params.samples
    association = params.association
    pdist_metric = params.metric
    noise_within, noise_between = params.noise_within, params.noise_between
    fdr_alpha = params.fdr_alpha
    output_pref = params.output
    dataset_dir = 'simulation/%s' % output_pref
    result_dir = 'simulation_out/%s' % output_pref

    start_time = time.time()

    for i in range(num_iters):
        print('Iteration', i+1)
        dataset_iter_path = '%s_%d' % (dataset_dir, i)
        # 1.1) create a synthetic dataset
        os.system('python tools/synthetic_data.py -a %s -n %d -xf %d -yf %d -nw %f -nb %f  -o %s' % (
            association, sample_num, x_feat_num, y_feat_num, noise_within, noise_between,
            dataset_iter_path))

        X_file = join(dataset_iter_path, 'X_%s_%d_%d.txt' % (association, x_feat_num, sample_num))
        Y_file = join(dataset_iter_path, 'Y_%s_%d_%d.txt' % (association, y_feat_num, sample_num))
        A_file = join(dataset_iter_path, 'A_%s_%d_%d.txt' % (association, x_feat_num, y_feat_num))
        A = pd.read_table(A_file, index_col=0).to_numpy()

        # 1.2) run HAllA
        test_halla = HAllA(discretize_func='equal-freq', discretize_num_bins=3,
                      pdist_metric=pdist_metric, fnr_thresh=0.2, fdr_alpha=fdr_alpha,
                      out_dir=result_dir)
        test_halla.load(X_file, Y_file)
        test_halla.run()
        # 1.2) compute power and FDR
        halla_power.append(compute_result_power(test_halla.significant_blocks, A))
        halla_fdr.append(compute_result_fdr(test_halla.significant_blocks, A))

        # 1.3) run AllA; avoid repeating the same computation
        test_alla = AllA(discretize_func='equal-freq', discretize_num_bins=3,
                      pdist_metric=pdist_metric, fnr_thresh=0.2, fdr_alpha=fdr_alpha,
                      out_dir=result_dir)
        test_alla.similarity_table = test_halla.similarity_table
        test_alla.pvalue_table     = test_halla.pvalue_table
        test_alla.fdr_reject_table = test_halla.fdr_reject_table
        test_alla.qvalue_table     = test_halla.qvalue_table
        test_alla.X, test_alla.Y   = test_halla.X, test_halla.Y
        test_alla._find_dense_associated_blocks()
        # test_alla.load(X_file, Y_file)
        # test_alla.run()

        # 1.3) compute power and FDR
        alla_power.append(compute_result_power(test_alla.significant_blocks, A))
        alla_fdr.append(compute_result_fdr(test_alla.significant_blocks, A))
    
    # remove all generated directories
    # shutil.rmtree(iter_path)
    # shutil.rmtree(result_dir)

    # 2) store results in a csv file
    pd.DataFrame(data={
       'type' : ['halla']*num_iters + ['alla']*num_iters,
       'power': halla_power + alla_power,
       'fdr'  : halla_fdr + alla_fdr, 
    }).to_csv('%s.csv' % output_pref)

    total_time = time.time() - start_time
    print('Total time is', total_time)
'''Run halla on console environment
'''

import argparse
import sys
import numpy as np
from os.path import join
import pkg_resources

from halla import HAllA

def parse_argument(args):
    parser = argparse.ArgumentParser(
        description='HAllA: Hierarchical All-against-All significance association testing version ' + pkg_resources.require('HAllA')[0].version,
    )
    # --load parameters--
    parser.add_argument(
        '-x', '--x_file',
        help='Path to X dataset: a tab-delimited input file, one row per feature, one column per measurement',
        required=True)
    parser.add_argument(
        '-y', '--y_file',
        help='Path to Y dataset: a tab-delimited input file, one row per feature, one column per measurement',
        required=False)
    
    # --halla parameters--
    parser.add_argument(
        '--max_freq_thresh',
        help='The maximum frequency threshold - features with max frequences >= the threshold will be removed',
        default=1, type=float, required=False)
    parser.add_argument(
        '--transform_data_funcs',
        help='Continuous data transformation function - a list',
        nargs='+',
        default=None, required=False)
    parser.add_argument(
        '--disable_bypass_discretization_if_possible',
        help='Disable bypassing discretization when all features are continuous',
        action='store_true', required=False)
    parser.add_argument(
        '--discretize_func',
        help='Discretization - function {None, quantile, kmeans, uniform, jenks}',
        default=None, choices=['None', 'quantile', 'kmeans', 'uniform', 'jenks'], required=False)
    parser.add_argument(
        '--discretize_num_bins',
        help='Discretization - number of bins',
        default=None, type=int, required=False)
    parser.add_argument(
        '-m', '--pdist_metric',
        help='Distance/similarity metric {spearman, pearson, dcor, nmi}',
        default='pearson', choices=['spearman', 'pearson', 'dcor', 'nmi'], required=False)
    parser.add_argument(
        '--sim2dist_set_abs',
        help='Hierarchical clustering - set similarity scores as absolute when computing distance',
        action='store_true', required=False)
    parser.add_argument(
        '--linkage_method',
        help='Hierarchical clustering linkage method - check scipy.cluster.hierarchy.linkage()',
        default='average', required=False)
    parser.add_argument(
        '--permute_func',
        help='P-value approximation function in the p-value permutation test',
        default='gpd', choices=['gpd', 'ecdf'], required=False)
    parser.add_argument(
        '--permute_iters',
        help='# iterations in the p-value permutation test',
        default=1000, type=int, required=False)
    parser.add_argument(
        '--disable_permute_speedup',
        help='Disable breaking early in the permutation test if p-value is insignificant',
        action='store_true', required=False)
    parser.add_argument(
        '--fdr_alpha',
        help='FDR threshold',
        default=0.05, type=float, required=False)
    parser.add_argument(
        '--fdr_method',
        help='FDR method - check statsmodels.stats.multitest.multipletests()',
        default='fdr_bh', required=False)
    parser.add_argument(
        '--fnr_thresh',
        help='FNR threshold',
        default=0.2, type=float, required=False)
    parser.add_argument(
        '--rank_cluster',
        help='Procedure to rank cluster using the p-values within the cluster {best, average}',
        default='best', choices=['best', 'average'], required=False)
    parser.add_argument(
        '-o', '--out_dir',
        help='Directory path to store results', required=True)
    parser.add_argument(
        '--disable_verbose',
        help='Disable verbose',
        action='store_true', required=False)
    parser.add_argument(
        '--seed',
        help='Randomization seed',
        default=None, type=float, required=False)

    # --hallagram parameters--
    parser.add_argument(
        '--disable_hallagram',
        help='Disable generating hallagram',
        action='store_true', required=False)

    # --clustermap parameters--
    parser.add_argument(
        '--disable_clustermap',
        help='Disable generating clustermap',
        action='store_true', required=False)

    # --hallagram/clustermap parameters--
    parser.add_argument(
        '--x_dataset_label',
        help='Hallagram/clustermap: label for X dataset',
        default='', required=False)
    parser.add_argument(
        '--y_dataset_label',
        help='Hallagram/clustermap: label for Y dataset',
        default='', required=False)
    parser.add_argument(
        '--cbar_label',
        help='Hallagram/clustermap: label for the colorbar',
        default='', required=False)

    # --diagnostic-plot parameters--
    parser.add_argument(
        '--disable_diagnostic_plot',
        help='Disable generating diagnostic plot',
        action='store_true', required=False)
    params = parser.parse_args()
    if params.discretize_func == 'None':
        params.discretize_func = None
    return(parser.parse_args())

def main():
    params = parse_argument(sys.argv)
    halla_instance = HAllA(max_freq_thresh=params.max_freq_thresh,
                 transform_data_funcs=params.transform_data_funcs,
                 discretize_bypass_if_possible=not params.disable_bypass_discretization_if_possible,
                 discretize_func=params.discretize_func, discretize_num_bins=params.discretize_num_bins,
                 pdist_metric=params.pdist_metric, linkage_method=params.linkage_method,
                 sim2dist_set_abs=params.sim2dist_set_abs,
                 permute_func=params.permute_func, permute_iters=params.permute_iters,
                 permute_speedup=not params.disable_permute_speedup,
                 fdr_alpha=params.fdr_alpha, fdr_method=params.fdr_method,
                 fnr_thresh=params.fnr_thresh, rank_cluster=params.rank_cluster,
                 out_dir=params.out_dir, verbose=not params.disable_verbose,
                 seed=params.seed)
    halla_instance.load(params.x_file, params.y_file)
    halla_instance.run()
    if not params.disable_clustermap:
        halla_instance.generate_clustermap(x_dataset_label=params.x_dataset_label,
                                           y_dataset_label=params.y_dataset_label,
                                           cbar_label=params.cbar_label)
    if not params.disable_hallagram:
        halla_instance.generate_hallagram(x_dataset_label=params.x_dataset_label,
                                           y_dataset_label=params.y_dataset_label,
                                           cbar_label=params.cbar_label)
    if not params.disable_diagnostic_plot:
        halla_instance.generate_diagnostic_plot()

if __name__ == "__main__":
    main()
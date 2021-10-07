'''Run halla on console environment
'''

import argparse
import sys
import numpy as np
from os.path import join, basename, splitext, isdir
import pkg_resources

from halla import HAllA, AllA

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
        '--alla',
        help='Use AllA instead of HAllA',
        action='store_true', required=False)
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
        help='Distance/similarity metric {spearman, pearson, dcor, mi, nmi, xicor}',
        default='spearman', choices=['spearman', 'pearson', 'dcor', 'mi', 'nmi', 'xicor'], required=False)
    parser.add_argument(
        '--sim2dist_disable_abs',
        help='Hierarchical clustering - disable setting similarity scores as absolute when computing distance',
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
        '--hallagram',
        help='Generates hallagram',
        default=True,action='store_true', required=False)
    parser.add_argument(
        '--no_hallagram',
        help='Turn off the automatically generated hallagram',
        dest='hallagram',
        action='store_false')

    # --clustermap parameters--
    parser.add_argument(
        '--clustermap',
        help='Generates clustermap',
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
    parser.add_argument(
        '-n', '--block_num',
        help='Number of top clusters to show (for hallagram only); if -1, show all clusters',
        default=50, type=int, required=False)
    parser.add_argument(
        '--trim',
        help='Trim hallagram to features containing at least one significant block',
        default=True,
        type=bool, required=False)
    parser.add_argument(
        "--plot_file_type",
        help = "File type of hallagram output",
        default = "pdf",
        dest="plot_type",
        required = False
    )

    # --other options--
    parser.add_argument(
        '--no_progress',
        help="Turn off the progress bar for p-value table calculations",
        dest='no_progress',
        default=False,
        action = 'store_true',required=False)
    parser.add_argument(
        '--dont_copy',
        help="Don't make a copy of the data files in the output directory",
        dest='dont_copy',
        default=False,
        action = 'store_true',required=False)
    parser.add_argument(
        '--force_permutations',
        help="If turned on, force permutation testing",
        dest='force_permutations',
        default=False,
        action = 'store_true',required=False)
    parser.add_argument(
        '--num_threads',
        help='Number of threads to use when running permutation tests in parallel, default=4',
        default=1, type=int, required=False)
    parser.add_argument(
        '--version',
        action='version',
        version=pkg_resources.require('HAllA')[0].version
    )
    parser.add_argument(
        '--dont_skip_large_blocks',
        required=False,
        help="Don't skip very large (>45 features) blocks in diagnostic plots",
        dest='dont_skip',
        default=False,
        action='store_true')
    parser.add_argument(
        '--large_diagnostic_subset',
        help = "Subset the feature pairs plotted in large block (>15, <45) diagnostic plots.",
        required=False,
        dest='large_diagnostic_subset',
        default=105
    )
    parser.add_argument(
        '--splitting_diagnostic_mode',
        required=False,
        dest='splitting_diagnostic_mode',
        help="Diagnostic mode to write out tree descent algorithm progress. Prints branches being considered at each step and Gini score improvement (the latter only if applicable). Values listed in brackets are indices (0-indexed) of features in X and Y datasets respectively.",
        default=False,
        action='store_true')
    parser.add_argument(
        '--gini_uncertainty_level',
        required=False,
        dest='gini_uncertainty_level',
        type=float,
        help="Gini uncertainty mode opts to split larger hierarchical branches when the difference in Gini impurity improvement is less than the given level.",
        default=0.02)

    # --diagnostic-plot parameters--
    parser.add_argument(
        '--diagnostic_plot',
        help='Generates diagnostic plot',
        action='store_true', required=False)
    params = parser.parse_args()
    if params.discretize_func == 'None':
        params.discretize_func = None
    return(parser.parse_args())

def main():
    params = parse_argument(sys.argv)

    if isdir(params.out_dir):
        sys.exit("Error: output directory already exists.")

    if params.alla:
        instance = AllA(max_freq_thresh=params.max_freq_thresh,
                 transform_data_funcs=params.transform_data_funcs,
                 discretize_bypass_if_possible=not params.disable_bypass_discretization_if_possible,
                 discretize_func=params.discretize_func, discretize_num_bins=params.discretize_num_bins,
                 pdist_metric=params.pdist_metric,
                 permute_func=params.permute_func, permute_iters=params.permute_iters,
                 permute_speedup=not params.disable_permute_speedup,
                 fdr_alpha=params.fdr_alpha, fdr_method=params.fdr_method,
                 out_dir=params.out_dir, verbose=not params.disable_verbose,
                 no_progress=params.no_progress, dont_copy=params.dont_copy, force_permutations=params.force_permutations, 
                 num_threads=params.num_threads, 
                 splitting_diagnostic_mode=params.splitting_diagnostic_mode,
                 gini_uncertainty_level=params.gini_uncertainty_level,
                 seed=params.seed)
    else:
        instance = HAllA(max_freq_thresh=params.max_freq_thresh,
                 transform_data_funcs=params.transform_data_funcs,
                 discretize_bypass_if_possible=not params.disable_bypass_discretization_if_possible,
                 discretize_func=params.discretize_func, discretize_num_bins=params.discretize_num_bins,
                 pdist_metric=params.pdist_metric, linkage_method=params.linkage_method,
                 sim2dist_set_abs=not params.sim2dist_disable_abs,
                 permute_func=params.permute_func, permute_iters=params.permute_iters,
                 permute_speedup=not params.disable_permute_speedup,
                 fdr_alpha=params.fdr_alpha, fdr_method=params.fdr_method,
                 fnr_thresh=params.fnr_thresh, rank_cluster=params.rank_cluster,
                 out_dir=params.out_dir, verbose=not params.disable_verbose,
                 no_progress=params.no_progress, dont_copy=params.dont_copy, force_permutations=params.force_permutations,
                 dont_skip=params.dont_skip,
                 num_threads=params.num_threads,
                 splitting_diagnostic_mode=params.splitting_diagnostic_mode,
                 gini_uncertainty_level=params.gini_uncertainty_level,
                 seed=params.seed)
    instance.load(params.x_file, params.y_file)
    instance.run()
    if params.clustermap:
        if params.x_dataset_label=='':
            params.x_dataset_label = splitext(basename(params.x_file))[0]
        if params.y_dataset_label=='':
            params.y_dataset_label = splitext(basename(params.y_file))[0]
        if params.alla:
            print('AllA does not produce clustermap.', file = sys.stderr)
        else:
            instance.generate_clustermap(x_dataset_label=params.x_dataset_label,
                                           y_dataset_label=params.y_dataset_label,
                                           cbar_label=params.cbar_label)
    if params.hallagram:
        if params.x_dataset_label=='':
            params.x_dataset_label = splitext(basename(params.x_file))[0]
        if params.y_dataset_label=='':
            params.y_dataset_label = splitext(basename(params.y_file))[0]
        instance.generate_hallagram(x_dataset_label=params.x_dataset_label,
                                           y_dataset_label=params.y_dataset_label,
                                           cbar_label=params.cbar_label, trim = params.trim,
                                           plot_type = params.plot_type)
    if params.diagnostic_plot:
        if params.alla:
            print('AllA does not produce diagnostic plot.', file = sys.stderr)
        else:
            instance.generate_diagnostic_plot()

if __name__ == "__main__":
    main()

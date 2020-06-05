from .config_loader import config, update_config
from .hierarchy import HierarchicalTree
from .utils.data import preprocess, eval_type, is_all_cont
from .utils.distance import get_distance_function
from .utils.stats import get_pvalue_table, pvalues2qvalues
from .utils.tree import compare_and_find_dense_block

import pandas as pd
import numpy as np
import scipy.spatial.distance as spd

class HAllA(object):
    def __init__(self, discretize_bypass_if_cont=config.discretize['bypass_if_cont'],
                 discretize_func=config.discretize['func'], discretize_num_bins=config.discretize['num_bins'],
                 pdist_metric=config.hierarchy['pdist_metric'], pdist_args=config.hierarchy['pdist_args'],
                 permute_func=config.permute['func'], permute_iters=config.permute['iters'],
                 fdr_alpha=config.stats['fdr_alpha'], fdr_method=config.stats['fdr_method'],
                 fnr_thresh=config.stats['fnr_thresh'],
                 seed=None):
        # update config settings
        update_config('discretize', bypass_if_cont=discretize_bypass_if_cont, func=discretize_func, num_bins=discretize_num_bins)
        update_config('hierarchy', pdist_metric=pdist_metric, pdist_args=pdist_args)
        update_config('permute', func=permute_func, iters=permute_iters)
        update_config('stats', fdr_alpha=fdr_alpha, fdr_method=fdr_method, fnr_thresh=fnr_thresh)

        self.reset_attributes()
        self.seed = seed
    
    def reset_attributes(self):
        self.X, self.Y = None, None
        self.X_hierarchy, self.Y_hierarchy = None, None
        self.similarity_table = None
        self.pvalue_table, self.qvalue_table = None, None
        self.fdr_reject_table = None
        self.significant_blocks = None

    def load(self, X_file, Y_file=None):
        # TODO: currently assumes no missing value and header+index col are provided
        X, X_types = eval_type(pd.read_table(X_file, index_col=0))
        Y, Y_types = eval_type(pd.read_table(Y_file, index_col=0)) if Y_file \
            else (X.copy(deep=True), np.copy(X_types))

        # if not all types are continuous but pdist_metric is only for continuous types
        # TODO: add more appropriate distance metrics
        if not (is_all_cont(X_types) and is_all_cont(Y_types)) and config.hierarchy['pdist_metric'] != 'nmi':
            raise ValueError('pdist_metric should be nmi if not all features are continuous...')
        # if all features are continuous, discretization will be bypassed
        if is_all_cont(X_types) and is_all_cont(Y_types) and config.discretize['bypass_if_cont']:
            print('All features are continuous; bypassing discretization and updating config...')
            update_config('discretize', func=None)

        # filter tables by intersect columns
        intersect_cols = sorted(list(set(X.columns) & set(Y.columns)))
        X, Y = X[intersect_cols], Y[intersect_cols]

        # clean and preprocess data
        self.X = preprocess(X, X_types, discretize_func=config.discretize['func'], discretize_num_bins=config.discretize['num_bins'])
        self.Y = preprocess(Y, Y_types, discretize_func=config.discretize['func'], discretize_num_bins=config.discretize['num_bins'])

    def run_clustering(self):
        self.X_hierarchy = HierarchicalTree(self.X, feature_names=list(self.X.index))
        self.Y_hierarchy = HierarchicalTree(self.Y, feature_names=list(self.Y.index))
    
    def compute_pairwise_similarities(self):
        confh = config.hierarchy
        n, m = self.X.shape[0], self.Y.shape[0]
        X, Y = self.X.to_numpy(), self.Y.to_numpy()

        # obtain similarity matrix
        if confh['pdist_args']:
            self.similarity_table = spd.cdist(X, Y, metric=get_distance_function(confh['pdist_metric']), **confh['pdist_args'])
        else:
            self.similarity_table = spd.cdist(X, Y, metric=get_distance_function(confh['pdist_metric']))
        # obtain p-values
        confp = config.permute
        self.pvalue_table = get_pvalue_table(X, Y, pdist_metric=confh['pdist_metric'], pdist_args=confh['pdist_args'],
                                                   permute_func=confp['func'], permute_iters=confp['iters'], seed=self.seed)
        # TODO: similarity rank?
        
        # obtain q-values
        self.fdr_reject_table, self.qvalue_table = pvalues2qvalues(self.pvalue_table.flatten(), config.stats['fdr_alpha'])
        self.qvalue_table = self.qvalue_table.reshape(self.pvalue_table.shape)
        self.fdr_reject_table = self.fdr_reject_table.reshape(self.pvalue_table.shape)

    def find_dense_associated_blocks(self):
        self.significant_blocks = compare_and_find_dense_block(self.X_hierarchy, self.Y_hierarchy, self.similarity_table,
                                     self.qvalue_table, self.fdr_reject_table, fnr_thresh=config.stats['fnr_thresh'])

    def run(self):
        # computing pairwise similarity matrix
        self.compute_pairwise_similarities()

        # hierarchical clustering
        self.run_clustering()
        
        # iteratively finding densely-associated blocks
        self.find_dense_associated_blocks()
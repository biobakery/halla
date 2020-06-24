from .config_loader import config, update_config
from .hierarchy import HierarchicalTree
from .utils.data import preprocess, eval_type, is_all_cont
from .utils.similarity import get_similarity_function
from .utils.stats import get_pvalue_table, pvalues2qvalues
from .utils.tree import compare_and_find_dense_block
from .utils.plot import generate_hallagram, generate_clustermap

import pandas as pd
import numpy as np
import scipy.spatial.distance as spd

class HAllA(object):
    def __init__(self, discretize_bypass_if_possible=config.discretize['bypass_if_possible'],
                 discretize_func=config.discretize['func'], discretize_num_bins=config.discretize['num_bins'],
                 pdist_metric=config.hierarchy['pdist_metric'], linkage_method=config.hierarchy['linkage_method'],
                 permute_func=config.permute['func'], permute_iters=config.permute['iters'],
                 fdr_alpha=config.stats['fdr_alpha'], fdr_method=config.stats['fdr_method'],
                 fnr_thresh=config.stats['fnr_thresh'],
                 seed=None):
        # update config settings
        update_config('discretize', bypass_if_possible=discretize_bypass_if_possible, func=discretize_func, num_bins=discretize_num_bins)
        update_config('hierarchy', pdist_metric=pdist_metric, linkage_method=linkage_method)
        update_config('permute', func=permute_func, iters=permute_iters)
        update_config('stats', fdr_alpha=fdr_alpha, fdr_method=fdr_method, fnr_thresh=fnr_thresh)

        self._reset_attributes()
        self.seed = seed
    
    '''Private functions
    '''
    def _reset_attributes(self):
        self.X, self.Y = None, None
        self.X_hierarchy, self.Y_hierarchy = None, None
        self.similarity_table = None
        self.pvalue_table, self.qvalue_table = None, None
        self.fdr_reject_table = None
        self.significant_blocks = None
        self.has_loaded = False
        self.has_run = False
    
    def _run_clustering(self):
        self.X_hierarchy = HierarchicalTree(self.X)
        self.Y_hierarchy = HierarchicalTree(self.Y)
    
    def _compute_pairwise_similarities(self):
        confh = config.hierarchy
        X, Y = self.X.to_numpy(), self.Y.to_numpy()

        # obtain similarity matrix
        self.similarity_table = spd.cdist(X, Y, metric=get_similarity_function(confh['pdist_metric']))
        # obtain p-values
        confp = config.permute
        self.pvalue_table = get_pvalue_table(X, Y, pdist_metric=confh['pdist_metric'],
                                                   permute_func=confp['func'], permute_iters=confp['iters'], seed=self.seed)
        
        # obtain q-values
        self.fdr_reject_table, self.qvalue_table = pvalues2qvalues(self.pvalue_table.flatten(), config.stats['fdr_alpha'])
        self.qvalue_table = self.qvalue_table.reshape(self.pvalue_table.shape)
        self.fdr_reject_table = self.fdr_reject_table.reshape(self.pvalue_table.shape)

    def _find_dense_associated_blocks(self):
        self.significant_blocks = compare_and_find_dense_block(self.X_hierarchy.tree, self.Y_hierarchy.tree,
                                     self.fdr_reject_table, fnr_thresh=config.stats['fnr_thresh'])
        # convert block feature indices to feature names
        self.significant_blocks_feature_names = []
        x_features, y_features = list(self.X.index), list(self.Y.index)
        for block in self.significant_blocks:
            x_feat_indices, y_feat_indices = block[0], block[1]
            x_feat_names = [x_features[feat] for feat in x_feat_indices]
            y_feat_names = [y_features[feat] for feat in y_feat_indices]
            self.significant_blocks_feature_names.append([x_feat_names, y_feat_names])

    '''Public functions
    '''
    def load(self, X_file, Y_file=None):
        # TODO: currently assumes no missing value and header+index col are provided
        X, X_types = eval_type(pd.read_table(X_file, index_col=0))
        Y, Y_types = eval_type(pd.read_table(Y_file, index_col=0)) if Y_file \
            else (X.copy(deep=True), np.copy(X_types))

        # if not all types are continuous but pdist_metric is only for continuous types
        # TODO: add more appropriate distance metrics
        if not (is_all_cont(X_types) and is_all_cont(Y_types)) and config.hierarchy['pdist_metric'] != 'nmi':
            raise ValueError('pdist_metric should be nmi if not all features are continuous...')
        # if all features are continuous and distance metric != nmi, discretization can be bypassed
        if is_all_cont(X_types) and is_all_cont(Y_types) and \
            config.hierarchy['pdist_metric'].lower() != 'nmi' and config.discretize['bypass_if_possible']:
            print('All features are continuous; bypassing discretization and updating config...')
            update_config('discretize', func=None)

        # filter tables by intersect columns
        intersect_cols = [col for col in X.columns if col in Y.columns]
        X, Y = X[intersect_cols], Y[intersect_cols]

        # clean and preprocess data
        self.X = preprocess(X, X_types, discretize_func=config.discretize['func'], discretize_num_bins=config.discretize['num_bins'])
        self.Y = preprocess(Y, Y_types, discretize_func=config.discretize['func'], discretize_num_bins=config.discretize['num_bins'])

        self.has_loaded = True

    def run(self):
        '''Run all 3 steps:
        1) compute pairwise similarity matrix
        2) cluster hierarchically
        3) find densely-associated blocks iteratively
        '''
        if self.has_loaded == False:
            raise RuntimeError('load function has not been called!')

        # step 1: computing pairwise similarity matrix
        self._compute_pairwise_similarities()

        # step 2: hierarchical clustering
        self._run_clustering()
        
        # step 3: iteratively finding densely-associated blocks
        self._find_dense_associated_blocks()
    
    def generate_hallagram(self, cmap='RdBu_r', **kwargs):
        '''Generate a hallagram
        '''
        generate_clustermap(self.significant_blocks,
                            self.X.index.to_numpy(),
                            self.Y.index.to_numpy(),
                            self.X_hierarchy.linkage,
                            self.Y_hierarchy.linkage,
                            self.similarity_table,
                            cmap=cmap, **kwargs)
    
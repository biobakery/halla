from .config_loader import config
from .utils.distance import get_distance_function

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd
import numpy as np
import itertools

class Hierarchy(object):
    def __init__(self, matrix, feature_names):
        conf = config.hierarchy
        if conf['pdist_args']:
            self.distance_matrix = spd.pdist(matrix, metric=get_distance_function(conf['pdist_metric']), **conf['pdist_args'])
        else:
            self.distance_matrix = spd.pdist(matrix, metric=get_distance_function(conf['pdist_metric']))
        self.distance_matrix_sqr = spd.squareform(self.distance_matrix)
        self.feature_names = feature_names
        self.generate_hierarchical_clusters()
    
    def generate_hierarchical_clusters(self):
        # perform hierarchical clustering
        Z = sch.linkage(self.distance_matrix, method=config.hierarchy['linkage_method'])
        self.tree = sch.to_tree(Z)
    
def compare_and_find_dense_block(X, Y, sim_table, qvalue_table, fdr_reject_table, fnr_thresh=0.1):
    '''Given another Hierarchy object Y, compare and find
    densely-associated block from the top of the hierarchy;

    Densely-associated block = (1 - FNR)% of pairwise association are
      FDR significant
    Args:
    - X               : X hierarchical tree (Hierarchy object)
    - Y               : Y hierarchical tree (Hierarchy object)
    - sim_table       : pairwise-similarity table between X and Y
    - qvalue_table    : qvalue table for the pairwise-similarity
    - fdr_reject_table: a boolean table where True = reject H0
    - fnr_thresh      : false negative rate threshold
    '''

    def _is_densely_associated(block):
        return(np.sum(block) / block.size >= 1 - fnr_thresh)
    
    def _bifurcate(tree):
        '''Bifurcate a tree to its two branches
        '''
    
    def _bifurcate_one(tree1, tree2):
        '''Given two trees, bifurcate only one based on GINI impurity
        '''
        

    def _check_iter_block(X_tree, Y_tree, final_blocks=[]):
        '''Check block iteratively until:
        - a densely-associated block is found
        - X_tree and Y_tree are leaves
        Append densely-associated blocks to final_blocks array
        '''
        X_features = X_tree.pre_order()
        Y_features = Y_tree.pre_order()
        block_fdr_reject = fdr_reject_table[X_features,:][:,Y_features]
        if _is_densely_associated(block_fdr_reject):
            final_blocks.append([X_features, Y_features])
            return
        if X_tree.is_leaf() and Y_tree.is_leaf(): return
        if X_tree.is_leaf():
            X_trees, Y_trees = [X_tree], _bifurcate(Y_tree)
        elif Y_tree.is_leaf():
            X_trees, Y_trees = _bifurcate(X_tree), [Y_tree] 
        X_trees, Y_trees = _bifurcate_one(X_tree, Y_tree)
        for x, y in itertools.product(X_trees, Y_trees):
            _check_iter_block(X_tree, Y_tree, final_blocks)

    final_blocks = []
    _check_iter_block(X.tree, Y.tree, final_blocks)
    return(final_blocks)

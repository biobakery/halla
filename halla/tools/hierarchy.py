from .utils.similarity import get_similarity_function, similarity2distance

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd
import numpy as np

# TODO: if turns out we only need tree - no need to have this class
class HierarchicalTree(object):
    def __init__(self, matrix, pdist_metric, linkage_method):
        '''Args:
        - matrix        : a pandas DataFrame object or a mxn array
        - pdist_metric  : the pairwise distance metric
        - linkage_method: the hierarchical linkage method
        '''
        self.distance_matrix = similarity2distance(spd.pdist(matrix, metric=get_similarity_function(pdist_metric)), pdist_metric)
        self.distance_matrix_sqr = spd.squareform(self.distance_matrix)
        self._generate_hierarchical_clusters(linkage_method)
    
    '''Private functions
    '''
    def _generate_hierarchical_clusters(self, linkage_method):
        # perform hierarchical clustering
        self.linkage = sch.linkage(self.distance_matrix, method=linkage_method)
        self.tree = sch.to_tree(self.linkage)
    
    '''Public functions
    '''
    def get_clust_indices(self):
        return(self.tree.pre_order())

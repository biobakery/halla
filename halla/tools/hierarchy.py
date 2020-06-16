from .config_loader import config
from .utils.similarity import get_similarity_function, similarity2distance

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd
import numpy as np

# TODO: if turns out we only need tree - no need to have this class
class HierarchicalTree(object):
    def __init__(self, matrix):
        '''Args:
        - matrix       : a pandas DataFrame object or a mxn array
        - feature_names: the names of the features in the matrix in an array
        '''
        conf = config.hierarchy
        self.distance_matrix = similarity2distance(spd.pdist(matrix, metric=get_similarity_function(conf['pdist_metric'])), conf['pdist_metric'])
        self.distance_matrix_sqr = spd.squareform(self.distance_matrix)
        self._generate_hierarchical_clusters()
    
    '''Private functions
    '''
    def _generate_hierarchical_clusters(self):
        # perform hierarchical clustering
        self.linkage = sch.linkage(self.distance_matrix, method=config.hierarchy['linkage_method'])
        self.tree = sch.to_tree(self.linkage)
    
    '''Public functions
    '''
    def get_clust_indices(self):
        return(self.tree.pre_order())

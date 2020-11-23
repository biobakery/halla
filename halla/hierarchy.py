from .utils.similarity import get_similarity_function, similarity2distance

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd
import numpy as np

class HierarchicalTree(object):
    def __init__(self, matrix, pdist_metric, linkage_method, sim2dist_set_abs=True, sim2dist_func=None):
        '''Args:
        - matrix        : a pandas DataFrame object or a mxn array
        - pdist_metric  : the pairwise distance metric
        - linkage_method: the hierarchical linkage method
        '''
        if pdist_metric == 'xicor':
            treemetric = get_similarity_function('symmetric_xicor')
        else:
            treemetric = get_similarity_function(pdist_metric)

        self.distance_matrix = similarity2distance(spd.pdist(matrix, metric=treemetric),
                                                   sim2dist_set_abs,
                                                   sim2dist_func)
        self.distance_matrix = np.clip(self.distance_matrix, a_min=0, a_max=None)
        self.distance_matrix_sqr = spd.squareform(self.distance_matrix)
        self._generate_hierarchical_clusters(linkage_method)

    '''Private functions
    '''
    def _generate_hierarchical_clusters(self, linkage_method):
        # perform hierarchical clustering
        self.linkage = sch.linkage(self.distance_matrix, method=linkage_method)
        self.tree = sch.to_tree(self.linkage)

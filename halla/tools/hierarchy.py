from .config_loader import config
from .utils.distance import get_distance_function

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd

# TODO: if turns out we only need tree - no need to have this class
class HierarchicalTree(object):
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
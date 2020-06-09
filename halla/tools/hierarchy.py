from .config_loader import config
from .utils.similarity import get_similarity_function, similarity2distance

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd

# TODO: if turns out we only need tree - no need to have this class
class HierarchicalTree(object):
    def __init__(self, matrix, feature_names):
        conf = config.hierarchy
        self.distance_matrix = similarity2distance(spd.pdist(matrix, metric=get_similarity_function(conf['pdist_metric'])), conf['pdist_metric'])
        self.distance_matrix_sqr = spd.squareform(self.distance_matrix)
        self.feature_names = feature_names
        self.generate_hierarchical_clusters()
    
    def generate_hierarchical_clusters(self):
        # perform hierarchical clustering
        Z = sch.linkage(self.distance_matrix, method=config.hierarchy['linkage_method'])
        self.tree = sch.to_tree(Z)
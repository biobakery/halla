from .config_loader import config

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd

class Hierarchy(object):
    def __init__(self, matrix):
        self.matrix = matrix
        self.distance_matrix = spd.pdist(matrix, metric=config.hierarchy['distance_metric'])
        self.distance_matrix_sqr = spd.squareform(self.distance_matrix)
        self.generate_hierarchical_clusters()
    
    def generate_hierarchical_clusters(self):
        # perform hierarchical clustering
        Z = sch.linkage(self.distance_matrix, method=config.hierarchy['linkage_method'])
        self.tree = sch.to_tree(Z)
        self.dendrogram = sch.dendrogram(Z)
    
    def get_tree(self):
        return(self.tree)
    
    def get_leaves(self):
        return(self.dendrogram['leaves'])
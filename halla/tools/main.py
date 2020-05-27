from .config_loader import config
from .hierarchy import Hierarchy
from .data_utils import preprocess, eval_type, is_all_cont
from .distance_metrics import get_distance_function

import pandas as pd
import numpy as np
import scipy.spatial.distance as spd

def update_config(attribute, **args):
    vals = getattr(config, attribute)
    for key in args:
        if key not in vals:
            raise KeyError('%s not found in config.%s' % (key, attribute))
        vals[key] = args[key]
    print('Updating config.%s to:' % attribute, vals)
    setattr(config, attribute, vals)

class HAllA(object):
    def __init__(self, pdist_metric='euclidean', pdist_args=None):
        # update config settings
        update_config('hierarchy', pdist_metric=pdist_metric, pdist_args=pdist_args)
        self.reset_attributes()        
    
    def reset_attributes(self):
        self.X, self.Y = None, None
        self.X_hierarchy, self.Y_hierarchy = None, None
        self.similarity_table = None
        self.pvalue_table, self.qvalue_table = None, None
        self.rank_index = None

    def load(self, X_file, Y_file=None):
        # TODO: currently assumes no missing value and header+index col are provided
        X, X_types = eval_type(pd.read_table(X_file, index_col=0))
        Y, Y_types = eval_type(pd.read_table(Y_file, index_col=0)) if Y_file \
            else (X.copy(deep=True), np.copy(X_types))

        # TODO: what are the suitable pdist_metric(s)?
        # if not all types are continuous but pdist_metric is only for continuous types
        # if not (is_all_cont(X_types) and is_all_cont(Y_types)) and config.hierarchy['pdist_metric'] == ...:
        #     raise ValueError('pdist_metric only works for continuous data but not all features are continuous')

        # filter tables by intersect columns
        intersect_cols = list(set(X.columns) & set(Y.columns))
        X, Y = X[intersect_cols], Y[intersect_cols]

        # clean and preprocess data
        self.X = preprocess(X)
        self.Y = preprocess(Y)

    def run_clustering(self):
        self.X_hierarchy = Hierarchy(self.X)
        self.Y_hierarchy = Hierarchy(self.Y)
    
    def compute_pairwise_similarities(self):
        conf = config.hierarchy
        n, m = self.X.shape[0], self.Y.shape[0]
        self.pvalue_table, self.qvalue_table = np.zeros((n, m)), np.zeros((n, m))
        self.rank_index = np.zeros((n*m, 2), dtype=int)
        X, Y = self.X.to_numpy(), self.Y.to_numpy()

        if conf['pdist_args']:
            self.similarity_table = spd.cdist(X, Y, metric=get_distance_function(conf['pdist_metric']), **conf['pdist_args'])
        else:
            self.similarity_table = spd.cdist(X, Y, metric=get_distance_function(conf['pdist_metric']))

    def run(self):
        # computing pairwise similarity matrix
        self.compute_pairwise_similarities()

        # hierarchical clustering
        self.run_clustering()
        
        # iteratively finding densely-associated blocks
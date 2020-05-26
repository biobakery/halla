from .config_loader import config
from .hierarchy import Hierarchy
from .data_utils import preprocess, eval_type, is_all_cont

import pandas as pd
import numpy as np

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
        self.X = None
        self.Y = None
    
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
        # clean and preprocess data
        self.X = preprocess(X[intersect_cols])
        self.Y = preprocess(Y[intersect_cols])
    
    def get_tables(self):
        '''Returns X and Y in pandas DataFrame
        '''
        return(self.X, self.Y)

    def run_clustering(self):
            self.X_hierarchy = Hierarchy(self.X)
            self.Y_hierarchy = Hierarchy(self.Y)

    def run(self):
        # hierarchical clustering
        self.run_clustering()
        # computing pairwise similarity matrix

        # iteratively finding densely-associated blocks
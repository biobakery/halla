from .config_loader import config
from .hierarchy import Hierarchy

import pandas as pd

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
        X = pd.read_table(X_file, index_col=0)
        Y = pd.read_table(Y_file, index_col=0) if Y_file else X.copy(deep=True)

        intersect_cols = list(set(X.columns) & set(Y.columns))
        # filter tables by intersect_cols
        self.X = X[intersect_cols]
        self.Y = Y[intersect_cols]
    
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
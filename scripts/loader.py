import pandas as pd
from os.path import join, isfile
import numpy as np
import scipy.cluster.hierarchy as sch
from re import search

from halla.utils.data import eval_type

class HAllAPartialLoader(object):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.load_datasets()
        self.load_similarity_table()
        self.load_significant_clusters()
        self.load_linkages()
    
    def load_datasets(self):
        def fix_type(df, types):
            updated_df = df.copy(deep=True)
            for row_i in range(updated_df.shape[0]):
                updated_df.iloc[row_i] = updated_df.iloc[row_i].to_numpy().astype(types[row_i])
            return(updated_df)
        self.X_ori, self.X_types = eval_type(pd.read_table(join(self.input_dir, 'X_original.tsv'), index_col=0))
        self.Y_ori, self.Y_types = eval_type(pd.read_table(join(self.input_dir, 'Y_original.tsv'), index_col=0))
        self.X = fix_type(pd.read_table(join(self.input_dir, 'X.tsv'), index_col=0), self.X_types)
        self.Y = fix_type(pd.read_table(join(self.input_dir, 'Y.tsv'), index_col=0), self.Y_types)
        self.X_feat_map = { name: i for i, name in enumerate(self.X.index.to_list()) }
        self.Y_feat_map = { name: i for i, name in enumerate(self.Y.index.to_list()) }
        self.X_features = self.X.index.to_numpy()
        self.Y_features = self.Y.index.to_numpy()
        
    def load_similarity_table(self):
        df = pd.read_table(join(self.input_dir, 'all_associations.txt'))
        with open(join(self.input_dir, 'performance.txt')) as f:
            perf_lines = f.readlines()
        self.sim_table = np.zeros((self.X.shape[0], self.Y.shape[0]))
        perf_match = [search("fdr alpha", line) for line in perf_lines]
        perf_i = [i for i,v in enumerate(perf_match) if v != None]

        fdr_thresh = float(str.split(perf_lines[perf_i[0]], ' ')[-1][:-1]) # TODO make it find the line that says fdr alpha
        self.fdr_reject_table = np.zeros((self.X.shape[0], self.Y.shape[0]))
        for row in df.to_numpy():
            x, y = self.X_feat_map[row[0]], self.Y_feat_map[row[1]]
            self.sim_table[x][y] = row[2]
            self.fdr_reject_table[x][y] = row[4] < fdr_thresh

    def load_significant_clusters(self):
        df = pd.read_table(join(self.input_dir, 'sig_clusters.txt'))
        self.significant_blocks = []
        for row in df.to_numpy():
            X_feats, Y_feats = row[1].split(';'), row[2].split(';')
            block_0 = [self.X_feat_map[feat] for feat in X_feats]
            block_1 = [self.Y_feat_map[feat] for feat in Y_feats]
            block = [block_0, block_1]
            self.significant_blocks.append(block)
    
    def load_linkages(self):
        if not isfile(join(self.input_dir, 'X_linkage.npy')):
            self.name = 'AllA'
            return
        self.name = 'HAllA'
        self.X_linkage = np.load(join(self.input_dir, 'X_linkage.npy'))
        self.Y_linkage = np.load(join(self.input_dir, 'Y_linkage.npy'))
        self.X_tree = sch.to_tree(self.X_linkage)
        self.Y_tree = sch.to_tree(self.Y_linkage)
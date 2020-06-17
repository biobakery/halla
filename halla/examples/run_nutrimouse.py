from os.path import dirname, abspath, join
import sys
import pandas as pd

sys.path.append(dirname(dirname(abspath(__file__))))

from tools import HAllA

gene_file  = join(dirname(abspath(__file__)), '../data', 'nutrimouse', 'gene.txt')
lipid_file = join(dirname(abspath(__file__)), '../data', 'nutrimouse', 'lipid.txt')

pdist_metric = 'spearman'

halla = HAllA(pdist_metric=pdist_metric, seed=123)
halla.load(lipid_file, gene_file)
halla.run()
halla.generate_hallagram(figsize=(20, 5), dendrogram_ratio=(0.05, 0.15), cbar_pos=None)
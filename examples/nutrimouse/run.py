from os.path import dirname, abspath, join
import sys
import pandas as pd
import seaborn as sns

sys.path.append(dirname(dirname(abspath(__file__))))

from halla import HAllA

gene_file  = join(dirname(abspath(__file__)), 'gene.txt')
lipid_file = join(dirname(abspath(__file__)), 'lipid.txt')

pdist_metric = 'spearman'

halla = HAllA(pdist_metric=pdist_metric, out_dir='out', seed=123, verbose=False)
halla.load(lipid_file, gene_file)
halla.run()

sns.set(font_scale = 0.7)
halla.generate_hallagram(figsize=(20, 5), dendrogram_ratio=(0.05, 0.15), cbar_pos=None)
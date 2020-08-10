from os.path import dirname, abspath, join
import sys
import seaborn as sns

from halla import HAllA

gene_file  = join(dirname(abspath(__file__)), 'gene.txt')
lipid_file = join(dirname(abspath(__file__)), 'lipid.txt')

pdist_metric = 'spearman'

halla = HAllA(pdist_metric=pdist_metric, out_dir='out', seed=123, verbose=False)
halla.load(lipid_file, gene_file)
halla.run()

sns.set(font_scale = 0.7)
halla.generate_clustermap(figsize=(35, 10), text_scale=8, dendrogram_ratio=(0.05, 0.1),
                          cbar_pos=(0, 0, .01, 1), mask=True,
                          cbar_kws={'ticklocation': 'left', 'label': 'Spearman Correlation'},
                          x_label='Gene', y_label='Lipid')
halla.generate_hallagram(figsize=(35, 10), block_border_width=1, text_scale=10,
                         x_label='Gene', y_label='Lipid', mask=True,
                         cbar_kws={'ticklocation': 'left', 'label': 'Spearman Correlation'})
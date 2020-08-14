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
halla.generate_clustermap(x_dataset_label='Lipid', y_dataset_label='Gene', cbar_label='Spearman correlation')
halla.generate_hallagram(block_border_width=2, cbar_label='Spearman correlation',
                          x_dataset_label='Lipid', y_dataset_label='Gene')

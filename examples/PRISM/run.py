from os.path import dirname, abspath, join
from halla import HAllA

X_file = join(dirname(abspath(__file__)), 'X_dataset.txt')
Y_file = join(dirname(abspath(__file__)), 'Y_dataset.txt')

pdist_metric = 'spearman'

halla = HAllA(max_freq_thresh=1, pdist_metric=pdist_metric, out_dir='prism_out', fdr_alpha=0.05, fnr_thresh=0.2, seed=123)
halla.load(X_file, Y_file)
halla.run()

halla.generate_hallagram(cbar_label='Spearman Correlation', y_dataset_label='Metabolites', x_dataset_label='Microbiomes', mask=True, output_file='hallagram_mask.png')
halla.generate_hallagram(cbar_label='Spearman Correlation', y_dataset_label='Metabolites', x_dataset_label='Microbiomes')
halla.generate_diagnostic_plot()
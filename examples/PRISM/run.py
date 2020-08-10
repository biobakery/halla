from os.path import dirname, abspath, join
import sys
import pandas as pd

sys.path.append(dirname(dirname(abspath(__file__))))

from tools import HAllA, AllA
from tools.utils.report import generate_hallagram

dir_path = '../data/PRISM'
X_file = join(dirname(abspath(__file__)), dir_path, 'X_dataset.txt') # microbes
Y_file = join(dirname(abspath(__file__)), dir_path, 'Y_dataset.txt') # metabolites
conversion_class_file = join(dirname(abspath(__file__)), dir_path, 'standards.txt')

pdist_metric = 'spearman'

test_halla = HAllA(pdist_metric=pdist_metric, out_dir='PRISM_out', seed=123)
test_halla.load(X_file, Y_file)
test_halla.run()

# retrieve the corresponding putative chemical classes - create a dict object
put_classes_df = pd.read_table(conversion_class_file, names=['feature', 'class'])
put_classes = { put_classes_df.iloc[i]['feature']: put_classes_df.iloc[i]['class'] \
                for i in range(put_classes_df.shape[0]) }
Y_classes = [put_classes[feat] for feat in test_halla.Y.index.to_numpy()]

# trim by the top 20 clusters
generate_hallagram(test_halla.significant_blocks[:20],
                   test_halla.X.index.to_numpy(), Y_classes,
                   test_halla.X_hierarchy.tree.pre_order(),
                   test_halla.Y_hierarchy.tree.pre_order(),
                   test_halla.similarity_table,
                   x_label='Metabolites', y_label='Microbiomes',
                   label_args={'labelpad': 15, 'weight': 'bold', 'fontsize': 'large'},
                   figsize=(75, 50),
                   text_scale=8, output_file='PRISM-hallagram.png')
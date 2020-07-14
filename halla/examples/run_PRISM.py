from os.path import dirname, abspath, join
import sys

sys.path.append(dirname(dirname(abspath(__file__))))

from tools import HAllA, AllA
from tools.utils.report import generate_hallagram

X_file = join(dirname(abspath(__file__)), '../data/PRISM', 'X_dataset.txt')
Y_file = join(dirname(abspath(__file__)), '../data/PRISM', 'Y_dataset.txt')

pdist_metric = 'spearman'

test_halla = HAllA(pdist_metric=pdist_metric, out_dir='local_tests/PRISM_out', seed=123)

test_halla.load(X_file, Y_file)
test_halla.run()

# trim by the top 20 clusters
generate_hallagram(test_halla.significant_blocks[:20],
                   test_halla.X.index.to_numpy(), test_halla.Y.index.to_numpy(),
                   test_halla.X_hierarchy.tree.pre_order(),
                   test_halla.Y_hierarchy.tree.pre_order(),
                   test_halla.similarity_table,
                   figsize=(50, 35),
                   text_scale=8, output_file='PRISM-init-updated.png')
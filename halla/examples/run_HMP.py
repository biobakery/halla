from os.path import dirname, abspath, join
import sys
import pandas as pd

sys.path.append(dirname(dirname(abspath(__file__))))

from tools import HAllA

dir_path = '../data/HMP'
X_file = join(dirname(abspath(__file__)), dir_path, 'abundance_by_species_stool.tsv')
Y_file = join(dirname(abspath(__file__)), dir_path, 'metadata.tsv')

pdist_metric = 'nmi'

test_halla = HAllA(pdist_metric=pdist_metric, out_dir='local_tests/HMP_out', discretize_func='equal-freq', seed=123)
test_halla.load(X_file, Y_file, remove_na=True)
test_halla.run()
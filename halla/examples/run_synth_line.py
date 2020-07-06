from os.path import dirname, abspath, join
import sys

sys.path.append(dirname(dirname(abspath(__file__))))

from tools import HAllA

X_file = join(dirname(abspath(__file__)), '../data', 'synthetic_line_80_100_50', 'X_line_80_50.txt')
Y_file = join(dirname(abspath(__file__)), '../data', 'synthetic_line_80_100_50', 'Y_line_100_50.txt')

pdist_metric = 'pearson'

test_halla = HAllA(discretize_func='equal-freq', discretize_num_bins=4,
                  pdist_metric=pdist_metric, out_dir='local_tests/out', seed=123)

test_halla.load(X_file, Y_file)
test_halla.run()
test_halla.generate_hallagram()
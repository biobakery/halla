#TODO: move to the appropriate directory

from tools import HAllA
from os.path import dirname, abspath, join

X_file = join(dirname(abspath(__file__)), 'data', 'X_5_10.txt')
Y_file = join(dirname(abspath(__file__)), 'data', 'Y_4_10.txt')

pdist_metric, pdist_args = 'minkowski', { 'p': .2 }
# pdist_metric, pdist_args = 'pearson', None

test_halla = HAllA(discretize_func='equal-freq', discretize_num_bins=4,
                  pdist_metric=pdist_metric, pdist_args=pdist_args, seed=123)

test_halla.load(X_file, Y_file)
test_halla.run()
# print(test_halla.X_hierarchy.distance_matrix)